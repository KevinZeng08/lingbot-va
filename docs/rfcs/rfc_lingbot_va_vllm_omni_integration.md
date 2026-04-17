# RFC: LingBot-VA 集成到 vllm-omni

- **版本**: 0.2
- **日期**: 2026-04-13
- **状态**: Draft
- **作者**: Kevin Zeng

---

## 1. 背景与动机

LingBot-VA 是一个基于 Wan2.2 改造的自回归扩散机器人控制模型，通过双流 MoT（Mixture-of-Transformers）架构同时生成视频预测和动作序列。当前推理依赖自定义的 `VA_Server` + FSDP 分布式，缺乏高效的工程化推理框架。

[vllm-omni](https://github.com/vllm-project/vllm-omni) 是 vLLM 的多模态推理引擎，已支持多种 diffusion pipeline（Wan2.2、LTX2、StableAudio 等），并提供 TP 并行、CFG 并行、分布式 VAE 解码、OpenPI serving 等基础设施。

近期 [DreamZero PR #2162](https://github.com/vllm-project/vllm-omni/pull/2162) 成功将 DreamZero（另一个 VLA world model）集成到 vllm-omni 中，为 LingBot-VA 的集成提供了直接可参考的范本。

**目标**：将 LingBot-VA 集成到 vllm-omni，利用其分布式推理基础设施，同时保持与原始实现的数值一致性。

---

## 2. 范围

### 2.1 包含（In Scope）

- **Phase 1 — I2VA 模式**：给定初始图像 + 文本 prompt，自回归生成多 chunk 的视频 latent + 动作序列
- **Phase 2 — TP / CFG 并行优化**：ColumnParallelLinear、DistributedRMSNorm、CFGParallelMixin
- **Phase 3 — Server 模式**：WebSocket 闭环控制，对接仿真环境

### 2.2 不包含（Out of Scope）

- 训练流程迁移（`forward_train`、FlexAttention 训练 mask）
- SageAttn / SlidingTileAttn 等高级 attention 后端适配
- 量化（INT8/FP8）

### 2.3 为什么先 I2VA 再 Server？

| 维度 | I2VA 模式 | Server 模式 |
|------|----------|------------|
| **复杂度** | 低 — 单次调用，无外部交互 | 高 — 需仿真环境交互、session 管理、KV cache 生命周期 |
| **可验证性** | 强 — 固定输入固定输出，容易做数值对比 | 弱 — 依赖仿真环境，难以离线对比 |
| **代码覆盖** | 覆盖 85% 核心路径（编码、去噪循环、解码、action 归一化） | 额外覆盖 `_compute_kv_cache`、`clear_pred_cache`、obs 循环 |
| **调试难度** | 低 — 可用固定图像做 e2e 对比 | 高 — 需联调 WebSocket、仿真环境 |
| **依赖** | 无外部依赖 | 依赖 serving 层（OpenPI / WebSocket） |

**结论：先实现 I2VA 模式是正确的路径**。I2VA 是 Server 模式的子集——`_infer()` 方法在两个模式中完全共用。先把 I2VA 跑通、数值对齐，Server 模式只需额外实现 `_compute_kv_cache()`（将真实观测写入 KV cache）和 serving 层接入。

---

## 3. 设计总览

### 3.1 文件结构

参照 DreamZero PR 的组织方式，在 vllm-omni 中新建：

```
vllm_omni/diffusion/models/lingbot_va/
├── __init__.py
├── pipeline_lingbot_va.py           # 主 Pipeline（DiffusionEngine 入口）
├── state_lingbot_va.py              # 跨 forward() 的持久状态（KV cache、VAE 缓存、帧计数）
├── modeling/
│   ├── __init__.py
│   ├── lingbot_va_transformer.py    # 双流 MoT Transformer（TP 适配）
│   ├── streaming_vae.py             # WanVAEStreamingWrapper（流式编码）
│   └── scheduler.py                 # FlowMatchScheduler（从原始代码移植）
└── utils.py                         # get_mesh_id、data_seq_to_patch 等工具函数
```

### 3.2 架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LingBotVAPipeline (nn.Module, CFGParallelMixin)       │
│                                                                         │
│  ┌──────────┐ ┌──────────────┐ ┌───────────────────────────────────┐   │
│  │Tokenizer │ │ TextEncoder  │ │   LingBotVATransformer3DModel     │   │
│  │(T5)      │ │ (UMT5)      │ │   (双流 MoT, 30层, TP-parallel)    │   │
│  └──────────┘ └──────────────┘ └───────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────┐ ┌─────────────────────────────────────┐  │
│  │StreamingVAE (full-res)  │ │ StreamingVAE (half-res, T-shape)   │  │
│  │ encode_chunk / clear     │ │ (可选, env_type=robotwin_tshape)    │  │
│  └──────────────────────────┘ └─────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────┐ ┌──────────────────────────┐             │
│  │ FlowMatchScheduler      │ │ FlowMatchScheduler       │             │
│  │ (video, shift=5.0)      │ │ (action, shift=1.0)      │             │
│  └──────────────────────────┘ └──────────────────────────┘             │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ LingBotVAState                                                    │  │
│  │  - kv_cache (transformer slot pool)                               │  │
│  │  - streaming_vae caches (encoder causal conv state)               │  │
│  │  - init_latent, frame_st_id, prompt_embeds                        │  │
│  │  - action_mask, norm_stats                                        │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ DistributedAutoencoderKLWan (vllm-omni, 仅解码)                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 关键模块设计

### 4.1 Pipeline — `LingBotVAPipeline`

#### 职责
- 拥有所有子组件（tokenizer、text_encoder、vae、streaming_vae、transformer、schedulers）
- 实现 `forward(req: OmniDiffusionRequest) -> DiffusionOutput`
- 管理双循环（video → action）× 多 chunk 自回归
- 提供 `load_weights()` 做权重映射

#### forward() 流程（I2VA 模式）

```
forward(req):
  1. 从 req 中提取 prompt、images
  2. encode_prompt → prompt_embeds, negative_prompt_embeds
  3. _encode_obs(images) → init_latent    # 通过 streaming_vae
  4. state.reset()                         # 清理 KV cache + VAE 缓存
  5. transformer.create_empty_cache(...)   # 初始化 slot 池
  6. for chunk_id in range(num_chunks):
       actions, latents = _infer_one_chunk(init_latent, frame_st_id)
       all_latents.append(latents)
       all_actions.append(actions)
  7. state.cleanup()                       # 清理 KV cache + streaming VAE
  8. full_latent = cat(all_latents)
  9. video = vae.decode(full_latent)       # 使用分布式 VAE 解码
  10. return DiffusionOutput(output={"video": video, "actions": actions_np})
```

#### _infer_one_chunk() 流程

```
_infer_one_chunk(init_latent, frame_st_id):
  # --- 阶段 1: Video 去噪 (25 步) ---
  latents = randn(1, 48, frame_chunk_size, H, W)
  video_scheduler.set_timesteps(num_inference_steps)
  timesteps = pad(video_scheduler.timesteps, extra_one_step)
  for i, t in enumerate(timesteps):
      last_step = (i == len(timesteps) - 1)
      input_dict = _prepare_latent_input(latents, None, t, frame_st_id)
      input_dict = _repeat_input_for_cfg(input_dict)
      noise_pred = predict_noise_maybe_with_cfg(
          positive_kwargs={...input_dict, action_mode=False},
          negative_kwargs={...},
          update_cache=1 if last_step else 0
      )
      if not last_step:
          noise_pred = data_seq_to_patch(noise_pred) → CFG combine
          latents = video_scheduler.step(noise_pred, t, latents)
      latents[:,:,0:1] = cond if frame_st_id==0 else latents[:,:,0:1]

  # --- 阶段 2: Action 去噪 (50 步) ---
  actions = randn(1, action_dim, frame_chunk_size, action_per_frame, 1)
  action_scheduler.set_timesteps(action_num_inference_steps)
  action_timesteps = pad(action_scheduler.timesteps, extra_one_step)
  for i, t in enumerate(action_timesteps):
      last_step = (i == len(action_timesteps) - 1)
      input_dict = _prepare_latent_input(None, actions, t, frame_st_id)
      input_dict = _repeat_input_for_cfg(input_dict)
      noise_pred = predict_noise_maybe_with_cfg(
          positive_kwargs={...input_dict, action_mode=True},
          negative_kwargs={...},
          update_cache=1 if last_step else 0
      )
      if not last_step:
          noise_pred = rearrange → CFG combine
          actions = action_scheduler.step(noise_pred, t, actions)
      actions[:,:,0:1] = cond if frame_st_id==0 else actions[:,:,0:1]

  return postprocess_action(actions), latents
```

#### 与 DreamZero 的关键区别

| 维度 | DreamZero | LingBot-VA |
|------|-----------|------------|
| 去噪 | 单循环（video + action 同步，transformer 一次调用同时输出两者） | **双循环串行**（先跑完 video 25 步，再跑 action 50 步） |
| Chunk | 单 chunk per forward() | 多 chunk 自回归（10 chunk，在 forward() 内循环） |
| Scheduler | FlowUniPCMultistep (两个 copy) | 自定义 FlowMatchScheduler（欧拉法，extra_one_step） |
| KV Cache | 简单 torch.cat 拼接 | Slot 分配池 + LRU 淘汰 + update_cache 模式 0/1/2 |
| CFG | video 做 CFG，action 只取 positive | video 和 action 各自独立 guidance_scale |
| Transformer forward | 单次调用返回 (video_pred, action_pred) | 按 `action_mode` flag 分两次调用 |

**影响**：不能直接使用 DreamZero 的 `VideoActionScheduler` wrapper；需要在 pipeline 内部自行管理两个串行循环。`predict_noise()` 每次只返回一种输出（video 或 action），而非 tuple。

### 4.2 State — `LingBotVAState`

借鉴 DreamZero 的 `state_dreamzero.py`，将所有跨 forward() 调用的状态集中管理：

```python
class LingBotVAState:
    """跨 forward() 调用的持久状态"""

    def __init__(self):
        self.reset()

    def reset(self):
        # 推理状态
        self.frame_st_id: int = 0
        self.init_latent: torch.Tensor | None = None
        self.prompt_embeds: torch.Tensor | None = None
        self.negative_prompt_embeds: torch.Tensor | None = None

        # 推理参数（从 config 初始化）
        self.action_mask: torch.Tensor | None = None
        self.actions_q01: torch.Tensor | None = None
        self.actions_q99: torch.Tensor | None = None

        # 环境布局参数
        self.latent_height: int = 0
        self.latent_width: int = 0

        # 标记是否需要重新初始化
        self.is_initialized: bool = False

    def should_reset(self, prompt: str | None) -> bool:
        """判断是否需要重置（prompt 变化、session 切换等）"""
        ...

    def cleanup_caches(self, transformer, streaming_vae, streaming_vae_half=None):
        """清理所有缓存"""
        transformer.clear_cache("pos")
        streaming_vae.clear_cache()
        if streaming_vae_half:
            streaming_vae_half.clear_cache()
```

### 4.3 Transformer — `LingBotVATransformer3DModel`

#### 设计决策：独立实现 vs 复用 wan2_2_transformer

**选择独立实现**（与 DreamZero 的 `CausalWanModel` 相同策略），理由：
1. 双流结构（action_embedder + action_proj_out + condition_embedder_action）是结构性差异
2. 自定义 KV Cache（slot 池、allocate_slots、restore_cache）与 vllm-omni 的 Attention 后端不兼容
3. forward() 接口不同（input_dict + action_mode vs hidden_states + timestep）

#### 从原始代码适配的要点

| 原始 (model.py) | vllm-omni 适配 |
|-----------------|--------------|
| `ConfigMixin` + `ModelMixin` | 去掉，使用 pipeline 的 `load_weights()` |
| `nn.Linear` (Q/K/V) | `ColumnParallelLinear(gather_output=False)` |
| `nn.Linear` (O/to_out) | `RowParallelLinear(input_is_parallel=True)` |
| `nn.Linear` (FFN up) | `ColumnParallelLinear` |
| `nn.Linear` (FFN down) | `RowParallelLinear` |
| `torch.nn.RMSNorm` | `DistributedRMSNorm`（全局 all_reduce + rsqrt） |
| `FeedForward` (diffusers) | 替换为 ColumnParallel + GELU + RowParallel |
| `custom_sdpa` / `flash_attn_func` | vllm-omni 的 `Attention` 层 **或** 保留原始实现 |
| `FlexAttnFunc` | **不移植**（仅训练用） |

#### KV Cache 策略

**Phase 1：保留原始 slot 分配池**。原始 `WanAttention` 的 KV cache 逻辑（`init_kv_cache`、`allocate_slots`、`update_cache`、`restore_cache`、`clear_pred_cache`）完整保留，attention 计算使用 `F.scaled_dot_product_attention` 或 `flash_attn_func`。

**Phase 2（可选）：迁移到 vllm-omni Attention 后端**。参考 DreamZero `CausalWanSelfAttention` 的做法——手动管理 K/V tensor，但使用 `vllm_omni.diffusion.attention.layer.Attention` 做实际计算。

### 4.4 Streaming VAE — `WanVAEStreamingWrapper`

直接从 `wan_va/modules/utils.py` 移植，保持原始逻辑不变：
- `encode_chunk(x_chunk)` — 利用 encoder 的 `feat_cache` 实现因果卷积流式编码
- `clear_cache()` — 重置 `feat_cache`
- 解码端使用 vllm-omni 的 `DistributedAutoencoderKLWan.decode()`

对于 `robotwin_tshape` 环境，需要两个 streaming VAE（全分辨率 + 半分辨率），在 pipeline 初始化时按 `env_type` 条件创建。

### 4.5 Scheduler — `FlowMatchScheduler`

直接从 `wan_va/utils/scheduler.py` 移植（136 行，自包含，无外部依赖）。

关键行为：
- `extra_one_step=True`：timesteps 多一步（`linspace(start, end, N+1)[:-1]`）
- `shift` 参数：video=5.0, action=1.0
- `step()` 实现单步欧拉法：`prev = sample + noise_pred * (sigma_next - sigma_curr)`

---

## 5. 框架接入

### 5.1 模型注册

```python
# registry.py
_DIFFUSION_MODELS = {
    ...
    "LingBotVAPipeline": (
        "lingbot_va",
        "pipeline_lingbot_va",
        "LingBotVAPipeline",
    ),
}

# 不需要 post/pre process func（pipeline 内部处理）
# 或可选注册一个 video processor
```

### 5.2 模型检测

```python
# hf_utils.py
def _looks_like_lingbot_va(model_name: str) -> bool:
    """检测 LingBot-VA checkpoint"""
    try:
        # LingBot-VA checkpoint 有 transformer/config.json 含 action_dim
        cfg_path = os.path.join(model_name, "transformer", "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            return "action_dim" in cfg and "patch_embedding_mlp" in str(cfg.get("_class_name", ""))
    except:
        pass
    return False

# is_diffusion_model() 末尾添加:
return ... or _looks_like_lingbot_va(model_name)
```

### 5.3 输出格式

DreamZero PR 已将 `DiffusionOutput.output` 扩展为 `torch.Tensor | tuple | dict`，且 `DiffusionEngine` 已支持从 dict 中提取 `actions` 字段。直接使用：

```python
return DiffusionOutput(
    output={
        "video": decoded_video,           # [B, C, F, H, W] tensor
        "actions": actions_np,            # numpy array
    }
)
```

### 5.4 CFG 并行

继承 `CFGParallelMixin`，覆写：

```python
class LingBotVAPipeline(nn.Module, CFGParallelMixin):

    def predict_noise(self, **kwargs) -> torch.Tensor:
        """单模态输出（video 或 action，由 action_mode 决定）"""
        return self.transformer(
            input_dict=kwargs["input_dict"],
            update_cache=kwargs.get("update_cache", 0),
            cache_name="pos",
            action_mode=kwargs["action_mode"],
        )

    def combine_cfg_noise(self, positive, negative, cfg_scale, cfg_normalize=False):
        """根据当前模态选择对应的 guidance_scale"""
        return super().combine_cfg_noise(positive, negative, cfg_scale, cfg_normalize)
```

**注意**：LingBot-VA 的 video CFG 和 action CFG 使用不同的 `guidance_scale`（video=5, action=1），需要在调用 `predict_noise_maybe_with_cfg` 时传入对应的 `true_cfg_scale`。

### 5.5 权重加载

> ⚠️ **前置要求（与 README 保持一致）**
>
> 推理/评测前，必须将 `<model_path>/transformer/config.json` 中的 `attn_mode` 从 `"flex"` 改为 `"flashattn"`（或 `"torch"`）。
> `"flex"` 仅用于训练，若不修改会在推理阶段报错。

LingBot-VA 的 checkpoint 布局为标准 diffusers 格式（与 Wan2.2 基本兼容）：

```
lingbot-va-posttrain-robotwin/
├── text_encoder/         # UMT5 权重
├── tokenizer/            # T5TokenizerFast
├── transformer/          # WanTransformer3DModel (含 action_* 新增键)
│   ├── config.json
│   └── diffusion_pytorch_model-*.safetensors
└── vae/                  # AutoencoderKLWan
```

权重映射要点：
- **大部分键名与标准 Wan2.2 兼容**：`blocks.*.attn1.*`, `blocks.*.attn2.*`, `blocks.*.ffn.*`, `condition_embedder.*`, `norm_out.*`, `proj_out.*`
- **新增键**（需特殊处理）：
  - `patch_embedding_mlp.weight/bias` — 原始为 `nn.Linear`，需映射到 `ColumnParallelLinear`（若做 TP）或保留原始
  - `action_embedder.weight/bias`
  - `action_proj_out.weight/bias`
  - `condition_embedder_action.*` — 完整的 `WanTimeTextImageEmbedding` 副本
  - `scale_shift_table` — 保持不变

```python
def load_weights(self, weights):
    # text_encoder: 直接加载（标准 UMT5）
    # vae: 直接加载（标准 AutoencoderKLWan）
    # transformer: 逐键映射
    # 注意: 加载前需确保 transformer/config.json 的 attn_mode != "flex"
    for name, tensor in weights:
        if name.startswith("transformer."):
            # 处理 TP 分片
            new_name = name  # 键名基本兼容
            if new_name in params:
                default_weight_loader(params[new_name], tensor)
```

---

## 6. Serving 层（Phase 3）

### 6.1 Server 模式额外需要的组件

| 组件 | 说明 |
|------|------|
| `_compute_kv_cache(obs)` | 将真实观测编码并写入 KV cache（`update_cache=2`） |
| `clear_pred_cache()` | 清理预测生成的 KV cache，保留观测写入的 |
| `frame_st_id` 管理 | 每次观测写入后递增 |
| WebSocket serving | 对接 OpenPI 或自定义 WebSocket 协议 |
| Transform | 处理 RoboTwin 的 obs schema（多相机、T-shape 布局） |

### 6.2 复用 DreamZero 的 serving 架构

DreamZero PR 新增了完整的 serving 层：
- `openpi_connection.py` — WebSocket 协议（msgpack 编解码）
- `openpi_serving.py` — `ServingRealtimeRobotOpenPI`（obs → transform → engine → actions）
- `transform/base.py` — 抽象 transform 接口

LingBot-VA 的 Server 模式可以：
1. 复用 `openpi_connection.py`（WebSocket 协议层）
2. 复用 `openpi_serving.py`（serving 调度层）
3. 新增 `transform/robotwin.py`（处理 RoboTwin 的 obs schema、多相机 T-shape 布局）

### 6.3 Server 模式的 forward() 扩展

```python
def forward(self, req, **kwargs):
    extra_args = req.sampling_params.extra_args or {}

    if extra_args.get("reset"):
        self.state.reset()
        self._initialize_state(prompt=extra_args.get("prompt"))
        return DiffusionOutput(output={"actions": empty_array})

    elif extra_args.get("compute_kv_cache"):
        obs = extra_args["obs"]
        self._compute_kv_cache(obs)
        return DiffusionOutput(output={"actions": empty_array})

    else:  # infer one chunk
        obs = extra_args.get("obs")
        actions, latents = self._infer_one_chunk(obs, self.state.frame_st_id)
        return DiffusionOutput(output={"actions": actions})
```

---

## 7. 需要从 LingBot-VA 移植的文件清单

| 源文件 | 目标文件 | 说明 | 修改量 |
|--------|---------|------|-------|
| `wan_va/modules/model.py` → `WanAttention`, `WanTransformerBlock`, `WanTransformer3DModel` | `modeling/lingbot_va_transformer.py` | 去掉 ConfigMixin/ModelMixin，TP 适配 | 大 |
| `wan_va/modules/utils.py` → `WanVAEStreamingWrapper` | `modeling/streaming_vae.py` | 包装 vllm-omni 的 VAE 实例 | 小 |
| `wan_va/utils/scheduler.py` → `FlowMatchScheduler` | `modeling/scheduler.py` | 直接复制 | 无 |
| `wan_va/utils/utils.py` → `get_mesh_id`, `data_seq_to_patch` | `utils.py` | 直接复制 | 无 |
| `wan_va/wan_va_server.py` → `_infer()`, `_encode_obs()`, `_prepare_latent_input()`, `_repeat_input_for_cfg()`, `preprocess/postprocess_action()`, `encode_prompt()` | `pipeline_lingbot_va.py` | 适配 vllm-omni 接口 | 中 |
| `wan_va/configs/va_robotwin_cfg.py` | 配置映射 | `norm_stat`, `used_action_channel_ids` 等 | 小 |

---

## 8. 配置映射

LingBot-VA 使用 `EasyDict` 配置，需要映射到 vllm-omni 的 `OmniDiffusionConfig.model_config` dict：

```python
# vllm serve 启动参数中通过 model_config 传入:
vllm serve ./lingbot-va-posttrain-robotwin \
  --omni \
  --model-config '{
    "env_type": "robotwin_tshape",
    "height": 256,
    "width": 320,
    "action_dim": 30,
    "action_per_frame": 16,
    "frame_chunk_size": 2,
    "num_chunks_to_infer": 10,
    "num_inference_steps": 25,
    "action_num_inference_steps": 50,
    "guidance_scale": 5.0,
    "action_guidance_scale": 1.0,
    "snr_shift": 5.0,
    "action_snr_shift": 1.0,
    "attn_window": 72,
    "obs_cam_keys": ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"],
    "used_action_channel_ids": [0,1,2,3,4,5,6,28,7,8,9,10,11,12,13,29],
    "action_norm_method": "quantiles",
    "norm_stat": { "q01": [...], "q99": [...] },
    "infer_mode": "i2va"
  }'
```

Pipeline 从 `od_config.model_config` 读取这些参数。

---

## 9. 测试策略

### 9.1 数值一致性测试（最关键）

```
1. 用原始 VA_Server 在 robotwin_i2va 配置下生成:
   - latents_0.pt ~ latents_9.pt (每 chunk 的 video latent)
   - actions_0.pt ~ actions_9.pt (每 chunk 的 action tensor)
   - demo.mp4 (解码视频)

2. 用 vllm-omni LingBotVAPipeline 在相同配置、相同随机种子下生成

3. 对比:
   - latent MSE < 1e-5 (bf16 精度)
   - action MSE < 1e-5
   - 视频帧 PSNR > 40dB
```

### 9.2 单元测试

| 测试 | 验证点 |
|------|-------|
| Transformer forward (video mode) | 输出 shape、dtype、无 NaN |
| Transformer forward (action mode) | 输出 shape、dtype、无 NaN |
| KV cache init → update → clear → restore | slot 分配正确、mask 状态正确 |
| StreamingVAE encode_chunk | 输出与非流式编码一致 |
| FlowMatchScheduler step | 与原始实现数值一致 |
| Pipeline e2e (1 chunk, 2 steps) | 端到端 smoke test |

### 9.3 集成测试

| 测试 | 验证点 |
|------|-------|
| `vllm serve` 启动 + API 请求 | 服务能正常启动和响应 |
| TP=2 数值对比 | 与 TP=1 的 action MSE < 1e-3 |
| CFG parallel 对比 | 与非 parallel 的 action MSE < 1e-6 |

---

## 10. 实施计划

### Phase 1: I2VA 最小可用（3-5 天）

| # | 任务 | 预计时间 |
|---|------|---------|
| 1.1 | 创建目录骨架 + `__init__.py` | 0.5 天 |
| 1.2 | 移植 `FlowMatchScheduler`、`get_mesh_id`、`data_seq_to_patch` | 0.5 天 |
| 1.3 | 移植 `WanVAEStreamingWrapper` | 0.5 天 |
| 1.4 | 实现 `LingBotVATransformer3DModel`（非 TP，保留原始 nn.Linear + KV cache） | 1 天 |
| 1.5 | 实现 `LingBotVAState` | 0.5 天 |
| 1.6 | 实现 `LingBotVAPipeline.forward()`（I2VA 完整流程） | 1 天 |
| 1.7 | 注册模型 + 模型检测 + 权重加载 | 0.5 天 |
| 1.8 | 数值一致性对比 + 修 bug | 1 天 |

**里程碑**：单 GPU 上 `vllm serve` 跑通 I2VA，生成结果与原始 VA_Server 数值一致。

### Phase 2: TP / CFG 并行优化（1-2 周）

| # | 任务 | 预计时间 |
|---|------|---------|
| 2.1 | Transformer TP 适配（ColumnParallel / RowParallel / DistributedRMSNorm） | 2 天 |
| 2.2 | CFGParallelMixin 接入（predict_noise / combine_cfg_noise） | 1 天 |
| 2.3 | 分布式 VAE 解码对接 | 1 天 |
| 2.4 | TP=2 数值对比调试 | 2 天 |
| 2.5 | 性能基准测试（latency、throughput） | 1 天 |

**里程碑**：TP=2 + CFG parallel 下 I2VA 跑通，精度损失在可接受范围内。

### Phase 3: Server 模式（1-2 周）

| # | 任务 | 预计时间 |
|---|------|---------|
| 3.1 | 实现 `_compute_kv_cache()` + `clear_pred_cache()` 生命周期管理 | 2 天 |
| 3.2 | 新增 `transform/robotwin.py`（多相机 T-shape 布局） | 1 天 |
| 3.3 | 对接 OpenPI serving（WebSocket endpoint） | 1 天 |
| 3.4 | 仿真环境联调（RoboTwin client） | 3 天 |
| 3.5 | 闭环控制 E2E 测试 | 2 天 |

**里程碑**：通过 `vllm serve` 启动 Server 模式，RoboTwin 客户端能通过 WebSocket 连接并完成完整的操控任务。

---

## 11. 风险与注意事项

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| TP 下 RMSNorm 精度漂移 | action 精度下降 | 使用 `DistributedRMSNorm`（DreamZero 已验证可行）|
| bf16 运算顺序差异（`* inv_std` vs `/ std`） | latent 编码偏差 | 严格复用原始代码的运算顺序 |
| `Conv3dLayer` GEMM fast path | patch_embedding 数值差异 | 设置 `enable_linear=False`（参考 DreamZero）|
| slot KV cache 内存占用 | OOM 风险 | `attn_window=72` 控制 slot 池大小，约 15M params/层 |
| FlexAttention 依赖 | 编译失败 | I2VA 推理不需要 FlexAttention，**不移植** |
| 自定义 KV cache 与 vllm-omni cache_dit / tea_cache 不兼容 | 无法使用缓存加速 | 列入 `_NO_CACHE_ACCELERATION` 集合 |

---

## 12. 开放问题（Open Questions）

1. **是否需要支持多 request 并发？** — I2VA 模式下单个 request 需要 10 chunk × (25+50) = 750 个 transformer forward，耗时较长。是否需要 step-level interleave？（建议：Phase 1 不需要，后续按需评估）

2. **是否将 FlowMatchScheduler 上游贡献到 vllm-omni？** — 如果其他模型也需要此 scheduler，可以放到 `vllm_omni/diffusion/models/schedulers/` 下。

3. **半分辨率 VAE 是否需要分布式？** — `streaming_vae_half` 仅处理腕部相机的较小分辨率图像，可能不需要分布式解码。

4. **Phase 3 是否继续使用原始的 `broadcast_object_list` 分布式通信？** — vllm-omni 使用 TP 替代 FSDP，原始的 worker_loop 模式不再适用。Server 模式需要重新设计分布式通信方式。

---

## 13. 参考资料

- [DreamZero vllm-omni PR #2162](https://github.com/vllm-project/vllm-omni/pull/2162)
- [LTX2 vllm-omni PR #2160](https://github.com/vllm-project/vllm-omni/pull/2160)（VideoAudioScheduler 模式）
- [LingBot-VA 推理流程分析文档](../inference_pipeline.md)
- [LingBot-VA vllm-omni 集成方案分析](../vllm_omni_integration.md)
- [vllm-omni DiffusionEngine 源码](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/diffusion_engine.py)

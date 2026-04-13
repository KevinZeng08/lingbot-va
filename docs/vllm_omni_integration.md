# LingBot-VA 集成到 vllm-omni 方案分析

## 1. 目标

将 LingBot-VA 的 **I2VA 模式**（Image-to-Video-Action）集成到 vllm-omni 的 diffusion 推理框架中，使其能够：

1. 给定初始图像 + 文本 prompt，自回归地生成多个 chunk 的**视频 latent**和**动作序列**
2. 利用 vllm-omni 已有的分布式推理、调度、VAE 分布式解码等基础设施
3. 输出同时包含解码后的视频帧和 action 数组

> **注意**：Server 模式（闭环控制）涉及仿真环境交互和 KV Cache 的 pred/obs 生命周期管理，复杂度高于 I2VA，后续再集成。

---

## 2. 核心差异：LingBot-VA vs. 标准 Wan2.2

LingBot-VA 基于 Wan2.2 改造，但有以下**关键差异**需要在 vllm-omni 中支持：

### 2.1 模型架构差异

| 维度 | 标准 Wan2.2 | LingBot-VA |
|------|-----------|------------|
| **Transformer** | 单流（video only） | **双流 MoT**：video 流 + action 流，共享 30 层 Transformer blocks |
| **嵌入层** | `patch_embedding` | `patch_embedding_mlp`（video）+ `action_embedder`（action） |
| **投影层** | `proj_out` | `proj_out`（video）+ `action_proj_out`（action） |
| **条件嵌入** | 单个 `condition_embedder` | `condition_embedder`（video）+ `condition_embedder_action`（action） |
| **新增参数** | — | `action_dim=30` |
| **KV Cache** | 无 | 每层 self-attention 有自定义 KV Cache 池（`attn_caches`），支持 `update_cache` 模式 0/1/2 |
| **VAE 编码** | 标准 `AutoencoderKLWan` | `WanVAEStreamingWrapper`（流式编码，维护因果卷积缓存）|
| **Attention** | 无 KV Cache | 自定义 KV Cache slot 分配 + 滑动窗口淘汰 |
| **去噪流程** | 单循环（video） | **两阶段串行**：先 video 去噪 → 后 action 去噪（共享 Transformer） |
| **调度器** | 单个 scheduler | 两个独立 `FlowMatchScheduler`（video: shift=5.0, action: shift=1.0） |
| **CFG** | 标准 CFG | Video CFG + Action CFG（独立的 guidance_scale） |

### 2.2 推理流程差异

标准 Wan2.2 的 `forward()` 是一个**单循环去噪**：
```
for t in timesteps:
    noise_pred = transformer(latents, t, prompt_embeds)
    latents = scheduler.step(noise_pred, t, latents)
return vae.decode(latents)
```

LingBot-VA 的 I2VA `generate()` 是一个**双循环、多 chunk 自回归**：
```
for chunk_id in range(num_chunks):
    # 阶段1: Video 去噪 (25 步)
    for t in video_timesteps:
        video_noise_pred = transformer(latents, t, ..., action_mode=False)
        latents = video_scheduler.step(video_noise_pred, t, latents)
        # 最后一步: update_cache=1 写入 Pred Video KV

    # 阶段2: Action 去噪 (50 步) — 能 attend to 已写入的 Video KV
    for t in action_timesteps:
        action_noise_pred = transformer(actions, t, ..., action_mode=True)
        actions = action_scheduler.step(action_noise_pred, t, actions)
        # 最后一步: update_cache=1 写入 Pred Action KV

# 拼接所有 chunk 的 latent → VAE 解码
return vae.decode(all_latents), all_actions
```

---

## 3. 集成方案

### 3.1 文件结构

在 vllm-omni 中新建如下文件：

```
vllm_omni/diffusion/models/lingbot_va/
├── __init__.py
├── pipeline_lingbot_va.py              # 主 Pipeline（对标 pipeline_wan2_2.py）
├── lingbot_va_transformer.py           # 双流 Transformer（对标 wan2_2_transformer.py）
└── streaming_vae.py                    # 流式 VAE 包装器
```

### 3.2 注册模型

在 `vllm_omni/diffusion/registry.py` 中添加：

```python
_DIFFUSION_MODELS = {
    # ... existing entries ...
    "LingBotVAPipeline": (
        "lingbot_va",
        "pipeline_lingbot_va",
        "LingBotVAPipeline",
    ),
}

_DIFFUSION_POST_PROCESS_FUNCS = {
    # ...
    "LingBotVAPipeline": "get_lingbot_va_post_process_func",
}

_DIFFUSION_PRE_PROCESS_FUNCS = {
    # ...
    "LingBotVAPipeline": "get_lingbot_va_pre_process_func",
}
```

### 3.3 Transformer 模型适配

#### 3.3.1 需要解决的问题

LingBot-VA 的 `WanTransformer3DModel` 与 vllm-omni 中已有的 `WanTransformer3DModel` 有以下差异：

1. **新增模块**：`action_embedder`, `action_proj_out`, `condition_embedder_action`（独立的时间步嵌入）
2. **KV Cache 自定义实现**：LingBot-VA 用自定义的 slot 分配 / LRU 淘汰管理 KV Cache，而 vllm-omni 有自己的 `Attention` 后端（FlashAttn / SDPA 等）
3. **forward 签名不同**：LingBot-VA 接收 `input_dict` + `action_mode` 标志，vllm-omni 接收 `hidden_states` + `timestep` + `encoder_hidden_states`
4. **双次调用**：每个去噪步中 Transformer 只被调用一次（video 或 action 二选一），但 video 和 action 去噪是串行的两个循环

#### 3.3.2 推荐方案：独立实现 `LingBotVATransformer3DModel`

**不复用** vllm-omni 的 `wan2_2_transformer.py`，而是独立实现一个新的 Transformer 类。原因：

- LingBot-VA 的 KV Cache 逻辑（slot 分配、`update_cache` 模式、`clear_pred_cache`）与 vllm-omni 的 Attention 后端完全不兼容
- 双流设计（action_embedder + action_proj_out）是结构性变化，不是简单参数差异
- I2VA 模式仍然需要 KV Cache 跨 chunk 传递（自回归），不能简单去掉

```python
# lingbot_va_transformer.py

class LingBotVATransformer3DModel(nn.Module):
    """
    双流 MoT Transformer，从 LingBot-VA 原始代码适配。

    关键修改（相对于原始 lingbot-va/wan_va/modules/model.py）：
    1. 去掉 ConfigMixin/ModelMixin，使用 vllm-omni 的权重加载模式
    2. 使用 vllm-omni 的 ColumnParallelLinear / RowParallelLinear 实现 TP
    3. 保留自定义 KV Cache 逻辑（init/update/clear/restore）
    4. forward() 保持原始的 input_dict + action_mode 接口
    """

    def __init__(self, ...):
        # 保留原始结构：
        # - patch_embedding_mlp, action_embedder
        # - condition_embedder, condition_embedder_action
        # - blocks (WanTransformerBlock × 30)
        # - proj_out, action_proj_out
        # - 自定义 KV Cache (WanAttention.attn_caches)
        ...

    def forward(self, input_dict, update_cache=0, cache_name="pos",
                action_mode=False, train_mode=False):
        # 保持原始逻辑
        ...
```

**权重映射**：LingBot-VA 的权重键名与标准 Wan2.2 大部分兼容（共享 `blocks.*`、`patch_embedding_mlp`、`condition_embedder` 等），新增的键为：
- `action_embedder.weight`, `action_embedder.bias`
- `action_proj_out.weight`, `action_proj_out.bias`
- `condition_embedder_action.*`

需要在 `load_weights` 中处理这些新增键的加载。

### 3.4 Pipeline 实现

#### 3.4.1 `LingBotVAPipeline` 类

```python
# pipeline_lingbot_va.py

class LingBotVAPipeline(nn.Module, CFGParallelMixin, ProgressBarMixin):
    """
    LingBot-VA I2VA Pipeline for vllm-omni.

    输入：初始图像 + 文本 prompt
    输出：DiffusionOutput（包含视频 tensor + action 数组）
    """

    support_image_input = True    # 需要输入初始观测图像
    color_format = "RGB"

    def __init__(self, *, od_config: OmniDiffusionConfig):
        super().__init__()
        self.od_config = od_config

        # 加载组件（与 Wan22Pipeline 类似）
        self.tokenizer = AutoTokenizer.from_pretrained(...)
        self.text_encoder = UMT5EncoderModel.from_pretrained(...)
        self.vae = DistributedAutoencoderKLWan.from_pretrained(...)

        # 流式 VAE 包装器
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        # 如果是 robotwin_tshape 环境，还需要加载半分辨率 VAE
        self.streaming_vae_half = None  # 按 env_type 条件加载

        # LingBot-VA 双流 Transformer
        self.transformer = LingBotVATransformer3DModel(...)

        # 双调度器
        self.video_scheduler = FlowMatchScheduler(shift=5.0, ...)
        self.action_scheduler = FlowMatchScheduler(shift=1.0, ...)

        # LingBot-VA 特有配置
        self.frame_chunk_size = 2
        self.action_per_frame = 16
        self.action_dim = 30
        self.num_chunks_to_infer = 10
        # ... 更多配置从 od_config 读取

    def forward(self, req: OmniDiffusionRequest, ...) -> DiffusionOutput:
        """I2VA 模式的完整推理流程"""

        # 1. 编码文本 prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(...)

        # 2. 编码初始观测图像 → init_latent
        init_latent = self._encode_obs(image)

        # 3. 初始化 KV Cache
        self.transformer.create_empty_cache(...)

        # 4. 多 chunk 自回归推理
        all_latents = []
        all_actions = []
        for chunk_id in range(self.num_chunks_to_infer):
            frame_st_id = chunk_id * self.frame_chunk_size
            actions, latents = self._infer_one_chunk(
                init_latent, prompt_embeds, negative_prompt_embeds,
                frame_st_id
            )
            all_latents.append(latents)
            all_actions.append(actions)

        # 5. 拼接所有 chunk
        full_latent = torch.cat(all_latents, dim=2)
        full_action = torch.cat(all_actions, dim=1)

        # 6. 清理 KV Cache
        self.transformer.clear_cache("pos")
        self.streaming_vae.clear_cache()

        # 7. VAE 解码 (利用 vllm-omni 的分布式 VAE)
        video = self._decode_latent(full_latent)

        return DiffusionOutput(
            output=video,
            custom_output={
                "action": full_action.cpu().numpy(),
                "latent": full_latent if output_type == "latent" else None,
            }
        )

    def _infer_one_chunk(self, init_latent, prompt_embeds,
                          negative_prompt_embeds, frame_st_id):
        """
        推理一个 chunk，包含：
        1. Video 去噪循环 (25 步)
        2. Action 去噪循环 (50 步)

        直接复用 wan_va_server.py 中 _infer() 的逻辑。
        """
        # 采样随机噪声
        latents = torch.randn(1, 48, self.frame_chunk_size, ...)
        actions = torch.randn(1, self.action_dim, self.frame_chunk_size, ...)

        # === Video 去噪 ===
        self.video_scheduler.set_timesteps(self.num_inference_steps)
        for i, t in enumerate(self.video_scheduler.timesteps):
            last_step = (i == len(timesteps) - 1)
            input_dict = self._prepare_latent_input(latents, None, t, ...)
            input_dict = self._repeat_input_for_cfg(input_dict)

            video_noise_pred = self.transformer(
                input_dict, update_cache=1 if last_step else 0,
                action_mode=False
            )

            if not last_step:
                # CFG + scheduler step
                latents = self.video_scheduler.step(noise_pred, t, latents)

        # === Action 去噪 ===
        self.action_scheduler.set_timesteps(self.action_num_inference_steps)
        for i, t in enumerate(self.action_scheduler.timesteps):
            last_step = (i == len(timesteps) - 1)
            input_dict = self._prepare_latent_input(None, actions, t, ...)
            input_dict = self._repeat_input_for_cfg(input_dict)

            action_noise_pred = self.transformer(
                input_dict, update_cache=1 if last_step else 0,
                action_mode=True
            )

            if not last_step:
                actions = self.action_scheduler.step(noise_pred, t, actions)

        actions = self.postprocess_action(actions)
        return actions, latents
```

### 3.5 FlowMatchScheduler 适配

vllm-omni 已有 `FlowUniPCMultistepScheduler`，但 LingBot-VA 使用自定义的 `FlowMatchScheduler`（单步欧拉法），两者不兼容：

- LingBot-VA 的 scheduler 支持 `extra_one_step=True`（多一步以确保完全去噪到 t=0）
- `snr_shift` 对 video 和 action 分别设置

**推荐**：将 LingBot-VA 的 `FlowMatchScheduler` 直接复制到 `vllm_omni/diffusion/models/lingbot_va/` 下，不修改其逻辑。

### 3.6 流式 VAE 编码器

LingBot-VA 使用 `WanVAEStreamingWrapper` 实现逐 chunk 编码。vllm-omni 的 `DistributedAutoencoderKLWan` 支持分布式解码但不支持流式编码。

**方案**：
1. 编码端使用 LingBot-VA 原始的 `WanVAEStreamingWrapper`（包装 vllm-omni 的 VAE 实例）
2. 解码端使用 vllm-omni 已有的 `DistributedAutoencoderKLWan.decode()`

```python
# streaming_vae.py - 直接从 lingbot-va 移植
class WanVAEStreamingWrapper:
    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv
        # ... 缓存管理 ...

    def encode_chunk(self, x_chunk):
        # 保持原始逻辑
        ...
```

---

## 4. 关键技术挑战

### 4.1 KV Cache 与 vllm-omni Attention 后端的兼容

**问题**：LingBot-VA 的 KV Cache 是自定义的 slot 分配池（每个 attention 层独立维护 `attn_caches` dict），而 vllm-omni 使用统一的 `Attention` 后端（FlashAttn / SDPA / SageAttn 等）。

**方案**：I2VA 模式下，**保留 LingBot-VA 原始的 KV Cache 实现**，不使用 vllm-omni 的 Attention 后端：

```python
# 在 LingBotVATransformer3DModel 中使用原始的 WanAttention
# 而非 vllm_omni.diffusion.attention.layer.Attention
class WanAttention(nn.Module):
    def __init__(self, ...):
        self.attn_op = F.scaled_dot_product_attention  # 或 flash_attn_func
        self.attn_caches = {}  # 自定义 KV Cache 池
    ...
```

这样做的**代价**是：
- 无法使用 vllm-omni 的 SageAttn / SlidingTileAttn 等优化后端
- 无法使用 vllm-omni 的 cache acceleration（tea_cache / cache_dit）

这是一个合理的权衡——I2VA 模式的 chunk 数有限（默认 10 个），KV Cache 的 slot 管理开销不大。后续可以逐步将 KV Cache 迁移到 vllm-omni 的统一后端。

### 4.2 双调度器 + 双循环

**问题**：vllm-omni 的 `StepScheduler` 假设每个 request 有一个去噪循环，但 LingBot-VA 每个 chunk 有**两个串行循环**（video + action），且多个 chunk 是自回归的。

**方案**：

**方案 A（推荐）：Pipeline 内部管理循环**

不使用 vllm-omni 的 `StepScheduler`，在 `LingBotVAPipeline.forward()` 内部自行管理所有循环。这与 `Wan22Pipeline.forward()` 的模式一致——vllm-omni 支持 pipeline 在 `forward()` 中自行跑完整个去噪流程。

```python
def forward(self, req, ...):
    for chunk_id in range(num_chunks):
        for t in video_timesteps:       # 25 步 video 去噪
            ...
        for t in action_timesteps:      # 50 步 action 去噪
            ...
    return DiffusionOutput(...)
```

**方案 B（进阶）：利用 `SupportsStepExecution` 协议**

如果后续需要 step-level 调度（如多 request 交错执行），可以实现 `SupportsStepExecution` 接口：

```python
class LingBotVAPipeline(SupportsStepExecution):
    def prepare_encode(self, state, **kwargs):
        # 编码 prompt + 初始图像 + 初始化 KV Cache
        ...
    def denoise_step(self, state, **kwargs):
        # 根据 state.phase (video/action) 和 state.step_idx 执行一步
        ...
    def step_scheduler(self, state, noise_pred, **kwargs):
        # 选择 video_scheduler 或 action_scheduler
        ...
    def post_decode(self, state, **kwargs):
        # VAE 解码 + 打包输出
        ...
```

但这需要大量的状态管理（当前 chunk_id、phase、step_idx），初始集成不建议采用。

### 4.3 输出格式扩展

**问题**：vllm-omni 的 `DiffusionOutput.output` 设计为单个 tensor（视频/图像/音频），而 LingBot-VA 同时输出 video + action。

**方案**：使用 `DiffusionOutput.custom_output` 字段传递 action：

```python
return DiffusionOutput(
    output=decoded_video,           # [B, C, F, H, W] 视频 tensor
    custom_output={
        "action": action_array,     # numpy array, shape (C, F_total, action_per_frame)
        "action_metadata": {
            "action_dim": 16,
            "action_per_frame": 16,
            "frame_chunk_size": 2,
            "num_chunks": 10,
            "norm_method": "quantiles",
        }
    }
)
```

post_process_func 中处理视频输出：

```python
def get_lingbot_va_post_process_func(od_config):
    from diffusers.video_processor import VideoProcessor
    video_processor = VideoProcessor(vae_scale_factor=1)

    def post_process_func(video, output_type="np"):
        if output_type == "latent":
            return video
        return video_processor.postprocess_video(video, output_type=output_type)

    return post_process_func
```

### 4.4 Tensor Parallelism (TP)

LingBot-VA 原始代码使用 FSDP 进行分布式推理，而 vllm-omni 使用 TP（Tensor Parallelism）。

**需要替换的层**：
- `nn.Linear` → `ColumnParallelLinear` / `RowParallelLinear`（在 attention 的 Q/K/V 投影和 FFN 中）
- `WanAttention` 中的 `to_q/to_k/to_v` → `QKVParallelLinear`
- `to_out` → `RowParallelLinear`
- `patch_embedding_mlp` → `ColumnParallelLinear`
- `action_embedder` → `ColumnParallelLinear`

可参考 vllm-omni 中已有的 `wan2_2_transformer.py` 的 TP 实现。

### 4.5 多相机 + T-Shape 布局

LingBot-VA 的 `robotwin_tshape` 环境使用 T 形 latent 布局（高相机全分辨率 + 腕部相机半分辨率拼接）。这在 `_encode_obs` 中实现，需要两个 VAE 实例（全分辨率 + 半分辨率）。

**方案**：在 pipeline 初始化时根据 `env_type` 配置加载第二个 VAE：

```python
if self.env_type == 'robotwin_tshape':
    self.streaming_vae_half = WanVAEStreamingWrapper(
        DistributedAutoencoderKLWan.from_pretrained(...)
    )
```

---

## 5. 配置映射

LingBot-VA 的 `EasyDict` 配置需要映射到 vllm-omni 的 `OmniDiffusionConfig` / `OmniDiffusionSamplingParams`：

| LingBot-VA 配置字段 | 映射到 vllm-omni | 说明 |
|---|---|---|
| `wan22_pretrained_model_name_or_path` | `od_config.model` | 模型路径 |
| `height`, `width` | `sampling_params.height/width` | 图像尺寸 |
| `num_inference_steps` | `sampling_params.num_inference_steps` | Video 去噪步数 |
| `action_num_inference_steps` | **新增字段** | Action 去噪步数 |
| `guidance_scale` | `sampling_params.guidance_scale` | Video CFG |
| `action_guidance_scale` | **新增字段** | Action CFG |
| `frame_chunk_size` | **新增字段** | 每 chunk 帧数 |
| `num_chunks_to_infer` | **新增字段** | I2VA chunk 数 |
| `action_dim` | **新增字段** | 动作维度 |
| `action_per_frame` | **新增字段** | 每帧动作子步数 |
| `attn_window` | **新增字段** | KV Cache 滑动窗口 |
| `snr_shift` | `od_config.flow_shift` | Video SNR shift |
| `action_snr_shift` | **新增字段** | Action SNR shift |
| `env_type` | **新增字段** | 环境类型（影响 VAE 布局）|
| `obs_cam_keys` | **新增字段** | 观测相机键名 |
| `norm_stat` | **新增字段** | 动作归一化统计量 |
| `used_action_channel_ids` | **新增字段** | 使用的动作通道 |
| `video_exec_step` | **新增字段** | Video 提前终止步数 |

需要在 `OmniDiffusionConfig` 或 `OmniDiffusionSamplingParams` 中扩展这些字段，或者使用一个独立的 `LingBotVAConfig` dataclass：

```python
@dataclass
class LingBotVAConfig:
    """LingBot-VA specific configuration."""
    action_num_inference_steps: int = 50
    action_guidance_scale: float = 1.0
    frame_chunk_size: int = 2
    num_chunks_to_infer: int = 10
    action_dim: int = 30
    action_per_frame: int = 16
    attn_window: int = 72
    action_snr_shift: float = 1.0
    env_type: str = "robotwin_tshape"
    obs_cam_keys: list[str] = field(default_factory=lambda: [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist"
    ])
    video_exec_step: int = -1
    norm_stat: dict = field(default_factory=dict)
    used_action_channel_ids: list[int] = field(default_factory=list)
    action_norm_method: str = "quantiles"
```

---

## 6. 实施步骤

### Phase 1：最小可用（~3-5 天）

1. **复制核心代码**：
   - 将 `wan_va/modules/model.py` 中的 `WanAttention`, `WanTransformerBlock`, `WanTransformer3DModel` 复制到 `lingbot_va_transformer.py`，去掉 `ConfigMixin/ModelMixin`，改用 vllm-omni 的权重加载
   - 将 `wan_va/modules/utils.py` 中的 `WanVAEStreamingWrapper` 复制到 `streaming_vae.py`
   - 将 `wan_va/utils/scheduler.py` 中的 `FlowMatchScheduler` 复制到 pipeline 目录下
   - 将 `wan_va/utils/utils.py` 中的 `get_mesh_id`, `data_seq_to_patch` 等工具函数复制过来

2. **实现 Pipeline**：
   - 从 `wan_va/wan_va_server.py` 的 `generate()` 和 `_infer()` 提取 I2VA 推理逻辑
   - 封装到 `LingBotVAPipeline.forward()`
   - 输入适配：从 `OmniDiffusionRequest` 中提取图像和 prompt
   - 输出适配：返回 `DiffusionOutput`

3. **注册 + 测试**：
   - 在 `registry.py` 中注册 `LingBotVAPipeline`
   - 编写 `model_index.json`（或在 pipeline 代码中硬编码 `model_class_name`）
   - 用 RoboTwin I2VA 配置做端到端测试

### Phase 2：性能优化（~1-2 周）

4. **TP 适配**：将 Transformer 中的 Linear 层替换为 vllm-omni 的 TP 版本
5. **分布式 VAE 解码**：利用 `DistributedAutoencoderKLWan` 的分布式解码能力
6. **Attention 后端**：探索将 KV Cache 逻辑迁移到 vllm-omni 的 Attention 后端

### Phase 3：Server 模式（后续）

7. **仿真环境集成**：在 vllm-omni 中添加仿真环境调用接口
8. **KV Cache 闭环管理**：实现 `_compute_kv_cache` 的 pred/obs 生命周期
9. **WebSocket 通信**：对接仿真客户端

---

## 7. 需要从 LingBot-VA 复制的文件清单

| 源文件 | 目标位置 | 说明 |
|-------|---------|------|
| `wan_va/modules/model.py` | `lingbot_va_transformer.py` | 双流 Transformer（需要适配） |
| `wan_va/modules/utils.py` (部分) | `streaming_vae.py` | `WanVAEStreamingWrapper` 类 |
| `wan_va/utils/scheduler.py` | `scheduler.py` 或内联 | `FlowMatchScheduler` |
| `wan_va/utils/utils.py` (部分) | `utils.py` | `get_mesh_id`, `data_seq_to_patch` |
| `wan_va/wan_va_server.py` (逻辑) | `pipeline_lingbot_va.py` | `_infer()`, `_encode_obs()`, `_prepare_latent_input()`, `preprocess_action()`, `postprocess_action()` 等方法 |
| `wan_va/configs/va_robotwin_cfg.py` | 配置映射 | `norm_stat`, `used_action_channel_ids` 等 |

---

## 8. 风险与注意事项

1. **权重兼容性**：LingBot-VA 的预训练权重是在 `diffusers` 的 `ModelMixin` 下保存的，需要确认 vllm-omni 的 `AutoWeightsLoader` 能正确加载。可能需要写 weight mapping
2. **内存占用**：I2VA 模式同时需要 Transformer + VAE + KV Cache + 多 chunk 的 latent/action 缓冲区。10 个 chunk 的 video latent = `(1, 48, 20, H, W)` 不算大，但 KV Cache 的 slot 池可能占用较多显存
3. **FlexAttention 依赖**：LingBot-VA 的训练模式使用 `torch.nn.attention.flex_attention`，但 I2VA 推理不需要（只用 `torch` 或 `flashattn` 模式），可以安全忽略
4. **数值一致性**：由于 vllm-omni 使用 TP 而 LingBot-VA 使用 FSDP，RMSNorm 在 TP 下需要特殊处理（参考 vllm-omni 的 `DistributedRMSNorm`），否则会有数值偏差
5. **仿真环境不在 I2VA scope 内**：I2VA 模式不需要仿真环境调用（`take_action` / `get_obs`），只需要初始图像。仿真集成留到 Server 模式

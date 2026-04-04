# LingBot-VA 推理流程分析文档

## 1. 概述

LingBot-VA 是一个基于 **自回归扩散 (AR Diffusion)** 框架的机器人控制模型，能够在一个统一的模型中同时完成 **视觉世界建模 (Video Prediction)** 和 **动作推理 (Action Inference)**。其核心架构是一个基于 Wan 视频生成模型改造的 **双流混合 Transformer (Dual-Stream Mixture-of-Transformers, MoT)**，通过交错的 video-action 序列实现自回归推理。

### 支持的推理模式

| 模式 | 配置字段 `infer_mode` | 说明 |
|------|----------------------|------|
| **I2VA (Image-to-Video-Action)** | `i2va` | 给定初始图像和文本 prompt，自回归地生成多个 chunk 的视频预测和动作序列 |
| **Server 模式** | `server` | WebSocket Server-Client 架构，从仿真环境接收实时观测，逐 chunk 推理动作 |

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        VA_Server (主入口)                        │
│  wan_va/wan_va_server.py                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  组件加载:                                                       │
│  ┌──────────┐ ┌──────────────┐ ┌───────────┐ ┌───────────────┐ │
│  │   VAE    │ │ T5 Tokenizer │ │T5 TextEnc │ │ Transformer   │ │
│  │(编码/解码)│ │  (文本分词)   │ │(文本编码)  │ │ (核心模型)    │ │
│  └──────────┘ └──────────────┘ └───────────┘ └───────────────┘ │
│                                                                 │
│  推理接口:                                                       │
│  ┌──────────┐ ┌───────────────────┐ ┌───────────────────────┐  │
│  │ _reset() │ │ _compute_kv_cache()│ │ _infer() (核心推理)   │  │
│  └──────────┘ └───────────────────┘ └───────────────────────┘  │
│                                                                 │
│  Scheduler:                                                     │
│  ┌──────────────────────┐ ┌──────────────────────────┐         │
│  │ FlowMatchScheduler   │ │ FlowMatchScheduler       │         │
│  │ (video 去噪)          │ │ (action 去噪)             │         │
│  └──────────────────────┘ └──────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 模型组件详解

### 3.1 VAE（变分自编码器）

- **类型**: `AutoencoderKLWan`（来自 `diffusers` 库）
- **作用**: 将 RGB 图像编码到 latent 空间，以及将 latent 解码回像素空间
- **流式编码**: 通过 `WanVAEStreamingWrapper` 实现流式 VAE 编码，支持逐 chunk 编码并维护因果卷积的缓存
- **latent 维度**: 48 通道（编码后 mu + logvar 各 24 通道，取 mu）
- **下采样倍率**: 空间 16x，时间通过 patch

```python
# 编码流程 (位于 _encode_obs)
videos → [0,255] → [-1,1] 归一化 → streaming_vae.encode_chunk() 
→ chunk(mu, logvar) → normalize_latents(mu) → video_latent
```

### 3.2 T5 文本编码器

- **Tokenizer**: `T5TokenizerFast`
- **Text Encoder**: `UMT5EncoderModel`
- **最大序列长度**: 512 tokens
- **作用**: 将文本指令 (prompt) 编码为 embedding，用于 Transformer 的 cross-attention
- **支持 CFG**: 当 `guidance_scale > 1` 时，同时计算正向和负向 prompt embeddings

### 3.3 WanTransformer3DModel（核心 Transformer）

- **架构**: 30 层 `WanTransformerBlock`，每层包含：
  1. **Self-Attention** (带 KV Cache 的因果注意力)
  2. **Cross-Attention** (与文本 embedding 交互)
  3. **Feed-Forward Network** (GELU 激活)
- **参数**:
  - 24 个注意力头，每头 128 维 → inner_dim = 3072
  - FFN 维度: 14336
  - Patch size: `(1, 2, 2)` (时间不 patch，空间 2x2 patch)
- **双流设计**:
  - **视频流**: `patch_embedding_mlp` 嵌入 → `proj_out` 输出
  - **动作流**: `action_embedder` 嵌入 → `action_proj_out` 输出
  - 两条流共享相同的 Transformer blocks，但通过 `action_mode` 标志切换嵌入和投影层
  - 使用独立的 `condition_embedder` 和 `condition_embedder_action` 进行时间步嵌入

### 3.4 位置编码 (RoPE)

- **类型**: `WanRotaryPosEmbed`（3D 旋转位置编码）
- **维度分配**: `(f_dim, h_dim, w_dim)` 其中 `f_dim = dim - 2*(dim//3)`, `h_dim = w_dim = dim//3`
- **Grid ID 构造** (`get_mesh_id`):
  - 视频 token: `(frame_id, height_id, width_id, type=0)`
  - 动作 token: `(frame_id + offset, -1, -1, type=1)` — 动作 token 没有空间位置，只有时间位置

### 3.5 KV Cache 机制

- **位置**: 在每个 `WanAttention` 的 `self-attention` 层中
- **实现**: 预分配固定大小的 KV 池（由 `attn_window` 决定），通过 slot 分配和释放管理
- **关键操作**:
  - `init_kv_cache()`: 预分配 `(batch, total_token, num_head, head_dim)` 大小的 K/V 缓存
  - `update_cache()`: 将新的 K/V 写入可用 slot，标记为已用
  - `clear_pred_cache()`: 清除预测帧的缓存（保留观测帧）
  - `restore_cache()`: 推理步骤中不保留的临时缓存恢复
- **update_cache 模式**:
  - `0`: 不更新缓存（中间去噪步骤）
  - `1`: 更新缓存并标记为 `is_pred=True`（最后一步去噪，写入预测结果）
  - `2`: 更新缓存并标记为 `is_pred=False`（写入真实观测的 KV）

### 3.6 FlowMatchScheduler

- **类型**: Flow Matching 调度器
- **Video Scheduler**: `snr_shift=5.0`，用于视频 latent 去噪
- **Action Scheduler**: `snr_shift=1.0`（可配置），用于动作去噪
- **核心公式**: `x_{t-1} = x_t + model_output * (sigma_{t-1} - sigma_t)`

---

## 4. 推理流程详解

### 4.1 启动与初始化

**入口**: `wan_va/wan_va_server.py :: main()`

```
main() → run(args)
  ├── 读取配置 (VA_CONFIGS[config_name])
  ├── 初始化分布式环境 (init_distributed)
  ├── 创建 VA_Server 实例
  │   ├── 加载 VAE, Tokenizer, TextEncoder, Transformer
  │   ├── 使用 FSDP 分片 Transformer (shard_model)
  │   └── 创建 FlowMatchScheduler (video & action)
  └── 根据 infer_mode 选择:
      ├── 'i2va' → model.generate()
      └── 'server' → run_async_server_mode(model, ...)
```

### 4.2 Server 模式推理流程

Server 模式下，通过 WebSocket 接收仿真环境的观测数据，每一帧的推理分三步：

#### 步骤 1: 重置 (`_reset`)

当客户端发送 `reset=True` 时触发：

```
_reset(prompt)
  ├── 清空 Transformer KV Cache
  ├── 清空 VAE 流式缓存
  ├── 重置 frame_st_id = 0
  ├── 计算 latent 空间尺寸 (height/16, width/16 * num_cameras)
  ├── 创建空 KV Cache (create_empty_cache)
  │   └── 总 token 数 = (attn_window/2) * latent_token + (attn_window/2) * action_token
  ├── 加载动作归一化统计量 (q01, q99)
  └── 编码文本 prompt (encode_prompt → T5 Encoder)
```

#### 步骤 2: 计算 KV Cache (`_compute_kv_cache`)

当客户端发送 `compute_kv_cache=True` 时触发，用于将真实观测写入 KV Cache：

```
_compute_kv_cache(obs)
  ├── 清除上一次预测的 KV 缓存 (clear_pred_cache)
  ├── 编码观测图像 → latent (_encode_obs)
  │   ├── 多相机图像 resize 到指定尺寸
  │   ├── 归一化到 [-1, 1]
  │   ├── VAE 流式编码 (streaming_vae.encode_chunk)
  │   ├── 取 mu，标准化 latent
  │   └── 多相机 latent 拼接 (在宽度维度)
  ├── 预处理动作 (preprocess_action)
  │   ├── padding 到 action_dim
  │   ├── 通道重排 (inverse_used_action_channel_ids)
  │   └── quantile 归一化到 [-1, 1]
  ├── 构建输入字典 (_prepare_latent_input)
  │   ├── latent 部分: timestep=0 (clean), grid_id 构建
  │   └── action 部分: timestep=0 (clean), grid_id 构建
  └── Transformer 前向 (update_cache=2, 写入观测 KV)
      ├── 先写入 latent KV (action_mode=False)
      └── 再写入 action KV (action_mode=True)
```

#### 步骤 3: 推理一个 Chunk (`_infer`)

当既非 reset 也非 compute_kv_cache 时触发，执行实际的去噪推理：

```
_infer(obs, frame_st_id)
  ├── [首帧] 编码初始观测 → init_latent
  ├── 采样随机噪声
  │   ├── latents: (1, 48, frame_chunk_size, H_lat, W_lat)
  │   └── actions: (1, action_dim, frame_chunk_size, action_per_frame, 1)
  │
  ├── === 阶段 1: Video 去噪循环 ===
  │   for t in video_timesteps:    # 默认 25 步
  │   │   ├── 构建条件 (首帧用 init_latent 作为 condition)
  │   │   ├── _prepare_latent_input(latents, None, t, ...)
  │   │   │   └── 设置 noisy_latents, timesteps, grid_id, text_emb
  │   │   ├── [CFG] _repeat_input_for_cfg (batch 维度翻倍)
  │   │   ├── Transformer forward (action_mode=False)
  │   │   │   ├── patch_embedding_mlp 嵌入 latent
  │   │   │   ├── 文本 embedding 通过 text_embedder
  │   │   │   ├── RoPE 旋转位置编码
  │   │   │   ├── 时间步 embedding (condition_embedder)
  │   │   │   ├── 30 层 WanTransformerBlock
  │   │   │   │   ├── Self-Attn (带 KV Cache 的因果注意力)
  │   │   │   │   ├── Cross-Attn (与文本交互)
  │   │   │   │   └── FFN
  │   │   │   ├── norm_out + scale_shift
  │   │   │   └── proj_out → 噪声预测
  │   │   ├── data_seq_to_patch (序列→patch 空间重排)
  │   │   ├── [CFG] guidance: pred = neg + scale * (pos - neg)
  │   │   ├── scheduler.step (去噪一步)
  │   │   └── [最后一步] update_cache=1 (写入预测 KV)
  │
  ├── === 阶段 2: Action 去噪循环 ===
  │   for t in action_timesteps:   # 默认 50 步
  │   │   ├── 构建条件 (首帧动作条件为零)
  │   │   ├── _prepare_latent_input(None, actions, t, ...)
  │   │   ├── [CFG] _repeat_input_for_cfg
  │   │   ├── Transformer forward (action_mode=True)
  │   │   │   ├── action_embedder 嵌入动作
  │   │   │   ├── RoPE 编码
  │   │   │   ├── 时间步 embedding (condition_embedder_action)
  │   │   │   ├── 30 层 WanTransformerBlock (共享)
  │   │   │   └── action_proj_out → 噪声预测
  │   │   ├── rearrange → action 空间
  │   │   ├── [CFG] guidance
  │   │   ├── action_scheduler.step
  │   │   └── [最后一步] update_cache=1 (写入预测 KV)
  │
  ├── 动作后处理 (postprocess_action)
  │   ├── quantile 反归一化
  │   └── 通道重排 (used_action_channel_ids)
  │
  └── 返回 actions, latents
```

### 4.3 I2VA 模式推理流程

I2VA 模式用于离线推理，给定初始图像生成完整的视频和动作序列：

```
generate()
  ├── _reset(prompt)
  ├── load_init_obs() — 从 input_img_path 加载初始图像
  ├── for chunk_id in range(num_chunks_to_infer):
  │   └── _infer(init_obs, frame_st_id=chunk_id * frame_chunk_size)
  ├── 拼接所有 chunk 的 latent 和 action
  ├── 释放 Transformer 和 TextEncoder 显存
  ├── VAE 解码 latent → 视频
  └── 保存为 demo.mp4
```

---

## 5. 核心闭环：预测 → 执行 → 观测 → 再预测

> **这是 LingBot-VA 推理的最核心逻辑。**

### 5.1 总览

LingBot-VA 的 Server 模式推理不是一次性的前向推理，而是一个与仿真环境（或真实机器人）持续交互的 **闭环控制循环**。每个循环包含三个阶段：模型预测、仿真执行、真实观测回写。这三者通过 KV Cache 紧密耦合。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         闭环控制循环 (每个 Chunk)                        │
│                                                                         │
│  ┌─────────────┐     ┌─────────────────┐     ┌──────────────────────┐  │
│  │  ① 模型预测  │ ──→ │  ② 仿真环境执行   │ ──→ │  ③ 真实观测回写      │  │
│  │  (_infer)   │     │  (TASK_ENV)      │     │  (_compute_kv_cache) │  │
│  │             │     │                  │     │                      │  │
│  │ Video 去噪  │     │ take_action()    │     │ clear_pred_cache()   │  │
│  │   ↓         │     │ × action_per_    │     │ encode_obs() → V_obs │  │
│  │ Action 去噪 │     │   frame × frames │     │ preprocess() → A_obs │  │
│  │   ↓         │     │   ↓              │     │ 写入 KV Cache        │  │
│  │ 返回 action │     │ get_obs() 拍照   │     │ (update_cache=2)     │  │
│  └─────────────┘     └─────────────────┘     └──────────────────────┘  │
│         │                                              │                │
│         └──────────────── 下一个 Chunk ←───────────────┘                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Client 端完整循环（以 RoboTwin 评测为例）

以下是从 `evaluation/robotwin/eval_polict_client_openpi.py` 提取的核心循环逻辑：

```python
# ===== 第一步：初始化 =====
model.infer(dict(reset=True, prompt=prompt))    # 清空 KV Cache，编码 prompt
observation = TASK_ENV.get_obs()                 # 获取初始观测
first_obs = format_obs(observation, prompt)      # 格式化为模型输入

# ===== 核心闭环 =====
while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:

    # ① 模型预测：Video 去噪 → Action 去噪 → 返回 action
    ret = model.infer(dict(obs=first_obs, prompt=prompt, ...))
    action = ret['action']   # shape: (C_action, F, action_per_frame)

    # ② 仿真执行：逐步将 action 送入仿真环境
    key_frame_list = []
    for i in range(action.shape[1]):             # 遍历每一帧 (frame_chunk_size=2)
        for j in range(action.shape[2]):         # 遍历帧内每一步 (action_per_frame=16)
            ee_action = convert_action(action[:, i, j])
            TASK_ENV.take_action(ee_action)      # 仿真环境执行单步

            if (j+1) % action_per_frame == 0:    # 每执行完一帧的所有步骤
                obs = format_obs(TASK_ENV.get_obs(), prompt)  # 拍照
                key_frame_list.append(obs)        # 收集关键帧

    # ③ 真实观测回写 KV Cache：用真实观测替换模型的想象
    model.infer(dict(
        obs=key_frame_list,         # 真实图像（仿真环境中拍的）
        compute_kv_cache=True,      # 触发 _compute_kv_cache
        state=action                # 实际执行的动作
    ))

    # 检查是否完成任务
    if TASK_ENV.eval_success:
        break
```

### 5.3 KV Cache 在闭环中的角色

#### 5.3.1 KV Cache 数据结构

每个 `WanAttention` 的 self-attention 层维护一个固定大小的 KV 池（video token 和 action token 混合存储）：

```python
attn_caches[cache_name] = {
    'k':      [batch, total_token, num_head, head_dim]  # K 向量池
    'v':      [batch, total_token, num_head, head_dim]  # V 向量池
    'id':     [total_token]        # 递增 ID（用于 LRU 淘汰最老的 slot）
    'mask':   [total_token]        # bool，标记 slot 是否被占用
    'is_pred':[total_token]        # bool，区分「模型预测」还是「真实观测」
}
```

池的总大小 = `(attn_window / 2) × latent_token_per_chunk + (attn_window / 2) × action_token_per_chunk`

#### 5.3.2 `update_cache` 三种模式

| `update_cache` 值 | `is_pred` | 行为 | 使用场景 |
|---|---|---|---|
| `0` | — | **临时借用**：写入 → attention → 立即撤回 | 去噪循环的中间步骤 |
| `1` | `True` | **持久写入（Pred）**：写入后不撤回 | 去噪循环的**最后一步** |
| `2` | `False` | **持久写入（Obs）**：写入后不撤回 | `_compute_kv_cache` 写入真实观测 |

#### 5.3.3 Pred KV 与 Obs KV 的关系

**Pred KV 是「对未来的想象」，Obs KV 是「对过去的记忆」，两者在时间维度上互补。**

| | Pred KV (`is_pred=True`) | Obs KV (`is_pred=False`) |
|---|---|---|
| **时间** | 当前 chunk（**未来**） | 历史 chunk（**过去**） |
| **内容** | 模型想象的下一帧 video + action | 仿真环境的真实观测 |
| **作用** | 让 action 去噪 attend to 未来的视觉预测 | 提供历史上下文 |
| **生命周期** | 下一轮 `_compute_kv_cache` 开头被 `clear_pred_cache` 清除 | 一直保留到被滑动窗口淘汰 |
| **写入方式** | `_infer()` 去噪最后一步 `update_cache=1` | `_compute_kv_cache()` 中 `update_cache=2` |

`is_pred` 标志的**唯一作用**是在 `clear_pred_cache()` 中实现**选择性清除**：

```python
def clear_pred_cache(self, cache_name):
    cache = self.attn_caches[cache_name]
    is_pred = cache['is_pred']
    cache['mask'][is_pred] = False   # 只释放 pred 的 slot，保留 obs
```

#### 5.3.4 Video 去噪与 Action 去噪的联系

**Video 去噪和 Action 去噪共享同一个 KV Cache 池，两者通过 KV Cache 建立因果依赖：**

1. Video 去噪先完成，最后一步 (`update_cache=1`) 将去噪后的 clean video latent 的 K/V **持久写入** cache
2. Action 去噪时，每一步的 attention 都能**看到 cache 中已经存在的 video KV**
3. 因此 **action 的推理是以预测的 video 为条件的**

这本质上是一种串行的自回归依赖：

```
历史真实观测 (Obs Video KV + Obs Action KV)
    → 预测 Video (先去噪，写入 Pred Video KV)
        → 预测 Action (后去噪，能 attend to Pred Video KV)
```

#### 5.3.5 `update_cache=0` 的「临时借用」机制

去噪循环的中间步骤（非最后一步）使用 `update_cache=0`：

```python
# WanAttention.forward() 中:
slots = self.update_cache(cache_name, key, value, is_pred=...)  # 先写入
# ... 做 attention（Q 能看到 cache 中所有 valid 的 KV，包括刚写入的自身）
if update_cache == 0:
    self.restore_cache(cache_name, slots)  # 做完立即撤回
```

中间的 noisy 状态不应该污染 cache，只有最终去噪干净的结果才值得保留。但在做 attention 的那一刻，token 需要能看到自身（同一 chunk 内的 self-attention），所以必须先临时写入。

#### 5.3.6 KV Cache 完整生命周期示例

用 3 个 chunk 的例子说明（`V`=video KV, `A`=action KV）：

```
══════════ Chunk 0 ══════════

_reset():
  Cache: [ 空 ]

_infer():  (首次推理，使用初始观测)
  Video 去噪:
    步 1~24: update_cache=0, 临时借用后撤回
    步 25(最后): update_cache=1, Pred Video KV 持久写入
  Cache: [ V₀ᵖʳᵉᵈ ]

  Action 去噪:  ← 能看到 V₀ᵖʳᵉᵈ
    步 1~49: update_cache=0, 临时借用
    步 50(最后): update_cache=1, Pred Action KV 写入
  Cache: [ V₀ᵖʳᵉᵈ  A₀ᵖʳᵉᵈ ]

  → 返回 action → 仿真环境逐步执行 → 每帧拍照

_compute_kv_cache(obs_0):
  clear_pred_cache → 清掉 V₀ᵖʳᵉᵈ 和 A₀ᵖʳᵉᵈ
  Cache: [ 空 ]
  写入真实观测 V₀ᵒᵇˢ 和 A₀ᵒᵇˢ (update_cache=2)
  Cache: [ V₀ᵒᵇˢ  A₀ᵒᵇˢ ]

══════════ Chunk 1 ══════════

_infer():
  Video 去噪: Q attend to [ V₀ᵒᵇˢ  A₀ᵒᵇˢ ] + 自身
    最后一步 → 写入 V₁ᵖʳᵉᵈ
  Cache: [ V₀ᵒᵇˢ  A₀ᵒᵇˢ  V₁ᵖʳᵉᵈ ]

  Action 去噪: Q attend to [ V₀ᵒᵇˢ  A₀ᵒᵇˢ  V₁ᵖʳᵉᵈ ] + 自身
    最后一步 → 写入 A₁ᵖʳᵉᵈ
  Cache: [ V₀ᵒᵇˢ  A₀ᵒᵇˢ  V₁ᵖʳᵉᵈ  A₁ᵖʳᵉᵈ ]

  → 返回 action → 仿真执行 → 拍照

_compute_kv_cache(obs_1):
  clear_pred_cache → 清掉 V₁ᵖʳᵉᵈ 和 A₁ᵖʳᵉᵈ
  Cache: [ V₀ᵒᵇˢ  A₀ᵒᵇˢ ]
  写入 V₁ᵒᵇˢ 和 A₁ᵒᵇˢ
  Cache: [ V₀ᵒᵇˢ  A₀ᵒᵇˢ  V₁ᵒᵇˢ  A₁ᵒᵇˢ ]

══════════ Chunk 2 ══════════
  _infer(): Q 能看到 4 组历史 Obs KV + 自身的 Pred KV
  ...如此循环
```

#### 5.3.7 滑动窗口淘汰

当 cache 池满时，`allocate_slots` 按 `id`（写入顺序）淘汰最老的 slot，**不区分 pred/obs**——这实现了一个基于时间的滑动窗口，确保模型始终能看到最近 `attn_window` 个 chunk 的历史。

---

## 6. 仿真执行详解

### 6.1 仿真环境与物理引擎

当前代码库中的仿真执行基于 **RoboTwin-2.0** 平台，使用 **SAPIEN** 物理引擎：

- **环境类型**: 双臂机器人操作任务（left arm + right arm）
- **动作空间**: 末端执行器 (End-Effector) 控制，每个臂 `[x, y, z, quat(4), gripper]` = 8 维，双臂共 16 维
- **观测空间**: 3 个 RGB 相机（cam_high + cam_left_wrist + cam_right_wrist）+ 关节状态

#### SAPIEN 渲染管线

从 `test_render.py` 可以看到 SAPIEN 的渲染配置：

```python
sapien.render.set_camera_shader_dir("rt")              # 使用光线追踪着色器
sapien.render.set_ray_tracing_samples_per_pixel(32)     # 每像素 32 次采样
sapien.render.set_ray_tracing_path_depth(8)             # 光线追踪路径深度 8
sapien.render.set_ray_tracing_denoiser("oidn")          # 使用 OIDN 去噪器
```

SAPIEN 使用**光线追踪 (Ray Tracing)** 渲染器而非光栅化，这提供了更高质量的图像（包括全局光照、软阴影、反射等），但渲染耗时也比光栅化更高。渲染管线可以配置最大材质和纹理数量（`max_num_materials=50000, max_num_textures=50000`），支持复杂场景。

#### 专家轨迹验证

评测脚本在策略执行前会先用专家策略（`TASK_ENV.play_once()`）运行一次完整的 episode，验证该 seed 的场景设置是可解的：

```python
# 第一步：专家验证（确保场景有效）
TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
episode_info = TASK_ENV.play_once()          # 专家策略完整执行
TASK_ENV.close_env()

# 只有专家能完成的场景才会进入策略评测
if TASK_ENV.plan_success and TASK_ENV.check_success():
    # 第二步：用同一个 seed 重新设置场景，让策略执行
    TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
    # ... 策略评测 ...
```

这确保了评测的公平性——模型不会在物理上不可能完成的场景中被扣分。

### 6.2 Action 执行细节

模型每次推理输出的 action 形状为 `(C_action, frame_chunk_size, action_per_frame)`，即 `(16, 2, 16)` (RoboTwin 配置)。Client 端将这些 action **逐步**送入仿真。

#### 6.2.1 首个 Chunk 的特殊处理

第一个 chunk 存在 `start_idx = 1 if first else 0` 的逻辑——**第一帧的 action 被跳过**，只执行从第 2 帧开始的 action：

```python
start_idx = 1 if first else 0
for i in range(start_idx, action.shape[1]):   # 首 chunk 跳过第 0 帧
    for j in range(action.shape[2]):
        ...
```

原因：第一个 chunk 的第 0 帧对应初始观测（`init_latent`），其 action 是以零向量为条件的 padding（见 `_infer` 中 `action_cond = torch.zeros(...) if frame_st_id == 0`），不包含有意义的控制信号。

#### 6.2.2 动作格式转换

模型输出的 action 是归一化后的相对量，需要经过多步转换才能送入仿真：

```python
# 模型输出 → postprocess_action (quantile 反归一化)
# → Client 端格式转换:

if action.shape[0] == 14:
    # 14 维模式：euler 角 → quaternion
    ee_action = np.concatenate([
        ee_action[:3],                                          # 左臂 xyz
        euler2quat(ee_action[3], ee_action[4], ee_action[5]),   # euler → quat
        ee_action[6:10],                                        # 左夹爪 + 右臂 xyz
        euler2quat(ee_action[10], ee_action[11], ee_action[12]),# euler → quat
        ee_action[13:14]                                        # 右夹爪
    ])
elif action.shape[0] == 16:
    # 16 维模式：相对 pose → 绝对 pose（叠加初始 EE 姿态）
    ee_action = add_init_pose(ee_action, inint_eef_pose)
    # 四元数归一化（保证旋转合法，避免数值漂移）
    ee_action[3:7] /= np.linalg.norm(ee_action[3:7])
    ee_action[11:15] /= np.linalg.norm(ee_action[11:15])
```

**`add_init_pose`** 的逻辑是将模型预测的**相对位移**和**相对旋转**叠加到初始 EE 姿态上：
- 位移：`out_trans = new_pose[:3] + init_pose[:3]` （简单加法）
- 旋转：`out_rot = init_R * new_R`（四元数乘法，`scipy.spatial.transform.Rotation`）

这意味着模型学习的是**相对于初始姿态的增量**，而非绝对世界坐标。

#### 6.2.3 完整执行循环

```python
# action shape: (C=16, F=2, H=16)
# 即 2 帧 × 每帧 16 个子步骤 = 32 个控制步（首 chunk 为 16 步）
for i in range(start_idx, F):         # 遍历帧
    for j in range(action_per_frame): # 遍历帧内子步骤
        ee_action = action[:, i, j]   # 取出 16 维的 EE pose
        ee_action = convert_action(ee_action)  # 格式转换
        TASK_ENV.take_action(ee_action, action_type='ee')  # 仿真执行单步

        # 每执行完一帧的所有步骤后拍照
        if (j+1) % action_per_frame == 0:
            obs = TASK_ENV.get_obs()     # 获取 RGB 图像 + 关节状态
            key_frame_list.append(obs)   # 收集为下一轮的真实观测
```

### 6.3 观测采集时机

不是每个 action 子步骤都采集观测，而是**每帧结束时**采集一次关键帧：

```
一个 Chunk (frame_chunk_size=2):

  帧 0:  action_step_0, action_step_1, ..., action_step_15 → 拍照 → key_frame_0
  帧 1:  action_step_0, action_step_1, ..., action_step_15 → 拍照 → key_frame_1

  key_frame_list = [key_frame_0, key_frame_1]
  → 发送给 _compute_kv_cache()
```

这与模型的 `frame_chunk_size` 对齐：每个 chunk 预测 2 帧视频 + 2 帧动作，仿真执行后也收集 2 帧真实观测回写 KV Cache。

**注意**：首个 chunk 因跳过第 0 帧，只拍 1 张照片（`key_frame_list` 长度为 1）。但 `_compute_kv_cache` 中通过 `init_latent` 拼接弥补了第 0 帧的 latent（`torch.cat([self.init_latent, latent_model_input], dim=2)`）。

### 6.4 仿真观测的质量保证

#### 6.4.1 图像观测 (V_obs) 的可靠性

**仿真环境拍到的图像是物理引擎的「ground truth」渲染结果：**

1. **物理一致性**：SAPIEN 的光线追踪渲染器在每次 `get_obs()` 调用时，根据当前物理仿真状态（物体位姿、关节角、光源）重新渲染。物体位置、光照、遮挡完全由物理引擎决定，不存在生成误差。
2. **Domain Randomization**：评测脚本支持多种随机化以增强鲁棒性：
   - `cluttered_table`：杂乱桌面干扰物
   - `random_background`：随机背景（含 `clean_background_rate` 控制比例）
   - `random_light`：随机光照条件（含 `crazy_random_light_rate` 控制极端情况比例）
   - `random_table_height`：随机桌面高度
   - `random_head_camera_dis`：随机头部相机距离
3. **多视角一致性**：3 个相机（head / left_wrist / right_wrist）在同一物理步后同步渲染，保证多视角间的时间一致性

因此，**V_obs 的质量完全由物理引擎保证，不会引入额外噪声**。唯一的「质量」问题是 sim-to-real gap——仿真渲染的视觉风格与真实世界存在分布差异，但这在仿真评测中不构成问题。

#### 6.4.2 动作状态 (A_obs) 的语义

写入 KV Cache 的 `state=action` **并非关节编码器读数**，而是**模型上一轮预测的 action 原始值**（经过 quantile 反归一化后的结果）：

```python
# Client 端发送：
model.infer(dict(
    obs=key_frame_list,         # 真实图像
    compute_kv_cache=True,
    state=action                # ← 就是模型预测的 action 本身
))

# Server 端接收后：
action_model_input = self.preprocess_action(obs['state'])  # 重新做 quantile 归一化
```

这样设计的原因：模型训练时，KV Cache 中的 action token 就是训练数据中的 clean action（归一化后的），所以推理时也应该用 clean action（模型自己的预测）来保持 KV Cache 的数据分布一致。如果用关节编码器读数，会引入 IK 求解误差和控制延迟带来的分布偏移。

#### 6.4.3 质量风险：误差累积

观测本身的质量没有问题，但**闭环执行的关键风险在于 action 误差累积**：

```
模型预测 action 不准 → 仿真执行后 obs 偏离预期
  → obs 被写入 KV Cache → 下一轮预测基于偏移的历史
    → 进一步加大 action 偏差 → 发散
```

LingBot-VA 通过两个机制缓解此问题：
1. **World Model (Video 预测)**：Video 去噪为 action 去噪提供了「对未来的想象」，相当于一种内在的 model-based planning，比纯 action prediction 有更好的预测基础
2. **真实观测回写**：每个 chunk 结束后用真实 obs 替换 pred KV（`clear_pred_cache` + `update_cache=2`），防止模型想象与现实的持续偏离——这本质上是一种 **Model Predictive Control (MPC) 的思路**

### 6.5 延迟分析

#### 6.5.1 单 Chunk 延迟分解

Server 模式下，一个完整 chunk 的端到端延迟 = **模型推理 + 网络通信 + 仿真执行 + KV Cache 回写**：

| 阶段 | 操作 | 耗时构成 | 典型量级 |
|------|------|---------|---------|
| **① 模型推理 (`_infer`)** | Video 去噪 25 步 + Action 去噪 50 步 | 每步 1 次 Transformer forward (或 2 次如果开 CFG) | **数秒~数十秒** (主要瓶颈) |
| **② 网络通信** | WebSocket 发送 obs + 接收 action | msgpack 序列化 + 网络传输 | <10 ms |
| **③ 仿真执行** | `take_action()` × 32 步 + `get_obs()` × 2 次 | SAPIEN 物理步进 + RT 渲染 | 数百 ms |
| **④ KV Cache 回写 (`_compute_kv_cache`)** | VAE 编码 + 2 次 Transformer forward | 无去噪循环，单次 forward | 数百 ms |

#### 6.5.2 模型推理延迟细分

步骤①是主要瓶颈。假设单次 Transformer forward 耗时 `T_fwd`：

| 配置项 | 值 | 等效 Forward 次数 |
|-------|-----|-----------------|
| Video 去噪 | 25 步 | 25 × T_fwd |
| Action 去噪 | 50 步 | 50 × T_fwd |
| **合计（无 CFG）** | | **75 × T_fwd** |
| **合计（CFG batch=2）** | | **≈ 150 × T_fwd** (batch 维度翻倍) |

其中 `T_fwd` 取决于：
- **模型大小**：30 层 Transformer，3072 维 hidden，约 1.3B 参数
- **序列长度**：当前 chunk 的 token 数（video + action）
- **KV Cache 大小**：历史 token 参与 attention 计算
- **GPU 数量/型号**：FSDP 多卡并行可线性降低 `T_fwd`

#### 6.5.3 仿真执行延迟

仿真执行（步骤③）的延迟来源：

1. **物理步进 (`take_action`)**: SAPIEN 的 PhysX 后端执行一个时间步的刚体动力学、碰撞检测、关节约束求解。单步通常 <1 ms，32 步 ≈ 数十 ms
2. **渲染 (`get_obs`)**: 光线追踪渲染器是仿真端的主要耗时。每次渲染 3 个相机视角 × 32 SPP（samples per pixel），分辨率由 camera config 决定。单次渲染约 50~200 ms，2 次渲染 ≈ 100~400 ms
3. **总计**: 仿真执行耗时通常在 **200~500 ms** 量级，远小于模型推理

#### 6.5.4 KV Cache 回写延迟

步骤④包含：
1. **VAE 编码**: 对 2 帧 × 3 相机的图像做流式编码。如果 VAE 被 offload 到 CPU，需要先搬回 GPU
2. **2 次 Transformer forward**: 分别写入 video KV 和 action KV（`update_cache=2`），每次只跑 1 遍（无去噪循环）
3. **总计**: 通常在 **200~500 ms** 量级

#### 6.5.5 优化手段

| 优化策略 | 原理 | 效果 |
|---------|------|------|
| **KV Cache** | 避免重复计算历史帧，每次 forward 只处理当前 chunk 的 token | 将 attention 计算量从 O(N²) 降为 O(N×W)，W=当前 chunk token 数 |
| **`video_exec_step`** | 提前终止 video 去噪（如只跑 10 步而非 25 步） | 牺牲 video 质量，减少 40~60% 的 video forward 次数 |
| **多 GPU FSDP** | 8 GPU 并行，参数和计算分片 | 近线性加速，`T_fwd` 降为约 1/8 |
| **VAE/TextEncoder Offload** | 推理阶段卸载到 CPU | 释放 GPU 显存给 Transformer，允许更大的 KV Cache |
| **禁用 Video CFG** | 设 `guidance_scale=1`，不做 CFG | batch 从 2 降为 1，forward 等效次数减半 |
| **WebSocket ping 禁用** | `ping_interval=None` | 防止长推理过程中 WebSocket 超时断连 |

#### 6.5.6 仿真 vs. 真实部署的延迟差异

| | 仿真环境 | 真实机器人 |
|---|---|---|
| **Action 执行** | 瞬时（物理引擎步进 <1 ms/步） | 真实物理运动，数百 ms~数秒 |
| **观测采集** | RT 渲染 50~200 ms/帧 | 相机拍照 <10 ms |
| **是否阻塞** | 同步执行，Client 等待仿真完成 | **可与 KV Cache 回写并行**：机械臂执行的同时异步编码 |
| **总体瓶颈** | 模型推理（数秒~数十秒） | 取决于 action chunk 对应的物理执行时间 vs 推理时间 |

在真实部署中，一个常见的优化是：**在机械臂执行当前 chunk 的 action 的同时，异步进行下一轮的 KV Cache 回写和模型推理**。这样只要推理时间 < 机械臂执行时间，就能实现无缝衔接。以 RoboTwin 的配置为例，每个 chunk 包含 32 个控制步，如果控制频率为 10 Hz，物理执行需要 3.2 秒——这给模型推理留出了充足的时间窗口。

---

## 7. 注意力机制与因果掩码

### 7.1 推理模式注意力 (`attn_mode = "torch"` 或 `"flashattn"`)

推理时不使用 FlexAttention，而是通过 **KV Cache** 实现因果性：
- 每一步只将当前 chunk 的 token 作为 query
- KV 来自 cache 中存储的历史 token + 当前 token
- 通过 slot 分配/释放管理滑动窗口

### 7.2 训练模式注意力 (`attn_mode = "flex"`)

训练时使用 `FlexAttnFunc` 实现复杂的因果掩码，包含以下规则：
- **Clean→Clean (观测→观测)**: 因果掩码（只能看到当前及之前的帧）
- **Noise→Clean (噪声→观测)**: 严格因果掩码（只能看到之前的帧，不含当前帧）
- **Noise→Noise (噪声→噪声)**: 只能看到同一帧的噪声 token
- **滑动窗口**: 所有注意力受限于 `window_size` 范围内
- **序列隔离**: 不同 batch 样本之间不可见

---

## 8. 数据流与张量形状

### 8.1 视频 Latent

| 阶段 | 张量形状 | 说明 |
|------|---------|------|
| 原始图像 | `(B, C_cam, F, H, W)` | 多相机 RGB 图像 |
| VAE 编码后 | `(1, 48, F, H/16, W/16*N_cam)` | Latent 空间，多相机拼接在宽度维 |
| Patch 后 | `(B, F*H'*W', 48*p1*p2*p3)` | 展平为 token 序列 |
| Transformer 输出 | `(B, L, C)` | 噪声预测序列 |
| 反 Patch | `(B, 48, F, H/16, W/16*N_cam)` | 还原 latent 空间 |

### 8.2 动作

| 阶段 | 张量形状 | 说明 |
|------|---------|------|
| 原始动作 | `(C_action, F, H_action)` | 多通道多帧动作 |
| 归一化后 | `(1, action_dim, F, action_per_frame, 1)` | 5D 张量 |
| 展平 token | `(B, F*action_per_frame, action_dim)` | Token 序列 |
| Transformer 输出 | `(B, F*action_per_frame, action_dim)` | 噪声预测 |
| 反归一化 | `(C_used, F, action_per_frame)` | 最终动作输出 |

### 8.3 典型配置 (RoboTwin)

| 参数 | 值 |
|------|-----|
| 图像尺寸 | 256 × 320 |
| Latent 尺寸 | 24 × 20 (高相机), 12 × 10 (腕部相机) |
| Frame chunk size | 2 |
| Action per frame | 16 |
| Action dim | 30 (使用 16 通道) |
| Video 去噪步数 | 25 |
| Action 去噪步数 | 50 |
| Attention window | 72 chunks |
| Patch size | (1, 2, 2) |

---

## 9. Server-Client 通信协议

### 9.1 架构

```
┌────────────────┐    WebSocket     ┌──────────────────────┐
│   仿真环境      │ ◄──────────────► │  VA_Server (GPU)      │
│  (RoboTwin等)   │    msgpack       │  ├─ Rank 0: Server    │
│                │                  │  ├─ Rank 1: Worker    │
│  Client Policy │                  │  ├─ ...               │
└────────────────┘                  │  └─ Rank N: Worker    │
                                    └──────────────────────┘
```

### 9.2 通信流程

1. **Client 连接** → Server 返回 metadata
2. **Client 发送观测** (msgpack 序列化的 dict):
   - `reset=True, prompt="..."` → 触发 `_reset()`
   - `compute_kv_cache=True, obs={...}, state={...}` → 触发 `_compute_kv_cache()`
   - 常规推理请求 → 触发 `_infer()`，返回 `{action: np.ndarray}`

### 9.3 分布式推理

- Rank 0 运行 WebSocket Server，接收请求后通过 `dist.broadcast` 将观测广播给所有 Worker
- 所有 Rank 使用 FSDP 分片的 Transformer 进行并行推理
- Rank 0 收集结果返回给 Client

---

## 10. 关键配置说明

### 10.1 配置层级

```
shared_config.py (基础配置)
  └── va_robotwin_cfg.py / va_demo_cfg.py (环境配置)
       └── va_robotwin_i2va.py / va_demo_i2va.py (I2VA 推理配置)
```

### 10.2 重要参数

| 参数 | 说明 | RoboTwin 默认值 |
|------|------|----------------|
| `attn_window` | KV Cache 滑动窗口大小 | 72 |
| `frame_chunk_size` | 每次推理的帧数 | 2 |
| `num_inference_steps` | Video 去噪步数 | 25 |
| `action_num_inference_steps` | Action 去噪步数 | 50 |
| `video_exec_step` | Video 提前终止步数 (-1=不提前) | -1 |
| `guidance_scale` | Video CFG 系数 | 5 |
| `action_guidance_scale` | Action CFG 系数 | 1 |
| `snr_shift` | Video 调度器 SNR shift | 5.0 |
| `action_snr_shift` | Action 调度器 SNR shift | 1.0 |
| `enable_offload` | 是否将 VAE/TextEncoder offload 到 CPU | True |
| `attn_mode` | 注意力实现方式 | 推理用 `torch`/`flashattn`，训练用 `flex` |

---

## 11. 文件索引

| 文件路径 | 功能 |
|---------|------|
| `wan_va/wan_va_server.py` | 推理主入口，`VA_Server` 类定义 |
| `wan_va/modules/model.py` | `WanTransformer3DModel` 核心模型定义 |
| `wan_va/modules/utils.py` | 模型加载工具、VAE 流式编码包装器 |
| `wan_va/utils/scheduler.py` | `FlowMatchScheduler` 调度器 |
| `wan_va/utils/utils.py` | Grid ID 构造、数据格式转换工具 |
| `wan_va/utils/sever_utils.py` | 分布式推理 & WebSocket Server 包装 |
| `wan_va/utils/Simple_Remote_Infer/` | WebSocket Server/Client 实现 |
| `wan_va/configs/` | 各环境的推理/训练配置 |
| `wan_va/distributed/fsdp.py` | FSDP 分片 & 激活检查点 |
| `script/run_launch_va_server_sync.sh` | 多 GPU 启动脚本 |

---

## 12. 推理优化策略

1. **KV Cache**: 避免重复计算历史帧的 attention，大幅减少推理延迟
2. **流式 VAE 编码**: `WanVAEStreamingWrapper` 缓存因果卷积状态，逐 chunk 编码而非全量重编码
3. **FSDP 分片**: 多 GPU 分片参数，支持大模型推理
4. **Video/Action 分离去噪**: 先完成 video 去噪，再进行 action 去噪，共享 Transformer 但独立调度
5. **显存 Offload**: VAE 和 Text Encoder 可 offload 到 CPU，只在需要时上 GPU
6. **异步保存**: 使用线程池异步保存中间结果，不阻塞推理
7. **滑动窗口注意力**: 通过 `attn_window` 限制 KV Cache 大小，支持长序列推理

# LingBot-VA → vllm-omni 集成开发计划

> 基于 [RFC v0.2](rfcs/rfc_lingbot_va_vllm_omni_integration.md) 生成

---

## Roadmap

```
Phase 1: I2VA 单 GPU 跑通           ██████████░░░░░░░░░░  (当前)
Phase 2: TP / CFG 并行              ░░░░░░░░░░░░░░░░░░░░
Phase 3: Server 模式 + Serving      ░░░░░░░░░░░░░░░░░░░░
```

**一句话目标**：把 LingBot-VA 的 I2VA 推理搬进 vllm-omni，复用其 TP / CFG / serving 基础设施，保持数值一致。

---

## 当前进度

| 文件 | 状态 | 说明 |
|------|:----:|------|
| `modeling/scheduler.py` | 🔨 | FlowMatchScheduler，已写未测试 |
| `modeling/streaming_vae.py` | 🔨 | WanVAEStreamingWrapper，已写未测试 |
| `modeling/lingbot_va_transformer.py` | 🔨 | Transformer + KV cache，已写未测试 |
| `utils.py` | 🔨 | get_mesh_id / data_seq_to_patch，已写未测试 |
| `state_lingbot_va.py` | 🔨 | 跨 forward() 状态管理，已写未测试 |
| `pipeline_lingbot_va.py` | ❌ | **核心 Pipeline — 下一步** |
| 模型注册 / 检测 / 权重加载 | ❌ | 需要在 vllm-omni 侧修改 |
| 数值对齐验证 | ❌ | 需要 reference outputs |

---

## Phase 1: I2VA 最小可用

### 1.1 实现 `pipeline_lingbot_va.py`
- `LingBotVAPipeline(nn.Module)` — 持有 tokenizer / text_encoder / vae / streaming_vae / transformer / schedulers / state
- `forward(req) → DiffusionOutput` — I2VA 完整流程：encode_prompt → encode_obs → multi-chunk loop → decode → return
- 内部方法：`_infer_one_chunk`、`_prepare_latent_input`、`_repeat_input_for_cfg`、`_encode_obs`、`preprocess_action`、`postprocess_action`、`encode_prompt`
- 关键点：**双循环串行**（video 25 步 → action 50 步），不能复用 DreamZero 的 VideoActionScheduler

### 1.2 模型注册 + 检测 + 权重加载
- `registry.py` 注册 `"LingBotVAPipeline"`
- `hf_utils.py` 添加 `_looks_like_lingbot_va()` 检测
- `load_weights()` 处理 action_embedder / action_proj_out / condition_embedder_action 等新增键

### 1.3 数值一致性验证
- 用原始 VA_Server + `robotwin_i2va` 配置生成 reference latents & actions
- 同配置同种子在 vllm-omni pipeline 下复现，对比 MSE < 1e-5

### 1.4 里程碑
> 单 GPU `vllm serve` 跑通 I2VA，输出与原始 VA_Server 数值一致。

---

## Phase 2: TP / CFG 并行

- Transformer TP 适配：nn.Linear → ColumnParallel / RowParallel，RMSNorm → DistributedRMSNorm
- CFGParallelMixin 接入：predict_noise / combine_cfg_noise，video 和 action 各自 guidance_scale
- 分布式 VAE 解码对接
- TP=2 数值对比 + 性能基准

---

## Phase 3: Server 模式

- 实现 `_compute_kv_cache()` + `clear_pred_cache()` 生命周期管理
- 新增 `transform/robotwin.py`（多相机 T-shape 布局）
- 对接 OpenPI serving（复用 DreamZero 的 WebSocket 层）
- 仿真环境联调 + 闭环 E2E 测试

---

## 关键技术决策摘要

| 决策 | 结论 | 理由 |
|------|------|------|
| 先 I2VA 还是先 Server？ | **先 I2VA** | 覆盖 85% 核心路径，固定输入易验证 |
| Transformer 复用 wan2_2 还是独立？ | **独立实现** | 双流结构 + 自定义 KV cache 不兼容 |
| KV cache 用 vllm-omni 的还是原始？ | **Phase 1 保留原始 slot 池** | 降低风险，Phase 2 可选迁移 |
| FlexAttention 是否移植？ | **不移植** | 仅训练用，推理不需要 |
| Scheduler 用 diffusers 的还是自定义？ | **移植原始 FlowMatchScheduler** | extra_one_step 行为不兼容 diffusers |

---

## 下一步行动

**立即开始**: 编写 `pipeline_lingbot_va.py`，从 `wan_va_server.py` 的 `generate()` / `_infer()` / `_encode_obs()` 适配到 vllm-omni 的 `forward(req) → DiffusionOutput` 接口。

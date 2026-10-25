# R2 Target 模型 DFlash 只读路由探针结果

R2 阶段需要确认 Qwen3.6 目标模型配 DFlash 投机解码在 310P3 上会走到哪条实现路径，但此前只有源码阅读结论、没有可留档的统一证据。本次在锁定提交上运行只读探针 `r2_target_route_probe.py`，产出一份脱敏枚举报告：Target 判为公开 Qwen3.5 MoE 架构、上游 V2 DFlash 输入准备走 Triton 直发路径、hybrid ModelState 存在条件性 Triton 风险。本报告只做静态文本与配置字段检查，不加载模型、不运行推理，不构成设备或静态准入 PASS。

## 1. Git 版本核对

| 项目 | 值 |
|---|---|
| 仓库 | `/home/w***46/310P/vllm-ascend` |
| 远程 | `git@github.com:10wvw01/vllm-ascend.git`（origin） |
| 现有工作分支 | `perf/310p-f1c-stock-fp16-hccl-vs-memfabric-abba-20261025`，干净，全程未切换、未修改 |
| 锁定提交 | `0d68b3b29c93d46d8f9c66bde425ad98245dcb76` |
| 分支核对 | 该提交即 `origin/work/310p-v2-r2-implementation-20261025` 的 tip，二者一致 |
| 获取方式 | `git fetch` + 独立 detached worktree `/tmp/opencode/r2-probe-worktree`，`rev-parse` 复核一致 |

锁定提交说明：`audit(310p-r2): fingerprint frozen Qwen3.5 MoE IsHybrid class without private draft data`。

## 2. 探针输入与环境

- `--target-config`：本地 `Qwen3.6-35B-A3B-w8a8` 的 `config.json`，探针返回 `READ`（读取为有界 JSON，无回显）。
- `--vllm-root`：`/vllm-workspace/vllm`，git tag `v0.27.1`、工作区干净，探针所需 5 个上游源文件全部存在，返回 `REQUESTED`。
- `--draft-config`：按任务要求未提供，自研 DFlash Draft 配置未被读取，返回 `NOT_PROVIDED`。
- `--repo-root`：未显式传入，探针默认使用其所在的锁定提交 worktree。
- `mamba_cache_mode`：保持默认 `ALIGN`，这是 R2 计划声明值，不是观察到的进程参数。
- 探针仅使用标准库（argparse/json/pathlib），未导入 vllm、torch、torch-npu 或任何模型代码。

## 3. 探针脱敏输出（仅工具预定义枚举）

```json
{
  "schema_version": 1,
  "purpose": "STATIC_METADATA_AND_SOURCE_ONLY",
  "contract": {
    "target_id": "Eco-Tech/Qwen3.6-35B-A3B-w8a8",
    "device": "310P3_DUAL_DIE",
    "tp": 2,
    "input_tokens": 4096,
    "output_tokens": 2048,
    "batch": 10,
    "speculative_method": "dflash",
    "k": 7
  },
  "inputs": {
    "target_config": "READ",
    "draft_config": "NOT_PROVIDED",
    "frozen_vllm_source": "REQUESTED"
  },
  "target": {
    "architecture_compatibility": "PUBLIC_QWEN35_MOE_ARCH_COMPATIBLE",
    "local_artifact_identity": "UNVERIFIED"
  },
  "draft": {
    "attention_layout": "UNVERIFIED",
    "private_architecture_and_weights": "NOT_COLLECTED"
  },
  "declared_flags": {
    "mamba_cache_mode": "ALIGN",
    "actual_process_flags": "NOT_OBSERVED"
  },
  "routes": {
    "legacy_310p_dflash_input": "ASCENDC_SOURCE_BINDING_PRESENT",
    "v2_candidate_dflash_input": "SCOPED_NON_TRITON_CANDIDATE_PRESENT",
    "frozen_gpu_v2_dflash_input": "DIRECT_TRITON_SOURCE_PATH_PRESENT",
    "v2_worker": "NOT_IMPLEMENTED_GUARD_PRESENT",
    "qwen35_moe_hybrid_declaration": "ISHYBRID_PUBLIC_MODEL_CLASS_PRESENT",
    "gpu_hybrid_model_state": "CONDITIONAL_FACTORY_PATH_PRESENT",
    "effective_runtime_runner": "NOT_OBSERVED",
    "effective_model_state": "NOT_OBSERVED",
    "actual_dflash_triton_launch": "NOT_OBSERVED",
    "actual_device_kernel": "NOT_RUN"
  },
  "disclosure": "ALLOWLIST_ENUMS_ONLY_NO_PRIVATE_CONTENT",
  "evidence_level": "STATIC_ONLY_NOT_RUNTIME_OR_ONDEVICE_GO"
}
```

## 4. 分层解读

### 4.1 源码事实（探针对锁定提交与冻结 vLLM 0.27.1 的文本检查）

- Target 架构声明：冻结 vLLM 的 `qwen3_5.py` 中，`Qwen3_5ForConditionalGeneration` 继承 `Qwen3VLForConditionalGeneration` 与 `IsHybrid`，`Qwen3_5MoeForConditionalGeneration` 在此基础上叠加 MoE；`config/model.py` 提供 `is_hybrid()`。枚举 `ISHYBRID_PUBLIC_MODEL_CLASS_PRESENT`。
- V1（legacy 310P）DFlash 路由：本仓库 `patch_idex_310.py` 与 `dflash_proposer_310.py` 存在 AscendC 源绑定（`AscendDflashProposer(.310).set_inputs_first_pass`、`npu_copy_and_expand_dflash_inputs`）。枚举 `ASCENDC_SOURCE_BINDING_PRESENT`。
- V2 候选路由：本仓库 `v2/dflash_speculator.py` 存在 scoped 的非 Triton 输入准备候选。枚举 `SCOPED_NON_TRITON_CANDIDATE_PRESENT`。
- 上游 GPU V2 DFlash 路由：冻结 vLLM 0.27.1 的 GPU speculator 中 `propose` 直接发射 Triton kernel（`_prepare_dflash_inputs_kernel[`）。枚举 `DIRECT_TRITON_SOURCE_PATH_PRESENT`。
- V2 Worker 守卫：`worker_310p.py` 仍含 `NOT_IMPLEMENTED` 保护，本次未解除。枚举 `NOT_IMPLEMENTED_GUARD_PRESENT`。

### 4.2 配置推断（本地 Target config.json 的字段枚举）

- 本地 w8a8 目标模型的 `config.json` 声明 `model_type=qwen3_5_moe`、存在 `text_config`、`architectures` 含 `Qwen3_5MoeForConditionalGeneration`，探据此判为公开 Qwen3.5 MoE 架构兼容。枚举 `PUBLIC_QWEN35_MOE_ARCH_COMPATIBLE`。
- 局限：`local_artifact_identity=UNVERIFIED`。未做哈希或来源核验，无法确认该本地目录与 `Eco-Tech/Qwen3.6-35B-A3B-w8a8` 发布工件逐位一致，"架构兼容"仅指 config 字段匹配，不覆盖量化方案与权重完整性。

### 4.3 ModelState 条件风险（静态存在性，非运行时结论）

上游 model_states 工厂按 `is_hybrid` 条件创建 `MambaHybridModelState`，其内部含 Triton kernel 发射（`preprocess_mamba_align_fused_kernel`、`_scatter_num_accepted_kernel`）。由于 Target 声明为 IsHybrid，即使 DFlash speculator 的输入准备已换成非 Triton 候选，走上游 ModelState 工厂时仍可能在 NPU 上触达 Triton 路径。枚举 `CONDITIONAL_FACTORY_PATH_PRESENT`。该路径是否实际执行取决于运行时路由，本次未观察。

## 5. 真实路由尚缺的证据

1. 运行时实际生效的 runner（V1 legacy 还是 V2）：`NOT_OBSERVED`。
2. 实际创建的 ModelState 类型（是否命中 `MambaHybridModelState`）：`NOT_OBSERVED`。
3. DFlash 输入准备是否真的发射 Triton kernel：`NOT_OBSERVED`。
4. 设备上 kernel 是否执行：`NOT_RUN`。
5. 本地 Target 工件身份核验（哈希对比发布工件）：`UNVERIFIED`。
6. Draft 注意力布局：`UNVERIFIED`（按保密要求未读取 Draft 配置）。
7. `mamba_cache_mode=ALIGN` 为计划声明值，实际进程 flags：`NOT_OBSERVED`。

## 6. 保密检查结论

- 未传入 `--draft-config`，自研 DFlash Draft 配置、权重、逐层架构均未被读取或披露（`draft_config=NOT_PROVIDED`，`private_architecture_and_weights=NOT_COLLECTED`）。
- 探针输出仅含预定义枚举字段，无路径、无原始配置内容、无版本号之外的任何私有信息。
- 全程未安装依赖、未加载模型、未启动服务、未执行 NPU 推理、未修改设备环境。
- 未修改任何算子、Runner、模型及业务代码；未解除 V2 Worker 的 `NOT_IMPLEMENTED` 保护。
- 报告中 Target 架构字段来自公开 Qwen3.5 架构的枚举判断，不涉及私有内容。

## 7. 准入声明

本报告的证据级别为 `STATIC_ONLY_NOT_RUNTIME_OR_ONDEVICE_GO`：仅为静态源码与配置字段检查，不构成设备准入或静态准入的 PASS。真实路由结论需补充第 5 节所列运行时证据后方可给出。

---

- 探针脚本：锁定提交 worktree 内 `tools/310p_mm_ar/r2_target_route_probe.py`
- 执行日期：2026-10-25，310P3 服务器
- 敏感内容已于 2026-10-25 人工审核通过（本地路径脱敏处理后入库）

# R1-B 现场裁决：Stock+DFlash Eager smoke NO-GO（草稿模型架构门禁，根因已证据闭环）

> 2026-10-25；执行者 OpenCode；唯一主线 `dev/310p-memfabric-mm-ar` @ `3ad10c0a5d5838333f29e43b24e6e94addc82d20`（隔离 worktree `/home/w00804146/310P/vllm-ascend-r1b`，任务分支 `work/310p-r1b-stock-eager-smoke-nogo-20261025`）。执行合同：[DFLASH_STOCK_V2_R1_GO_GATE_20261025.md](DFLASH_STOCK_V2_R1_GO_GATE_20261025.md) 第 8 节（R1-B）。
>
> **裁决：G0 全项 PASS（含 import-smoke）；R1-1 Stock+DFlash Eager smoke = FAIL/NO-GO——目标模型双 Rank 完整加载后，DFlash 草稿模型在构造期被上游架构门禁拒绝。R1-2（V2 smoke）按合同阻断未运行；R1-3/R1-4（两组 full）、R1-5/R1-6（Graph）全部 NOT_RUN。未出现 507014/507899/数值错误/挂死/二进制身份不一致；现场已恢复并核验。**

## 1. 四组运行状态（独立、不得互相替代）

| 组 | 状态 | 依据 |
|---|---|---|
| G0 只读依赖+设备核对 | **执行完毕，全项 PASS** | `evidence/r1b_20261025/g0/g0_dflash.json`（static_go=true、blockers=[]、import probe exit=0）、`g0_hccl_probe.log`（本轮重跑 RANK0/RANK1 PASS result0=3.0） |
| R1-1 Stock+DFlash Eager smoke | **执行完毕，FAIL/NO-GO**（草稿构造期架构门禁） | `stock_eager_smoke/`：server.log(.gz)、root_cause_traceback.txt、run.outcome.json |
| R1-2 V2+DFlash Eager smoke | **NOT_RUN（前置 Stock smoke 未通过，按合同阻断）** | V2 SO 未编译（阻断前置已解除必要性）；无 server.log/bench.json |
| R1-3/R1-4 Stock/V2 Eager full | **NOT_RUN（前置 smoke 未通过）** | 同上 |
| R1-5/R1-6 Graph smoke | **NOT_RUN（前置 Eager 均未通过）** | 同上 |

无任何 DFlash 接受率、Target Forward、融合覆盖率、physical M、吞吐/TTFT/TPOT/ITL 数据——本轮一律不填写、不推算。Bench 未启动（服务未达 healthy）。

## 2. G0 通过项（真实证据，全部本轮新采）

- **现场保护**：9 个既有 worktree 全部 clean 并快照（`g0/worktree_list_before.txt`、`worktree_status_each_before.txt`）；本轮仅新增隔离 worktree `vllm-ascend-r1b` @ `3ad10c0a5`（detached→任务分支）与证据目录；未 checkout/reset/clean 任何在用工作树。
- **源码冻结**：隔离 worktree 取自远程唯一主线 HEAD `3ad10c0a5`（`git fetch` 后核对，与任务指定 SHA 一致）；主工作树（f1c 分支 bd422330a）与主线在 `csrc/_310P/memfabric_mm_ar_v2/src`、`vllm_ascend/{_310p,ops,device,quantization}` 生产源码 diff 为空。
- **Python/CLI 身份**：`/usr/local/python3.12.13/bin/python3`（3.12.13）；vllm CLI `/usr/local/python3.12.13/bin/vllm` shebang 指向同一解释器；vllm 0.27.1+empty（editable ← `/vllm-workspace/vllm` = 官方上游 @ `6e448d0ea9bf`）；torch 2.10.0、torch-npu 2.10.0.post4、transformers 5.17.0（`g0_versions.txt`）。Serve 时 `vllm_ascend` 经 PYTHONPATH 前缀解析到本隔离 worktree（trial 的 runtime identity 校验通过后才放行启动）。
- **同卡双 Die 与独占**：NPU43520 = 设备 6/7（同 Bus `0000:AB:00.0`、同 310P3、Driver 26.1.1、CANN 9.1.0），本轮 npu-smi 确认无进程、无 hugepage 占用（不依赖上一轮快照）；NPU5664(0/1)、NPU16384(2/3) 被不可见 PID 占用（未触碰）；NPU38144(4/5) 设备 4 历史残留显存 13095MB 无属主，维持不动。
- **HCCL**：本轮在设备 6/7 重跑独立双进程 HCCL all_reduce FP16 探针：`RANK0/RANK1 HCCL_PROBE PASS result0=3.0`，rc=0（`g0_hccl_probe.log`）。
- **端口/进程**：8000/8581/4077 全部空闲；宿主无 vllm/serve 进程。
- **SO/KVER 指纹**（`g0_so_sha256.txt`，与上一轮核准基线逐字节一致）：KVER11 `libmf_sdma_orch_v11.json` `25e16b0a…ebc78`；MemFabric Host 双路径 `libmf_hybm_core.so` 同 SHA `88d39285…7bc1`；主工作树 `vllm_ascend_C…so` `8eb65de9…bea1`、`libmf310p_v2_trial_device.so` `b8ca96f0…6811`、`libmf310p_device.so` `8447a55a…55df`。隔离 worktree 内 SO 构建产物缺失（gitignored），符合 R1-B §8.5 预期；未复制、未替换任何已安装库。
- **CPU 合同测试**（真实 serve Python、隔离 worktree 根、PYTHONPATH 前缀指向 r1b）：`unittest discover tools/310p_mm_ar` **33/33 PASS**；`pytest csrc/_310P/memfabric_mm_ar_v2/test/ut` **54/54 PASS**。
- **只读 preflight + import-smoke**：`g0_dflash.json`：`static_go=true`、`blockers=[]`、pip check `KNOWN_SERVER_STATE_WAIVED`（20 条已核验豁免、0 条 unexpected）、真实 serve 顺序导入探针 exit=0、目标/草稿 config.json 均在。
- **KFC/首写 guard**：`MF_KFC_BYPASS`、`MF310P_V2_NOWARMUP`、`MF310P_V2_AR_ONLY_NOWARMUP` 均未设置。

## 3. R1-1 失败根因（本轮核心产出，证据闭环）

### 3.1 现象

服务以合同参数启动（TP2、K=7、`cudagraph_mode=NONE`、用户原始对齐参数），**目标模型 Qwen3.6-35B-A3B-w8a8 在双 Rank 完整加载**（1324 条 weight load 日志，含 GDN 层），随后 `drafter.load_model` 阶段两个 Worker 同时失败：

```
NotImplementedError: DFlash drafters with mixed sliding/full attention require
the V2 model runner; relaunch with VLLM_USE_V2_MODEL_RUNNER=1.
```

完整链路（`root_cause_traceback.txt`）：`model_runner_v1.py:4030 load_model → _310p/spec_decode/llm_base_proposer_310.py:249 → spec_decode/llm_base_proposer.py:335 _get_model → get_model → qwen3_dflash.py:674 → DFlashQwen3Model.__init__ → qwen3_dflash.py:123 _resolve_layer_attention` → 上游 raise。服务 rc=1 退出，trial 记录 `service exited before health, rc=1`（`run.outcome.json`）。

### 3.2 根因链（每一环均有源码/实测证据，`g0/root_cause_source_evidence.txt`）

1. **草稿模型是混合滑窗/全注意力架构**：`/home/models/Qwen/Qwen3.6-35B-A3B-DFlash/config.json` 的 `layer_types = [sliding_attention ×5, full_attention ×1]`（`sliding_window=4096`、`max_window_layers=6`）。上游注释明确该形态属新生态格式（如 z-lab/gemma-4-31B-it-DFlash），**需要多 KV 组**。
2. **上游 vllm 0.27.1 的 V1 runner 不支持该形态**：`qwen3_dflash.py:115-127` 对 `0 < num_sliding < len(layer_types)` 且非 V2 runner 直接 `NotImplementedError`——这是上游有意的架构限制，不是缺依赖。
3. **自动强制 V2 的上游路径被本插件的 310P 策略关闭**：上游 `VllmConfig.use_v2_model_runner` 会在 DFlash 需要多 KV 组时自动返回 True；但 `vllm_ascend/patch/platform/patch_use_v2_model_runner.py`（PR #11389）将其覆盖为**只读 `VLLM_USE_V2_MODEL_RUNNER` 环境变量，未设置即 False**——因为 NPU V2 runner 尚不能承接上游默认启用的全部模型/特性。因此本次启动在 V1 runner 下进入草稿构造并触发第 2 条门禁。
4. **显式 `VLLM_USE_V2_MODEL_RUNNER=1` 被上游 Triton 校验硬拒绝**：真实 serve 导入顺序 + 补丁就位后的 CPU 探针（`g0/probe_v2_runner_gate.log`）实测：`VllmConfig()` 构造即抛 `ValueError: Model Runner V2 requires Triton.`（`config/vllm.py:2257 _validate_v2_model_runner`）。本服务器 310P Docker **有意不装 Triton**（G0 已核验的已知状态）。
5. **装 Triton 也救不了**：插件自身注释（`patch/worker/patch_idex_310.py:105-110`）："310P has no functional Triton: kernels cannot be compiled (bishengir-compile fails on Ascend310P3)"；310P 各模块因此强制 `HAS_TRITON=False` 走 PyTorch 回退。即 310P 上不存在“让 V2 runner 通过校验且真实可用”的 Triton 安装路径。

### 3.3 结论

**Stock+DFlash（K7、本草稿模型、vllm 0.27.1、310P 无 Triton 环境）在合同冻结的四个自由度内无解**：V1 runner 被上游草稿架构门禁拒绝；V2 runner 被上游 Triton 校验拒绝，且 310P 无功能性 Triton。这不是 pip 缺包、不是我们集成代码缺陷、也不是设备/通信问题——**历史“Stock+DFlash 启动失败、疑似缺依赖”的根因自此从 UNKNOWN 变为已证据闭环**：用户草稿模型为混合滑窗/全注意力新格式，而本环境只能提供 V1 runner。

目标模型加载、ascend w8a8 量化、TP2 双 Rank、HCCL、GDN 路径均正常——失败边界严格限定在草稿模型架构门禁。

## 4. 现场恢复确认（实测）

服务进程组由 trial 正常 SIGTERM 清理；复核：宿主无 vllm/trial 进程；设备 6/7 无进程、hugepages 0/0、显存回到基线（1612/1338MB，与介入前 1613/1338 同量级）；端口 8000/8581 空闲。本轮对系统的全部改动：新增隔离 worktree（可 `git worktree remove`）、`/tmp/opencode/r1b_k7_20261025_001/`（全量证据，保留）、本分支文档/证据提交。**未安装/卸载任何包、未替换任何 SO、未清理设备 4 残留、未触碰不可见 PID 占用的 0-3 卡。**

## 5. 需要所有者裁决的选项（本轮不擅自执行）

| 选项 | 内容 | 代价/风险 |
|---|---|---|
| A | 换用**全 full-attention** 的 DFlash 草稿模型（如 z-lab/Qwen3.5-9B-DFlash 形态，上游注释确认 V1 可跑） | 偏离用户冻结的草稿模型与 K7 业务口径；需用户重新指定并核验模型 |
| B | 给上游 `qwen3_dflash.py` 写 V1 多 KV 组支持补丁 | 上游明确以“V1 无多 KV 组”为由拒绝；工程量大且背离上游，长期维护差 |
| C | 在 310P 上启用 V2 runner 路线：需同时补丁 Triton 校验并解决 310P 无功能性 Triton 的全部运行时路径 | 插件自身已确认 310P Triton 不可编译；V2 runner + DFlash 在 310P 从未验证；属于新研发方向而非本轮适配 |
| D | 接受现状：DFlash 投机在 310P/本草稿组合上暂不可用，R1-B 以 NO-GO 归档，主线回到大 M/小 M 非投机目标 | 无额外风险；P0-2（DFlash 薄适配）暂停待上游或模型侧变化 |

## 6. 原始证据索引（`csrc/_310P/memfabric_mm_ar_v2/evidence/r1b_20261025/`；全量同结构保留于 `/tmp/opencode/r1b_k7_20261025_001/`）

| 文件 | 内容 |
|---|---|
| g0/g0_dflash.json（+stdout） | R1-B 只读 preflight + import-smoke：static_go=true、blockers=[] |
| g0/g0_unittest.log / g0_pytest_ut.log | CPU 合同 33/33、54/54 PASS |
| g0/g0_hccl_probe.log | 本轮设备 6/7 HCCL 双进程探针 PASS |
| g0/g0_versions.txt / g0_so_sha256.txt / g0_npusmi.txt / ports_models_processes.txt / python_vllm_identity.txt | 版本、SO/KVER SHA、设备表、端口/模型/进程身份 |
| g0/worktree_list_before.txt / worktree_status_each_before.txt | 现场保护快照（9 worktree 全 clean） |
| g0/probe_v2_runner_gate.py/.log | 决定性探针：env=1 时 VllmConfig 构造即 `requires Triton` 失败 |
| g0/root_cause_source_evidence.txt | 上游/插件源码摘录 + 草稿 layer_types 实测 |
| stock_eager_smoke/manifest.json | 冻结的 R1-1 计划（含 env、完整 server/bench 命令） |
| stock_eager_smoke/root_cause_traceback.txt | 草稿构造失败完整 traceback（双 Worker + EngineCore） |
| stock_eager_smoke/server.log.gz（37MB 全量）/ server.pid / run.outcome.json / execute.stdout.log | 完整服务日志与 trial 执行记录 |

# R1 现场G0裁决：Stock/V2+DFlash K7 依赖准入 NO-GO（四组全部未启动）

> 2026-10-25；执行者 OpenCode；唯一主线 `dev/310p-memfabric-mm-ar` @ `07586f2f22303c61d65c063241074b34e9dd42b7`（隔离 worktree `/home/w00804146/310P/vllm-ascend-r1`）。执行合同：[DFLASH_STOCK_V2_R1_GO_GATE_20261025.md](DFLASH_STOCK_V2_R1_GO_GATE_20261025.md)。
>
> **裁决：G0 = NO-GO（只读依赖 preflight 6 blockers，static_go=false）。按合同"G0 未全部通过禁止启动模型"与用户安全约束"环境版本不一致立即停止"，R1-1～R1-6 全部 NOT_RUN，未启动任何模型服务，未占用任何 NPU Die 做推理，未修改任何正式 Kernel/通信协议/安装库/系统 SO。**

## 1. 四组运行状态（独立、不得互相替代）

| 组 | 状态 | 依据 |
|---|---|---|
| G0 只读依赖+设备核对 | **执行完毕，NO-GO**（6 blockers） | `evidence/r1_g0_20261025/g0_stock_dflash.json`、`g0_stock_dflash_importsmoke.json` |
| R1-1/R1-2 Stock/V2+DFlash Eager smoke | **NOT_RUN（被 G0 阻断）** | 本轮无任何 server.log/bench.json，无模型加载 |
| R1-3/R1-4 Stock/V2+DFlash Eager full | **NOT_RUN（被 G0 阻断）** | 同上 |
| R1-5/R1-6 Stock/V2+DFlash Graph smoke | **NOT_RUN（前置 Eager 均未运行）** | 同上 |

无任何 DFlash 接受率、Target Forward 次数、融合覆盖率、Graph capture/replay 数据——这些只能来自实机运行，本轮一律不填写、不推算。

## 2. G0 通过项（真实证据）

- **现场保护**：7 个既有 worktree 全部干净并记录（`worktree_list_before.txt`、`all_worktrees_status_before.txt`）；本轮仅新增隔离 worktree `vllm-ascend-r1`（detached @ 07586f2f2）与证据目录，未 checkout/reset/clean 任何在用工作树。
- **CPU 合同测试**：`python3 -m unittest discover -s tools/310p_mm_ar` 26/26 PASS；`python3 -m pytest -q csrc/_310P/memfabric_mm_ar_v2/test/ut` 54/54 PASS（实际 serve 所用 Python `/usr/local/python3.12.13/bin/python3`，R1 worktree 根目录，PYTHONPATH 指向 R1）。
- **模型文件**：目标 `/home/models/Qwen/Qwen3.6-35B-A3B-w8a8`（arch `Qwen3_5MoeForConditionalGeneration`）与草稿 `/home/models/Qwen/Qwen3.6-35B-A3B-DFlash`（arch `DFlashDraftModel`，6 层）config 均在。**注意：真实路径比 GO_GATE/工具常量多一级 `Qwen/`**（见 §4-F）。
- **同卡双 Die 映射**：NPU43520 = 设备 6/7（同 Bus `0000:AB:00.0`，同 310P3），当前空闲、无进程，为本轮计划使用对（与 F1-C 相同）；NPU38144 = 设备 4/5（设备 4 有 13096MB 残留显存与 5860/5860 hugepages，npu-smi 进程表与 /proc 均无属主，判定为历史泄漏残留，未动）；NPU5664(0/1)、NPU16384(2/3) 被不可见 PID 895887/1420543 占用（跨命名空间/容器，未触碰）。
- **版本/SO 指纹**（`g0_versions.txt`、`g0_so_sha256.txt`）：Python 3.12.13；torch 2.10.0；torch_npu 2.10.0.post4；transformers 5.17.0；vllm 0.27.1+empty（editable ← `/vllm-workspace/vllm` = **vllm-project/vllm 官方上游** @ `6e448d0ea9bf`，2026-08-11，DFlash 文件在树内）；CANN toolkit 9.1.0（V100R001C11SPC001B243）；Driver 26.1.1；KVER11 `libmf_sdma_orch_v11.json` SHA256 `25e16b0a…ebc78`；MemFabric Host 双路径（`/usr/local/memfabric_hybrid/1.2.0/.../lib64` 与 `site-packages/memfabric_hybrid/lib`）`libmf_hybm_core.so` 同 SHA256 `88d39285…7bc1`；主 worktree `vllm_ascend_C...so` SHA256 `8eb65de9…bea1`、`libmf310p_v2_trial_device.so` `b8ca96f0…6811`、`libmf310p_device.so` `8447a55a…55df`。
- **HCCL 健康**：引用 F1-C 同设备(6,7) HCCL 探针 PASS 证据（2026-10-24，`g0_hccl_probe_f1c_reference_20261024.log`，RANK0/RANK1 result0=3.0）；本轮因 G0 fail-stop 未重复执行设备探针。
- **端口/进程**：8000/8581/4077 空闲；无 vllm/python 服务进程。

## 3. G0 blockers 与逐项根因（本轮核心产出）

冻结合同工具 `dflash_stock_preflight.py`（K7）原始裁决（`g0_stock_dflash.json`）：`static_go=false`，blockers 5 项；追加 `--import-smoke`（`g0_stock_dflash_importsmoke.json`）后第 6 项。

### B1/B2/B3 版本 pin 不匹配（vllm 0.27.1+empty≠0.24.0；torch-npu 2.10.0.post4≠post2；transformers 5.17.0≠5.13.0）

**根因：工具/打包元数据停留在上游 vllm v0.24.0 时代，而服务器实际工作基线已经演进到 vllm 0.27.1，且我方代码已完成适配。** 证据链：

1. 本仓 `requirements.txt:17,38,39` 仍声明 `torch-npu==2.10.0.post2`、`transformers==5.13.0`、`fastapi<0.124.0`（editable 安装元数据同源，pip check 亦据此报冲突）——这是 preflight PINNED 的出处。
2. 但主线历史已含 `1b6abadc8 fix(worker): adapt worker and patches to vLLM v0.26/v0.27 API changes`；`vllm_ascend/__init__.py:_install_fla_compat_alias()` 明确注释"vLLM v0.27.0 moved the FLA ops package"，`model_runner_310p.py:49-55` 有 cp_utils 新旧双 lane 兼容 shim。代码是面向 0.27 写的。
3. 近期全部实机成功记录（F1-C、SEL-01、F1/F2/D4，2026-10-21～25）均运行于 vllm 0.27.1+empty / post4 / 5.17.0。
4. 服务器 vllm 是官方 vllm-project/vllm@6e448d0（非魔改分支），`qwen3_dflash.py`、`v1/spec_decode/dflash.py` 等 DFlash API 文件在树内，preflight 的 6 项 API 文件检查全部 True。

**结论：不是环境缺依赖，而是 G0 工具的 pin 基线与服务器已验证工作基线脱节。**

### B4 `MISSING_SOURCE_API:vllm/cp_utils.get_total_cp_world_size`

vllm 0.27 将该函数改经 `vllm.distributed.get_dcp_group`（安装版 `cp_utils.py:9,57` 可证）。我方 `model_runner_310p.py:49-55` 的 try/except shim 已兼容此改动，真实导入验证（§3-B6 测试F）通过。**preflight 的 AST 顶层函数检查不认识兼容 shim，属工具检查逻辑缺陷。**

### B5 `PIP_CHECK_UNEXPECTED_FAILURE_OR_UNREADABLE`

完整 `pip check` 输出（`g0_pip_check_full.txt`）分三类：a) CANN 工具链（ms-service-profiler/mindstudio-kpp/affinity-sched）缺 matplotlib/pandas/plotly 等宿主机工具依赖，与 serving 无关；b) vllm-ascend 元数据缺 arctic-inference/memcache-hybrid/pandas/pandas-stubs/quart/triton-ascend（上游声明、310P Docker 有意不装，arctic/memcache 本就在工具 OBSERVE 清单）；c) 版本冲突 fastapi 0.136.3、torch-npu post4、transformers 5.17.0——全部源于 B1/B2/B3 的同一份过时元数据。**无一是运行时缺包：该 pip 状态与所有近期成功 serving 完全相同。** 工具分类器只豁免 Triton 缺失，无法区分"元数据过时"与"真实破坏"。

### B6 `ISOLATED_IMPORT_FAILED_OR_TIMED_OUT`（探针循环导入）——导入顺序取证

探针真实 traceback：`model_runner_310p:59 → dflash_full_and_piecewise:28 → attention_v1:68 → device_op:31 → ops/__init__:21 → fused_moe:41 → experts_selector:25 → device_op(部分初始化) → ImportError: cannot import name 'DeviceOperator'（circular import）`。

受控导入顺序实验（全部原始日志已归档 `g0_import_*.log`）：

| 测试 | 顺序 | 结果 |
|---|---|---|
| A | 探针原顺序（冷进程直导 model_runner_310p） | FAIL：循环导入 |
| B | 先 `import vllm_ascend.ops` | FAIL：`vllm.model_executor.layers.fla` 缺失（别名未装） |
| C | 先装 fla 别名再直导 model_runner_310p | FAIL：仍循环导入 |
| D | 先 `import vllm_ascend.platform` 再导 worker_310p | FAIL：仍循环导入（platform 模块导入不触发 ops） |
| E | 先 `adapt_patch(is_global_patch=True)` 再导 worker_310p | model_runner_310p **OK**（ops/device_op 已完整），后续失败仅为本轮测试者 PYTHONPATH 覆盖失误造成的 `acl` 伪缺失（真实环境 `PYTHONPATH` 含 CANN site-packages，`/usr/local/Ascend/cann-9.1.0/python/site-packages/acl` 存在） |
| F | **真实 serve 顺序**：`vllm` → `NPUPlatform.pre_register_and_update(None)`（platform.py:184-188，serve 早期平台钩子）→ `worker_310p` → DFlash 完整链 | **全链 OK**：ops/device_op/fla 别名就位，worker_310p、DFlashQwen3ForCausalLM、DFlashProposer、patch_idex_310 全部导入成功 |

**结论：循环导入是探针合成导入顺序的伪影，不是环境缺陷。** 真实 serve 经 `pre_register_and_update` 先行 `adapt_patch` 完整初始化 ops，规避了 device_op→ops→…→device_op 环。工具探针的导入序列不忠实于 serve 路径（缺陷），但它作为只读子进程探针**忠实地复现了"冷进程直导 DFlash runner 路径会崩"这一事实**——这正是历史上"Stock+DFlash 启动失败、疑似缺依赖"最可能的直接形态（ImportError 易被误读为缺包）。历史失败确切 traceback 仍缺失，本轮只能给出高置信候选根因，不宣称已证实。

## 4. 附加现场发现（G0 应记录项）

- **F. 工具模型路径常量错误**：`dflash_v2_trial.py` 硬编码 `/home/models/Qwen3.6-35B-A3B-w8a8`（GO_GATE 文本同），服务器实际为 `/home/models/Qwen/Qwen3.6-35B-A3B-w8a8`。preflight 因支持 `--target/--draft` 参数可用真实路径执行；trial 工具不加修复将直接拒绝启动（`target or DFlash draft model config missing`）。
- **G. R1 隔离 worktree 无 SO 构建产物**：`vllm_ascend_C.cpython-312-aarch64-linux-gnu.so`、`libmf310p_v2_trial_device.so` 等为 gitignored 构建产物，只存在于主 worktree。已验证主 worktree(f1c 分支 bd422330a) 与主线在 `csrc/_310P/memfabric_mm_ar_v2/src`、`vllm_ascend/_310p|ops|device|quantization` 生产源码 diff 为空，主 worktree 现存 SO 与主线源码对应（SHA 见 §2）。**若 G0 通过，V2 组启动前仍需在 R1 worktree 重建或经批准后复制 SO 并留存 SHA，否则 V2 serve 会因 `import vllm_ascend.vllm_ascend_C` 失败。**
- **H. editable 安装指向主 worktree**：site-packages 的 editable finder 将 `vllm_ascend` 映射到 `/home/w00804146/310P/vllm-ascend`（f1c 分支）。R1 试验必须显式 `PYTHONPATH=<r1-worktree>:$PYTHONPATH`（已验证可覆盖，注意不可裸覆盖——会丢 CANN acl 路径）。本轮所有 R1 侧检查均带此前缀。
- **I. 设备 4 残留显存**：13096MB + hugepages 满，无属主进程；未清理（保护现场），后续若需用 4/5 卡需先经批准处置。

## 5. 最小修复（随本报告同分支提交，供评审，不改变本轮裁决）

分支 `work/310p-mm-ar-r1-g0-nogo-20261025`，全部为工具/测试/文档，**未触碰任何正式 Kernel、Host ABI、Serving 运行时代码**：

1. `dflash_stock_preflight.py`：PINNED 更新为服务器已验证工作基线（vllm 0.27.1、torch-npu 2.10.0.post4、transformers 5.17.0，附证据注释）；cp_utils 检查接受 v0.26/0.27 `get_dcp_group` lane（要求我方 shim 同时在树）；pip check 分类器精确豁免"已验证服务器的已知可选缺包/过时元数据冲突"并逐条落 observations；`--import-smoke` 探针改为真实 serve 导入顺序（含 `pre_register_and_update`）。
2. `dflash_v2_trial.py`：TARGET/DRAFT 更正为服务器真实路径；execute 增加 worktree 一致性守卫（`vllm_ascend` 解析根必须位于本工具所在 worktree，否则拒绝启动并提示 PYTHONPATH），防 R1 manifest 下静默运行主 worktree 代码。
3. 同步更新两个纯 CPU 合同测试；本地复跑 30/30 unittest + 54/54 pytest 全部 PASS。
4. **修复后 preflight 对同一未改变服务器环境的实测结果**（`g0_stock_dflash_after_fix.json`，供评审复核，不用于放行本轮）：`static_go=true`、blockers 空、20 条已知状态逐条落 observations、0 条 unexpected、真实 serve 顺序 import 探针 exit=0。NO-GO 与 GO 的差异完全位于工具合同层，不在环境层。
5. GO_GATE 文末附现场勘误附录（模型真实路径、G0 工具发现），冻结正文未改动。

**修复后的工具未经评审批准不得用于放行本轮任何模型启动；本轮裁决以冻结合同原始输出为准。** requirements.txt 的过时 pin（torch-npu/transformers/fastapi）属生产打包元数据，按约束需项目所有者批准后另行更新，本轮未改。

## 6. 现场恢复确认

本轮对系统的全部改动：新增 R1 worktree（git worktree，可 `git worktree remove`）、`/tmp/opencode/r1_k7_20261025_001/`（证据，保留）、`/home/w00804146/310P/r1_run_evidence/g0/`（保护快照，保留）、本分支文档/工具提交。**未安装/卸载任何包、未替换任何 SO、未写任何 NPU 设备状态、未删除任何现场文件。设备 0-7 占用状态与介入前一致。恢复方案 = 保留全部证据 + 按需移除新增 worktree；F1-C 备份目录 `/tmp/opencode/f1c_g0/backup_20261024T165834` 未动。**

## 7. 原始证据索引（`csrc/_310P/memfabric_mm_ar_v2/evidence/r1_g0_20261025/`，另有 `/tmp/opencode/r1_k7_20261025_001/` 全量）

| 文件 | 内容 |
|---|---|
| g0_stock_dflash.json / g0_stock_dflash_importsmoke.json | 冻结合同 preflight 原始裁决（6 blockers + 真实 traceback） |
| g0_stock_dflash_after_fix.json | 修复后工具对同一环境的复测（static_go=true，供评审） |
| g0_unittest.log / g0_pytest_ut.log | CPU 合同 26/26、54/54 |
| g0_import_alias_first/platform_first/adapt_patch/real_serve_order.log | 导入顺序取证 A-F |
| g0_pip_check_full.txt | 完整 pip check（exit=1 全文） |
| g0_versions.txt / g0_so_sha256.txt / g0_npusmi.txt | 版本、SO/KVER SHA256、设备快照 |
| g0_hccl_probe_f1c_reference_20261024.log / g0_f1c_env_reference.sh | F1-C HCCL PASS 引用与现场环境口径 |
| worktree_list_before.txt / all_worktrees_status_before.txt | 现场保护快照 |

## 8. 下一轮准入条件（建议，待批准）

1. 评审并合入本分支工具修复；或项目所有者明确选择"把服务器对齐回 0.24.0 pin"路线（需重装 vllm/torch-npu/transformers，风险与工作量远高于修 pin，且与代码已适配 0.27 的事实矛盾）。
2. 用修复后 preflight 重跑 G0（预期 static_go=true），并人工复核设备/SO/KVER/HCCL（含设备 4 残留处置决策）。
3. V2 组启动前解决 §4-G 的 R1 worktree SO 供给（重建或经批准复制+SHA 记录）。
4. Stock Eager smoke → V2 Eager smoke → full → Graph 的顺序不变；Graph 仍仅受控诊断。

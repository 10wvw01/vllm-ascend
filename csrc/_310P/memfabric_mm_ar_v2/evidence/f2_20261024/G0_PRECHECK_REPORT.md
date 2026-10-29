# 310P3 F2 实机 G0 准入审查报告（2026-10-24）

> 执行人：OpenCode（现场实机核验）。裁决对象：F2 分支 `perf/310p-f2-mm-sdma-overlap-sweep-20261024` 的 18 场 Chunk×Overlap 扫描准入。
> 依据：`docs/F1_F2_ONDEVICE_PREFLIGHT_GATES_20261024.md`（G0-1~G0-8）。
> **总结论：G0 全部 PASS（含两项经记录的受控部署动作），允许进入 F2 性能采样。**

## G0-1 代码版本 / 工作树 — PASS

| 项 | 证据 |
|---|---|
| 分支 / HEAD | `perf/310p-f2-mm-sdma-overlap-sweep-20261024` @ `43fe496312a7bedae2bfd56ee55d7fad50627093`（`git fetch` 后 fast-forward 到 origin 最新） |
| 工作树 | `git status --porcelain` 输出为空（tracked 全干净；`vllm_ascend/*.so` 为未跟踪运行时构件，见 G0-3） |
| 对比 D4 基线 | `git diff 33e3f2cfc81e HEAD --stat`：仅新增 6 个文件（F2 计划/门禁文档 2、test/perf 3、test/ut 1），**`src/op_kernel`、`src/op_host`、协议、Host ABI 零改动**（`git diff 33e3f2cfc81e HEAD -- csrc/_310P/memfabric_mm_ar_v2/src/` 输出为空） |
| 未混入 C3 | 工作树无 C3 分支内容；Kernel 源含 `kProductionAddTile = 2048`（UT `test_protocol_kernel_not_edited_on_f2_branch` PASS） |

## G0-2 硬件：单卡双 Die 独占 — PASS

- `npu-smi info`（快照 `npu_smi_pre_sweep.txt`）：选用 **卡 43520（Bus-Id 0000:AB:00.0）的 Device 6/7**，两 Die 同属一张 310P3 卡；
  dev6：AICore 0%、1612MB（空闲基线水位）；dev7：AICore 0%、1339MB；进程表 "No running processes found in NPU 43520"。
- 其他卡对：5664(dev0) 有外部租户 PID 895887（13GB）；16384(dev2/3) 有外部租户 PID 285327（36GB×2、dev3 AICore 96%）；38144(dev4) 有 13GB 残留占用——均不触碰。
- **物理↔逻辑映射**：`ASCEND_RT_VISIBLE_DEVICES=6,7` → 进程内 local_rank 0 = 物理 dev6、local_rank 1 = 物理 dev7（orch_deploy_helper 以逻辑 0/1 探测分别命中两 Die 的 KVER11 kernel，映射正确）。
- dmesg 全量扫描 dev6/7：0 条错误记录。

## G0-3 编译 / 部署 / 运行时 SO — PASS（经两项受控部署，见"部署动作"）

| 二进制 | 路径 | SHA256 | 判定 |
|---|---|---|---|
| Python Host 扩展（插件 SO） | `/home/w00804146/310P/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so` | `8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1` | PASS：与 D4 全程使用的 Gate C 诊断 ON 构建逐字节同源；`nm -D` 导出 `mf310p_v2_trial_create/ib_probe/run/check_status/destroy`（另有 `run_ar_only`、`run_nz_direct_small`） |
| **设备 SO（F2 装载物）** | `/home/w00804146/310P/vllm-ascend/vllm_ascend/libmf310p_v2_trial_device.so` | `2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19` | PASS：**D4 已实机验证的 Tile2048 诊断 ON 候选**（`/tmp/opencode/d4/build/device_candidate_diag_on.so`，源 `5adbf1810`，2185952B）；非 D4 结束时还原的旧版 `b8ca96f0`，非 C3 |
| Host SO（MemFabric 核心） | `/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/libmf_hybm_core.so` 与 site-packages 双路径 | `1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30` | PASS：D4 dev6/7 迁移实证的物理设备号修复版，两路径一致（G0-5） |
| KVER11 JSON | `/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json` | `25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78` | PASS：与 D4/D1-A 同源 |
| 装载链验证 | 扩展 RUNPATH `$ORIGIN:…` → 实际加载 repo `vllm_ascend/` 下设备 SO；`LD_LIBRARY_PATH=/usr/local/memfabric_hybrid/latest/aarch64-linux/lib64`（`latest→1.2.0`） | — | PASS：`import vllm_ascend.vllm_ascend_C` 成功（替换前 `88d39285` 原版与替换后 `1dc46d7d` 补丁版均验证可装载） |

**部署动作（受控，含备份与恢复方案）：**

1. 设备 SO：D4 结束时按恢复清单还原为 pre-D4 `b8ca96f0`（G0 实测确认）。F2 按 G0-3 要求重新部署 D4 Tile2048 `2a19390c`。原状备份：`/tmp/opencode/f2_g0/backup_20261024T151617/libmf310p_v2_trial_device.so.pre_f2`（`b8ca96f0…`）。
2. Host SO：装机原版 `88d39285` 不含 dev6/7 物理号修复（非 0/1 可见列表下 `aclrtMemSetAccess` 必现 507899）。按 D4 先例部署补丁版 `1dc46d7d` 至 `/usr/local/.../lib64` 与 site-packages 双路径。原状备份：同目录 `libmf_hybm_core.so.usr_local_orig` / `.site_packages_orig`（均为 `88d39285…`，另存 D4 备份 `/tmp/opencode/d4/c0/backup_hostso_20261023T153335/`）。
3. 恢复方案：F2 全部结束、证据落盘后，将备份原件拷回两路径并复核 `88d39285`、设备 SO 还原 `b8ca96f0`（见 F2 报告收尾节）。
4. 替换前 fd 扫描确认无进程持有上述 SO；替换后逐路径 `sha256sum` 复核（上表值即复核结果）。

## G0-4 F1 小 M 诊断 ABI — PASS（预检，F1 切换时复核）

- 当前插件 SO 为 **Gate C 诊断 ON 构建**（`8eb65de9`，D4 环境记录明确"diagnostics-ON build, unchanged since D1-A"）；`nm -D` 实测导出 `mf310p_v2_trial_run_nz_direct_small`。
- F1 执行前将在 F1 分支复核 `V2_HOST_LIB` 指向同一构建并重跑 `nm -D` 取证。

## G0-5 MemFabric 物理设备号修复 — PASS

- dev6/7 为非 0/1 卡对，必须使用 `fix/310p-setaccess-physical-devid-20261023` 修复版 Host SO：已部署 `1dc46d7d`（G0-3 表），双安装路径一致。
- 实机验证：M2048/C1024/O1 试运行 `mf310p_v2_trial_create` 建池成功（含 meta GVA/own 段/imported 段全部 `aclrtMemSetAccess` 授权），**未出现 507899**。

## G0-6 KVER11 / KFC / HCCL — PASS

- `MF_SDMA_ORCH_JSON=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json`（真实装载路径，SHA `25e16b0a`）。
- 设备侧 KVER11（经 KFC 通道在 AICPU 侧执行，宿主机无该目录属预期）：`orch_deploy_helper <0|1> <json>` 双 Die 探测均输出 `kernel already present: /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/libmf_sdma_orch_v11.so`（rc=3），D4 部署物仍在位；helper 二进制 `849f93c0…` 与 Gate C/D1-A/D4 同源。
- KFC 通道功能性：helper KFC init/exec 全部成功（上述探测即经 KFC 通道完成）。
- HCCL 探针（`hccl_probe.py`，dev6/7，HCCL all_reduce + gloo barrier）：`HCCL_PROBE_PASS sum=128.0`，rc=0。
- 安全机制：环境无 `MF_KFC_BYPASS`、无 `MF310P_V2_NOWARMUP`、无 `MF_SDMA_ORCH_DEPLOY_HELPER` 包装器残留（`env` 快照 `ambient_env.txt`）；驱动脚本解析期亦 fail-closed 拒绝绕过变量。

## G0-7 本地工具验证（310P3 实机环境） — PASS

| 命令 | 结果 |
|---|---|
| `bash -n test/perf/f2_run_sweep.sh` | rc=0 |
| `python3 -m py_compile test/perf/f2_chunk_overlap_perf.py test/perf/f2_sweep_analyze.py` | rc=0 |
| `python3 -m pytest -q test/ut/test_f2_sweep_contract.py` | **3 passed**（参数网格/配对拒绝/协议源未改） |
| 受控试运行（正确性先行） | `M2048/C1024/O1`，torchrun TP2：3 预热 + 3 采样，**双 Rank 逐样本 bit-exact、status=0**，rc=0；warmup0 13.2s（首写保护+建池后首次 launch，预期），warmup1/2 event ≈1.38–1.41ms；会话日志见服务器 `/tmp/opencode/f2_g0/trial_m2048_c1024_o1/session.log` 与逐 Rank JSON |

## G0-8 采样隔离 — PASS

- Profiler：驱动不加载 Profiler（`profile: false` 写入每份 Rank JSON）；无 `MS_PROFILER*`/`PROFILING*` 环境变量。
- 无 Graph/DFlash 干扰：试运行进程仅运行 trial 驱动；未启动 vllm serve。
- 环境指纹：CANN 9.1.0（`V100R001C11SPC001B243`）、驱动 26.1.1、torch 2.10.0+cpu、torch_npu 2.10.0.post4、vllm 0.27.1、内核 `6.6.0-72.0.0.76.oe2403sp1.aarch64`（192 核）、主机时间 2026-10-24 15:18 Asia。
- F1/F2 不并行：F2 扫描期间独占 dev6/7；每场独立 torchrun 进程，15 分钟超时 + 30 秒强杀保护。

## 环境偏差记录（如实披露）

1. `libhcom.so`/`MEMFABRIC_HYBRID_EXTEND_LIB_PATH`、`sdid/server id/super pod id query failed on 310P (ret:207000)` 等 WARN 为 310P 平台已知无害告警（D4 同款日志，不影响功能）。
2. 空闲 Die 显存 ~1.3–1.6GB 为运行时基线水位（与历史会话一致）。
3. dev0/dev4 有外部租户残留占用、dev2/3 被外部租户高负载占用——均与本实验无关，不触碰、不清理。

## G0 裁决

**PASS** —— 允许执行 F2 18 场扫描（`f2_run_sweep.sh`）。F2 全部完成、设备与 KFC/HCCL 状态复核清洁后，方允许切换 F1 分支。

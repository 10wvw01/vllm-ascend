# F1-C G0 实机准入报告（2026-10-25）

> 分支：`perf/310p-f1c-stock-fp16-hccl-vs-memfabric-abba-20261025` @ `4af36b6fa0fd`（与任务指定已审查提交一致）。
> 结论：**G0 全项 PASS**（硬门槛逐项如下，原始证据见本目录及 `/tmp/opencode/f1c_g0/`）。
> G0-5 中的受控短跑（Stock/T1 Smoke）按计划文档 §4.1 在本报告 PASS 后、八场 ABBA 前独立执行，其结果记录于本目录 `g0_smoke_*.log` 与第 4 节。

## G0-1 代码与分支身份：PASS

| 项 | 证据 |
|---|---|
| HEAD | `4af36b6fa0fd`（`git pull --ff-only` 后 Already up to date；`git status --porcelain` 为空，工作树干净） |
| 相对 F1-1 HEAD 差异 | `git diff --stat b14fbe2f863e8 4af36b6fa0fd`：仅新增 5 个文件（1 计划文档 + `test/perf/f1c_{arm,analyze}.py`、`test/perf/f1c_run_abba.sh`、`test/ut/test_f1c_abba_contract.py`），**839 行全部为 docs/test，无任何正式 Kernel、Host ABI、通信协议或 Stock 默认路由改动**；不含 C3 优化、不含正式 V1 Decode 路由变更 |
| 生产源码未修改 | `git status --porcelain --untracked-files=no` 输出为空（`vllm_ascend/*.so` 均在 `.gitignore:32 *.so`，部署二进制不污染跟踪树） |

## G0-2 资源与物理映射：PASS

| 项 | 证据（`g0_npu_smi.txt`） |
|---|---|
| dev6/7 同卡双 Die | 卡 `43520`（Bus `0000:AB:00.0`）Chip0=Device6、Chip1=Device7，均为 310P3、Health OK |
| 独占性 | 进程表：`No running processes found in NPU 43520`；AICore 0%/0%；显存 1614/1337MB（空闲基线，与 F1-1 收尾记录一致） |
| 其他租户 | 5664(dev0/1)、16384(dev2/3)、38144(dev4/5) 有外部占用，与本次目标卡无关，不触碰 |
| 设备映射 | `ASCEND_RT_VISIBLE_DEVICES=6,7`；torchrun `--nproc-per-node=2`：local_rank0→物理 dev6、local_rank1→物理 dev7 |
| dmesg | dev6/7 无新增错误（最近 dev6/7 条目 uptime 72378/120712s，当前 uptime 144699s；最近 devmm 错误在其他租户设备 devid=1） |

## G0-3 二进制身份与受控部署：PASS

**部署前状态**（F1-1 收尾已恢复装机旧态，实测核对一致）：
- 设备 SO `vllm_ascend/libmf310p_v2_trial_device.so` = `b8ca96f0…`（装机旧版）
- Host SO 双路径 = `88d39285…`（装机原版）

**备份**（先备份后部署，`/tmp/opencode/f1c_g0/backup_20261024T165834/`，SHA 见 `g0_backup_sha256sums.txt`）：
- `libmf310p_v2_trial_device.so.pre_f1c` = `b8ca96f0…`
- `libmf_hybm_core.so.usr_local_orig` = `88d39285…`
- `libmf_hybm_core.so.site_packages_orig` = `88d39285…`
- fd 扫描：无任何进程持有上述 SO（`FD_SCAN_DONE` 无输出）

**受控部署**（部署源均为历史实机验证过的构建产物，逐字节 SHA 复核）：

| 目标路径 | 部署源 | 部署后 SHA256（实测） |
|---|---|---|
| `vllm_ascend/libmf310p_v2_trial_device.so` | `/tmp/opencode/d4/build/device_candidate_diag_on.so`（D4 记录 `d4_build_record.json` 候选 diag-ON，源 5adbf1810） | `2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19` ✓ |
| `/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/libmf_hybm_core.so` | `/tmp/opencode/mf-fix/build/src/hybm/csrc/libmf_hybm_core.so`（物理设备号修复版） | `1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30` ✓ |
| `/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_core.so` | 同上 | `1dc46d7d…` ✓（双路径一致） |
| `vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so` | 未改动（Gate-C 诊断 ON 构建一直在位） | `8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1` ✓ |
| `/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json` | 未改动（KVER11 保留） | `25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78` ✓ |

**ABI 与运行进程实际加载路径实证**（不止文件存在性）：
- `nm -D vllm_ascend_C…so` 实测导出：`mf310p_v2_trial_create`、`mf310p_v2_trial_ib_probe`、`mf310p_v2_trial_run_nz_direct_small`、`mf310p_v2_trial_check_status`、`mf310p_v2_trial_destroy`（T1 ABI 在位）；
- 源化 `set_env.sh` 后 Python 实际导入并从 `/proc/self/maps` 证明进程加载：
  - `…/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so`（`8eb65de9`）
  - `…/vllm-ascend/vllm_ascend/libmf310p_v2_trial_device.so`（即部署的 D4 `2a19390c`）
  - `/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/libmf_hybm_core.so`（即部署的 `1dc46d7d`）
- 注意事项：`libmf_hybm_core.so` 不在 ldconfig 缓存，必须 `source /usr/local/memfabric_hybrid/set_env.sh`（历史轮次同口径）。

## G0-4 协议与资源前置：PASS

| 项 | 证据 |
|---|---|
| HCCL 独立探针（不建 MemFabric 池） | `torchrun --nproc-per-node=2 /tmp/opencode/f1c_g0/hccl_probe.py`，dev6/7 HCCL all_reduce FP16：`RANK0/RANK1 HCCL_PROBE PASS result0=3.0`，rc=0（`g0_hccl_probe.log`） |
| 端口 8581 | 宿主无 `ss`/`netstat`/`ip`（极简环境，初次 G0 用 `ss` 取证无效，如实更正）。改用新增 `test/perf/f1c_port_probe.py` 解析 `/proc/net/tcp{,6}`：**listeners: NONE**（全系统零 LISTEN 套接字，8581 空闲，fail-closed rc=0）（`g0_port_listeners.txt`，重新取证）。该修复已随分支提交并通过 bash -n/py_compile/UT 8/8 回归（含新增端口守卫合同测试） |
| KFC | `MF_KFC_BYPASS`、`MF310P_V2_NOWARMUP` 均未设置（`printenv` 为空）；KFC/AICPU KVER11 部署在 T1 Smoke 建池时由 deploy helper 实证（预期 `already present`） |
| 设备侧 KVER11 | F1-1 收尾按 D4 先例保留；JSON `25e16b0a` 在位（上表）；T1 Smoke 建池时探测 |
| 首写/stream guard | 由未设置 `MF310P_V2_NOWARMUP` 保证开启（库默认） |

## G0-5 本地代码质量：PASS

| 项 | 结果 |
|---|---|
| `bash -n csrc/_310P/memfabric_mm_ar_v2/test/perf/f1c_run_abba.sh` | PASS |
| `python3 -m py_compile f1c_arm.py f1c_analyze.py` | PASS |
| `python3 -m pytest -q test/ut/test_f1c_abba_contract.py` | **7 passed**（含 fail-closed、per-index max、审计门、2ms 长尾、manifest 指纹合同） |
| 短跑冒烟 | 见第 4 节（Stock/T1 各一场 30 样本，独立进程分别执行） |

## G0-6 可恢复性：PASS

- 回滚命令（实验结束执行，恢复到本报告"部署前状态"）：
  ```
  cp /tmp/opencode/f1c_g0/backup_20261024T165834/libmf310p_v2_trial_device.so.pre_f1c \
     /home/w00804146/310P/vllm-ascend/vllm_ascend/libmf310p_v2_trial_device.so
  cp /tmp/opencode/f1c_g0/backup_20261024T165834/libmf_hybm_core.so.usr_local_orig \
     /usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/libmf_hybm_core.so
  cp /tmp/opencode/f1c_g0/backup_20261024T165834/libmf_hybm_core.so.site_packages_orig \
     /usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_core.so
  # 复核：b8ca96f0… / 88d39285…（双路径）
  ```
- 不修改外部租户或系统任务；故障退出后不自动切下一任务。

## 运行环境口径（供复现）

- CANN 9.1.0（`ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0`）、torch 2.10.0+cpu、torch_npu 2.10.0.post4、Python 3.12.13、npu-smi 26.1.1
- `source /usr/local/memfabric_hybrid/set_env.sh` 后设置：
  - `ASCEND_RT_VISIBLE_DEVICES=6,7`
  - `MF_SDMA_ORCH_JSON=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json`
  - `V2_HOST_LIB=/home/w00804146/310P/vllm-ascend/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so`
  - `F1C_DEVICE_SO_PATH=/home/w00804146/310P/vllm-ascend/vllm_ascend/libmf310p_v2_trial_device.so`
  - `F1C_CORE_SO_LIB64=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/libmf_hybm_core.so`
  - `F1C_CORE_SO_PY=/usr/local/python3.12.13/lib/python3.12/site-packages/memfabric_hybrid/lib/libmf_hybm_core.so`

## 4. G0-5 附：受控短跑冒烟结果（§4.1，`g0_smoke_stock.log`/`g0_smoke_t1.log` 与 `smoke_stock/`、`smoke_t1/` JSON）

| 冒烟 | 结果 |
|---|---|
| Stock FP16（HCCL 默认组 + gloo CPU 参考组，不建 MemFabric 池） | rc=0；双 Rank 30/30 样本位精确（`torch.equal` vs CPU gloo FP32→FP16 参考）；ev_p50 317.94/326.38µs，无 ≥2ms 样本；进程正常退出 |
| T1 MemFabric（gloo-only + MemFabric 池，MM8/Add4，无 HCCL group） | rc=0；双 Rank 30/30 位精确、`protocol_status=0`；KFC helper 双 die `kernel already present`（KVER11 AICPU 部署实证，G0-4 补证）；ib_probe 1000 次 PASS；ev_p50 382.76/374.35µs，无 ≥2ms 样本 |
| 路由取证（独立诊断会话，profiler text 导出） | Stock：`hcom_allReduce` COMMUNICATION 任务（Stream=N/A，跨流）+ AI_CORE MatMul/TransData（stream 37），3/3 被捕获迭代 `event_us` ≥ 含 hcom 结束点全簇跨度（1064>706、436>253、410>261µs）→ **HCCL Event 覆盖完整通信**；T1：仅单 kernel `V2NzDirectSmallKernel<16,8>` 于调用者流（stream 41），**零通信任务**。详见 `route_proof/`（正式输出目录内）与 `/tmp/opencode/f1c_route_proof/route_proof_analysis.txt` |

**Stock FP16 与 FP32 参考位精确（无量化差异），未发生 STOP 条件。**

## 裁决

**G0 PASS + 双冒烟 PASS + 路由取证 PASS**：允许启动八场正式 ABBA（`f1c_run_abba.sh`）。


# V2 目录重构实机回归报告（V2_REFACTOR_REGRESSION_20261019.md）

> 回归对象：`feat/310p-memfabric-public-api` 分支目录重构 + 正式/诊断代码拆分 + 构建调整，
> 即提交区间 `decba8b5f`（V2 迁移为 CANN 直调 src/docs/test 工程）与
> `4a4ed4069`（按 CANN 层隔离正式 Kernel 与 T0/T1/T2 诊断代码，HEAD）。
> 基线：重构前 `c503616f4`（G3 归档证据冻结点 `6746898c6`，见
> [ARCHIVE_20261019_G3.md](ARCHIVE_20261019_G3.md)）。
> 执行时间：2026-10-19 15:35–19:13（机器时钟），全部结论以实际执行日志为准，
> 原始日志索引见 §12。

## 1. 总结论

| 项 | 结论 |
|---|---|
| 目录重构/拆分/构建调整整体 | **PASS（修复 1 个编译级重构 BUG 后）** |
| 发现的重构引入问题 | **2 项**：① 默认（DIAGNOSTICS=ON）构建编译失败（已最小修复并复测）；② 旧布局遗留 `__pycache__` 使布局 UT 失败（已清理） |
| 基线遗留 P1（T2/V1 回归缺符号） | **已解决**：重构后 ON 构建导出全部 T1/T2 试用 API，T2/V1 实测全部 PASS |
| 既有 P0（Eager 性能负收益、长稳 hang、Graph 旁路） | 未复测、未修复（按本次任务范围明确排除）；未观察到行为变化 |
| 真实 Qwen Eager 功能 | 40 目标层启用、V2/Stock 路由、V2 算子级位精确 **PASS**；serve 级 greedy 全量比对受**栈级非确定性**干扰（stock 自身可复现翻转，详见 §8），该非确定性为既有已归档行为，**非本次重构引入** |

**判定依据**：重构改动只移动/拆分文件与构建清单，不改 MM/SDMA/同步算法；
本次实机逐项验证编译、符号、Torch Op 注册、算子位精确性、混合路由、复用与冷启动，
与基线逐项对比无算法级回归；唯一功能性缺口（T2/V1 缺符号）由本次拆分构建补齐。

## 2. 环境与版本确认（实测）

| 项 | 实测值 | 与要求/基线一致性 |
|---|---|---|
| 硬件 | Ascend 310P3（dav-2002），8 卡整机，验收使用卡 5664 dev0/dev1，`ASCEND_RT_VISIBLE_DEVICES=0,1` | 与 G3 基线一致 |
| CANN | `/usr/local/Ascend/cann-9.1.0`，`version=9.1.0`（V100R001C11SPC001B243），bisheng dav-2002 | CANN 9.1 ✓ |
| MemFabric 运行包 | `/usr/local/memfabric_hybrid` 1.2.0，CommitId `2f59dedb`（与部署源一致），`libmf_sdma_orch_v11.json`（KVER=11） | KVER=11 ✓ |
| 编排 AICPU kernel | `/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/libmf_sdma_orch_v11.so` 在位（T2 日志 `[helper] kernel already present`，KFC 独立部署助手路径正常） | ✓ |
| 模型 | `/home/models/Qwen/Qwen3.6-35B-A3B-w8a8`（Qwen3_5Moe，full_attention_interval=4 → 10 层 Full Attention + 30 层 GDN） | 与 G3 一致（注：备注中 `/home/models/Qwen3.6-35B-A3B-w8a` 实际位于 `Qwen/` 子目录） |
| TP | 2（torchrun --nproc-per-node=2） | ✓ |
| /dev/shm | 2.0G（TP MessageQueue 需要 ≥160MiB） | ✓ |
| Python 包 | vllm_ascend 为 editable 安装指向本仓库树（已核实 `.pth` finder） | ✓ |

## 3. 旧构建产物清理与防旧 .so 加载

1. `rm -rf build/`（清除重构前 `build/temp*` 全部 CMake 产物）；ON/OFF 各自使用独立全新构建目录。
2. 旧 `vllm_ascend/vllm_ascend_C.*.so`（10-19 03:14，重构前构建）与
   `libmf310p_v2_trial_device.so`（10-18 09:02）被新构建覆盖，切换前后 md5 全程留痕
   （`/tmp/opencode/regression/on_so_before_off.md5`）。
3. **防旧 .so 关键措施**：基线 T2/V1 失败的 forensic 复核表明，旧试用目录
   `/tmp/opencode/v2/libmf310p_v2_trial_device.so` **不含** `mf310p_v2_ar_stage_launch_async`，
   而旧 `run_t1.sh` 曾导出 `LD_LIBRARY_PATH=/tmp/opencode/v2:$LD_LIBRARY_PATH`
   （LD_LIBRARY_PATH 优先于 RUNPATH）——加载旧设备库即为当时 ImportError 的成因链。
   本次回归：所有运行脚本内置守卫（`env.sh` 检测到 `/tmp/opencode/v2` 在 LD_LIBRARY_PATH 即拒绝执行），
   T0 驱动改用**从重构后源码全新构建**的独立 host/device 库（§6）。
4. Graph 编译缓存 `~/.cache/vllm/torch_compile_cache` 在 E2E 前清除（§11.4 纪律）。

## 4. 源码 UT

### 4.1 V2 工程静态 UT（`csrc/_310P/memfabric_mm_ar_v2/test/ut`）

- 首次执行 `test_v2_project_layout.py::test_v2_root_has_only_docs_src_test` **FAIL**：
  工程根残留 `__pycache__/`（内含 `bench_ar_only.cpython-312.pyc` 等**重构前旧布局**产物，
  decba8b5f 已将对应 .py 移入 `test/`）。删除该目录后 **16/16 全 PASS**。
  判定：重构搬迁遗留的未跟踪产物，非源码缺陷；已清理。

### 4.2 仓库 310P UT 套件（`tests/ut/_310p`，1056 项）

- 结果：**1038 passed / 2 skipped / 8 failed / 8 errors**。
- 失败集合：`test_memfabric_mm_ar_source.py`（2）、`test_dflash_mamba_scheduler_310.py`（4）、
  `test_prefix_cache_behavior.py`（1）、`test_worker_memory_behavior.py`（8 ERROR）。
- **基线复核**：在重构前 `c503616f4` worktree 上执行同一失败集合，**失败项完全一致**
  （`test_memfabric_mm_ar_source.py` 期望 V1 ABI v7 而源码已演进到 v9、
  dflash/prefix/worker_memory 均与重构无关——重构提交未触碰这些文件）。
- 判定：**全部为既有失败，非本次重构回归**。

## 5. 双构建验证（VLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS ON/OFF）

### 5.1 OFF（正式最小构建）

- 独立干净目录 `/tmp/opencode/regression/build_diag_off`（预置 CMakeCache
  `VLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS:BOOL=OFF`，`setup.py build_ext --build-temp` 指向该目录）。
- 编译/链接：**PASS**。仅编译 `op_host/mf310p_v2_trial_adapter.cpp` + `torch_adapter/mf310p_v2_serve.cpp`
  （build.make 清单核实），无诊断源。
- 导出符号：`nm -D` 仅有 5 个正式 C ABI（`trial_create/destroy/ib_probe/run/check_status`）；
  无 T1/T2/地址探针/`_debug_status`。
- 未定义符号仅 4 个正式 kernel 入口（`trial_launch_async/ib_probe_async/ib_fan_probe_async/
  ib_workspace_bytes`），与 OFF 设备库（仅含 `mf310p_v2_kernel.asc` 产物）导出**一一对应**，导入解析 PASS。
- Torch Op 注册（实测 import）：`memfabric_mm_ar_allreduce`（V1）、`memfabric_mm_ar_v2_allreduce`、
  `memfabric_mm_ar_v2_shutdown` 注册；`memfabric_mm_ar_v2_debug_status` **正确缺席**。
- **实机能力验证**：`off_smoke.py`（tp2 混合路由变体，不使用任何诊断 Op）：
  **PASS 60/60**（36 次 V2 正式算子位精确 + 24 次池存活后 HCCL Stock），零断言/零错误。
- 判定：OFF 构建保留正式 Serving 能力且不依赖诊断符号，**PASS**。

### 5.2 ON（默认构建，含诊断）

- **首建失败（本次回归发现的最严重问题，详见 §10 BUG-1）**：
  `mf310p_v2_trial_diagnostics.cpp:350/362` 编译错误
  `'mf310p_v2_trial_launch_async' was not declared in this scope`。
  最小修复（补 4 行 extern 声明）后重建 **PASS**。
- 干净默认目录 `build/temp.linux-aarch64-cpython-312`，CMakeCache `DIAGNOSTICS:BOOL=ON`；
  编译全部 4 个 V2 源文件（adapter/diagnostics/serve/serve_diagnostics）。
- 导出符号：正式 5 个 + 诊断 4 个（`trial_run_nz_direct_small/trial_run_ar_only/trial_diag/
  trial_read_send_arena`）全部 `T`；未定义符号 14 个（含 `mf310p_v2_ar_stage_launch_async` 等
  T1/T2 kernel 入口）与 ON 设备库（kernel.asc + diagnostics.asc）导出**全部对应**。
- Torch Op 注册（实测 import）：V1 allreduce、V2 allreduce/_shutdown/_debug_status 全部注册；
  `ctypes` 直取 `mf310p_v2_ar_stage_launch_async` 等 5 个符号可用。
- 判定（修复后）：**PASS**；修复前该分支**默认构建即不可编译**，属重构引入的编译级回归。

## 6. T0/T1/T2/V1 回归（全部在 ON 构建，设备 0,1）

T0 独立驱动说明：仓库 `test/g3_e2e_scripts/regression.sh` 不含 T0 驱动
（需 `MF310P_V2_DIAG_DRIVER_DIR` 指向含 `v2_trial.py` 的目录，否则 SKIP，见 §9）。
本次使用归档驱动 `/tmp/opencode/v2/v2_trial.py`（驱动本身零改动）+
**从重构后源码全新构建**的 `libmf310p_v2_trial_host.so`（adapter+diagnostics+V1 adapter）
与两台设备库（构建脚本 `/tmp/opencode/regression/build_t0_host.sh`），不复用任何旧 .so。

| 项 | 结果 | 关键证据/基线对比 |
|---|---|---|
| T0 G0（IB 探针 1000+10000 reps） | **PASS** | `G0 ALL PASS`；per_rep 8.55/7.69us（基线 8.50/7.33us，噪声内） |
| T0 G1（串行多 chunk 位精确 256/512/1024） | **PASS** | `G1 ALL PASS`（全矩阵位精确） |
| T0 G2（serial/overlap M=2048/4096/8192） | **PASS** | `G2 ALL PASS`；med 1.448/2.730/5.255ms（基线 1.518/2.667/5.286ms，噪声内） |
| T0 Stock 基线 | **PASS** | med 1.681/3.114/6.960ms（基线 1.726/3.140/7.003ms） |
| T1 full（M=1..256 矩阵 ×20） | **PASS** | `MATRIX ALL PASS` |
| T1 stress（30 唯一 M ×4 mm ×2 add ×4 seed） | **PASS** | `random_m 1200 calls PASS` + `seq_1000 1000 calls PASS`（基线 160+1000，本次覆盖更全） |
| T2 AR-only（chunk 1024 × repeat 100） | **PASS（基线未能执行）** | `STRESS PASS: 100 calls, unique_m=30, adds=[1,2,4,8], fresh_y=1`，466.3s，协议状态校验通过 |
| V1 层级 bench（M=255..8192 ×5） | **PASS（基线未能执行）** | 8 个 M 全部 `PASS max_abs_diff=0.000000 n_bad=0`，`ALL PASS` |

- T2/V1 为 G3 基线 P1（§9/§12.10：`ImportError: undefined symbol: mf310p_v2_ar_stage_launch_async`，
  当时记"未执行"）：本次在干净环境 + ON 构建下**实际执行并全部通过**，该遗留项闭环。
  需要说明：基线失败成因含环境成分（§3 LD_LIBRARY_PATH 旧库污染）与构建成分（当时 .so 未含
  T2 试用 API）；本次重构后源码+干净构建使两条成因均消除。
- 所有位精确判据均为实算（`torch.equal` / n_bad / max_abs_diff），非仅看 rc。

## 7. G3 Gate B（ON 构建）

| 项 | 结果 | 证据 |
|---|---|---|
| 边界 M + 混合路由（plan 20 项×3） | **PASS 60/60** | 36 V2 + 24 HCCL，含 1/10/2047/4097/5000→Stock、2048/2049/3000/3073/4095/4096→V2，全部位精确，协议 status=0 |
| chunk=256 | **PASS 60/60** | 同上，staged 预热对 256 档成立 |
| chunk=512 | **PASS 60/60** | 同上 |
| 1000 次混合复用（reps=50） | **PASS 1000/1000** | 600 V2 + 400 HCCL；4 输出槽循环复用，零协议错误、零数值错误 |
| 新进程冷启动 ×20（含 75s meta 排空） | **PASS 20/20** | `TOTAL pass=20 fail=0`；无 507899/507018；基线为 18/20（2 轮已知一次性失败：cycle3 SIGSEGV、cycle17 冷损坏，均已归档），本次无任何失败 |

Gate B 判据（位精确 + 协议状态 + 实际执行）全部满足，与基线 §3/§5 等价或更好。

## 8. 真实 Qwen3.6 TP2 Eager 功能回归（3 次完整执行）

入口：`pytest tests/e2e/_310p/test_memfabric_mm_ar_functional.py::test_fused_matches_stock_eager`
（`MEMFABRIC_MM_AR_E2E_ENABLED=1`、`MEMFABRIC_MM_AR_E2E_V2=1`、dev0,1）。

### 8.1 三次运行中稳定成立的部分（每次均验证）

| 检查项 | 结果 |
|---|---|
| 40 目标层启用 | **PASS ×3**：两 rank 合计 80 处 `Select MemFabricMmArV2LinearMethod310`（10 o_proj + 30 out_proj 每 rank），`count_enabled_layers` 断言 10/30 通过 |
| V2/Stock 路由 | **PASS ×3**：路由日志 `M=4095 within [2048,4096]: V2 MM+MemFabric (chunk=1024)`；M<2048（短 prompt prefill/decode）走 `native MM + HCCL`；dummy/profile 期 warmup fallback 生效 |
| V2 算子位精确（进程内首调校验） | **PASS ×3**：`[mf310p-v2] first fused call verified bit-exact vs stock (M=4095, aligned=False)` 双 rank 全部通过（含 staged 深度扫描预热） |
| long_4095（V2 融合 prefill）逐 token 输出 | run1/run2 与 stock **一致**；run3 翻转（见 8.2，翻转发生在 decode 段） |

### 8.2 serve 级 greedy 全量比对：3 次均 FAIL —— 判定为既有栈级非确定性，非重构回归

三次失败均为个别 prompt 的 greedy 翻转，且**方向可逆**：

| 运行 | 失败项 | 内容 |
|---|---|---|
| run1（17:33） | short_0 | stock=' Paris' vs fused=' Pariserta'（M=7，双侧均执行纯 Stock 代码） |
| run2（17:45） | short_1 | stock=' John' vs fused=' Johnrena' |
| run3（18:59） | short_2 + long_4095 | stock='0而降' vs fused='0'（**stock 侧出现垃圾后缀，方向反转**）；long decode 段翻转 |

**证据链（全部本次实测）**：

1. **stock vs stock 也翻转**：18:41 双 stock serve 对照，`batched[4]`（"Count from one to twenty,
   step 4"，即基线 §6.4 归档翻转的那条 prompt）A='1' vs B='1自达' 不一致；
2. **单一 stock serve 内部即翻转**：18:05 探针，short_1 五次重复出现 ' John' 与 ' Johnrena' 两种输出；
3. **翻转发生在 V2 不参与的路径**：short/decode M<2048，fused serve 执行与 stock 完全相同的
   `unquantized_gemm + HCCL AR`（V2 方法 stock 分支）；三次失败时 V2 池/算子在这些 step 零执行
   （路由日志核实）；
4. **V2 参与的 prefill 段位精确**：每次首调校验（算子级）与 run1/run2 long 全文比对均一致；
5. **代码等价性**：`git diff c503616f4..HEAD -- vllm_ascend/` 为空——serving 全部 Python 代码与
   重构前逐字节一致；重构 C++ 改动仅在 V2 算子内部（上述翻转 step 不执行）。

**结论**：40 目标层、V2/Stock 路由、V2 算子输出正确性 **PASS**；
serve 级 greedy 全量比对在当前时段受栈级近并列 logit 噪声支配（基线 2026-09-22/2026-10-19 §6.4
已归档同类行为，当时噪声窗口较低可复跑通过；本时段翻转率高，3 次均无法对齐）。
该非确定性为**既有环境/栈问题，非本次重构引入**；按任务要求不将其记为 V2 回归。
（备注：本节两轮辅助探针 short_probe/det_pair 曾漏设 `MEMFABRIC_MM_AR_E2E_V2=1`，
实际对照的是 V1 路由——不影响上述结论，反而额外提供了 V1 同类翻转与 stock 自身翻转证据；
正式 V2 判定均取自 3 次标准功能测试。）

## 9. regression.sh 覆盖范围核实

- 仓库 `test/g3_e2e_scripts/regression.sh` 逻辑：T0 G0/G1/G2/Stock **仅在**
  `MF310P_V2_DIAG_DRIVER_DIR` 指向含 `v2_trial.py` 的现存目录时执行，否则打印
  `SKIP G0/G1/G2/Stock` 继续 T1/T2/V1（脚本 18–45 行）。**裸跑会静默跳过 T0**——
  本次已用全新构建驱动目录补齐 T0 实测，未把 SKIP 记为 PASS。
- T1/T2/V1 由仓库脚本无条件执行，本次全部实际跑通（§6）。
- 归档基线 `/tmp/opencode/g3/regression/status.txt`（t0×4 rc=0、t1 rc=0、t2 rc=1、v1 rc=1）
  未被覆盖，保留为对照证据。

## 10. BUG 清单与修复（仅重构引入项）

| # | 级别 | 问题 | 根因 | 处置 |
|---|---|---|---|---|
| BUG-1 | **编译级（重构引入）** | `VLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS=ON`（默认）构建失败：`mf310p_v2_trial_diagnostics.cpp:350` `mf310p_v2_trial_launch_async` 未声明 | `4a4ed4069` 拆分翻译单元时，diag=5 路径调用的正式 Kernel 发射函数声明留在 `mf310p_v2_trial_adapter.cpp`，诊断 TU 未获得声明 | 最小修复：在诊断 TU 顶部补 `extern "C"` 声明（与正式 adapter 完全一致的签名，4 行）；不触碰任何算法。修复后 ON 构建编译/链接/符号/Torch Op/全部实机测试通过。**该 BUG 使重构提交的默认构建不可用，必须随本修复一起合入** |
| BUG-2 | 遗留产物（重构相关） | `test_v2_project_layout.py` 布局断言失败：工程根多出 `__pycache__/` | decba8b5f 搬迁 .py 后，旧布局字节码残留（未跟踪文件） | 删除该 `__pycache__`；UT 16/16 恢复 PASS。无源码改动 |

未修改项：MM/SDMA/同步协议/路由算法零改动；V1 源码与 serving Python 栈零改动（git diff 为空已核实）。

## 11. 与重构前基线的差异汇总

| 维度 | 基线（6746898c6 证据冻结） | 本次（4a4ed4069 + BUG-1 修复） | 差异判定 |
|---|---|---|---|
| 默认构建 | PASS（单文件布局） | 修复前 FAIL（BUG-1）→ 修复后 PASS | 重构引入后修复 |
| OFF 最小构建 | 未验证 | PASS（编译/链接/符号/Op 注册/实机 60/60） | 新增能力验证（README 承诺兑现） |
| T0 G0/G1/G2/Stock | PASS | PASS，性能噪声内（§6 表） | 等价 |
| T1 full | PASS | PASS（stress 覆盖 1200+1000 > 基线 160+1000） | 等价或更全 |
| T2 AR-only | **未执行（缺符号，P1）** | **PASS 100 calls** | **补齐/改善** |
| V1 层 bench | **未执行（同上）** | **PASS 8×M 位精确** | **补齐/改善** |
| Gate B（路由/边界/chunk/1000 次） | PASS | PASS（60×3 + 1000/1000） | 等价 |
| Gate B 冷启动 ×20 | PASS（18/20，2 轮已知一次性失败） | PASS 20/20 | 等价或更好 |
| Qwen Eager 功能（Gate C） | PASS（噪声窗口低，一次通过） | 40 层/路由/V2 位精确 PASS；greedy 全量比对受栈级非确定性干扰（stock 自身可复现，§8.2） | 无算法回归证据；环境噪声时段差异 |
| Graph 模式 | 无效（torch.compile 旁路，P0） | 未测（任务范围排除） | — |
| Eager 性能 / 长稳 | FAIL（P0，既有） | 未测（任务范围排除） | — |

## 12. 原始日志索引（`/tmp/opencode/regression/`）

- `logs/status.txt` 全项 rc 记录；`logs/t0_*.log`、`t1_full.log`、`t1_stress.log`、
  `t2_ar_stress.log`、`v1_bench.log`、`gateb_*.log`、`off_smoke.log`
- `cold_cycles/`（cycle_1..20.log + summary.txt：pass=20 fail=0）
- `logs/e2e_run1|2|3/`（stock/fused serve 日志：层选择、路由、首调校验证据）、
  `qwen_eager_functional*.log`（3 次 pytest 全文）
- `logs/short_probe.out`、`det_pair.out`、`stock_vs_stock.out`、`stock_vs_stock_det.out`
  （§8.2 非确定性证据链，含 V1 误配置探针的原样保留）
- `build_diag_off.log`、`build_diag_on.log`（双构建全量输出）；
  `on_so_before_off.md5`（ON/OFF 制品切换与还原校验）
- 对照基线（未覆盖）：`/tmp/opencode/g3/regression/`（2026-10-13:34–13:49 原始回归）

## 13. 复现命令索引

```bash
# UT
pytest -q csrc/_310P/memfabric_mm_ar_v2/test/ut
pytest -q tests/ut/_310p

# 双构建（先 OFF 后 ON，均需 CANN/MemFabric env + SOC_VERSION=ascend310p3
#        + VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1(+_V2=1)）
#   OFF：干净目录预置 CMakeCache VLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS:BOOL=OFF
#        后 python setup.py build_ext --inplace --build-temp <off_dir>
#   ON ：rm -rf build && python setup.py build_ext --inplace

# 回归（设备 0,1；MF_SDMA_ORCH_JSON=.../libmf_sdma_orch_v11.json；
#       STORE_URL=tcp://127.0.0.1:8581；T0 需 MF310P_V2_DIAG_DRIVER_DIR 指向
#       全新构建驱动目录，见 /tmp/opencode/regression/build_t0_host.sh）
bash csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts/regression.sh   # T0 需显式给驱动目录
torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/test/e2e/tp2_v2_mixed_route_correctness.py --reps 50
DEVS=0,1 bash csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts/cold_cycles.sh <out>

# Qwen Eager 功能
MEMFABRIC_MM_AR_E2E_ENABLED=1 MEMFABRIC_MM_AR_E2E_V2=1 \
  pytest -sv tests/e2e/_310p/test_memfabric_mm_ar_functional.py::test_fused_matches_stock_eager
```

# A1-REENTRY｜310P3 服务器切换其它分支后的环境复原、MemFabric 编译部署与高效批量实机准入

> 日期：2026-10-27；适用 [主线大M实验总账](https://github.com/10wvw01/vllm-ascend/blob/dev/310p-memfabric-mm-ar/csrc/_310P/memfabric_mm_ar_v2/docs/LARGE_M_PREFILL_TTFT_EXPERIMENT_LEDGER.md) G-01/A-01/A-02/L1-01～L3。实施分支 `work/310p-mm-ar-large-m-a1-20261026`，PR#9 Draft。本文件是**现场第一批委托的具体操作与证据门禁**，不是当前设备 READY 或性能 GO。服务器已切换其它分支，所有旧环境身份/G0必须作废重新检查。
>
> **批处理原则：一次OpenCode会话尽可能做多的无副作用工作**；使用隔离 git worktree + 独立目录完成环境盘点、CPU/静态测试、源码版本溯源、隔离编译和备份/回滚**方案**，但绝不将脚本“批量”理解为可以越过部署/设备/短Smoke的人工许可。本轮 E0/E1不执行任何设备计算、KFC、安装/替换 SO 或修改其它分支的现有服务。

## 0. 当前远程可信指纹（源自已归档实机与GitHub源码，不代表服务器现态）

| 资产 | 来源/现有已知身份 | 现场必须确认 |
|---|---|---|
| vllm项目权威主线 | `10wvw01/vllm-ascend:dev/310p-memfabric-mm-ar`；2026-10-27初始总账版本 `f3eafd76a668622fa1b8e7f8787d25e7bca594e4` | 另建隔离A1工作树；不能在别的任务的工作树 `git switch/pull/reset/clean` |
| A1具体实现 | `10wvw01/vllm-ascend:work/310p-mm-ar-large-m-a1-20261026`；以委托时`git rev-parse HEAD`为准 | 仓库工作树 tracked clean；记录本次精确SHA；U1脚本与工具存在 |
| D4 Device SO冻结 | `2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19` | 文件SHA、真实装载路径、与正式D4 Kernel一致 |
| Python vllm_ascend插件冻结 | `8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1` | `vllm_ascend_C.__file__`指向所核路径；隔离检查Py版本及ABI |
| MemFabric Host系统/ Python两份 SO冻结 | 历史通过非0/1映射验证的`1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30` | **两份的完整SHA一致**、有效装载来源一致、不得拿未修复`88d39285...`凑数 |
| KVER11 `libmf_sdma_orch_v11.json`冻结 | `25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78` | 对应Device AICPU编排核/launch JSON/符号 `HybmSdmaOrchEpoch`；全路径/版本 |
| MemFabric历史服务器修复源码 | 2026-10-23记录`/tmp/opencode/mf-fix/build`和`/tmp/opencode/d4/c0/backup_hostso_20261023T153335/`；旧源分支名称`fix/310p-setaccess-physical-devid-20261023`可能仅本地 | **先查本地对象、worktree、git log、git diff及所有备份**。不要假定目录还存在，也不要把历史路径当本次部署的绝对路径 |
| MemFabric GitHub现有镜像 | `10wvw01/memfabric_hybrid_310p` main = `131ee5ffd1a6ae9d4824cbb0cf4d32d99bbfdc5a`（与`10wvw01/memfabric_hybrid:fix-310p-kfc-deploy-helper`对应）；检查7个镜像分支的三个SetAccess处**均未发现物理映射修复** | 现场其他Git对象、未推送分支、备份可能存在修复；主仓新修复候选 `10wvw01/memfabric_hybrid:work/310p-setaccess-redeploy-20261027` 只能当待构建/静审代码，**不可宣称与历史1dc46d7d库二进制等价** |

**严重误区：**另一MemFabric文档中历史`AclrtMemSetAccess`“增加meta授权”是**增加SetAccess调用**（修复设备meta可见性），**不是**“逻辑DeviceID转物理ID”这一独立的非identity VMM修复；两者不能混为一谈。

## 1. Stage E0｜只读环境重建审计（本次已准备工具，可与CPU合批）

1. 在**全新独立目录**归档 run_id、当前操作用户、`uname`/`npu-smi info`/driver/CANN/Python/torch/torch_npu/vllm版本、系统进程/端口、4张物理卡2Die拓扑和MCU/Alarm，禁止停止别的任务的进程或隐式强制释放NPU。
2. 在所有服务器仓库**只读**`git worktree list`、HEAD、branch、`status --porcelain`，尤其`memfabric_hybrid*`与旧`/tmp/opencode/mf-fix`；寻找`git reflog`/`git log --all`的fix变更、`git stash list`（禁止stash pop）、`*.patch`、已归档`libmf_hybm_core.so`备份及全SHA。只读取证，不自动挑最新分支覆盖历史修复。
3. 确定独立A1与MemFabric**新 worktree**的工作路径、只读模型`/home/models/Qwen3.6-35B-A3B-w8a8`或现场真实位置、已有CANN/装机库/插件/设备SO/KVER11的真实路径和所属用户；记录包元数据与`readlink -f`。特别查`/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/libmf_hybm_core.so`与`site-packages/memfabric_hybrid/lib/libmf_hybm_core.so`两处是否均为正确`1dc46d7d…`；不是就记录原状，不擅自改。
4. 使用新增`test/perf/a1_reentry_readonly.py`写`environment_inventory.json`；该工具**绝不导入torch、执行Kernel、改SO或自动授予G0**。选卡清单`card_selection.json`必须人工/现场附加，不可仅凭工具自动判定同卡拓扑。
5. 查有毒环境变量：`MF_SDMA_ORCH_DEPLOY_HELPER`伪rc3、`MF_KFC_BYPASS`、`MF310P_V2_NOWARMUP`、`MF310P_V2_AR_ONLY_NOWARMUP`任一出现即隔离环境无效；常驻epoch`MF_SDMA_ORCH_EPOCH_BUDGET_S=4294967295`不得进入Serving；默认v11 15s Epoch是历史正式形态。
6. 确认A1程序所需的物理卡能从四卡中择优、同一板卡两Die健康、空闲、独占。旧43520 dev6/7 / Die7 MCU Alarm在现场未解除证据前排除。服务器正由其它分支使用：**同卡被占用=STOP/选其它空闲卡，不驱赶他人、不改他们的环境**。

E0结论必须分成`ENV_INVENTORY_COMPLETE`、`ENV_MISMATCH`、`CARD_AVAILABLE`、`G0_NO_GO`等不同字段；脚本生成结果永远`G0_APPROVED=false`。

## 2. Stage E1｜同一现场会话尽量完成CPU回归、源码溯源、隔离编译（不部署）

### 2.1 A1/vllm-ascend原仓CPU回归

按当前工作分支真实HEAD进入全新隔离worktree，设置取证目录：
```bash
P=csrc/_310P/memfabric_mm_ar_v2
python3 -m py_compile "$P/test/perf/a1_reentry_readonly.py" "$P/test/perf/a1_g0_readonly.py" "$P/test/perf/a1_large_m_assess.py"
python3 -m unittest discover -s "$P/test/ut" -p 'test_a1_*.py' -v
python3 -m pytest -q "$P/test/ut"
bash -n "$P/test/perf/u1_run.sh"
```
每一项保存stdout/stderr、exit code、测试总数、用到的Python、HEAD；失败不继续装机。额外静态核查`u1_run.sh` Gate `U1_G0_APPROVED/U1_EXCLUSIVE_APPROVED/U1_RESTORE_PLAN_VERIFIED`和五个SHA（见上表），不能通过修改冻结SHA绕过不匹配。

### 2.2 MemFabric历史源码优先复用（避免反复踩坑）

1. **路径 A（优先）**：找到完整、可验证的旧 `1dc46d7d…` Host SO（两份备份或可恢复归档）及其原始修复`mf-fix`工作树/diff/提交；固定源码commit、未提交差异`git diff`、build.ninja/CMakeCache和SO哈希，保存source-manifest。**不必无条件重编正确老库**。
2. **路径 B（仅 A 不可用）**：用当前已存在的`memfabric_hybrid_310p`镜像核实7分支的修复缺失；只在新的隔离worktree考虑 `10wvw01/memfabric_hybrid:work/310p-setaccess-redeploy-20261027` 的候选**三处SetAccess修复**。先编译其`tests/310p_device_id_mapping_test.cpp`做纯CPU映射边界测试，再按现场仓库`build.sh`/`script/build_and_pack_run.sh`、CANN与Python ABI查明**原本使用的编译参数**，编译进**隔离staging目录**；不得直接`pip install`或`cmake --install`到生产前缀。
3. 保持**Host仅3处SetAccess换算**；`aclrtMallocPhysical`、Device KVER11、launch JSON、SDMA与IB协议均不改；对源代码真实检查`HybmPhysicalDeviceIdOf`与3次授权调用，另核对历史实机探针：非identity上`MallocPhysical`逻辑ID、`SetAccess`物理ID在当时CANN 9.1才成立。若CANN升级必须重新评审probe合同，**不得自动执行具有设备操作的探针**。
4. 候选SO首次构建 SHA **不能假定等于** `1dc46d7d…`：相同功能不意味着字节级相同，必须单独提交`source_sha/build_flags/dependencies/ABI/host_so_new_sha`与`U1冻结SHA变更提案`等待研发/用户审查，**禁止自行编辑U1锁定指纹**；构建失败提交完整错误，不覆盖现装库。
5. vllm-ascend同理：若现场已有历史冻结`D4 Device SO 2a193…`、插件`8eb65…`且依赖版本兼容，优先验证后复用；如必须重建，另在独立staging完整记录CANN/bisheng/Python ABI及动态链接/符号/校验，不能在不确认部署前随手`pip install -e`覆盖其它项目运行环境。

E1结束只输出`CPU_PASS/FAIL`、`HOST_SOURCE_PROVENANCE_FOUND/UNKNOWN`、`ARTIFACTS_REUSABLE/STAGED_NEW/UNAVAILABLE`、`BUILD_PASS/FAIL/NOT_ATTEMPTED`、`NEEDS_REVIEW`；**不得设置任意U1批准旗标**。

## 3. Stage E2｜编译产物与受控部署评审（必须收到E0/E1报告后独立放行）

- **E2不是首批OpenCode自行执行的动作**。研发侧先审核所选卡同卡双Die/独占、四卡G0、旧SO完整哈希、两条Host装载路径、插件SO、设备SO、KVER11、CANN/ABI兼容、恢复备份及所有新产物SHA。
- 部署前对共享安装目录做**原样备份**并验证可恢复性；不得因安装新SO中断他人使用的服务/模型/任务。可使用隔离`LD_LIBRARY_PATH/PYTHONPATH`工作环境时优先隔离，但必须证明运行时真装载这一套路径（仅env变量设置不够）。若必须动共享系统/全局site-packages，确认窗口、占用、回滚操作并取得明确用户授权。
- 如果沿旧`1dc46d7d`二进制路线且SHA、CANN/ABI全部匹配，可从受审归档恢复**两份一致**；如果候选重新编译SHA不同，先作专门审查并更新测试合同才能放行，绝不为跑通直接改U1 `u1_spec.py`硬编码。
- KFC **不提前手工部署**。设备侧编排核v11由正式MemFabric在**E2部署通过后、单独授权的首轮Smoke建池时**自动执行：`LoadOrchestratorKernel→RunDeployHelper→KFC test -f→必要时CUST迁移复制→核验大小/符号→TryLoadKernel`。从两Die分别确认`libmf_sdma_orch_v11.so`和`HybmSdmaOrchEpoch`加载。孤立507018、0x2a、507014/507899、KFC helper异常、常驻epoch引发507017立即STOP。2026-10-23旧文档`mini_kfc rm -f`流程**仅用于独立授权的诊断**，不可当本轮正常自动部署脚本。

## 4. Stage E3｜最少模型启动的性能测试分批（仍要阶段准入）

| 批次 | 可以同一OpenCode会话批做的内容 | 必须卡住的门禁 | 回传 |
|---|---|---|---|
| **BATCH-R0（第一批，已准备）** | E0只读四卡/其它分支环境盘点 + A1全量CPU回归 + 旧MemFabric修复搜证 + 纯CPU DeviceID用例 + 可行的完全隔离staging编译 + 备份/恢复计划 | 不允许SO全局覆盖、KFC、建池、`torchrun`、NPU探针；CPU或环境异常后停提交证据 | `environment_inventory.json`、四卡清单、`source_provenance.json`、`build_manifest.json`、CPU日志、差异/阻塞清单 |
| **BATCH-R1（待R0验收授权）** | 审批后受控复原双Host SO/设备SO/插件SO/KVER11与真实装载校验；已批准首次M2048双RankStock+V2 Smoke，检查自动KFC，安全时一次完成绝对最小正确性矩阵 | 共享部署必须独立许可；Smoke失败立即STOP且保存第一次故障；**不能自动升到Screen** | 两Die KFC状态、SO/ABI/provenance、双Rank数值/日志、回滚结论 |
| **BATCH-R2（待R1验收授权）** | 同卡/同热态Stock–V2 `large screen`涵盖M1024诊断与M2048/4096正式、读取A1离线分析；独立HCCL跨流设备完成审计的可行性预判 | 先确认Smoke和版本完全对应；Screen只给性能方向不签L1或L3_GO | Screen原始Rank数据、M2048/4096真实全链相对Stock、下一步瓶颈 |
| **BATCH-R3（待R2/HCCL审计授权）** | 复用U1双ABBA `large confirm`8会话、可按需完成短Profiler定位 | Screen/路由与首写/设备安全合同全部合格；Confirm不能借Screen的正向直接放行 | 完整算子L1、双Rank p50/p90/p99与HCCL尾部证据 |
| **BATCH-R4（后续新计时工具验收及授权）** | 普通Qwen3.6 Eager 4096/1并发1无Profiler L3 Screen，正向才四次ABBA Confirm；按需短Profiler | 无Profiler真实Prefill设备Step完成边界尚待研发侧实现/校准。**当前不允许假装已具备此能力** | L2真实40站点、L3完整Step与Stock漂移/判定 |
| **BATCH-R5（仅L3_GO之后）** | 4096/2048/TP2/并发10/50请求/ignore-eos正式客户端 TTFT Stock–V2–V2–Stock 4场 | L3_GO+独立批准、禁Graph/DFlash混入Eager账 | L4 TTFT p50/p90/p99、传导差额 |

**任何一批都可以在同一OpenCode会话内收集大量只读信息与CPU证据，但不能在没有已审核环境身份的情况下自动连跑硬件批次。** 保存失败原始样本、所有Rank日志和开始/结束`npu-smi`，绝不在异常后自动换卡重复直到成功。

## 5. 首批委托准入与明确停止条件

**首批只授予BATCH-R0可执行级别（用户委托OpenCode后生效）。** 工作端现在可以pull本A1分支准备CPU和只读取证；该许可不包含修改正在跑别的任务的目录/进程/已安装环境。OpenCode必须：
- 输出原始证据与机器可读summary，附**确切仓库/分支/HEAD/所有SO SHA**；如敏感主机信息仅服务器保留，把脱敏摘要和manifest上传本仓工作分支。
- 若发现历史`mf-fix`未提交修复：先`git diff`/保存patch与源tree，不执行任何`git reset`、清理或切换覆盖；若旧1dc46d7d SO仍可验证，优先复用路径A的恢复计划而非重编。
- 若没有隔离工作区、CPU测试失败、编译目标不明确、Hash不匹配、共享目录被其他任务使用、卡健康/独占条件不成立，**标记NO_GO、提交报告，停止可能产生副作用的步骤**；允许把其它互不依赖的只读取证做完以提高单次信息密度。
- 当前A1原仓CPU = NOT_RUN，实机G0 = NOT_RUN，任何设备Smoke/Screen/Confirm/L3/L4 = NOT_RUN。**新一批设备结果必须由现场实际回传后才能改变本账本状态。**

# 310P3 F2→F1 实机准入（Go/No-Go）与操作规程｜2026-10-24 修订

> **状态：仅源码/脚本审视完成；310P3服务器硬件、Host/Device SO、KVER11部署、逻辑设备映射及CPU单元测试结果尚待OpenCode现场核验。**  
> **裁决：允许进入服务器进行G0准入检查（PRECHECK-GO），不允许未经G0通过直接开始F2/F1压测。**  
> 项目对象：310P3单卡双Die TP2上的FP16 NZ MM+MemFabric SDMA+Add；Stock保持默认，正式生产内核不修改。先F2后F1，同一对双Die不能并行占用。

## G0准入检查——所有硬门槛全部PASS才开始任何有性能意义的采样

| 编号 | 检查对象 | 必须拿到的证据 / FAIL动作 |
|---|---|---|
| G0-1 | 代码版本/工作树 | `git status --porcelain`确认无未经审阅的修改；F2/F1均以D4 Tile2048的`33e3f2cfc81e`为公共基线；保存branch、commit与diff；**不得跑C3分支或混合分支** |
| G0-2 | 真正的单卡双Die | `npu-smi info`确认物理卡对、归属同一310P3单卡的die0/die1、无其他租户活动/残余PID/异常占用；`ASCEND_RT_VISIBLE_DEVICES=x,y`和进程内逻辑local_rank0/1映射留证；**不要把“可见设备=0/1”误认为物理号也0/1** |
| G0-3 | 编译/部署/运行时SO | F2设备核必须是D4 Tile2048候选 `2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19`，不可默用D4结束时恢复的旧设备SO；记录插件扩展、Host、Device SO与已装载Kernel的路径和SHA256；不能只哈希某个未装载的旁路文件 |
| G0-4 | F1小M诊断ABI | 必须使用`-DVLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS=ON`编出的配套Host/Device构建；`nm -D "$V2_HOST_LIB"`确认`mf310p_v2_trial_run_nz_direct_small`导出，核对诊断kernel装载；F1不允许使用只含正式核的DIAGNOSTICS=OFF构建 |
| G0-5 | MemFabric物理号 | 若使用物理dev6/7等非0/1卡对，必须按D4的`fix/310p-setaccess-physical-devid-20261023`现场证据确认`libmf_hybm_core.so`物理设备号修复、双安装路径一致。D4已知旧Host在非0/1映射下建池会报`507899`；未通过则STOP/NO-GO，不可把改卡号当作修复 |
| G0-6 | KVER11/KFC/HCCL | `MF_SDMA_ORCH_JSON`指向真实`libmf_sdma_orch_v11.json`；检查两物理die AICPU`libmf_sdma_orch_v11.so`部署版本/注册、KFC可用、G0 IB probe、HCCL可用；**MF_KFC_BYPASS与MF310P_V2_NOWARMUP必须unset**；不能假定只是JSON文件存在就表示KVER11设备端部署正确 |
| G0-7 | 本地工具验证 | **实际在310P3所在环境**执行 `bash -n`、`python3 -m py_compile`、各分支对应`pytest`，检查pytest exit code/日志；再完成一次受控试运行，先正确性，后性能；失败先修复同分支并回归，保留失败证据 |
| G0-8 | 采样隔离 | 确认Profiler关闭、没有Graph/DFlash等非本期干扰，保存CANN、torch/torch_npu、vllm、环境变量、主机时间及设备占用；两个分支不同时运行，失败后设备不得未经检查立即重用 |

**重要：ChatGPT无310P3 Shell访问权。这里“源码准入”并不等于“现场G0通过”。OpenCode负责将每项G0结果、命令及其原始输出归档，不允许只写“环境OK”。**

## F2实机操作顺序（先执行）

1. `git fetch origin && git switch perf/310p-f2-mm-sdma-overlap-sweep-20261024 && git pull --ff-only`。不允许Git分支切换覆盖已有未提交实验修改；先记录SHA。
2. 核查生产核`src/op_kernel/mf310p_v2_kernel.asc`、协议、Host ABI相对D4基线均未变；必须部署经SHA验证的D4 Tile2048正确设备SO；如果当前安装版本不是D4，先备份、受控重新构建部署并取证。
3. CPU测试：
```bash
bash -n csrc/_310P/memfabric_mm_ar_v2/test/perf/f2_run_sweep.sh
python3 -m py_compile csrc/_310P/memfabric_mm_ar_v2/test/perf/f2_chunk_overlap_perf.py csrc/_310P/memfabric_mm_ar_v2/test/perf/f2_sweep_analyze.py
python3 -m pytest -q csrc/_310P/memfabric_mm_ar_v2/test/ut/test_f2_sweep_contract.py
```
4. 确认G0全部PASS，G0-3满足D4 SHA，所测Die独占，然后明确设置`ASCEND_RT_VISIBLE_DEVICES`、`MF_SDMA_ORCH_JSON`、`F2_DEVICE_SO_PATH`、`F2_EXCLUSIVE_APPROVED=YES`。**这些变量不能只复制示例路径，必须指向真实装载物**。
5. 先运行M2048/C1024/O1试运行和bit-exact核对，若失败终止。之后执行`bash csrc/_310P/memfabric_mm_ar_v2/test/perf/f2_run_sweep.sh /tmp/opencode/f2_sweep_<unique_run_id>`；脚本自动以每场独立进程顺序运行18场，每场3预热+30样本，15分钟超时+30秒进程结束宽限，不删除失败样本。
6. 保存每个Rank的Event/Wall、位精确、status、完整会话原始日志、SO/KVER/环境指纹；脚本生成`f2_screening.json`是**D4同SO参数筛选**，不是与Stock MM+HCCL的最终比较。如果发现稳定≥3%相对D4改善，才安排独立同设备Stock对照和重复ABBA，不许凭历史Stock数据直接GO。

## F1实机操作顺序（只在F2释放且设备清洁后）

1. 记录F2结束时双Die/KFC/HCCL健康，确认残留PID与端口、设备异常均已清理；任何507014/wedge都不得直接进入F1。
2. `git switch perf/310p-f1-smallm-fixed-cost-20261024 && git pull --ff-only`。必须使用诊断模块ON且ABI验证通过的构建，明确设置`V2_HOST_LIB`到实际导出T1 ABI的绝对路径、`F1_DEVICE_SO_PATH`、同卡`ASCEND_RT_VISIBLE_DEVICES`、`MF_SDMA_ORCH_JSON`。物理号补丁如使用必须两线一致。
3. CPU测试：
```bash
bash -n csrc/_310P/memfabric_mm_ar_v2/test/perf/f1_run_smallm.sh
python3 -m py_compile csrc/_310P/memfabric_mm_ar_v2/test/perf/f1_smallm_analyze.py
python3 -m pytest -q csrc/_310P/memfabric_mm_ar_v2/test/ut/test_f1_smallm_contract.py
```
4. 正确性先行、确认HCCL与G0 IB探针成功，设置`F1_EXCLUSIVE_APPROVED=YES`，执行`bash csrc/_310P/memfabric_mm_ar_v2/test/perf/f1_run_smallm.sh /tmp/opencode/f1_screen_<unique_run_id>`，36组（M1/8/10/16/32/64 × MM核4/8 × Add核1/4/8，每组30次）与建池前Stock MM+HCCL初筛；超时30分钟+30秒杀进程后立即保全环境。
5. **不要把初筛直接称为“同场严格公平Stock胜负”**，因为T1建池前/后与Host sync的测量偏差。必须先对M10开销做单独诊断，再为候选实施无Profiler、双Rank配对Event和真实Stock对照。

## 出现异常时（所有分支共用）

- ABI缺失、D4设备SO SHA不符、KVER11不匹配、物理号补丁错误、双Die不独占：**G0 NO-GO，禁止运行完整扫描。**
- 507014/507899、Rank一侧超时、首写/位精确/status错误：**停止当前场，保存Rank0/Rank1日志、npu-smi、Host/CANN错误、AICPU/KFC证据，记录已执行组合**；不要自动重跑、清洗慢样本或换卡继续算同一组。
- 任何超时强杀可能使KFC/HCCL处于脏状态；清理只针对本实验PID，未经授权不要`pkill`其他租户进程。重启、系统SO部署/恢复必须备份并完成回归；未恢复不得切分支跑F1。
- 发现测试脚本错误时，OpenCode先提交**最小测试代码修复**并跑CPU/正确性回归；若需要改变正式Kernel/IB/SDMA/DONE协议，要独立评审，不在本次诊断中直接动。

## 结束交付与裁决

- `G0_PRECHECK_REPORT.md`必须逐项列实际证据与PASS/FAIL、设备物理/逻辑映射、版本与SO/KVER SHA、编译/pytest结果。
- `F2_ONDEVICE_REPORT.md`：18会话每场双Rank配对p50/p90/p99、6ms以上长尾、相对同M D4基线变化、相邻重复锚漂移、失败记录、下一步优先关键路径；**别虚报Stock收益**。
- `F1_ONDEVICE_REPORT.md`：M10和邻近形状36组合正确性/最优分核/Stock对照，明确是**SCREEN_ONLY**还是已补上严格Event对照；固定协议MM/IB/signal/wait/Add/DONE的假设和下一步取证。
- 推送至各自F1/F2分支，并给出 commit SHA、证据路径、明确GO/NO-GO与下一轮一处一变量优化建议。**Stock产品默认不变**。最终真实Qwen三方工作负载固定输入4096/输出2048/TP=2/batch上限10/Eager，且留证实际步内batch。

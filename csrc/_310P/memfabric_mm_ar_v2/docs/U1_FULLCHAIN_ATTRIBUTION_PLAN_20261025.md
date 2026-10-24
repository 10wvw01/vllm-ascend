# U1：310P3单卡双Die MM+MemFabric AllReduce全链路性能归因

> 2026-10-25 · U1独立分支：`perf/310p-u1-fullchain-attribution-20261025`，基于F1-C实机NO-GO后的最新代码。**代码已提交但目标310P3实机尚未执行。状态仅PRECHECK-GO，完成G0现场验证、CPU测试、受控冒烟后才可开始性能采样。**
>
> **工程不改动正式Kernel/Host ABI/通信协议与Stock默认路由。** 单卡双Die，TP=2、K=N=2048、FP16，KVER11正常加载。所有实验保留失败和原始样本。最终Qwen三方验收：输入4096、输出2048、TP2、batch上限10、Eager。

## 1. 从多轮单点实验转为一次高信息量测试

历史F1-C：M10的真正FP16 Stock约308.2µs、T1约425.2µs，T1等效速率落后约27.5%；下一代方案要获得5%正向收益，需从当前425.2µs降至约293.5µs，节省约132µs。历史F2：M2048/4096的D4 Tile2048/Chunk1024/Overlap1优于更小Chunk与串行，继续扫256/512没有新价值。D4较旧V2单核+6.26%，不是与Stock的模型端收益。

| 阶段 | 内容 | 交付信息 | 严格限制 |
|---|---|---|---|
| G0 | 一次受控部署，锁定硬件、SO、KVER11、Host/Device、备份恢复 | 真实环境身份 | 文件SHA≠证明设备侧实际已注册/加载 |
| D0 | Stock与T1/D4的两组多M测试，各组Stock→融合 | 全M正确性、Event/Wall、Rank配对、热态回测 | 初筛永不GO |
| D1-A | 单Die原生F.linear与T1 NZ MM+Clean，MM4/8 | 小M计算路径是否低效 | MM诊断仍有CleanLines，不是纯MM |
| D1-B | 4Chunk正式D4重叠与4Chunk串行，以及单Chunk机制探针 | 大M重叠价值、局部同步路径线索 | 不可将不同诊断Kernel的中位数相减当阶段耗时 |
| D2 | 候选M独立进程两轮ABBA | 稳定性/收益门禁、漂移/长尾 | 需独立HCCL跨流Event取证 |
| D3 | 单独短Profiler/设备时间线 | 关键路径有证据则定位 | 不允许拿Profiler时间替代无Profiler性能 |
| D4 | 真实Serving实际M调用量和现有单算子结果 | 乐观模型收益预算 | 上界预算不是实测吞吐 |

### D0形状设计

- **小M / T1**：`M={1,8,10,16,17,64,255,256}`，固定MM8/Add4。涵盖Decode M10、模板边界16→17及255→256。
- **大M / D4**：`M={1024,2048,4096}`，Chunk1024、Overlap1。M1024是试验性的过渡形状，必须先正确性PASS；不代表模型Serving已路由此形状到V2。
- M257～1023在当前固定Chunk1024下不适用，**不许补假数据**。必要时另行独立合法Chunk实验，但不要重复F2大规模Chunk网格。
- 一个Stock进程覆盖全部同Scope M，一个MemFabric进程覆盖全部同Scope M；**没有同进程HCCL+MemFabric共存**。同一M固定两组交替输入，CPU gloo FP32求和再FP16用于逐次位精确，性能Stock只有原生FP16 F.linear+FP16 HCCL，无FP32计时转换。每个形状单独预热、双Rank逐样本配对，并在同Worker结束时**额外回测首M八个样本**判热态漂移。

## 2. G0不可跳过的准入与已知坑

1. 检查git分支HEAD、干净工作树、编译/实际Python加载路径；`git diff`确认U1只在`test/`和`docs/`。310P3同一卡的两个Die独占，并核实物理设备号与LOCAL_RANK映射，不能假定可见`6,7`就一定正确。
2. F1-C实验收尾曾恢复旧Host/Device SO。**部署前先备份原双路径SO及SHA/恢复命令**，再受控部署正确测试版本。U1锁定完整SHA：D4 Tile2048设备SO `2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19`；诊断ON Python扩展 `8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1`；物理设备号修复版MemFabric Host双路径 `1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30`；KVER11 JSON `25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78`。脚本核验实际哈希，但**设备端AICPU与加载地址仍需现场取证**。
3. 验证`libmf_sdma_orch_v11.so`已正确部署、KFC正常、HCCL独立探针、IB probe可用、无残留PID/端口/其他租户。禁止`MF_KFC_BYPASS`、`MF310P_V2_NOWARMUP`及`MF310P_V2_AR_ONLY_NOWARMUP`；不能为了取数字跳过首写和协议屏障。
4. G0真实执行CPU与Shell合同测试，保存exit code与原始日志：

```bash
git fetch origin
git switch perf/310p-u1-fullchain-attribution-20261025
git pull --ff-only
bash -n csrc/_310P/memfabric_mm_ar_v2/test/perf/u1_run.sh
bash -n csrc/_310P/memfabric_mm_ar_v2/test/perf/u1_campaign.sh
python3 -m py_compile csrc/_310P/memfabric_mm_ar_v2/test/perf/u1_*.py
python3 -m pytest -q csrc/_310P/memfabric_mm_ar_v2/test/ut/test_u1_matrix_contract.py
```

**CPU/abi/部署任一项失败立即NO-GO**，只允许最小测试代码或部署修复、重新审查/回归。U1代码目前尚未获得以上命令在服务器上的PASS证据，禁止直接声称准入已过。

## 3. D0一条命令跑小M和大M（4个新TP2进程，而非各形状单独八轮）

现场填入真实安装路径，不允许复制示例路径直接执行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=6,7  # 仅示例，实际须通过物理号取证
export MF_SDMA_ORCH_JSON=/实际安装路径/libmf_sdma_orch_v11.json
export V2_HOST_LIB=/实际安装路径/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
export U1_DEVICE_SO_PATH=/实际安装路径/libmf310p_v2_trial_device.so
export U1_CORE_SO_LIB64=/实际安装路径1/libmf_hybm_core.so
export U1_CORE_SO_PY=/实际安装路径2/libmf_hybm_core.so
# 以下标记只允许在逐项G0审核、真实卡对独占和备份恢复方案检查通过后设置：
export U1_G0_APPROVED=YES U1_EXCLUSIVE_APPROVED=YES U1_RESTORE_PLAN_VERIFIED=YES
bash csrc/_310P/memfabric_mm_ar_v2/test/perf/u1_campaign.sh /tmp/opencode/u1_campaign_001
```

默认先小M Stock+T1，再大M Stock+D4。每个子实验生成manifest、两个Rank JSON、真实session.log、before/after npu-smi、`u1_analysis.json`。`screen`每M3预热+12样本，仅用于快速筛选；任何漏采、非零协议、位精确错误、507014/507899、实际二进制SHA不符或挂死均停止整个活动、保全证据并隔离设备，不自动重试。

**U1-Campaign中`U1_DIAG_APPROVED=YES`为可选D1授权**：仅D0完全通过且确认KFC健康/权限后才可打开；否则只生成D0地图。不必为了运行诊断而多次部署版本，若已有D0目录则不可覆盖，应新建独立D1目录手工运行。

## 4. D1大信息量归因与D3有限设备剖析

- 小M：`u1_diag_mm.py`在逻辑设备0上测M10/16/64/256，原生FP16 F.linear vs T1 NZ N-split MM+Clean，分别MM核4/8，每种≥30样本。此工具不创建HCCL/MemFabric数据面，**其“MM+Clean”不是完整T1中的可直接相减阶段**，不能基于两条不等价路径的median做加法分解。
- 大M：`u1_diag_large.py --case formal`测4096行四Chunk正式重叠，`--case loop4`测四Chunk串行，每种独立TP2进程，CPU gloo先生成参考再建池。该脚本不创建HCCL group，按样本校验位精确和status。
- 可选单Chunk1024诊断：`diag0`纯诊断局部MM、`diag4`完整IB与DONE诊断、`diag5`单Chunk正式路径。每个Case独立TP2且必须先正确性PASS。**禁跑diag1**：其设计故意跳过IB就绪会合；不能为了测速度牺牲数据依赖。
- T2 AR-only包含额外AICore暂存/首写守护/清缓存，不能用于直接量化当前T1“纯通信时间”；历史Gate C单Chunk也不能与4Chunk正式D4逐段直接相减。
- 需要内部阶段精确归因时，只允许另外评审可证明安全、可校准开销的诊断插桩或完整设备时间线；如果只看到整个Kernel持续时间，就在报告写`PHASE_TIMING_NOT_OBSERVABLE`，**不允许从源码推算精确µs归因**。
- Profiler仅用于另开短窗口的独立诊断，不能污染正式Stock/T1/D4无Profiler性能地图与ABBA。Stock的Event包围HCCL跨流任务必须独立取证，不得重用F1-C的M10证据代替M4096证明。

## 5. D2升级门槛和收益预算

若D0提示明显正收益并且无错误，经人工审查后在**新的绝对目录**运行：

```bash
export U1_CONFIRM_APPROVED=YES
bash csrc/_310P/memfabric_mm_ar_v2/test/perf/u1_run.sh /tmp/opencode/u1_large_confirm_001 large confirm
# small需要新证据才批准，不因为F1-C已有NO-GO而自动重复昂贵8场：
# bash .../u1_run.sh /tmp/opencode/u1_small_confirm_001 small confirm
```

每Scope8场Stock/T1或Stock/D4交叉ABBA，每场全部所属M、每M5预热+40样本；形状顺序轮转且首M尾部回测。多维判定：两轮各≥5%速率改善、融合整体p90≤Stock×1.05、p99≤Stock×1.10；任一场首M锚点漂移或Stock跨会话变化>5%时只能判`INCONCLUSIVE`。独立HCCL路由/Event覆盖审计文件齐备后，才可能给`OPERATOR_CANDIDATE_ONLY`，不能直接给真实模型GO。

若另有**真实**Qwen固定负载实际M和调用次数证据，使用`u1_e2e_budget.py`计算乐观Amdahl上界；不能拿虚构的调用次数填表。负载固定：**输入4096、输出2048、TP=2、batch上限10、Eager**。模型级Stock/V1/V2三方指标仍需单独真模验证，所有失败会话和实际batch保留，绝不只统计成功样本。

## 6. 实机交付/退出

交付至少包括：`U1_G0_PRECHECK_REPORT.md`、`U1_D0_PERF_MAP.md`、`U1_D1_ATTRIBUTION.md`、`U1_GATES_AND_NEXT_OPTIMIZATION.md`、所有Rank JSON/原始session.log/Trace索引/SO哈希/失败文件/恢复记录。D1未运行填`NOT_RUN`。下一次正式Kernel修改建议必须带**目标µs收益范围、证据强度、合法通信/内存可见性实现方案及对Qwen业务命中率的影响**。如果F1通信下限已不足以回收约132µs，就停止小M微优化，把资源投入F2四Chunk大M关键路径。

退出前恢复G0备份Host/Device SO、核对双路径SHA、检查KFC/HCCL与设备无残留任务，提交GitHub U1独立分支和完整Commit SHA。

**状态总结：U1完成测试框架与离线合同准备；BiSheng/pytest/TP2实机均未由本助手执行。现场G0不通过时不能跑D0/D1，Stock始终默认。**

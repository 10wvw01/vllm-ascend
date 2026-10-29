# U1静态审查双轮记录与实机准入状态（2026-10-25）

> 本文记录**实际GitHub源码差异审查和静态合同审查**；没有在本对话中访问310P3 Shell。`bash -n` / `py_compile` / `pytest` / Bisheng和TP2真实运行仍为**NOT_RUN**。因此最终只能给`PRECHECK-GO`，不能给设备性能`GO`。

基线：`perf/310p-f1c-stock-fp16-hccl-vs-memfabric-abba-20261025`；U1分支：`perf/310p-u1-fullchain-attribution-20261025`。

## 第一轮 R1：源代码、通信与生命周期准入检查

实际对比分支文件清单，仅新增：`docs/U1_*.md`、`test/perf/u1_*.py`、`test/perf/u1_*.sh`、`test/ut/test_u1_matrix_contract.py`。未改：`src/op_kernel`、`src/op_host`、`src/torch_adapter`、Serving路由、MemFabric依赖库、Stock默认设置。**源码层面P0安全边界PASS。**

静态检查：
1. Stock在独立torchrun Worker中以HCCL默认组执行`F.linear(fp16) + all_reduce(fp16)`；另建CPU gloo subgroup只做FP32参考和barrier。融合进程只初始化gloo、建MemFabric池。**没有同Worker混用两数据面。**
2. T1用诊断`mf310p_v2_trial_run_nz_direct_small`、M≤256/MM8/Add4；D4用正式`mf310p_v2_trial_run`、M=1024/2048/4096且固定Chunk1024/Overlap1；**M1024正确性属新待验证点，不得假称历史已验证。**
3. KVER11、D4 Tile2048、诊断ON扩展、物理设备号修复MemFabric Host双路径SHA被启动脚本精确锁定，实际Python扩展路径也核对。**实际Device SO装载和AICPU KVER11注册仍需G0现场证据**。
4. 必须unset KFC绕过和首写绕过。双Rank每次校验位精确及status；G0、独占卡与恢复方案旗标必须三项同时确认，正式Confirm额外要求人工复核标记。
5. 每个Scope先两个独立短Smoke会话（M10或M2048），成功之后才进入D0多形状，每M完成就写入`partial`，任何下个形状的失败不丢已有样本；**不使用partial宣称完整通过**。
6. D1单Die MM诊断没有创建分布式组或MemFabric池；D1大M仅CPU gloo通信参考，各Case独立进程；禁用有风险的`diag1`无IB路径（不在可用选项中）。

**R1静态结论：PASS，正式工程源码隔离；不是实机通信/首写正确性已验证。**

## 第二轮 R2：方法学、伪收益防范、复现与取证检查

静态核验并通过以下合同：

- Stock与融合采用同形状、同固定种子、同两轮轮换输入，参考=每Rank局部FP16 MM再由CPU gloo做FP32两Rank规约→FP16；Stock计时窗口无FP32转换，K/N=2048、TP2固定。
- 同一Scope多M一进程执行、形状顺序随ABBA会话旋转；每个样本先`max(EventRank0[i], EventRank1[i])`再做p50/p90/p99，不是`max(p50Rank0,p50Rank1)`。
- 每场必须有完整Rank0+1 JSON及原始`session.log`、实际SO指纹、manifest、正确性/status、无NaN/丢样本；缺任何原始信息分析器立即拒绝。每M完成的`partial`只是保全，不算该Scope GO。
- 本次补充真实Worker尾部**重复测首M八次**，分析器用Rank配对early/late p50判热态漂移；任一场同进程漂移>5%或Stock跨会话漂移>5%，Confirm不宣告赢家。
- 新增Stock **仅screen阶段**同会话原生MM-only测试和双Rank配对分析，减少额外Stock进程；正式confirm不采额外MM-only，保持Stock/HCCL对照负载纯净；**Stock完整MM+HCCL与MM-only中位数不可直接相减得HCCL阶段时延**。
- 快速`smoke`六个样本只判断正确性，禁止统计GO；`screen`只有Stock+融合各一场，禁止统计GO；`confirm`必须A-B-B-A×2两轮、每轮≥5%正收益、融合p90/p99不显著退化，且须独立路由与HCCL Event全通信覆盖证明，才能列为`OPERATOR_CANDIDATE_ONLY`。
- Event仍是Host下发参与的设备流区间，不是纯核微周期。D1诊断路径阶段不等价，严禁简单差分或叠加。
- Eager上界脚本强制`input4096/output2048/TP2/batch10/Eager`与**真实Serving调用分布**，缺历史M调用数填UNKNOWN，不允许称估算为测得Qwen吞吐。
- CPU离线合成UT涵盖两Scope形状、两轮ABBA、真实Smoke顺序、缺日志/缺Rank、非零协议、NaN、错M序号、缺尾部锚、Stock MM-only样本缺失、未经Audit虚报GO和缺业务M调用量。不使用合成UT中的数值声称310P3性能。

**R2静态结论：PASS（按GitHub源码可见约束逐项检查）；pytest执行结果仍为NOT_RUN。**

## 审查中发现并已修复的关键设计问题

1. 旧初稿只有“anchor”描述却未真正重复测量：现追加首M末尾8样本、双Rank配对及漂移拒绝。
2. 一次多M长会话若末尾新M错误，原先没有中途保存完成形状：现每M位精确PASS后保存独立`partial`文件，同时完整会话仍实行fail-closed。
3. 首轮多M扫描可能同时承担第一次新驱动冒烟：现由U1 Campaign自动安排小M Stock/T1与大M Stock/D4各2场Smoke，总共四场Smoke先于四场性能Screen。
4. 测量有完整MM+HCCL却缺原生MM基础：现Stock同一进程采第二套MM-only样本，分析器核验但禁止不等价耗时差分。
5. 新增Smoke后，合成UT阶段计数/期待形状已同步更新；Smoke不再流入性能比较。

## 未关闭的硬门槛及现场操作顺序

- **G0-NO-GO/待验证**：两Die真实物理号映射、设备独占、Host双路径真实版本、KVER11设备端注册/首写、Python loaded SO、BiSheng构建以及设备SO实际加载。
- **CPU-NOT_RUN**：目标310P3运行环境的`bash -n`、`python3 -m py_compile`、`pytest`必须实际PASS；不能以本文源代码检查代替。详细命令见`U1_FULLCHAIN_ATTRIBUTION_PLAN_20261025.md`。
- **SMOKE-NOT_RUN**：M10、M2048两套后端四场冒烟都必须位精确/status=0后方可D0。
- **D0/D1/D2/D3/D4-NOT_RUN**：按G0→4Smoke→4Screen→按证据D1→候选形状Confirm→D3/模型预算推进；不得先启动八场ABBA。
- **退出门禁**：任何异常立刻保全，恢复G0备份Host/Device SO后验证SHA、KFC及无残留；所有脱敏原始日志推送GitHub本分支。

**准入判断：可让OpenCode在310P3进入G0与CPU合同验证；不得未经G0与Smoke成功立即压测。Stock继续默认。**

# A1-R2 最终报告（attempt3收官）：可完成阶段全部完成，Epoch四场被外部占用阻断（2026-10-27）

> **run_id:** `a1-r2-batch-20261027-001`（attempt3）｜**分支:** `work/310p-mm-ar-large-m-a1-20261026` @ `2718f41dc`（工具链修复见 `b25866222`、`215deb767`）
> **前序:** attempt1 NZ_ROUTE_NO_GO（路由门立功）与attempt2被外部击杀的完整记录见[STOP报告](A1_R2_BATCH_STOP_REPORT_20261027.md)。
> **设备:** 310P3 卡16384 dev2/3；attempt3进场时独占干净，Mechanism/E05起外部负载回归。
>
> **结构化裁决：**
> `NZ路由证明=NZ_STOCK_ROUTE_VERIFIED(4×6组真实aclnnMatmulWeightNz配对,独占)`｜`NZ Smoke=PASS`｜`NZ Screen=完成`｜`NZ 8场Confirm=INCONCLUSIVE_DRIFT(外部突发污染04/05场,分析器诚实拒绝,无任何GO)`｜`D4五Case+短Profiler=完整独占采集,DIAG_CASES_COMPLETE_INNER_PHASE_UNKNOWN`｜`R1离线复算=完成`｜`Epoch 15/5/5/15=NOT_RUN(E15A启动即被外部显存占用拒绝,批次自STOP)`｜`L4=未执行(合同禁止)`

## 1. 三个P0最终结论

| P0 | 结论 | 依据 |
|---|---|---|
| **NZ真实Stock对照收益** | **UNKNOWN**：数据已采集但被污染，不可签发任何方向结论 | 8场Confirm中04/05场遭外部突发污染（M4096 15340µs、M1024 13785µs，为正常值7.6×/13×），分析器判INCONCLUSIVE_DRIFT。干净场次趋势（D4 M4096 2490–2631µs vs NZ-Stock 2010–2335µs）方向与R1-L2一致但**不作为正式证据**。 |
| **D4内部瓶颈可信程度** | **外部观测可信、内部分段UNKNOWN**（合同判定） | 独占完成：formal 2636/2617µs、loop4 3234/3213µs、diag0 6379/412µs（双Rank不对称）、diag4 27033/26995µs、diag5 1021/1056µs；短Profiler证实每次调用=**单个**`V2TrialKernel<tiling1024,1024>`（中位2343µs），MM/SDMA/Add物理上不可从外部拆分；不跨Case相减冒充分段时间。 |
| **周期长尾根因可信程度** | **UNKNOWN（未测）** | Epoch四场NOT_RUN；R1离线复算再次确认V2臂尖峰index 0/6/12/19/25（间距6/6/7/6）、Stock干净、`REPRODUCIBLE_PERIODIC_V2_TAIL_CAUSE_UNKNOWN`。 |

## 2. 本批有效证据清单（全部独占采集）

- **NZ真实路由（正式）**：rank0/1×M2048/4096各6组`aclnnMatmulWeightNz_MatMulV3Common_MatMulV3`+相邻`hcom_allReduce`完成配对：M2048 MM 399–411µs+HCCL 601–614µs；M4096 MM 779–801µs+HCCL 1152–1206µs；与R1-L2真实站点证据吻合。
- **NZ Screen（M1024/2048/4096）**：NZ-Stock链p50 672/1067/2023µs，D4 917/1401/2520µs（screen阶段独占；M4096 NZ链≈2023µs与路由CSV MM+HCCL和2031µs互证）。
- **D4五Case**：见上表；`diag_assessment.json`=DIAG_CASES_COMPLETE_INNER_PHASE_UNKNOWN。
- **短Profiler**：3Case×双Rank chrome trace已采（原始trace SERVER_ONLY留档），`sp_trace_summary.json`给出单kernel计数与时长。
- **R1离线复算**：`REAL_SITE_EXTERIOR_MEASURED_INNER_PHASES_UNKNOWN` + `REPRODUCIBLE_PERIODIC_V2_TAIL_CAUSE_UNKNOWN`。

## 3. 外部占用证据链（本次阻断根因）

外部宿主进程（本容器PID命名空间不可见）在本卡反复突击，三次实测：
1. 07:02–07:06 pid4140162 占dev2 14841MB（短Profiler期间，Confirm的04/05场污染亦属此类中途突发，边界快照不可见）；
2. 07:05 E15A启动自检：dev2仅剩27.33GB<需求38.92GB → vllm干净拒绝启动（无驱逐、无伤害对方）→ 批次STOP_STAGE= EPOCH_ABBA；
3. 07:10 pid9371(dev2,15905MB)+pid18482(dev3,7419MB) 同时占两Die。
attempt2（04:23:56整树被静默击杀）+ 本批模式确认：**该外部负载可杀进程、可与任何正式测量冲突，R2的Serving/长窗实验在其存在期间不可能获得有效数据。**

## 4. 恢复条件与下一步

1. **Epoch四场重测**（P0-3唯一缺项）必须满足：卡16384两Die无外部进程、显存/hugepages基线、且有连续约80分钟不被打断的独占窗口（4场serve各约15–20分钟）；建议与管理员协调该外部任务（疑似凌晨04:00后启动的轮换负载）的排期后再执行。
2. **NZ Confirm重测**（P0-1正式结论）同窗口即可附带完成（约10分钟）。
3. 工具链已全部就绪：launcher `/tmp/opencode/a1-r2-launch.sh`（OUT需换新目录）、身份JSON按新证据刷新即可重跑`a1_r2_one_batch.sh`全程。

## 5. 下一版Kernel优化建议（基于本轮已固证事实，内部分段仍UNKNOWN）

1. **每Chunk(1024行)全链成本是硬底**：diag5单Chunk≈1021µs vs Stock同规模链631µs（+62%/千行）；4Chunk串行3234µs、重叠formal 2636µs——重叠已挽回约600µs，继续压Chunk间空隙的边际收益小于压**Chunk内MM段**（R1建议"MM段NZ化"方向继续成立：D4整kernel 2343µs profiled vs Stock链MM 752µs+AR 1166–1391µs≈1.9–2.1ms）。
2. **diag0双Rank不对称（6379 vs 412µs）**：单Chunk协议下rank0存在长时间等待——值得核查协议侧单Chunk会合/邮件等待语义（诊断用途，不代表正式分段时长）。
3. **diag4的27ms病态等待路径**：phase-4语义暴露了有界等待的极端形态，建议在Kernel内加**不改协议**的受控内部探针（先核CANN9.1对310P3的GetSystemCycle支持，产物必须隔离编译、不冻结D4 SO SHA）。
4. 任何内部分段结论在取得无扰动设备时戳证据前维持`INNER_PHASE_UNKNOWN`；不得以diagCase相减伪造分段。

## 6. 证据索引

- `attempt3_completed_stages/`：批次运行册、CPU回归、NZ路由证明（含4份正确路由CSV）、Screen/Confirm分析与8场原始Rank数据、`contamination_analysis.json`、五Case结果与`sp_trace_summary.json`、E15A失败serve.log与campaign.json、R1离线复算双JSON。
- `attempt1_nzroute_nogo/`、`nzroute_fix_diag/`、`attempt2_external_kill/`：见STOP报告§8。
- SERVER_ONLY：短Profiler原始chrome trace（6件，~10MB）留 `/tmp/opencode/a1-r2-batch-attempt3/mechanism/short_profiler/`；全批OUT目录 `/tmp/opencode/a1-r2-batch-attempt3/`。
- 边界声明：本报告判定仅限本批代码/环境/卡；Epoch未测、Confirm被污染，不外推任何性能结论；未执行L4；未修改任何共享SO。

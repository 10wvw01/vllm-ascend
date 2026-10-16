# memfabric_mm_ar_v2 研发计划

> 设计依据:`DESIGN.md`。v1(`../memfabric_mm_ar/`)冻结不动。
> 每阶段有明确验收标准;未达标不进入下一阶段。

## 阶段总览

| 阶段 | 内容 | 产物 | 验收标准 |
|------|------|------|---------|
| P0 | 代码骨架 + 编译通过 | 全部源文件 + 设备库 .so | bisheng dav-2002 编译 0 error;host 侧链接通过 |
| P1 | 契约探针(双卡) | probe op 双向通过 | `v2_probe` 返回 0;生产者扩展(槽分区写环+TAIL 推进)与邮差契约实测一致 |
| P2 | 单波正确性 | 串行路 parity | M ∈ {1..2048} 抽样 30 点 + 边界全集 vs 不融合参考位等价 |
| P3 | 交叠路正确性 | 交叠路 parity | M ∈ {2049..65536} 抽样 30 点(含 M%512≠0 尾、多波 M>8192)位等价 |
| P4 | graph + 压力 | 回放/压力通过 | torch.npu.graph 捕获-回放 ×1000 位等价;连续混合调用 1e4 次无 poison |
| P5 | 性能与发布 | 基准报告 | 对比 v1:串行带持平、交叠带 ≥ v1、R1/R7 量化入档 |

## P0 — 代码骨架与编译(本目录全部文件)

任务:

1. `mf310p_v2_adapter_api.h`:ABI 常量(段内 cells 偏移、本地格偏移、imm 编码、
   状态码)、`mf310p_v2_context_t`、create/destroy/get_layout/barrier/geometry、
   lanes/init/probe/quiet launch 声明。
2. `mf310p_v2_device.asc`:
   - dcci 纪律 helpers(ReadCell/WriteCell/CleanRegion/PollChanged/Fail);
   - 槽分区生产者(WriteMail+背压、MarkWritten、AdvanceTail、ArrivalPeek);
   - 8 模板 `Mf310pV2LanesKernel<CHUNK>`(MM+ship+poll+add+ack,§6 伪代码);
   - `V2InitKernel` / `V2ContractProbeKernel` / `V2QuietKernel`;
   - extern "C" launcher(模板 switch + no-op warmup 形状)。
3. `mf310p_v2_adapter.cpp`:链接复用 v1 `mf310p_create/destroy/exchange_geometry/
   control_barrier`;本地格/scratch 分配;launch 薄壳(参数校验 → kernel switch)。
4. `memfabric_mm_ar_v2_torch_adpt.h` + `memfabric_mm_ar_v2_binding.cpp`:
   op `memfabric_mm_ar_v2_allreduce(Tensor,Tensor,int)->Tensor`、
   `..._v2_shutdown()`、`..._v2_probe(int,int)->int`。
5. `memfabric_mm_ar_v2_runtime.cpp`:RuntimeState、路由(DESIGN §9)、
   生命周期(DESIGN §10)、毒化、atexit。
6. `cmake/memfabric_310p_v2.cmake` + CMakeLists 增量接线
   (v2 gate = v1 gate + `VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2=1`)。

验收:

- [ ] `bash format.sh ci` 通过(ruff/markdownlint);
- [ ] 310P 环境 `VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1
      VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2=1` 配置构建成功,
      `libmf310p_v2_device.so` 产出;
- [ ] feature-off 构建注册 stub、调用报清晰错误。

## P1 — 契约探针(双卡,拦截 DESIGN R2)

`V2ContractProbeKernel`(单块,lane0,对称调用):

1. 生产侧:槽分区生产者发 `iters` 封 8B 探针邮件(probeSrc → 对端 probeCell,
   imm=`0x200|i`,seq 从 globalSeqBase 顺延),每封后 MarkWritten+AdvanceTail;
2. 消费侧:peek-drain 消费对端 iters 封 + 首次 8 封 INIT,逐封校验
   dst/len/imm,并读 probeCell 验证载荷往返 `(rank<<32)|i` 位等价;
3. 收尾:globalSeqBase/drainSeq 写回,返回 0。

验收:

- [ ] 双卡 `iters=1e5` 通过(顺序、载荷、TAIL/HEAD 推进);
- [ ] 故意破坏用例(错 dst/len/imm)触发 MAIL_MISMATCH + Trap + runtime 毒化;
- [ ] 探针后首个 op 调用正常(序号空间衔接正确)。

## P2 — 串行路正确性(M ≤ 2048)

参考实现:同权重 `torch.matmul(x, w.T) + 对端 part` 的两卡集合等价
(或直接对比 v1 op 输出,位等价)。

M 网格:`{1,2,7,15,16,17,63,64,65,127,128,129,255,256,257,300,511,512,513,
1000,1023,1024,1025,1536,2047,2048}` + 随机 30 点。

验收:

- [ ] 全网格输出与参考**逐位相等**;
- [ ] `protocolStatus` 恒 0;无 Trap;
- [ ] M%256==0 的 direct-read 与 staged 路径都覆盖。

## P3 — 交叠路正确性(M > 2048)

M 网格:`{2049,2560,2561,4096,4097,6144,8191,8192,8193,12288,16384,16385,
32768,65536}` + 随机 30 点(强制覆盖:尾 chunk partial、多波、波尾=16 chunk)。

专项:

- [ ] 多波(M>8192)的槽 reread invalidate 正确(P0 规则);
- [ ] B=16 满波与 B<16 尾波邮件账一致(排空校验通过);
- [ ] 波门跨波推进正确(ack 影子)。

## P4 — Graph 与压力

- [ ] `torch.npu.graph` 捕获(M 固定)→ 回放 ×1000:位等价、
      globalSeqBase/影子/计数器跨 replay 演进正确;
- [ ] eager/graph 混合交替调用 ×1000;
- [ ] 双卡 1e4 次连续调用(串行/交叠混合)无 poison、RSS 稳定(无泄漏);
- [ ] 故意单卡慢化(sleep 注入)验证背压/波门不死锁(DESIGN §8.5)。

## P5 — 性能基准(dav-2002 实测)

| 场景 | 指标 | 对比基线 |
|------|------|---------|
| M ∈ [256,2048] 串行带 | 单次延迟 | v1 单波 A/B(预期持平 ±10%) |
| M < 256 | 单次延迟 | v1 N-split(预期回退,R7 量化) |
| M ∈ (2048,8192] 交叠 | 单次延迟 | v1 batch pipeline(预期 ≥,launch 26δ 收益) |
| M > 8192 多波 | 吞吐 | v1 batch pipeline |
| R1 尾轮询 | 独立微基准 | 决定是否保留切缝 |
| profiler | MM/SDMA 交叠度 | CANN 时间线 |

验收:报告落盘 `VALIDATION_REPORT.md`(仿 v1),全部风险项 R1-R7 有数据结论。

## 测试矩阵汇总

| 维度 | 取值 |
|------|------|
| M | 0 / 1 / 255/256/257 / 2047/2048/2049 / 8191/8192/8193 / 大 M 随机 |
| M%256, M%512 | 0 与非 0(staged / direct-read) |
| waves | 1 / 2 / ≥3 |
| 尾 chunk | full(512) / partial |
| 执行模式 | eager / graph replay / 混合 |
| 调用序列 | 单型重复 / 串行↔交叠交替 / 跨 shape 交替 |
| 故障注入 | 慢对端 / 校验破坏(仅探针) |

## 环境变量

| 变量 | 作用域 | 默认 | 说明 |
|------|--------|------|------|
| `VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2` | 构建(CMake gate) | 未设=不编译 v2 | 需同时设 v1 的 enable(复用建池层) |
| `VLLM_ASCEND_310P_MEMFABRIC_V2_TRACE` | 运行时 | 关 | host 入队 tracing(stderr) |
| `VLLM_ASCEND_310P_MEMFABRIC_V2_LOCAL_BYTES` | 运行时 | 96 MiB | 池 local_size(须 ≥ 64MiB+cells+余量) |

> Python 侧路由开关(model patch 选择 v1/v2 op)落地时按 AGENTS.md 登记进
> `vllm_ascend/envs.py`;C++ 侧沿用 v1 的 getenv 局部模式。

## 回退预案

- R2 探针失败 → 单发货者模式:lane 写 ship 描述格,lane0 代发(公共 signal),
  lane 间会合成本回到 v1 形态,其余设计不变;
- R3 UB 不足 → add tile 降至 1024 / basicM 降至 64,重测;
- R1 实测劣化超预算 → 在尾 chunk 标签轮询处恢复一次 launch 边界
  (合同不变,仅路由层加分叉)。

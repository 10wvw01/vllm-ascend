# memfabric_mm_ar_v2 设计方案 — 单内核 Lane 对称融合

> **CUSTOMIZED 310P3 ONLY**。v1(`csrc/_310P/memfabric_mm_ar/`)保持冻结不动;
> 本目录是全新设计,与本目录外的 v1 代码零耦合(仅链接复用 v1 adapter 的建池函数)。

## 1. 目标与范围

- Hardware: Ascend 310P3 / dav-2002,TP=2,Qwen3.6 `o_proj`/`out_proj`。
- Shape: x `[M, 2048]` FP16,weight `[2048, 2048]` NZ FP16,out `[M, 2048]`。
- **一次调用 = 1 个 AICore 内核 launch**(M≤2048 与 M>2048 同一内核模板族,
  差别只是运行期参数),消除 v1 的 27 launch/波与三路路由。
- **任意 M 无小M N-split 路**:M<256 补零到 256 模板(接受全量权重流地板),
  换取符号数、权重切片缓存、blocked add、graph 预热合同的全部删除。
- **2048 分界**:M ≤ 2048 串行单 chunk;M > 2048 交叠 chunk 流水。

## 2. 设计公理(继承自 v1/V2 尸检的实测结论)

| # | 公理 | 来源 |
|---|------|------|
| A1 | 同 stream 相邻 launch 边界 = 免费的全核 barrier + L1 flush | v1 依赖但未言明 |
| A2 | launch gap 是对端邮件的落地窗口;单内核 B=1 的 wait 自旋实测更差 | device.asc:705 (v1 V2 实验) |
| A3 | wait 只有躲在计算后面才免费 | v1 lookahead 流水线 |
| A4 | `signal()` 是**单生产者 API**(读 REQ_TAIL→写槽→写回,无原子) | smem_shm_aicore_base_sdma.h:127 |
| A5 | AICore L1 与 SDMA/对端互不侦听:每次跨引擎可见性必须显式 dcci | probe 内核 E0c 定论 |
| A6 | 请求环 64 槽 / 到达环 1024 槽 / 邮差每邮件 3 SQE,严格 FIFO | smem_shm_sdma_layout.h |

## 3. 三个对称内存技巧

**T1 引擎写信箱(跨 rank 同步 = 读本地内存)**
每封数据邮件配一封 8B 标签邮件,由**对端的 SDMA 引擎**直接写进我段的标签格子
(同一请求环 FIFO ⇒ 标签落地 ⇒ 数据已完整落地)。消费方轮询自己段内的 64B
格子(一次 dcci),不碰 mail FIFO、不等邮差转发、无需任何人广播。
v1 的 mail-ok 广播被此技巧整体消灭。

**T2 Lane 镜像(集体 → 8 条点对点流)**
两卡 M-partition 同构:我的核 b 只算每个 chunk 的第 b 个行片,对端核 b 亦然。
于是我的核 b 的 add 只依赖**对端核 b** 一份数据,我的核 b 的覆写安全只由
**对端核 b** 的确认保证。ready 会合、generation、done 会合、全局 credit、
全局 quiet——五样机制的存在理由(等 8 个核)全部消失。

**T3 影子比较(格子永不清零)**
所有格子用"值变化"语义:`cell != shadow` 即新事件,shadow 随消费推进。
配合每 lane 单调计数器(值流严格递增),格子**自清理**:

- 无每波清零、无 eager/graph 双轨 generation、无 prepare_wave。
- graph 捕获合同退化为"预热 + 烘焙指针",replay 期间计数器/影子在内存中自然演进。

## 4. 关键问题:`signal()` 多生产者竞态 → 槽分区生产者

**问题**。A4:`signal()` 内部是"读 REQ_TAIL → 写槽 → 写回 TAIL+1"的无原子
读改写。v1 靠"只有 core0 调 signal"规避。v2 里 8 条 lane 并发发货,
两 lane 读到同一 TAIL 会互相覆写邮件 → 丢邮件 = 死锁或脏数据。
本平台无已证实的 AICore 标量 test-and-set,自旋锁不可实现;
轮转票号(turnstile)会把信号顺序耦合到 lane 顺序,退化为每 chunk 屏障。

**解法:槽分区生产者(对公共布局头的生产者侧扩展)**。
请求环 64 槽 × 32B,每 64B cache line 恰好装两槽。给每封邮件分配
**确定性全局序号 seq**,使"序号 → cache line"映射的 line 属主恒等于
`line % 8 = lane`——**任何 cache line 在任何时刻只被一个核写**,消除
v1 式 false-sharing 写回竞态:

```text
波 w(内含 B 个 chunk),波内序号布局(waveSeq 为该波基准):
  chunk k, lane b:  data seq = waveSeq + 16k + 2b      ← 偶槽
                    tag  seq = waveSeq + 16k + 2b + 1  ← 奇槽(与 data 同 line)
  ack lane b:       seq = waveSeq + 16B + 2b           ← 连发两封相同邮件
                    seq = waveSeq + 16B + 2b + 1       ← (dup,幂等,凑满 line)
  波步长 = 16B + 16
  line(16k+2b) = 8k+b (mod 32) → 属主恒为 lane b ✓
```

生产流程(lane b):

1. **背压**:写槽 seq 前轮询 `REQ_HEAD`,直到 `HEAD > seq - 64`(该槽上一任
   邮件已被邮差消费),预算超时 → fail-stop。
2. **私有写槽**:直接写 32B 邮件 `{src,dst,len,imm}` 到
   `res + RS_REQ_RING + (seq%64)*32`,逐行 dcci clean。
3. **标记**:写 `writtenThrough[lane] = seq+1`(store + clean)。
4. **TAIL 推进(仅 lane0)**:每轮读 8 个 `writtenThrough`(dcci),
   `P = min_b writtenThrough[b]`,若 `P > REQ_TAIL` 则写 `REQ_TAIL = P`。
   单写者 ⇒ 无竞态;P 单调 ⇒ 只前进。

**与黑盒边界的关系(明确豁免声明)**:v1 的边界规则"不得读写 request ring /
head/tail"在 v2 **生产者侧**有意放宽:只使用已安装公共头
`smem_shm_sdma_layout.h` 的单一事实源常量、与公共 API 完全相同的 dcci 纪律、
邮差"按 [HEAD,TAIL) 序消费"这一文档化行为。**消费侧仍 100% 公共 API**
(`wait/quiet` + 到达戳 peek)。风险对冲:

- 布局变更 → 编译期断裂(单一事实源头文件);
- 行为变更 → `V2ContractProbeKernel` 启动期双向往返探针拦截(见 PLAN P0);
- 若探针失败,回退方案 = 单发货者模式(lane0 代发,lane 写 ship 描述格)。

**标签值单调性论证(防引擎滞后读旧值)**:标签 SQE 在**执行时**读
`tagSrc[lane]`。需要证明:同一标签格子按落地序读到的值严格递增。
传递链:我写 `tagSrc(k, w+2)` ⟹ 已过波 w+2 门 ⟹ 对端已发波 w+1 ack
⟹ 对端已 add 完波 w+1 全部 chunk ⟹ 对端已消费 `tag(k, w+1)`
⟹ 该标签邮件已执行。故 `tag(k,w+1)` 执行时读值 ∈ [v(k,w+1), v(k,w+2)),
`tag(k,w+2)` ∈ [v(k,w+2), v(k,w+3)) —— 区间不相交,影子比较恰触发一次。

## 5. 内存布局

### 5.1 对称池段(每卡一段,偏移对称;arena 8192 行 = 16 chunk × 512)

```text
段基址 + 0            send_arena   32 MiB
+ 32 MiB              recv_arena   32 MiB
+ 64 MiB + 8 KiB      cellsBase(本卡格子,对端引擎可写/可读):
   tagCell  [8 lane][16 slot] × 64B = 8 KiB    对端引擎写,本端 dcci 轮询
   ackCell  [8 lane]          × 64B = 512 B    对端引擎写(波门 + 兼作 lane quiet)
   tagSrc   [8 lane]          × 64B = 512 B    本端 lane 写 + clean,引擎读(SQE 源)
   ackSrc   [8 lane]          × 64B = 512 B    同上
   probeCell[1]               × 64B = 64  B    契约探针往返格
```

### 5.2 本地设备内存(vLLM 自有,非池内;全部 64B 行对齐)

```text
writtenThrough [8]×64B         lane 写 + clean,lane0 dcci 读(min-prefix)
laneCounter    [8]×64B         每 lane 单调计数器(tag/ack 值;初值 1,首值 2)
shadowAck      [8]×64B         波门影子(初值 0)
shadowTag      [8][16]×64B     每 slot 标签影子(初值 0)
seqState       4×u64           globalSeqBase / drainSeq / drainWave / drainPos
protocolStatus 64B             错误通道(Trap 前写码)
scratch        2048 行 × 4KB   尾 chunk/串行 staging,分配时清零一次
                               (垃圾行局部化:v1 已论证,原样继承)
```

### 5.3 状态初始化与自播种

- `V2InitKernel`(单块):`ackCell[b] ← 1`(本 AICore 写 + clean,= 种子);
  `tagCell` 全零;`laneCounter[b] ← 1`;经**公共 `signal()`**(单生产者,无竞态)
  向对端发 8 封 INIT ack 拷贝(imm `0x180|b`)→ 对端 ackCell 同步获种子 1;
  `globalSeqBase ← 8`;quiet。
- 首波门:`ackCell(b)=1 != shadow(0)` ✓ 立即放行,与对端进度无关。
- 波门正确性:`对端 lane b 的波 w ack ⟹ 对端已 add 完波 w 该 lane 全部片
  ⟹ 我可覆写对端 recv 槽,且我的 send 槽拷贝已落地(免 quiet 论证)`;
  ack 值 ≥ 2 严格大于种子 1,影子单调 ✓。
- tagCell 初值 0 = 影子初值 → 首个真实标签(≥2)恰触发一次。
- 不需要跨 rank 的 init 依赖:对端首个 ack 逻辑上必然晚于本端首波发货。

## 6. 唯一的计算内核:`Mf310pV2LanesKernel<CHUNK>`

8 个模板符号(CHUNK ∈ {256,512,768,1024,1280,1536,1792,2048},
每 lane 行片 = CHUNK/8 ∈ {32..256},basicM = min(行片,128),
tiling 参数与 v1 单波阶梯一致),blockDim=8,`lane = GetBlockIdx()`。
串行路(B=1)与交叠路(B≥5)是同一模板的运行期参数差异。

```cpp
// 参数:gva, x, tailSrc, w, out, ownSegment, peerSegment, peerViewOfMe,
//       localState, totalChunks, tailValidRows, enabled
lane = GetBlockIdx(); if (!enabled || lane >= 8) return;
res = smem_shm_sdma_reserved(gva);                    // 公共 inline
// 载入持久状态(寄存器):counter, shadowAck, shadowTag[0..15]
// waveSeq = globalSeqBase(本地格,上一 launch 的 lane0 写回;graph replay 自然演进)
for (w = 0; w < waves; ++w) {
  B = min(16, totalChunks - 16w);
  for (k = 0; k < B; ++k) {
    if (k == 0) gate: 轮询 ackCell[lane] != shadowAck → 更新影子
                    (lane0 自旋内夹带 AdvanceTail + PeekDrain,见 §7);
    // ---- 计算:零会合 ----
    srcRows = (本波尾 chunk && tailValid < CHUNK) ? scratch : x + g*CHUNK*ROW;
    mm(A=srcRows+lane*rpl 行, B=全量 NZ 权重, C=send+(g*CHUNK+lane*rpl) 行);
    CleanRegion(自己写的 C 行);                        // 唯一保留的 clean(物理必需)
    // ---- 发货:数据 + 标签(槽分区生产者) ----
    cnt++; tagSrc[lane] = cnt (写+clean);
    seqD = waveSeq + 16k + 2*lane;
    WriteMail(seqD, sendSlice → peerRecvSlice, rpl*ROW, imm=(k<<3)|lane);
    WriteMail(seqD+1, tagSrc[lane] → peerTagCell[lane][k], 8, imm=TAG|(k<<3)|lane);
    writtenThrough[lane] = seqD+2 (写+clean);
    if (lane==0) AdvanceTail();
    // ---- 消费上一 chunk(夹在 MM(k) 与 MM(k+1) 之间,A3) ----
    if (k > 0) {
      轮询 tagCell[lane][k-1] != shadowTag[k-1] → 更新影子
        (lane0 自旋内夹带 AdvanceTail + PeekDrain);
      if (w > 0) InvalidateRegion(recvSlice);          // P0:同 launch 槽重读
      add(out 行片 ← send + recv, 精确 validLaneRows 行);  // v1 位等价 FP16 加
    }
  }
  轮询 tagCell[lane][B-1];(w>0 则先 invalidate)add(B-1);   // 尾 chunk(暴露,见风险)
  // ---- lane 信用归还(兼作对端的 lane-quiet) ----
  cnt++; ackSrc[lane] = cnt (写+clean);
  seqA = waveSeq + 16B + 2*lane;
  WriteMail(seqA, ackSrc[lane] → peerAckCell[lane], 8, imm=ACK|lane);
  WriteMail(seqA+1, 同上, 幂等重复(凑满 line));        // imm/dst/len 全同
  writtenThrough[lane] = seqA+2; if (lane==0) AdvanceTail();
  waveSeq += 16B + 16;
}
// ---- 收尾:持久化 + 排空兜底 ----
写回 counter / shadowAck / shadowTag(本地格,own-core 读写);
if (lane == 0) { AdvanceTail(); PeekDrain 预算排空;
                globalSeqBase = waveSeq; drainSeq/Wave/Pos 写回; }
```

add 内核体复用 v1 的双缓冲 TQue 向量加(位等价论证原样成立),tile 缩到
2048 元素以让出 UB;尾 chunk 的 `validLaneRows = clamp(tailValid - lane*rpl, 0, rpl)`,
垃圾行只被计算/交换、永不归约。

## 7. 排空(Drain)与 Peek

`wait()` 是阻塞原语,无法在自旋中夹带 TAIL 推进 ⇒ 排空必须 **peek 化**:

```text
V2ArrivalPeek: 读 ARR_STAMP[drainSeq % 1024](dcci)== drainSeq+1 ⇒ 下一封已落地
V2DrainOne:  peek 通过 → 公共 wait_at() 取邮件 → 按确定性序校验 → drainSeq++
```

- **确定性校验序**:每波邮件 = `for k: for b: [data(16k+2b), tag(+1)]`,
  然后 `for b: [ack(16B+2b), dup(+1)]`;INIT 邮件(seq 0..7)最先。
  校验 dst(= peerViewOfMe + 偏移)/len/imm 三元组,错任一 → fail-stop。
- **排空调度点**:lane0 的每个自旋点(波门、标签轮询、收尾)都夹带
  `AdvanceTail + PeekDrain`;生产速率 ≈ 16 封/chunk,排空机会 ≈ 每 chunk 一次,
  到达环积压恒 ≪ 1024。
- **单消费者**:`wait` 只允许 lane0 调用(lanes 1-7 从不消费)。
- 波尾不做阻塞式"等 8 ack":门走格子、排空只是 FIFO 卫生 + 校验,
  两者解耦,消除了"双侧 lane0 互等"的死锁形态。

## 8. 正确性论证汇总

1. **数值**:MM 模板与 v1 逐位一致;FP16+FP16 在 FP32 精确、正确舍入加法唯一
   ⇒ 与不融合参考位等价(v1 已验证,原样继承)。
2. **顺序链**:同 lane 连续两封 signal → 请求环槽位递增 → 邮差按环序翻译
   → 同 SQ FIFO ⇒ 标签落地 ⇒ 数据完整。写侧 clean / 读侧 dcci 全程显式。
3. **槽复用安全**:波门(对端 ack)⟹ 对端已消费我上一波同名槽 ⟹ 覆写合法;
   对端 ack 同理蕴含我的 send 槽拷贝已落地(免 quiet)。
4. **标签单调性**:§4 传递链,影子比较无漏报/误报。
5. **无死锁**(关键非对称结构):
   - 每 lane 的依赖链是 (lane, wave) 偏序,无环;lane 间零依赖;
   - 双侧 lane0 的门/标签互不构成环:每侧 lane0 进入轮询前已使自己的
     chunk-(k-1) 邮件 TAIL 可见,所等的数据由对侧 lane0 的"轮询前推进"保证
     可见 —— 两个 lane0 锁步推进,各自永不等待对方未发布的数据;
   - 排空与门解耦(§7),不存在"等 ack 邮件可见"的循环;
   - 背压只针对**本端邮差**(本地独立推进),不构成跨 rank 环。
6. **容量**:在途请求 ≤ 单波 16B+16 ≤ 272 < 64 槽环(背压兜底);
   到达环积压 ≈ 一波 272 + 排空延迟 ≪ 1024。
7. **fail-stop**:任何轮询超时/校验失败 → 写 protocolStatus + `Trap()`;
   runtime 毒化,拒绝复用(v1 纪律)。

## 9. 路由(host,runtime.cpp)

```text
M == 0            → 空张量直接返回
M ≤ 2048 (串行)   → CHUNK = ceil(M/256)*256,totalChunks=1,tailValid=M,
                    waves=1;M%256!=0 时 stage x→scratch 后 launch
M > 2048 (交叠)   → CHUNK = 512,totalChunks = ceil(M/512),
                    waves = ceil(totalChunks/16),tailValid = M-(totalChunks-1)*512,
                    M%512!=0 时 stage 尾 chunk 有效行 → scratch 后 launch
两次 launch 之间无 host 同步(背压/门全部在内核内,论证见 §8.5)
```

## 10. 生命周期与合同

- **初始化(首调,host)**:v1 adapter `mf310p_create(rank,2,store,local,8192,512)`
  → layout 校验 → 本地格/scratch 分配清零 → 全符号 warmup(enabled=0)×2
  → `V2InitKernel`(真启动)→ stream sync → `control_barrier` →
  `exchange_geometry`(得 peerViewOfMe)→ sync。
- **warmup 合同**:8 模板 + init + probe + quiet 全符号,no-op 形状各两遍;
  graph 捕获前必须完成(eager 预热路径)。
- **graph 捕获**:烘焙 12 个内核参数(M 固定/图);seq 基、影子、计数器全部
  从内存读写 ⇒ replay 间自然演进,**无任何清零/代数烘焙合同**。
- **shutdown**:quiet 内核(公共 quiet)→ stream sync → 释放本地块 →
  `mf310p_destroy`;graph-used 上下文保持进程生命周期(v1 纪律)。
- **单池约束**:v1/v2 共用进程级 MemFabric 单例与 `mf310p_create(id=0)`;
  同进程只允许一个变体初始化,后者带明确报错拒绝。

## 11. 风险清单(P0/P1 实测拦截)

| # | 风险 | 量级估计 | 拦截手段 |
|---|------|---------|---------|
| R1 | 串行 B=1 尾 chunk 标签轮询暴露(v1 V2 教训残留) | ~10-50µs/次 | P3 微基准确认;退避轮询;唯一预留切缝=该轮询处加回 launch 边界 |
| R2 | 槽分区生产者 vs 邮差契约(灰区豁免) | 布局/行为变更即坏 | P0 契约探针;单一事实源头文件;回退=单发货者模式 |
| R3 | MM+add 共享 UB(TOTAL_UB_SIZE - 24KB 给 mm) | mm workspace 不足 | P1 编译+P2 首跑;不足则降 add tile / basicM |
| R4 | 同一 Matmul 对象跨 chunk 重复 SetTensorA+IterateAll | API 支持度 | P2 首跑;不支持则每 chunk 重建对象(开销可测) |
| R5 | 多波 reread 的 recv invalidate 开销(4096 dcci/lane/chunk) | 仅 M>8192 | P4 微基准;若 MTE2 不经 L1 可去掉 |
| R6 | lane0 TAIL 推进节拍 = 其他 lane 邮件可见性上界 | ≈ lane0 循环周期 | 论证已被流水吸收(§8.5);P4 压测验证 |
| R7 | M<256 补零的全量权重流地板(放弃 N-split 的代价) | ~100µs/call | 设计指令接受;扩展点保留(权重切片可后补) |

## 12. 文件清单

```text
csrc/_310P/memfabric_mm_ar_v2/
  DESIGN.md                        本文件
  PLAN.md                          研发计划与测试矩阵
  mf310p_v2_adapter_api.h          ABI v1:布局/本地格偏移常量 + create/launch 声明
  mf310p_v2_device.asc             8 模板 lanes 内核 + init/probe/quiet,槽分区生产者
  mf310p_v2_adapter.cpp            create/destroy/geometry/launch 薄壳(链接复用 v1 建池)
  memfabric_mm_ar_v2_torch_adpt.h  torch 侧校验
  memfabric_mm_ar_v2_binding.cpp   op 注册(v2_allreduce / v2_shutdown / v2_probe)
  memfabric_mm_ar_v2_runtime.cpp   路由/生命周期/warmup/graph/毒化
cmake/memfabric_310p_v2.cmake      bisheng dav-2002 编译 v2 设备库 + 接线
```

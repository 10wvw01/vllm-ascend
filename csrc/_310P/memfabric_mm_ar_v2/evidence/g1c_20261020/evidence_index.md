# G1 原始证据索引（evidence_index.md）

> 生成：2026-10-21（G1-C 原始证据整理轮）。
> 用途：为 G1-B B02、G1-C C2R1/C2R2 三次会话的原始证据建立可核验索引，
> 供独立复核并确定 G1-D 最小取证方案。**本轮不修改任何算子/通信代码，不做新长稳测试。**
>
> 证据分级：`observed`=日志直录；`derived`=带推导链的算术结论（推导依据逐条列出）；
> `unknown`=日志未记录且不可推导。禁止推测填充。

## 1. 原始会话与服务器位置（完整原始日志仅存服务器）

| 会话 | 服务器路径（已脱敏显示） | 性质 | 结果 |
|---|---|---|---|
| B02 | `/home/<user>/310P/vllm-ascend-diag/.../tmp/opencode/gate_e_forensics/B02/` | G1-B Host 飞行记录器（无 V2 阶段戳、无 SQ 直采） | 双 Rank AICore timeout（第 29 请求，13:22:04Z） |
| C2R1 | `/home/<user>/310P/vllm-ascend-g1c/.../tmp/opencode/gate_e_forensics/C2R1/` | G1-C 全诊断（V2 阶段戳 + CP SQ 直采） | 未复现（75/75，消费 15027 件/设备） |
| C2R2 | `/home/<user>/310P/vllm-ascend-g1c/.../tmp/opencode/gate_e_forensics/C2R2/` | G1-C 全诊断 | **复现**（第 3 请求，双 Rank V2 gen208 卡 stage 50/60） |

- 全量 164 文件的 SHA256 清单：`raw_manifest_SHA256SUMS_sanitized.txt`（本目录，
  路径已脱敏；服务器原件 `raw_manifest_SHA256SUMS.txt` 为绝对路径版）。
- 服务器端另有只读原始归档：`/home/<user>/310P/g1c_evidence_package/raw_archive_g1_20261021.tar.gz`
  （含三会话目录全量 + 归档时清单，SHA256 见 §6）。
- 传递用压缩包（脱敏）：`g1c_evidence_package_20261021.tar.gz`，内容 = 本目录全部
  脱敏/结构化产物，不含原始日志。

## 2. 结构化证据（由可复现脚本生成，确定性输出）

生成命令（在 g1c worktree）：

```bash
python3 csrc/_310P/memfabric_mm_ar_v2/test/g3_e2e_scripts/g1c_evidence_extract.py \
  --session B02=<B02目录> --session C2R1=<C2R1目录> --session C2R2=<C2R2目录> \
  --out <输出目录>
```

| 文件 | 内容 | 规模 |
|---|---|---|
| `timeline_B02.json/.csv` | 统一时间线（tick/epoch/V2/serve_error/device_hwts_timeout） | 1308 事件 |
| `timeline_C2R1.json/.csv` | 同上（健康对照） | 4528 事件 |
| `timeline_C2R2.json/.csv` | 同上（复现轮，含停滞窗全序列） | 1125 事件 |
| `session_stats_*.json` | 各会话统计（bench、MF 采样、纪元、不变量核验） | 3 份 |
| `boundary_1024_analysis.json` | **1024 边界专项核算**（observed/derived/unknown 分级 + 推导链） | 见 §4 |
| `extract_manifest.json` | 生成清单（会话→事件数/采样数/摘录窗口） | — |

时间线字段语义：`time_quality` = `exact`（微秒级日志时间戳）/ `interpolated`
（V2 行无时间戳，由相邻 MF 行线性插值）/ `lower_bound`（文件尾部无后锚点）/
`second_resolution`（vLLM 秒级行）/ `millisecond_resolution`（E10x 行）。
设备时间（`device_hwts_timeout` 的 `device_time_utc`）= 设备日志时间 −8h（时区对齐
依据：serve 启动时刻 vs 设备 run 日志首条，两会话一致核验）。

## 3. 脱敏日志摘录（`sanitized/`）

| 文件 | 窗口（UTC） | 内容 |
|---|---|---|
| `B02_132050_132210.log` | 13:20:50–13:22:10 | 纪 22 末次消费 → 纪 23-25 零消费 → 双 Rank E1020/507014 |
| `C2R1_170135_170145.log` | 17:01:35–17:01:45 | **健康 ARR/SQ 回绕穿越**（reqSeq 1016→1086，~70 封/秒） |
| `C2R1_171225_171235.log` | 17:12:25–17:12:35 | 健康终态（15027/15027，SQ 空） |
| `C2R2_171840_172030.log` | 17:18:40–17:20:30 | **停滞全窗**：gen197 退出 → gen208 双 Rank 卡 50/60 → 环排空 → 纪 6-9 正常 → 设备/宿主报错 |

脱敏规则（仅替换标识，技术内容零改动）：`/home/<user>`（用户路径）、`<host>`（主机名）、
`<container-ip>`/`<ip-lan-a>`/`<ip-lan-b>`（容器/内网 IP）。127.0.0.1 保留。
残留检查：三份摘录对原始敏感串 grep 均为 0 命中。

## 4. C2R2 统一时间线（停滞窗关键行，详见 timeline_C2R2.csv）

| UTC 时刻 | 质量 | 事件 | 双 Rank 状态 |
|---|---|---|---|
| 17:18:47.3 | exact | 纪 4 退出（负载前最后空转纪） | dev0/1 consumed=3（累计 9） |
| 17:18:47 | sec | 双 Rank 首调 V2 bit-exact（M=4096） | 纪 5 开始承载全部负载消费 |
| 17:18:59.3/.44 | exact | MF_G1C 采样 | dev0 reqSeq=931 / dev1=941（st=748/778，流动中） |
| 17:19:00.30/.44 | exact | MF_G1C 采样 | dev0 reqSeq=1006（st=973）/ dev1=1016（st=1003）；V2 gen197 双 Rank 全核 120（完成） |
| **17:19:00.7–17:19:01.0** | interp | **V2 gen208 进入并停滞** | rank0 core0=208:1:50（等对端 chunk1），cores1-7=60；rank1 core0=208:3:50（等对端 chunk3），cores1-7=60 |
| 17:19:01.31/.44 | exact | MF_G1C 采样 | **dev0 reqSeq=1025（st=3，hwH=hwT=3）/ dev1=1026（st=6，hwH=hwT=6）**：环排空、SQ 空、HAL rc=0、phase=8 空闲 |
| 17:19:02.41/.54 | exact | 纪 5 退出 / 纪 6 启动 | exitReason=0/mainRet=0（consumed 1016/1017） |
| 17:19:02→17:20:03 | exact | 纪 6-9 | 每 15.12s 干净轮转，consumed=0（控制面全程健康） |
| **17:20:10.216/.236** | exact | **设备 HWTS 看门狗击杀**（device_hwts_timeout） | rank0（slot12/stream33/task30929）/ rank1（slot16/stream8/task30818），间隔 20ms |
| 17:20:12.926/.932 | exact | 宿主表面化首错 507014 | 双 Rank 事件同步失败，间隔 5.6ms |

停滞→设备击杀间隔 ≈ 69.7s（aicore 看门狗周期）；B02 同构（停滞窗 ∈(13:20:55,13:21:10]，
设备击杀 13:22:05.635/.654，间隔 55–70s）。

## 5. 关键结论修正（相对 G1-C 报告的增量，均已写入 boundary_1024_analysis.json）

1. **“双 Rank 对称双丢失”表述修正为“单点丢失 + 下游等待”**：rank1 等待的 chunk3
   邮件，其源（rank0）停滞前从未投递 signal（仅投递 chunk0-2）——该等待是停滞的
   **下游结果**，非第二封丢失。唯一无法解释的等待是 rank0 等待 rank1 的 chunk1 邮件
   （该 signal 已被 dev1 消费、SQE 链已执行，见 §6 推导 D-loss）。
2. **几何判别**：overlap 流水（stage50 chunk 字段=prev）经双 Rank 锁步约束唯一化
   （非 overlap 假设与终态 reqSeq 不符，见 D-geometry）。
3. **B02 终态与 C2R2 同构**：dev0/dev1 消费 6146/6145（差 1 封），末封槽 1/0
   （跨第 6 次 wrap 后 1-2 封内）；C2R2 为 1025/1026（差 1 封），末封槽 0/1
   （**恰为第 1 次 wrap 后**）。两失败会话的终态游标均停在 1024 边界 ±1 封处，
   而健康 C2R1 终态在边界后 690 封（全程跨 14 次 wrap 无异常）。

## 6. 1024 边界专项（摘要；完整推导链见 boundary_1024_analysis.json）

**源码常量**（事实）：`ARR_SLOTS=1024`、`REQ_SLOTS=64`、`QUIET_SLOTS=64`
（`smem_shm_sdma_layout.h:45/49/52`）；每邮件 3 SQE（载荷→到达戳→完成戳，
`issueMailSqes`）；到达戳 1 基（stamp=seq+1），等待槽=ARR_HEAD%1024，轮询上限 1e8
（事实无限等待，`wait_at`）；**ARR_SLOTS=SQ 深度=1024 且 3 SQE/邮件 ⇒ seq≡1023 的
邮件同时占据 ARR 末槽(1023)与 SQ 末槽(1023)——两环边界被算术锁定**。

**C2R2 推导链**（每步依据见 JSON `derived[].basis`）：

| 步 | 结论 |
|---|---|
| D-geometry | overlap 模式；停滞代 signal 数 rank0=3 / rank1=4（锁步判别唯一） |
| D-seq | 停滞代 signal seq：rank0={chunk0:1022, chunk1:1023, chunk2:1024}；rank1={chunk0:1022, chunk1:1023, chunk2:1024, chunk3:1025}（基线 1022/1022） |
| D-loss | **丢失邮件 = rank1 的 chunk1，seq 1023（dev1 请求环），到达槽 dev0 ARR[1023]（末槽），期望戳 1024；其 SQE 链止于 dev1 SQ 槽 1021-1023（末槽）**；前一封（chunk0，seq 1022，槽 1022）已到达（rank0 通过 prev=0 等待）；同代下一封（seq 1024）为 wrap 邮件（槽 0） |
| D-wrap | 反方向（dev0→dev1）的 wrap 邮件（seq 1024，dev1 ARR 槽 0）**健康到达**（rank1 通过 prev=2 等待证明）——即同代内槽 0 正常、槽 1023 丢失 |
| D-inv | 双设备 shadowTail==3×reqSeq mod 1024（3/6）；C2R1/B02 同构核验全部成立 |

**跨会话模式**：两次失败（C2R2 第 1 次 wrap、B02 第 6 次 wrap）均以“游标停在边界
后 0-1 封、唯一/推断丢失邮件位于 ARR 末槽 1023”收场；健康会话 14 次穿越边界无异常。
**该模式是统计上显著的线索（若丢失位置与环位无关，两次均落在边界区间的概率量级
~1e-5），但样本 n=2，且 B02 的丢失邮件槽位依赖“同构假设”（无 V2 戳直证），
不构成根因结论——判定权在独立复核。**

## 7. UNKNOWN 登记（日志未记录，禁止推测填充）

| 字段 | 状态 | 说明 |
|---|---|---|
| arrHead（等待设备到达环游标）运行值 | UNKNOWN | 仅可由等待参数推导（C2R2 推导值=1023，见 D-loss）；无直采 |
| 到达槽 1023 的实际内存内容/戳值 | UNKNOWN | 停滞时未转储；无法区分“SQE#2 未写入”与“写入后不可见” |
| wrap 邮件（seq 1024）在 dev0 到达环的落地状态 | UNKNOWN | rank0 从未等待它，不可观察 |
| 每个 SQE 的硬件消费时刻 | UNKNOWN | 仅 1Hz 低频采样（phase8/32 封边界） |
| B02 在途邮件的 seq/槽位 | UNKNOWN | 无 V2 阶段戳（仅终态游标+同构推断） |
| F01/F02/F02R1/S01 的环游标 | UNKNOWN | 无 MF 记录（G1-A 无记录器/stock 不经环） |
| V2 行时间戳 | 无原生 | 时间线用线性插值（已标注 time_quality），±0.5s 级精度 |

## 8. 服务器端产物与哈希（2026-10-21）

- 原始归档：`/home/<user>/310P/g1c_evidence_package/raw_archive_g1_20261021.tar.gz`
- 传递包（脱敏）：`/home/<user>/310P/g1c_evidence_package/g1c_evidence_package_20261021.tar.gz`
- 两者 SHA256 与内部清单校验：见同目录 `package_SHA256SUMS.txt`（生成于打包时）。
- 结构化产物自身哈希（部分，全量见 `SHA256SUMS_extracted.txt`）：
  - `boundary_1024_analysis.json` `edb9ba472e7ae783bdfbca67dc5832114f944c45af60ab28839c11a9113c9119`
  - `timeline_C2R2.json` `c020d614e279f1000fc17d77c862eb20e5115efee547e77c4221c08781c95ea6`
  - （完整清单随包内 `SHA256SUMS_extracted.txt`）

## 9. 数据质量声明

- [MF_G1C] 的 hwHead/hwTail 为**最近一次** HAL 查询值（低频：空闲期 ~8192 循环/次、
  批次期 32 封边界/次），非采样时刻实时值；终态（phase=8、环空）语义下为精确值。
- [V2_G1C] 为每核最后写入的阶段字（64B 独占 cache line，写入即 clean）；
  D2H 采样可能落在两阶段之间，判读停滞以“连续多帧 changed=0 且非 120”为准。
- 双进程共享 stderr 存在少量行级穿插（G1-C 已修复 V2 行原子性；MF 行为 BM_LOG 单行写，
  观测到极少数穿插行，解析器跳过并计数）。
- 设备日志时区 = UTC+8（对齐依据见 §2）。

# A3 优化轮 A+B-2+C（2026-10-30 R2）：实现落地、六门禁全过、共享卡批次无可见墙钟收益

执行：OpenCode。用户授权的优化方向（前一轮数据裁决后确认）：**A（轮询 FlushLine 摊销）+ B-2（S 双槽化、去尾部最终 quiet）+ C（Host 相开销削减）**，只落在原 A3(3302) 主路径，Stock/V2/V3 基线内核不动。

## 本轮身份

| 项 | 值 |
|---|---|
| vllm-ascend HEAD | `eeb52f0ba`（起点 `b34901ba6` + 2 提交：`8a72abe88` 门禁修复、`eeb52f0ba` 本轮主体） |
| MemFabric | `db38d75e`（未修改，禁改约定不变） |
| 最终 manifest | `v3-build-r8`（head=`eeb52f0ba`，device_compile PASS；build_tensor + V2 freeze 完成后） |
| 设备/环境 | 卡 16384 dev2/3 物理直选，共享卡（用户授权 `--allow-shared-card`），aarch64 / CANN 9.1.0 |
| CPU 门禁 | 147 passed（v3 test 93 + v2/ut 54），ruff/codespell/typos/markdownlint 全过 |
| 三 A3 六门禁 | r8 manifest 上 3×PASS_QUICK_SCREEN_ONLY（`gates-r8/`） |
| 11 会话批次 | `campaign-shared-r5` 全 PASS_UNIFIED_ONLY → campaign PASS_ATTRIBUTION_ONLY（`campaign-r5-paired/`） |

## 代码改动（提交 eeb52f0ba，13 文件）

1. **A（摊销失效）**：A3Kernel Core0 波等待与 A3WaitSeats 自旋原来每圈 `FlushLine`；改为每 8 圈失效一次。标志只写一次不清零，最多引入 7 次陈旧读、不会漏检（死锁不可能：周期性失效持续重试）。
2. **B-2（S 双槽 + 去最终 quiet）**：池布局 `[Y0|Y1|S0|S1|ctrl]`（80MiB；MemFabric 源码实证 symmetric stride = ALIGN_UP(localSize, 1GiB)，恒为 2 的幂，几何校验恒过）。Host 传 per-slot S 指针；入站 header 地址从 `work+arena+64` 改为等价的 `ctrlSrc+64`。尾部 Drain→quiet→DONE 保留，**最终 quiet 删除**：DONE 交换（send+到达等待）证明对端已 Drain 全部入站 ADD ⇒ 本端出站 S 读已完成；且下一调用 seed 另一槽。V3 基线路径刻意不动。
3. **C（Host 热路径）**：ctypes 入口预绑定、reserve 输出 cell 复用、torch_npu 模块缓存（会话级），消除每调用的重复查找/分配。
4. 新增不变量测试（单 quiet、per-slot source、摊销失效计数），更新 host mock 几何（128MiB 对称）与 adapter fixture。

## 结论一：正确性全绿，改动可保留

CPU 147 + 设备 3×6 门禁 + 11 会话批次全部通过。B-2 的安全论证（DONE 交换 ⇒ 出站读完成）在门禁的交错/回绕用例下未出现任何反例。

## 结论二：共享卡墙钟无可见收益（诚实数据）

同批对照（r5 各臂 vs 本批 stock/v2 括号）：

| 配置 | r5 | r4 | 判读 |
|---|---|---|---|
| a3-c1024 M2048/comm1024 vs stock | −24.9%（stock 括号漂移 3.3%，unstable） | −22.7% | 噪声内 |
| a3-c1024 M4096/comm1024 vs stock | −11.5% | −8.3% | 噪声内 |
| a3-c1024 M4096/comm1024 vs v2 | 快 9.2% | 快 13% | 噪声内 |
| 全部 11 臂 r5 vs r4 绝对值 | ±3% 内随机散布 | — | stock 自身 before/after 漂移 −2.7%/+2.7% |

Host 五相 r5 比 r4 整体 +7~26%（含未触碰的 wrap/input_event 桶同步膨胀）→ 共享卡宿主 CPU 争抢，C 的 ~10µs 量级预期收益低于噪声地板，无法归因。

## 结论三：r7 管线计数器基线被污染（重要方法学发现）

r7 归档探针（16:29-16:31 窗口）与 r4 批次同核对照出现**不可能矛盾**：

- r4 批次 a3-c1024 M2048/comm1024 Event p50 = 1381µs；r7 探针同配置 kernel Duration p50 = 1443µs。**Event（含 kernel）< kernel 自身** ⇒ r7 探针 kernel 时长被同窗共租户干扰膨胀 ≥260µs。
- r8 探针自洽：kernel 1210µs < Event 1406µs，差 ≈196µs ≈ 既有 host gap 结构。
- 因此 r7→r8 的表观 a3 kernel 时长 −16~−22%、scalar_ratio 0.70→0.56 **不能归因代码改动**；上一轮的 728/913/1081µs scalar 管线时间同属污染读数，只保留方向性结论（scalar 占比高、mte2 ~70-76%、cube 91-92%）。

r8 干净基线（`pipe-profiler-r8/pipe_attribution_r8.txt`，18 配置 × 双 rank × 10 调用）可作为后续轮次的对照起点。

## 判读与建议

1. 三项优化攻击的成本（最终 quiet 在 DONE 时点本就近空转、FlushLine 指令开销、Host µs 级开销）**均低于共享卡 ±3% 噪声地板**——不是方向错误，是当前仪器分辨率不足。
2. 改动保留：零回归、结构上解除尾部对 S 复用的串行保护、为未来（独占卡/更高精度仪器/流级重叠设计）铺路。不签任何性能 GO（与既有一致）。
3. 后续如需证明 <100µs 级效果：独占双 Die 窗口 + 同批 A/B（同 manifest 两构建）+ Event/kernel 双仪器，或改用单卡直调微基准。

## 目录

- `gates-r8/`：三变体六门禁 result/g0（PASS_QUICK_SCREEN_ONLY ×3）。
- `campaign-r5-paired/`：11 会话 paired.json + campaign.json（PASS_ATTRIBUTION_ONLY）。
- `pipe-profiler-r8/`：18 份 kernel_details CSV（三变体 × c512/1024/2048 × 双 rank）+ 汇总 + 时长中位数。
- `probe-scenes/`：探针 console 日志、prof_index、lease_guard 留痕。

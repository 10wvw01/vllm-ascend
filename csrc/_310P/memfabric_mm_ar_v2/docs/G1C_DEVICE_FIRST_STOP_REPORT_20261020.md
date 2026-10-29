# G1-C：310P3 设备侧第一停止点取证执行报告（2026-10-20 实机）

> 方案依据：`docs/G1C_DEVICE_FIRST_STOP_DESIGN_20261020.md`（下称"设计文档"）。
> 执行：OpenCode @ 310P3 服务器，2026-10-20 16:00–17:35 UTC。
> 本报告只陈述实机事实与判别状态。**根因未定**：本轮目标是找到第一个设备侧停止点，
> 不是修复 BUG；证据不足处明确标记 UNKNOWN。
> 过程文件仅存服务器 `/tmp/opencode/g1c/`，未提交公网 GitHub；证据存本 worktree
> `tmp/opencode/gate_e_forensics/`（gitignored）。

## 1. 环境与版本（Gate C0）

### 1.1 分支与哈希

| 项 | 值 |
|---|---|
| vllm-ascend G1-C 分支 | `diag/310p3-g1c-device-first-stop-20261020` @ `3671dbb98`（自 G1-B 报告 HEAD `8f1eef3b4` 派生，git merge-base 已核验） |
| MemFabric G1-C 分支 | `10wvw01/memfabric_hybrid_310p` 同名分支 @ `6680424d`（自 `eb96eced` 飞行记录器版本派生；本机 origin=GitCode，仅用 github-dev 拉取） |
| 独立 worktree | `/home/w00804146/310P/vllm-ascend-g1c`、`/home/w00804146/310P/memfabric_hybrid-g1c`（原 `54c367ac`/`eb96eced` 工作树未触碰） |
| 主工作树隔离 | serve 经 `PYTHONPATH=<g1c-worktree>` 前置加载 G1-C 包（PathFinder 先于 editable finder，实测确认）；主树 `vllm_ascend/*.so` 全程未改动（构建前后 SHA256 复核） |
| CANN | `/usr/local/Ascend/cann-9.1.0`（9.1.0，V100R001C11SPC001B243），bisheng dav-2002 |
| 设备 | 310P3 ×2（device 0/1，同卡双 die），空闲起步，全程无第三方 NPU 任务 |

### 1.2 三类哈希（设计文档 §5 C0-1 要求）

| 产物 | SHA256 | 说明 |
|---|---|---|
| 原安装 Host `libmf_hybm_core.so`（双位置） | `88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1` | 与 G1-B 恢复后状态一致；备份于 `memfabric_hybrid-g1c/build-g1c/g1c_backup_20261020T164118Z/`（只读，SHA256SUMS 自校验 OK） |
| G1-C Host `libmf_hybm_core.so`（`6680424d` 构建） | `9d3c0cefab785205eec96622ec04c1f2702791233d6b4bf6f1eb85c5b46d6dca` | 内嵌版本串 `commit: 6680424d55c3...`；`[MF_G1C]`/`[MF_DIAG]` 字符串在位；ldd 依赖与原库一致 |
| 原 AICPU `libmf_sdma_orch.so`（KVER=11，设备侧） | `db7134b8a2fa56f7250097c36d6a530d38901743d5eb8055bf9168bb52aee401`（安装包侧，B02 provenance 同值） | **全程未覆盖**；设备侧 `libmf_sdma_orch_v11.so` 存活探针 sync=0 |
| G1-C AICPU kernel（`-DHYBM_G1C_DEVICE_TRACE=1`，KVER=12） | `02290083e630ff43470ea4441a15a78614b67b232b5ce3eed20f482dbe541597` | 以**新文件名** `libmf_sdma_orch_v12.so` 部署设备搜索路径（14072 字节，符号+尺寸校验通过），规避 CP1 对 v11 的 dlopen 缓存 |
| v12 launch JSON | `90172726965f4944ed955237c296faf1622762897f742ffbecf69a9e2fb554d0` | `kernelSo: libmf_sdma_orch_v12.so`，`MF_SDMA_ORCH_JSON` 显式指向 |
| G1-C V2 设备库 `libmf310p_v2_trial_device_g1c.so` | `d2317be047a8ee7ed81bcf2e3bcef06cd841edb49f841778c182eb8b888a4464` | bisheng 命令含 `-DMF310P_V2_G1C_TRACE=1`（build.make 复核） |
| G1-C `vllm_ascend_C...so` | `98bb081fedaf47d75a4bde6cd5293c458868108f5d92d3d10fc5e44e59e87371` | DT_NEEDED 指向 `libmf310p_v2_trial_device_g1c.so`（非默认名），RUNPATH `$ORIGIN` |
| 对照：默认构建（无 G1-C trace） | `libmf310p_v2_trial_device.so` 2185568 字节，`vllm_ascend_C` NEEDED 指向默认名 | 导出符号 9 个（5 正式 + 4 诊断）与 G1-C 构建完全一致，无退化 |

## 2. 编译与测试结果（Gate C0）

1. **CPU 单测**：V2 侧 `test_g1c_forensic_source.py` + `test_g1c_trace_summary.py`、MF 侧
   `test_g1c_device_trace_source.py`。首跑 1 项 FAIL（见 §3 兼容修复 #1），修复后 **9/9 PASS**。
2. **MemFabric CANN 9.1 编译**：全新 `build-g1c/`，Ninja RELEASE，
   `-DMF_SDMA_ORCH_KVER=12 -DMF_SDMA_ORCH_G1C_DEVICE_TRACE=ON`（其余参数与 10-19 现网构建一致）。
   产物即 §1.2 表；编译零错误零告警新增。
3. **V2 CANN 9.1 编译**：`setup.py build_ext --inplace --build-temp`（预置 CMakeCache
   `VLLM_ASCEND_310P_V2_G1C_TRACE:BOOL=ON`），产物只落 g1c worktree。**未出现** dav-2002
   不支持 trace 写 / Host ACL context 类编译错误，无需实机兼容修复。
4. **默认对照构建**：无 trace、独立构建目录，链接默认设备库名、导出符号不退化（§1.2 末行）。
   注：该构建的 `cmake --install` 会写入 worktree（setup.py install_path 行为），
   对照检查在 `/tmp` 产物上完成后已用 G1-C 构建重建 worktree 内文件（最终哈希以 §1.2 为准）。

## 3. 实机诊断兼容修复（仅诊断代码，未触碰任何算法）

| # | 文件 | 问题 | 修复 |
|---|---|---|---|
| 1 | `test/g3_e2e_scripts/g1c_trace_summary.py` | 不完整 V2 帧不计数，但 parser 在计数前 `continue`，rank 条目不存在，已提交 UT（`test_incomplete_v2_frame_does_not_count_as_evidence`）期望"见过该 rank 但 0 样本"语义 → KeyError FAIL | 帧识别后先建 rank 条目再判完整性；不完整帧保留 `v2_samples=0`/`v2_last={}` 条目，仍不作为证据 |
| 2 | `src/op_host/mf310p_v2_trial_adapter.cpp` | 1Hz 采样行由 10+ 次逐字段 `fprintf` 组成，双 rank 共享 stderr 时被互相穿插，产生不可解析证据行（C1 首跑实测大面积穿插） | 整行组入 512B 缓冲后单次 `fprintf` 输出（单次 write，行级原子）；修复后 C1/C2 全程 0 穿插行 |
| 3 | `test/g3_e2e_scripts/gate_e_eager.sh` | 脚本硬编码 KVER=11 launch JSON 并 unset `MF_SDMA_ORCH_JSON`，G1-C 无法指向 v12 | 保留调用方显式 `MF_SDMA_ORCH_JSON`（保存-还原），默认行为不变（仍固定 v11，缺失即报错退出） |

三处修复均为诊断/脚本层；MM、通信、SQE、Mailbox、IB、Epoch、KFC 算法零改动
（`git diff 8f1eef3b4..HEAD -- src/ 'csrc/**/src/**'` 仅含 #2 一个 hunk + 既有的 G1-C 埋点提交）。

## 4. Gate C1：取证链路自检（16:41–16:52 UTC）

**部署**：原库双位置备份（只读+自校验）→ v12 三件套落 `/tmp/opencode/g1c/kernel_v12_stage/`
→ G1-C Host 库替换双位置（哈希逐位置核对）→ 首次 C1 运行触发自动部署链，
v12 以新文件名落设备搜索路径（符号+精确尺寸校验通过）。
注：进程内 `dladdr` 定位部署助手失败（site-packages 位置无 helper），回退进程内部署链；
因 v12 已在位，serve 探测直接命中，无 KFC 非零退出风险；C2 额外显式指定
`MF_SDMA_ORCH_DEPLOY_HELPER` 指向 G1-C 构建的 helper 作纵深防御。

**最小正确性**（`torchrun --nproc-per-node=2 tp2_v2_mixed_route_correctness.py
--rows 2047 2048 2049 4095 4096 4097 --reps 3`，全诊断环境变量开启）：

| 检查项 | 结果 |
|---|---|
| 混合路由 bit-exact | **51/51 PASS**（含 Stock/HCCL 共存、4 chunk 轮转、V2 首调位精确） |
| `[G1B]` 日志级注入 | 7 进程 `hybm_set_log_level(1) rc=0` |
| `[MF_DIAG]` | start epoch 12、exit epoch 8（短测尾纪运行中属预期）、tick 连续 |
| `[MF_G1C]` | 133 行；`valid:1` 100%；`queryHeadRcLast/TailRcLast` 全 0；hwHead 12 个不同取值（0→438 实时推进） |
| `[V2_G1C]` | 148 完整帧（修复 #2 后 0 穿插），**8 核代际不一致帧 = 0**；观测到 stage 0/50/60/120（真实飞行中采样） |
| 采样扰动 | 无 507014、无 MTE3 首写丢失、无通信结果不一致、无 snapshot_error |

C1 通过，进入 C2。

## 5. Gate C2：原 Gate E 工况取证（16:56–17:30 UTC）

工况 = 原 G1-A Gate E：TP2 fused Eager、input 4096 / output 256、25 prompts/rep、
并发 10、620s 窗口、`MF_SDMA_ORCH_EPOCH_BUDGET_S` 未设置（15s 默认纪）。
执行入口 `test/g3_e2e_scripts/g1c_forensic_run.sh gate-e <会话>`（环境见其 `env` 子命令）。
**与 F01/B02 不混算复现率**（设计文档明令）。

### 5.1 C2R1（16:56–17:12）：未复现

- 75/75 请求完成（3 reps），0 错误，无 AICore timeout；`timeout_hits=0`。
- V2 采样 724/724（双 rank），MF 采样 721/722，0 invalid，0 snapshot_error。
- 编排消费 15027 件（dev0/dev1 各半），**超过 B02 死亡点（12291）**；V2 代际推进至 3008；
  SQ 游标观测到 1023→回绕（模 1024 干净回绕，与 B02 的 3 SQE/请求不变量一致）。
- 全程无一例中途停滞帧（唯一重复状态为 gen 完成后的 120 空闲与收尾 gen408 的 46 帧静默——
  负载进入 decode 段 M<2048 走 Stock，V2 无新调用，属正常）。
- 证据已归档：serve.log（`d7dba3dd...`）、3 个 bench JSON、设备 run/plog、
  `evidence/c2r1_sha256.txt`。

### 5.2 C2R2（17:13–17:28）：**复现，第一停止点证据落袋**

**会话时间线（双 Rank，UTC）**

| 时刻 | 事件 |
|---|---|
| 17:13:55 | fused serve 启动（全诊断环境） |
| 17:17:25 | 健康（V2 层标记 80 = 40×2 rank） |
| 17:18:47 | 双 rank 首调 V2 bit-exact 通过（M=4096） |
| **17:19:00.3–17:19:01.3** | **V2 代际 208 进入并停滞（双 rank 同一代际同一秒）**：core0 卡 **stage 50（`smem_shm_sdma_wait_at` 等待对端到达邮件）**，core1–7 卡 **stage 60（`ReleaseAdd` IB 核间汇合，被 core0 阻塞）**。rank0 等待对端 **chunk1** 邮件（chunk 字段=1，即 k=2 迭代 prev），rank1 等待对端 **chunk3** 邮件（收尾 last 迭代）——rank1 此前已消费 chunk0–2 邮件，rank0 已消费 chunk0 邮件 |
| 17:19:01.3/1.4 | 双设备 AICPU **消费完全部请求环**（reqSeq 1025/1026），转空闲（phase 8），SQ 空（hwHead==hwTail，回绕后 3/6），HAL 查询 rc=0，error=0 |
| 17:19:02 → 17:20:03 | 纪 5 正常退出（exitReason 0 / mainRet 0 / 15.12s），纪 6–9 每 15.1s 干净轮转——**计算面停滞期间控制面全程健康** |
| 17:20:12.926 / 17:20:12.932 | rank0（TP0 pid 4165430）/ rank1（TP1 pid 4165441）先后 `aclrtSynchronizeEvent` 507014 AICore timeout（间隔 **5.6ms**；F01 18ms、B02 31ms，同签名） |
| 17:19–17:25 | rep2 起连接失败；620s 窗口 32 reps，仅 rep1 完成 3 请求（比 B02 的第 29 请求更早） |

**死亡瞬间双 Rank 关键状态（最后有效帧）**

| 证据 | rank0 / dev0 | rank1 / dev1 |
|---|---|---|
| V2 8 核代际 | 208（8/8 核一致） | 208（8/8 核一致） |
| V2 core0 | `208:1:50` = 等 chunk1 邮件，**不再推进** | `208:3:50` = 等 chunk3 邮件，**不再推进** |
| V2 core1–7 | `208:1:60`（IB 汇合等待） | `208:3:60`（IB 汇合等待） |
| AICPU phase | 8（空闲环） | 8（空闲环） |
| reqSeq（累计消费） | 1025 | 1026 |
| 软件游标 | reqHead=reqTail=1025（环空） | reqHead=reqTail=1026（环空） |
| 硬件 SQ | hwHead=hwTail=3（空，rc=0） | hwHead=hwTail=6（空，rc=0） |
| Epoch | 纪 10 运行中，纪 5–9 全部 exitReason=0 | 同左 |
| 设备 plog | 17:19:00–17:20:12 窗口零 WARN/ERROR/SDMA 异常；仅 17:20:12.926 PeekLastErr 507014 | 同左（.932） |

采样链健康至进程死亡：V2 144 帧/rank、MF 145 帧/rank、0 invalid、0 snapshot_error；
`[MF_DIAG]` 最后 tick 距 rank0 首错 <1s（reqPending=0、cursorsValid=1）。

证据归档：`tmp/opencode/gate_e_forensics/C2R2/`（serve.log、32 个 bench JSON、
`evidence/g1c_trace_summary.json`（离线解析器输出）、设备 run/plog、provenance、samples/）。

## 6. Gate C3：判别与根因假设状态

按设计文档 §5 C3 判别树逐条对表：

| 判别分支 | 是否命中 | 依据 |
|---|---|---|
| Core0 卡 50 且 MemFabric 尚未生成对应 SQE → 查 AICPU 消费/提交 | **否** | 双 rank AICPU 已把请求环消费到空（reqSeq 1025/1026），phase 8 空闲，无积压 |
| SQE 已发但硬件 SQ Head 不推进 → 查 CP SQ / 固件 | **否** | 硬件 hwHead==hwTail（SQ 空），HAL 查询 rc=0，且停滞期间纪 6–9 照常 15.12s 轮转（CP 活着） |
| **硬件消费完成 → 查 V2 的 wait / 缓存可见性** | **命中** | CP SQ 排空 + AICPU 空闲 + 双 rank core0 永久等待"已被 CP 消费链处理"的到达邮件 |

**第一设备侧停止点（观测事实，非根因结论）**：
V2 融合算子（双 rank 同一代际 208）的 **core0 到达邮件等待**——即
"CP SQE 链执行完成（SQ 排空）" 与 "AICore 观察到对端到达邮件/stamp" 之间的链路。
停止点之前的全部前置环节（Host 控制面、请求环、AICPU 消费、SQE 提交、CP 硬件消费）
在同一时间线上均有"已推进到空闲/排空"的直接证据。

### 6.1 假设状态更新（对照 G1 设计 §2）

| ID | 假设 | G1-C 证据 | 状态 |
|---|---|---|---|
| H1 SQ 容量失控 | CP SQ 停滞瞬间为空（hwHead==hwTail），软件 shadowTail 干净回绕，无覆写签名；C2R1 全程 15027 件消费零异常 | **设备侧直接证据反驳**（较 G1-B 的"Host 层面反驳"升级；`submitSqe` 静态风险仍留档，但本轮未观测到其后果） |
| H2 Epoch 消费链停顿 | 停滞期间纪 5–9 正常轮转退出；纪 10 运行中；exitReason/mainRet 全 0 | **强证据反驳（再确认）** |
| H3 共享 Cache Line / 可见性 | 双 rank 同代际、对称双等待；对端链路"已发已消费"但邮件未达等待者；丢失点两个方向不同 chunk 索引（rank0←chunk1，rank1←chunk3）——更像时序相关的单件丢失而非固定槽位错位 | **显著增强（主嫌疑，仍未证实）** |
| H4 V2 IB/同步死锁 | core1–7 的 stage 60 是 core0 阻塞的**下游结果**（IB 汇合等 core0），core0 阻塞点在 MemFabric 邮件等待而非 IB 本身 | 原表述收窄：IB 核间汇合本身未观察到首停 |

### 6.2 无法用现有证据判别（UNKNOWN，禁止推测补全）

1. **丢失的是"邮件可见性"还是"SDMA 数据落地"**：SQE 排空证明 CP 已执行 SQE 三连，
   但无 per-SQE 完成戳，无法区分"SDMA 拷贝完成而到达 stamp/邮件未被 AICore 观察（缓存一致域问题）"
   与"邮件从未写入（SQE 执行不完整）"。
2. **双 rank 同代际对称丢失是否同因**：两方向丢失不同 chunk 索引，可能是两次独立丢失，
   也可能是同一时序窗口（如同一 wall-clock 附近 SDMA 写与 AICore 缓存状态竞争）的对称表现。
3. **无"近失"样本**：C2R1（3008 代际）与 C2R2 停滞前（207 代际）均无"卡数秒后自恢复"帧——
   等待要么正常完成要么永久挂起，支持"丢失型"而非"慢速型"；但样本量不足以排除低概率自恢复。
4. **观测扰动**：C2R2 故障点（第 3 请求）显著早于 F01/B02（第 29 请求），埋点的 GM/Cache 操作
   改变竞争时序（设计文档已预告）；G1-C 内部不构成复现率统计。
5. **宿主报错表面点漂移**：C2R2 首错在 async runner 事件同步（无 aclnnSwiGlu 字样），
   B02 在 aclnnSwiGlu——再次印证表面报错算子非根因。

## 7. 未解决问题与建议下一步（供方案负责人决策，本轮不执行）

1. **到达链路分段取证**：在 SQE 三连中为"到达 stamp 写入"与"邮件写入"增加独立的设备侧完成戳
   （需设计新诊断版本，涉及 SQE 配方区——**超出本轮"不改 SQE/配方"红线，须专项评审**）。
2. **AICore 侧等待点转储**：停滞时读取 AICore 等待的邮件槽地址/期望 tag 与到达环游标快照，
   判别"槽位错位"vs"内容不可见"。
3. **无诊断对照轮**：本轮未执行（时间预算）；如需量化观测扰动对复现点的影响，
   建议同日原环境（原库+v11）连跑两轮对照。
4. `submitSqe` 容量判断静态缺陷仍待修复决策（本轮证据显示其未构成本故障成因）。

## 8. 环境恢复证明（17:33–17:35 UTC，含 75s meta 排空等待）

| 步骤 | 结果 |
|---|---|
| 残留进程 | `pgrep vllm serve/torchrun/EngineCore/Worker_TP` 为空；设备无运行进程 |
| Host core.so 双位置 | 恢复 `88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1`（脚本内置断言比对通过），权限 644 |
| 备份清单 | `g1c_backup_20261020T164118Z/SHA256SUMS.txt` `sha256sum -c` OK，目录重新只读化 |
| 设备 v12 | `rm -f .../libmf_sdma_orch_v12.so` 后探针 507018（不存在，符合预期） |
| 设备 v11 | 探针 sync=0（在位，未触碰） |
| 诊断环境变量 | 无残留（新 shell `env` 复核） |
| 主工作树 | `git status` 干净，`vllm_ascend/*.so` 与构建前哈希一致 |
| 设备占用 | 双卡回基线（1661/1593 MB），无 507899 |

G1-C 证据、构建产物与备份仅存服务器本地：
- 证据：`vllm-ascend-g1c/csrc/_310P/memfabric_mm_ar_v2/tmp/opencode/gate_e_forensics/{C2R1,C2R2}/`
- 构建：`memfabric_hybrid-g1c/build-g1c/`、`/tmp/opencode/g1c/build_v2_{g1c,default}/`
- 过程脚本/日志：`/tmp/opencode/g1c/`（build_*.sh、deploy_g1c.sh、restore_g1c.sh、run_c*.sh、logs/）
- 备份：`memfabric_hybrid-g1c/build-g1c/g1c_backup_20261020T164118Z/`

## 9. 执行序列与合规声明

- 严格按 C0 → C1 → C2 → C3 顺序执行；C1 未通过前未启动 C2。
- 全程未修改：MM、通信、SQE、Mailbox、IB、Epoch、KFC 算法；未覆盖历史证据、
  原分支、原部署库（v11）与未经备份的运行库。
- 新设备库使用独立 KVER=12 文件名；测试后原库逐字节恢复并哈希核验。
- 提交到 GitHub G1-C 分支的内容仅含：3 处诊断兼容修复（§3）、可复现执行入口
  `g1c_forensic_run.sh`、本报告；`tmp/opencode` 过程文件未上传。

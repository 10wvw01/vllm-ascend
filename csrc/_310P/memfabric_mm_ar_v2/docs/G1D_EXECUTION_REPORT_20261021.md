# G1-D 精准诊断实机执行报告（2026-10-21，310P3）

> 方案依据：`docs/G1D_EXECUTION_PLAN_20261021.md`（下称"执行计划"）。
> 执行：OpenCode @ 310P3 服务器，2026-10-21 04:00–06:15 UTC。
> 证据分级约定：**observed**（实机直接观测）、**derived**（由观测推出）、**unknown**（无证据，禁止推测补全）。
> 原始完整证据（含 SHA256 自校验清单）仅存服务器本地
> `/home/w00804146/310P/g1d_evidence_20261021_R1/`；入库脱敏包见
> `csrc/_310P/memfabric_mm_ar_v2/evidence/g1d_20261021/`。

## 1. 执行总览与 Gate 状态表

| Gate | 内容 | 状态 | 关键结果 |
|---|---|---|---|
| D0 现场基线锁定 | 双仓库 SHA、原装库哈希、KVER/JSON、加载路径、备份 | **PASS** | 原装基线与 G1-C 恢复态逐字节一致（§2） |
| D1.0 静态与构建 | V2 G1-D + MF Host G1-D 构建、符号/ABI/宏核验 | **PASS** | v12 kernel 重构建与保全产物逐字节一致（§3） |
| D1.0 CPU UT | 双仓库离线解析器/源码合同测试 | **PASS** | V2 侧 32/32，MF 侧 3/3（含 1 处测试层修复） |
| D1.1 正确性 Gate | TP2 C1（诊断 OFF→ON） | **PASS** | OFF 51/51、ON 51/51 位精确（§4） |
| D2-A 最小压力 | 异步窗口 1/2/4 × 256 + Stock/HCCL 对照 | **PASS** | 三窗口全部位精确（w4 需 harness 修复，§5.2） |
| D2-B 边界取证 | 真实 seq mod 1024 边界/非边界对照 | **完成（健康侧）** | 边界槽/邮件实测正确，768 调用零失败（§5.3） |
| D2-C V1/V2 共性 | 同二进制 V1 对照 | **BLOCKED** | V1 兼容性无证据，按纪律不切 KVER9/10（§5.4） |
| D1.2/D3 Gate E 再现 | 原工况 620s ×2 + 1240s ×1，全取证 | **未复现** | 300/300 请求、0 错、0 超时（§6） |
| D4 报告与恢复 | 结构化证据、SHA256、恢复证明 | 本文件 + §9 | 环境已恢复并断言校验 |

**总结论（observed）**：G1-D 全取证环境下，正确性与受控压力全部通过；三次 Gate E
（含一次双倍时长长稳）均未复现 G1-C C2R2 型双 Rank 停滞。**不得据此宣布根因已修复**
（计划 D1.2 明示；埋点扰动效应未量化，见 §8 unknown）。

## 2. D0：现场基线锁定（observed）

### 2.1 分支与工作树

| 项 | 值 |
|---|---|
| vllm-ascend | `diag/310p3-g1d-minimal-forensics-20261021`，起点 `70734f715`，执行中追加诊断兼容提交（§7） |
| MemFabric | `10wvw01/memfabric_hybrid_310p` 同名分支，起点 `1f87d0f4a`，追加 1 提交（§7） |
| 工作树 | 主树直接检出 G1-D 分支；`*.so`/`build*`/`tmp` gitignore，porcelain 全程洁净；预存未跟踪资产经 `.git/info/exclude` 原地保留 |
| 隔离保证 | 未拉取 `wgm-dev-310p`；`feat/310p-memfabric-public-api` 分支未动（其原构建 .so 三件备份于 `/tmp/opencode/g1d/backup_main_tree_so_20261021T040553Z/` + SHA256SUMS） |

### 2.2 原装环境哈希（与 G1-C 恢复态一致）

| 产物 | SHA256 / 探针 | 判定 |
|---|---|---|
| Host `libmf_hybm_core.so`（/usr/local + site-packages 双位置） | `88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1` | = G1-C 恢复值 |
| AICPU v11 `libmf_sdma_orch.so`（安装侧）/ launch JSON | `db7134b8…` / `25e16b0a…` | 一致 |
| 设备侧 v11（`aicpu_kernels_device/libmf_sdma_orch_v11.so`） | mini_kfc 探针 dev0/dev1 sync=0 | 在位，全程未触碰 |
| 设备侧 v12 | 探针 507018 | 已清除（符合 G1-C 结束态） |
| 保全 v12 三件套（`/tmp/opencode/g1c/kernel_v12_stage/`） | `.so=02290083…` `v12.json=90172726…` `cust.json=74ad4668…` | = G1-C 记录 |
| 实际加载路径 | site-packages 包内 `lib/libmf_hybm_core.so`（ctypes CDLL）+ /usr/local 同步 | 双位置同哈希 |
| KVER | 安装 11；诊断线 12（显式 JSON） | - |

备份与恢复演练：G1-C 备份 `sha256sum -c` 复核 OK；G1-D 新建
`build-g1d/g1d_backup_20261021T041522Z/`（只读 + 自校验 + ownership）；恢复脚本为
G1-C 已验证模式的断言式改写。设备占用准入见 §6.1（外来占用按纪律挂起记录）。

## 3. D1.0：构建、静态核验与部署（observed）

### 3.1 MemFabric Host G1-D

- 配方与 G1-C 同构：Ninja RELEASE、`-DMF_SDMA_ORCH_KVER=12 -DMF_SDMA_ORCH_G1C_DEVICE_TRACE=ON`、BUILD_PYTHON=ON，独立 `build-g1d/`。
- 产物 `libmf_hybm_core.so` = `6e9dc2d699aa7b718a83f93397fbc2d0f18a799e384ffda9031d62f25195e55a`；内嵌 `commit: 1f87d0f4…`；`[MF_G1D]`/`[MF_G1C]` 字符串在位；ldd 依赖集合一致；导出 T 符号 1052 = 原装。
- **v12 可复现性（derived）**：G1-D 分支重构建的 KVER=12 `libmf_sdma_orch.so` 与保全产物 SHA256 逐字节一致（`02290083…`）——设备侧源码自 G1-C 起零改动。

### 3.2 V2 G1-D（vllm-ascend）

- `VLLM_ASCEND_310P_V2_BUILD_DIAGNOSTICS=ON + G1C_TRACE=ON + G1D_TRACE=ON`（CMakeCache 预置）。
- bisheng 与 Host CXX 命令均含 `-DMF310P_V2_G1C_TRACE=1 -DMF310P_V2_G1D_TRACE=1`（build.make/flags.make 复核）。
- 设备库独立名 `libmf310p_v2_trial_device_g1d.so`（`d9a6de66…`）；`vllm_ascend_C…so`（`bcbb4f67…`）DT_NEEDED → g1d 库；正式默认名库未被覆盖；导出符号 21/21 与默认构建一致。

### 3.3 部署（断言式）

原装双位置备份 → G1-D Host 替换（逐位置哈希断言）→ v12 经**独立进程** `orch_deploy_helper`
（G1-D 树构建，`849f93c0…` 与现网版一致）部署设备侧：dev0 rc=0 "deployed and verified
size=14072"，dev1 rc=3 已存在（双 die 共享设备文件系统）。校验级别 = 符号 + 精确字节数
（与 G1-C 同级；设备 OS 无哈希工具）。

### 3.4 CPU UT

V2 侧 `test_g1d_evidence_summary.py`/`test_g1c_trace_summary.py`/`test_gate_e_forensic_summary.py`/`test/ut/` 全量 **32/32**；MF 侧 `test_g1c_device_trace_source.py` **3/3**。

## 4. D1.1：TP2 C1 正确性 Gate（observed）

| 运行 | 配置 | 结果 |
|---|---|---|
| C1-OFF | G1-D 全套二进制，诊断环境变量全关 | **51/51 位精确 PASS**；无 507014 |
| C1-ON | 同二进制，全开关开启（g1d_run.sh c1 + pyinject 提级） | **51/51 位精确 PASS** |

C1-ON 取证链：`[G1B]` 7 进程；`[MF_DIAG]` 160 行（15s 纪干净轮转，exitReason/mainRet=0）；
`[MF_G1C]` 142；`[MF_G1D]` 142 帧 **0 snapshotError**；`[V2_G1C]` 138 帧；`[V2_G1D]` 112 条
同代邮件返回（如 `generation=38 captured=1 mailSeq=154 waitChunk=3 mailStatus=0`）。

**derived**：诊断开关不影响正确性行为；同代捕获机制工作正常。

## 5. D2：受控异步窗口与边界取证（observed）

### 5.1 D2-A 窗口压力（M=4096、chunk=1024、256 调用、含 Stock/HCCL 对照）

| 窗口 | 结果 | 耗时 |
|---|---|---|
| 1 | **PASS 256/256** | 139.3s |
| 2 | **PASS 256/256** | 114.1s |
| 4（修复 harness 后） | **PASS 256/256** | 103.1s |

### 5.2 窗口=4 失败定位（observed → derived）

- 现象：原版 harness 两 Rank 第 5 调抛 `V2 pooled outputs exhausted … Refusing unsafe
  alias/reuse`；前 4 调已验证；Epoch 健康；无 timeout；无 snapshotError。
- 探针 A（单窗口 ×4）：PASS → 失败锁定在**跨窗口边界**。
- 探针 B（解绑验证循环变量 + 二次 gc）：×8 PASS；修复后全量 256 PASS。
- **derived**：测试脚本侧残留 Python 引用（`result` 循环变量 + 单次 gc 不足以使池视图
  StorageImpl 引用计数回落）触发**保护性**槽耗尽；V2 池"拒绝不安全别名/复用"守卫行为正确。
  非通信/池缺陷。修复已提交（§7 #2）。

### 5.3 D2-B 边界取证（observed）

三窗口流量结束静默帧（dev0）：

| 字段 | 实测值 |
|---|---|
| arrHead | 1302 / 1307 / 1307 |
| stamp1020..1023 | **1021, 1022, 1023, 1024** |
| stamp0..2 | **1025, 1026, 1027** |
| mail1023Seq / mail0Seq | **1024 / 1025** |
| quiet60..63, quiet0..1 | 1277..1282 连续 |
| headSlot/headStamp | 278→279、283→284（与 arrHead 语义自洽） |

`[V2_G1D]` 返回 seq 覆盖 1..1300；`[MF_G1C]` reqSeq 至 1302；hwHeadLast 推进跨 1024。
768 次调用零失败，无失败侧对照样本（须由 Gate E 提供）。

**derived**：受控窗口工况下，"仅 1024 边界丢邮件/丢写"假设未获支持；不得外推到
prefill/decode 交叠工况。

### 5.4 D2-C：BLOCKED

V1 与 v11/v12 的 ABI/部署/路由兼容性无证据；按计划不切 KVER9/10 凑结果。本轮无 V1 侧结论。

## 6. D3：Gate E 再现（observed）

### 6.1 设备占用阻断（准入纪律执行）

04:41–05:00Z 设备 0 被外来命名空间作业间歇占用（AICore 0→87%、内存 2→9.7GB 周期，
npu-smi PID 宿主不可见）。按 D0 停止条件挂起并记录（`blockers/`），05:00:55 确认
双设备回基线（1352/2025MB）后才启动 D1R1。

### 6.2 三次会话结果（fused Eager 原工况：input 4096 / output 256 / 25 prompts / 并发 10）

| 会话 | 窗口 | 请求 | V2 末代际 | arrHead/reqSeq 终态 | 507014 | snapshotError |
|---|---|---|---|---|---|---|
| D1R1（05:01–05:20Z） | 620s | **75/75** | 3008 | 15027 | 0 | 0 |
| D1R2（05:22–05:38Z） | 620s | **75/75** | 3008 | 15027 | 0 | 0 |
| D1R3（05:41–06:09Z） | 1240s（长稳） | **150/150** | 6008 | 30027 | 0 | 0 |

- 三会话合计 300 请求 0 失败 0 超时；`[V2_G1C]` 全部会话末帧 8 核 stage=120（干净退出，
  g1c_trace_summary.json）；`[V2_G1D]` 同代邮件返回 1694/1390/2816 条/会话；
  `[MF_G1D]` 每会话 0 snapshotError。
- 边界静默帧（observed）：D1R1/D1R2 `stamp1020..1023=14333..14336、mail1023Seq=14336、
  stamp0..2=14337..14339`；D1R3 `…29693..29696、mail1023Seq=29696、mail0Seq=29697`
  ——第 14/29 圈边界连续正确。
- **未复现 G1-C C2R2 型停滞**（C2R2 于 rep1 第 3 请求即双 Rank gen208 卡 stage50/60；
  本轮三会话无任何中途停滞帧，V2 代际全程推进）。
- bench 窗口（D3 关联性能，BENCH_WINDOW_ONLY）：D1R3 六窗 output_tok/s
  22.9→30.3→28.5→30.0→30.0→29.9，TTFT p50 ~12s（并发排队），TPOT p50 ~250ms；
  与 G1-C C2R1 健康基线同量级。1Hz 采样下不作微秒级计时结论；正式性能须无取证版本另测。
- 判定（按计划）：**保留健康证据，不宣布修复**。三会话 ≈ 45k 件 AICPU 消费、44 次
  1024 边界穿越、~12k V2 代际零异常——构成对 H3"低概率单件丢失"假设的**压力反驳证据**
  （derived；非证明，见 §8）。

## 7. 诊断兼容修复与错误记录（均已提交各自 G1-D 分支）

| # | 仓库 | 提交 | 内容 | 定性 |
|---|---|---|---|---|
| 1 | vllm-ascend | `bc50a079d` | V2 目录合同测试未含 G1-C 证据打包提交入库的 `evidence/`，分支上从未可能通过 | 测试层 |
| 2 | vllm-ascend | `5118ac82d` | 异步窗口 harness 跨窗口残留引用释放（§5.2） | 测试层 |
| 3 | memfabric | `b4b45878` | tools/.gitignore 先于 orch_deploy_helper 存在，构建后污染洁净门禁 | 工具卫生 |

执行期环境问题（未构成代码修复，已按约定规避）：`g1d_run.sh` 不自带 CANN 环境；
调用方 source 后必须以**前置追加**方式设 PYTHONPATH（覆盖式 export 丢失 set_env.sh 的
CANN Python 路径 → `No module named 'tbe'` → ACL 500001 双 Rank 启动失败）。

## 8. 根因假设矩阵与 unknown

| ID | 假设 | 本轮证据 | 状态 |
|---|---|---|---|
| H3 缓存可见性/单件丢失（G1-C 主嫌疑） | 三次 Gate E + 全部 D2 零复现零异常；45081 件消费、44 次边界穿越正确 | **压力增强但未证实排除**（间歇性故障 1/2→0/3 样本量仍小） |
| "仅 1023 边界丢写" | 受控窗口 + Gate E 静默帧边界槽/邮件/回绕首槽全部连续正确 | 健康侧持续反驳；故障侧无样本 |
| H1 SQ 容量失控 | 承接 G1-C 反驳；本轮无异常 | 维持反驳 |
| H2 Epoch 停顿 | 全部 exitReason=0 | 维持反驳 |
| 槽耗尽（新增） | w4 原失败 = harness 残留引用；守卫正确 | 关闭（非通信类） |

**unknown（本轮不能回答）**：
1. 故障窗口的第一异常交接：未获得故障侧 G1-D 证据（未复现）。
2. G1-D 埋点扰动是否掩盖故障（1Hz D2H + 设备槽写 vs G1-C 同级埋点下 C2R2 可复现；
   两者埋点面接近，但无成对对照，无法量化差异）。
3. 诊断 OFF 会话（C1-OFF 51/51）仅覆盖正确性，未做 OFF 长稳对照。
4. 外来作业与 G1-D 会话无共享设备时段（挂起纪律），但主机侧共享资源未隔离测量（derived 边界）。

## 9. 环境恢复证明（observed，06:11–06:13Z）

| 步骤 | 结果 |
|---|---|
| 残留进程 | `pgrep vllm serve/torchrun/EngineCore/Worker_TP` 为空 |
| Host core.so 双位置 | 恢复 `88d392850733d7f166ce28615a8fdbf164b604979df9d3fc8cd0c10c64fd7bc1`（断言通过），权限 644 |
| 设备 v12 | `rm -f` 后探针 507018（不存在，符合预期） |
| 设备 v11 | dev0/dev1 探针 sync=0（在位，未触碰） |
| 备份重封闭 | `g1d_backup_20261021T041522Z/SHA256SUMS.txt` 校验 OK |
| 诊断环境变量 | 新 shell 无残留 |
| 设备占用 | dev0 1353MB / dev1 1594MB（≈G1-C 基线 1661/1593） |
| 主工作树 | G1-D 分支 porcelain 洁净；分支切换回 `feat/310p-memfabric-public-api` 后如需原 .so 可自 `/tmp/opencode/g1d/backup_main_tree_so_20261021T040553Z/` 恢复（或重新构建） |

## 10. 交付物与下一步建议

交付物：本报告；脱敏证据包 `evidence/g1d_20261021/`（索引、会话统计、边界 JSON、
脱敏节选、原始清单 SHA256）；本地完整原始证据 `g1d_evidence_20261021_R1/`（不入库）；
双分支诊断兼容提交（§7）。

下一步（不越权执行）：
1. 交独立复核裁决：G1-D 全取证三会话未复现的解读（埋点扰动 vs 环境差异 vs 样本量）
   需要成对设计——建议后续同日执行"无埋点对照轮 + G1-D 复现轮"交替（每轮 ≥2 会话）。
2. 若后续复现：`[V2_G1D]` 同代邮件 + stage52/53 + `[MF_G1D]` 动态头快照即按计划 §5
   判定表定位第一异常交接。
3. D2-C V1 对照需先完成 V1 ABI/部署兼容性核验（单独评审）。
4. 性能终验与"根因已解决"结论必须由关闭埋点的全矩阵长稳给出（计划 D4 纪律）。

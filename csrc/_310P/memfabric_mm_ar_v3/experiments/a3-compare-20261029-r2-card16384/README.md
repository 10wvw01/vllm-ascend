# A3/A3-2/A3-tail 同批对照（卡16384）：六会话PASS后被外部租户干扰STOP，c512组完整（2026-10-29 R2）

执行：OpenCode（按 `A3_SEAT_SCHEDULE.md` 同批合同；用户于 11:30 授权换卡 16384/dev2,3——
38144 dev4 被外部租户持续占用）。批次工具 `compare_a3.py`（ORDER：compute512 四会话 →
compute1024 四会话，每会话六门禁 + 六配置两轮短测），守护器 2s fail-closed 轮询。

## 总结论

| 阶段 | 结果 |
|---|---|
| 私有构建（本批次实际使用） | HEAD `4d5ce62e8`（docs提交后按工具合同在v3-build-r2重建；Device SO与81780f342构建逐字节一致`fd6807c9…`，证实源码未变）；tensor适配冻结`6be03080…` |
| G0（16384双Die健康） | PASS（双Die Health OK；11:42–12:06 连续24分钟空闲窗口） |
| 首次v28部署（dev2/3） | PASS：`DEPLOY_WINDOW_TRANSIENT`（4 PID，12:54:43）→`HEALED`（3.6s）——orch_deploy_helper fork+exec 按设计工作 |
| **六会话（同批同manifest同卡）** | **全部 PASS_QUICK_SCREEN_ONLY**：a3-before-c512、a3-2-c512、a3-tail-c512、a3-after-c512、a3-before-c1024、a3-2-c1024；每会话六门禁双Rank全PASS、24个计时文件完整、全部样本correct=true |
| 第7会话 a3-tail-c1024 | **STOP(97) 守护器**：池创建阶段被杀（12:06:42，`external cross-card PID on target: [3301645]`——外部进程同时占用5664两Die与16384 dev2；非代码问题） |
| 第8会话 a3-after-c1024 | 未启动（批次按合同STOP） |
| 重试批次 batch-r2 | **PREFLIGHT STOP**：外部 vLLM EngineCore（PID 3316995）恰在启动瞬间占用dev2 |
| 最终卡态 | 16384 dev2：vLLM EngineCore 21.8GB 稳定serving（12:15起20+分钟）+ 新TP2对；38144原租户未走；43520 dev7 Alarm——全服务器无可用独占窗口，停止盲试 |

## 同批 compute_rows=512 组结果（完整夹逼，六配置全部稳定）

口径：逐样本双Rank max → 每轮30样本分位 → 两轮p50中位为臂中心；A3基线=前后中位；
漂移>5%判不稳定。完整p90/p99与Host Wall见 `batch-r1-partial-analysis.json` 与下文。

| M | comm | A3前p50 µs | A3-2 p50 µs | A3-tail p50 µs | A3后p50 µs | 前后漂移 | A3-2两轮 | tail两轮 | A3-2 vs A3 | A3-tail vs A3 | 稳定 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2048 | 512 | 1542.6 | 1622.6 | 1787.2 | 1540.7 | 0.12% | 0.33% | 0.77% | **-5.25%** | **-15.93%** | YES |
| 2048 | 1024 | 1531.0 | 1609.0 | 1914.6 | 1567.2 | 2.36% | 1.65% | 0.54% | **-3.86%** | **-23.59%** | YES |
| 2048 | 2048 | 1667.3 | 1809.1 | 1656.9 | 1703.9 | 2.19% | 2.12% | 3.06% | **-7.33%** | +1.70% | YES |
| 4096 | 512 | 2525.1 | 2604.6 | 2875.5 | 2540.8 | 0.62% | 0.13% | 0.40% | **-2.83%** | **-13.52%** | YES |
| 4096 | 1024 | 2405.2 | 2493.2 | 2985.4 | 2460.5 | 2.30% | 0.47% | 0.75% | **-2.48%** | **-22.71%** | YES |
| 4096 | 2048 | 2546.3 | 2702.2 | 3201.5 | 2564.3 | 0.71% | 0.19% | 0.61% | **-5.75%** | **-25.29%** | YES |

（负值=候选更慢。p50均为两轮中位。）

**同批结论（仅c512组，六配置全稳定，漂移0.12%–3.06%）：**

1. **原A3（3302，8核计算/Core0兼通信）在6/6配置上快于A3-2（3303专职通信）**，
   差距2.5%–7.3%：专职通信Core0损失一核MM吞吐，未被通信专职收益抵偿。
2. **原A3在5/6配置上快于A3-tail（3304末尾座位）**，差距13.5%–25.3%；
   唯一例外M2048/comm2048：A3-tail快+1.70%，但该case前后夹逼漂移2.19%，
   与收益同量级，**不判稳定收益**。该粒度整矩阵单批次无前段交叠（席位单预告），
   与实测一致。
3. c512组夹逼一致性良好（前后漂移≤2.36%），说明24分钟窗口内环境稳定，
   数据具备同批可信度；但**整批八会话合同未完成（缺a3-tail-c1024、a3-after-c1024），
   本轮不签正式候选筛选GO，c1024结论待完整批次**。

### compute_rows=1024（不完整，2/4会话，无夹逼——仅方向性参考，不签）

| M | comm | A3前p50 µs | A3-2 p50 µs | A3-2 vs A3前 |
|---|---|---|---|---|
| 2048 | 512 | 1614.4 | 1575.9 | +2.38% |
| 2048 | 1024 | 1338.3 | 1484.8 | -10.94% |
| 2048 | 2048 | 1534.4 | 1637.0 | -6.69% |
| 4096 | 512 | 2747.3 | 2532.3 | +7.83% |
| 4096 | 1024 | 2222.4 | 2239.8 | -0.79% |
| 4096 | 2048 | 2279.9 | 2361.0 | -3.56% |

无A3后夹逼、无A3-tail，方向不一致（+7.8%~-10.9%），不构成任何结论。
MM粒度方向性观察（同批跨组，仅参考）：g1024/2048下c1024(tile128)比c512(tile64)
快约8–13%（如M2048g1024：1531→1338µs），g512相反（c512略快）。

## 干扰时间线与守护器行为（全部正常工作）

| 时间 | 事件 |
|---|---|
| 11:35–11:42 | 间歇外部TP2对（宿主PID 3154693/4、3161740/1，561–633MB，python3）——启动批次前观察到，等待至空闲 |
| 11:42–12:06 | **24分钟干净窗口**：六会话全部完成（每会话约2.2分钟） |
| 12:06:42 | 外部跨卡进程3301645（同占5664两Die+16384 dev2，101MB级探测形态）→ 守护器`STOP external cross-card PID on target`，杀自有PGID，a3-tail-c1024死于池创建（deploy_window flags在场，无任何门禁/ADD执行） |
| 12:08:14 | batch-r2 PREFLIGHT STOP：vLLM EngineCore 3316995（dev2，21.8GB）恰在预检时在卡 |
| 12:15起 | vLLM EngineCore 3349738（21.8GB）稳定serving 20+分钟；12:35另有新TP2对出现 |
| 全程 | 未杀任何外部租户；两次STOP现场均完整保留；未在失败会话重试ADD |

## 指纹

| 项 | 值 |
|---|---|
| vllm-ascend HEAD | `4d5ce62e867525f9dbef442c98a5f04d4945ada9`（批次实际HEAD；起点81780f342+docs提交4d5ce62e8，源码零改动——Device SO逐字节一致佐证） |
| MemFabric | `db38d75e`（tree 4fe624ca，未修改） |
| V3 Host/Device SO | r2重建：host `9f7f4cc2…`（含新构建路径）、device `fd6807c94a6d36149510d552b9db940129a15ce12338f612796747dd97e67484`（与起点构建逐字节一致） |
| Tensor适配 | `6be0308072673a3f92cdef8f61726ed9e179ca3c66d78ddbbd8cf5acb8ddf2ff` |
| manifest SHA256 | `99653b579f5e5c6c5cf5a726bf5238b62360c204726b20573f40ab4c4bf0a55e`（g0.json记录） |
| 进程内实际装载 | 双Rank `/proc/self/maps`：SDK `libmf_hybm_core.so`(`1cab72c1…`)、`libmf_smem.so`(`0a0cbd00…`)、私有`libmf310p_v3_host/device.so`——与manifest一致 |
| 设备/环境 | 16384双Die dev2/3物理直选（rank0→physical_device=2）；aarch64 / CANN 9.1.0 / torch 2.10.0+cpu / torch_npu 2.10.0.post4 / Ascend310P3 |
| variant/profile身份 | 每会话rank JSON记录variant/compute_rows/mm_tile_rows，compare_a3批内校验通过；variant→create符号→profile 3302/3303/3304链经首READY参数校验（协议门禁PASS佐证） |

## 复现命令

```bash
V3_PYTHON=/tmp/opencode/310p-a3cmp-20261029-001/venv/bin/python
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd /home/w00804146/310P/vllm-ascend-a3v3   # @4d5ce62e8
# 构建（如产物缺失；SDK在 $RUN_ROOT/memfabric-sdk 已冻结）
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/build.py \
  --memfabric-source /tmp/opencode/310p-a3cmp-20261029-001/memfabric-src \
  --memfabric-home /tmp/opencode/310p-a3cmp-20261029-001/memfabric-sdk \
  --cann /usr/local/Ascend/cann-9.1.0 --out <新build目录>
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/build_tensor.py \
  --manifest <新build目录>/manifest.json --out <新tensor目录>
# 完整八会话批次（需要双Die约20分钟真实独占窗口）
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/compare_a3.py start \
  --manifest /tmp/opencode/310p-a3cmp-20261029-001/v3-build-r2/manifest.json \
  --out <全新批次目录> --card 16384 --devices 2,3 \
  --lease-note '<现场真实独占窗口记录>'
$V3_PYTHON csrc/_310P/memfabric_mm_ar_v3/tools/compare_a3.py status --out <全新批次目录>
```

## 证据索引（均在本目录）

| 内容 | 位置 |
|---|---|
| 批次原始数据（六会话×gate×2Rank×24计时JSON + 失败会话现场 + batch_result/launch/console） | `batch-r1/`（`batch-r1/a3-tail-c1024/`为守护器STOP现场） |
| 重试批次STOP现场 | `batch-r2/a3-before-c512/`（PREFLIGHT STOP） |
| 配对统计与可复算脚本 | `batch-r1-partial-analysis.json`、`analyze_batch_r1_partial.py` |
| G0/占用快照（批前、批后全卡） | `npu-16384-before.txt`、`health-16384-before.txt`、`npu-allcards-stop-final.txt`、`proc-mem-16384-stop-final.txt`、`health-16384-stop-final.txt` |
| 批次构建身份 | `v3_build_manifest_r2.json` |
| 首轮38144 STOP记录 | `../a3-compare-20261029-r1-device-stop/`（R1轮，含CPU/SDK/编译证据） |

## 遗留（交研发侧）

1. **完整八会话批次未完成**：c512组已给出同批稳定结论（原A3占优）；c1024组需
   完整批次补齐a3-tail与A3后夹逼。恢复条件：双Die约20分钟真实独占窗口（当前16384 dev2
   被外部vLLM serving占用、38144 dev4原租户未走、43520 dev7 Alarm）。manifest已冻结，
   窗口可用后无需重建。
2. c512组结论对c1024的外推依赖MM粒度交互（专职通信损失一核MM吞吐在tile128下占比更大），
   需完整数据裁决；不据c512组宣布A3-2/A3-tail全场景否决。
3. KVER28占位自拷贝差异仍在，不签长稳；本轮为短测（30样本×2轮），不签正式性能GO。

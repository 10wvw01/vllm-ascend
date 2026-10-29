# 310P3 统一计时 11 会话批次（共享卡、用户授权豁免漂移门禁）：PASS_ATTRIBUTION_ONLY（2026-10-29 R1）

执行：OpenCode。用户于本轮明确授权：**不等独占窗口、共享卡直接跑、忽略漂移门禁**；据此新增
`--allow-shared-card`（c874babe4，显式传参并全程留痕，健康/告警/解析熔断保留）。默认行为不变（EXCLUSIVE，fail-closed）。

## 本轮身份

| 项 | 值 |
|---|---|
| vllm-ascend HEAD | `ecda8642f`（起点 `1aebe9c1e` + 5 个修复提交） |
| MemFabric | `db38d75e`（未修改） |
| 最终 manifest | `343a5bb0…`（v3-build-r7；V2 spec 冻结后） |
| V2 冻结 | D4 源 `5adbf1810`、插件 `8eb65de9`、Host `1dc46d7d`、D4 Device `2a19390c`、KVER11 `25e16b0a`（E3 G0 审核产物，全 SHA 对账通过） |
| 设备/环境 | 卡 16384 dev2/3 物理直选，**共享**（外部 TP2 对持续在场，762MB×2 轮换）；aarch64 / CANN 9.1.0 / torch 2.10.0+cpu / torch_npu 2.10.0.post4 |
| 批次 | `unified_campaign.py` 11 会话全部 PASS_UNIFIED_ONLY → campaign `PASS_ATTRIBUTION_ONLY` |
| 三 A3 六门禁 | r7 最终 manifest 上 3×PASS_QUICK_SCREEN_ONLY（IB/裸ADD/生命周期/协议/融合/尾块） |

## 本轮修复（HEAD 1aebe9c1e 集成点缺陷，全部推送至工作分支）

1. `029e1222e` test/conftest.py：importlib 模式下兄弟导入收集失败 → 所有会话死于自家 CPU 门禁。
2. `d96035006` 移除 worker 不存在的 `--run-id` → 每个统一会话 exit 2。
3. `8fcc60204` Storage SO 按 `mf310p_v32_tensor` 扩展名加载（原 `unified_storage` → PyInit 缺失）。
4. `0d9230020`+`ecda8642f` **设备 ticks 归因降级**：310P3 无核内时钟（见下）。

## 硬件事实：310P3 (dav-2002) 无 AICore 可读时钟（非实现 BUG）

- `GetSystemCycle()` 在 dav_m200 实现为 `ASCENDC_REPORT_NOT_SUPPORT`，恒返 0（CANN 9.1 源头实证）。
- `SYS_CNT` 汇编被 dav-2002 ISA 拒绝（探针编译：`instruction requires: DAV_L210…DAV_L510`，2002 不在列）。
- 设备端 trace 机制本身工作（头区/epoch/回取/flush 全对），仅时间戳全 0 → `UNAVAILABLE_DAV2002_NO_AICORE_CLOCK`。
- 历史佐证：G1C 分支在 310P3 上即用相位序号+Host 计时，未用核内时钟。
- 结论：MM/seed/就绪等待/协议/排空的**核内 ticks 归因在本硬件上不可得**；需 AICPU 时戳（MemFabric 变更，本轮禁改）或硬件 Profiler，交研发侧。证据：`evidence-index/syscnt_*`、`allzero_trace_stop_rank0.json`。

## 主要结果（wall p50 µs，双 Rank max 配对，off 前后中位；漂移为豁免门禁后仍如实记录）

基线（comm1024）：Stock M2048=1187.2 / M4096=2109.5；V2 M2048=1472.4 / M4096=2614.6。

| 臂 | 最佳配置 | M2048 | vs Stock | M4096 | vs Stock | vs V2(M4096) |
|---|---|---|---|---|---|---|
| V3 | c1024 | 1570.7 | −32.3% | 2538.7 | −20.3% | +2.9% |
| **A3** | **c1024** | **1463.2** | −23.3% | **2278.0** | −8.0% | **+12.9%** |
| A3-2 | c1024 | 1620.7 | −36.5% | 2361.1 | −11.9% | +10.0% |
| A3-tail | c2048/c1024 | 1600.1 | −34.8% | 2607.6 | −22.8% | +0.3% |

（负值=比 Stock 慢。**全部候选仍慢于 Stock NZ MM+HCCL**；A3 是最接近的，M4096/c1024 仅慢 8%。）

### 三 A3 同批对比（同 comm 匹配）

- 原 A3(3302) **5/6 优于 A3-2(3303)**（c512 组），**6/6 优于**（c1024 组）；与 R2 结论一致。
- A3-tail(3304) 几乎处处最慢；例外 M2048/c2048 量级与漂移同阶，不判收益。
- comm 敏感度：多数配置 c1024 最优；c2048 对 A3-2 明显不利。

### Host 归因（A3 每调用 ~158–184µs，含在计时窗口内）

validate_reap≈35%、enqueue≈34%、wrap≈18–20%、input_event≈6%、reserve≈5%；Wall−Event 稳定在 70–90µs（Host 下发空档，不作 Host 耗时解释）。

### 扰动与漂移

- 打点阶段 vs 无打点：全部 |Δ|≤3.8%（多数 ≤2%）→ trace 写入本身便宜，**无打点基线未被污染**。
- 前后夹逼漂移：Stock 前 2.6/2.9%，其余 ≤3.8% → 虽共享卡，本窗口外部负载轻且稳定；但按合同本数据**不签稳定收益/发布 GO**，共享卡+豁免门禁仅作方向性对照。
- 对照 R2 干净窗口：A3-c512/M2048/g1024 1531→1639µs（≈+7%），与共租噪声量级一致。

## 限制与遗留（交研发侧）

1. 共享卡 + 漂移豁免为用户单方授权，本报告不升级为正式对照或发布依据。
2. 设备 ticks 归因不可得（硬件），A3 内部 MM/seed/等待/排空占比本轮**无数据**；替代方案需研发侧决策。
3. KVER28 占位自拷贝差异仍在；短测 30×2 样本，不签长稳。
4. 失败现场（attempt1/attempt2、dbg1、campaign-shared-r1/r2/r3、shared-gates 及 abort 场景）全部保留在 run root，索引见 `evidence-index/server_artifact_index.txt`。

## 复现命令

```bash
V3_PY=<venv>/bin/python; source /usr/local/Ascend/cann-9.1.0/set_env.sh
$V3_PY csrc/_310P/memfabric_mm_ar_v3/tools/quick.py start --a3 --a3-variant <a3|a3-2|a3-tail> \
  --manifest <v3-build-r7>/manifest.json --out <新目录> --card 16384 --devices 2,3 \
  --store-url tcp://127.0.0.1:<独立端口> --allow-shared-card
$V3_PY csrc/_310P/memfabric_mm_ar_v3/tools/unified_campaign.py start \
  --manifest <v3-build-r7>/manifest.json --out <新目录> --card 16384 --devices 2,3 --allow-shared-card
$V3_PY csrc/_310P/memfabric_mm_ar_v3/tools/unified_campaign.py status --out <该目录>
```

## 追加：A3 管线级硬件归因（用户授权 Profiler 采集，2026-10-29 晚）

用户单独授权 msprof/硬件 Profiler 采集。用 `torch_npu.profiler` + `_ExperimentalConfig(Level1, AiCMetrics.PipeUtilization)`
对三 A3 变体（cr1024）各 3×comm×2M×10 样本独立采集（守护器共享模式，独立会话）。数据与脚本：`pipe-profiler/`。

**管线占比（aicore_time 归一，管线并发重叠，不可相加）——c1024/M2048 中位：**

| 变体 | kernel 时长 | mac(MM) | scalar(等待/协议) | mte2(拷贝/轮询缓存) | mte1(装数) | vec | cube_util |
|---|---|---|---|---|---|---|---|
| a3 | ~1200µs | **229µs (21%)** | **728µs (69–73%)** | 803µs (73%) | 146µs | 51µs | 91% |
| a3-2 | ~1319µs | 229µs (19%) | 913µs (75%) | 849µs (70%) | 146µs | 51µs | 92% |
| a3-tail | ~1501µs | 229µs (17%) | 1081µs (78%) | 1015µs (73%) | 146µs | 51µs | 92% |

**结论：**
1. **MM 不是瓶颈也不是变体差异来源**：三变体 mac_time 完全一致（229µs/M2048，460µs/M4096，×2 精确缩放）；cube_utilization 91–92%，候选 tiling 本身高效。
2. **scalar 管线（就绪等待+协议轮询+提交循环）是主导与差异来源**：a3 728 < a3-2 913 < a3-tail 1081µs；增量 +185/+353µs 与批次 wall 差（+157/+353µs）吻合——**变体排名几乎 100% 由等待/协议开销解释**，否定了"A3-2 损失 MM 吞吐"假说（mac 不变）。
3. **MTE2 稳定 ~70–73%**：seed/输出拷贝与轮询缓存操作混合，无法进一步分离（相位缺失）。
4. M2048→M4096：mac/mte1 ×2.0，scalar ×1.5，mte2 ×1.7——更大 M 摊薄等待，与 M4096 更接近 Stock 一致。
5. **SDMA 跨 Die 链路占用仍未测得**（本 Profiler 级无 AICPU/orch 任务明细）；核内相位墙钟仍不可得。
6. 优化指向：等待/协议路径的优化空间（~700–1100µs scalar）是 MM 路径（≤229µs）的 3–4 倍。

方法学提醒：kernel Duration（Profiler 开启）与批次 Event−空档吻合（~1200 vs 1388−~180µs），扰动可接受；占比为管线重叠口径，不代表互斥时间分段。

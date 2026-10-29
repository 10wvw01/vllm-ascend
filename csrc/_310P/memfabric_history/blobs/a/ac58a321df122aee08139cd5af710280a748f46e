# Gate E S3-R2 受控冒烟报告：部署助手显式路径方案——507018 阻断解除，融合路径全绿（2026-10-26，310P3）

> **结论先行：S3-R2 PASS。** S3-R 的 507018 级联根因（KFC 部署助手发现失败 → 进程内回退 →
> 已知上下文击杀）经本轮**单变量实验证实**；仅新增 `MF_SDMA_ORCH_DEPLOY_HELPER` 显式路径
> env 后，全新卡 16384 上助手模式完成探测-部署-校验，双 Rank 首调 bit-exact，25/25 请求
> 零错误完成，**507014（Gate E 目标故障）与 507899 均零出现**。修复版 Host SO 的 507899
> 回归 PASS 结论保持。

## 1. 执行背景与授权

- 上游链：S0–S3（507899 阻断）→ S3-R（507899 回归 PASS，新阻断 KFC 发射 507018）→
  只读审计（commit `ed63bf32e`：根因判定 = LD_PRELOAD 私有目录装载使 `FindDeployHelper()`
  的 dladdr 同目录查找落空 → 进程内回退 → 首探非零退出打死 torch_npu Worker 设备上下文；
  卡 16384 硬件与 KFC 通道经 S3-R 行数算术证明正常）。
- 用户于 2026-10-26 授权 S3-R2 受控冒烟（"授权执行 S3-R2"），范围：设备 2,3、90 秒、
  修复版 SO（LD_PRELOAD）+ 显式助手路径 + 监护 v2；S4/S5 不在本轮范围。
- 发射窗口 2026-10-26T16:50:06Z–16:58:26Z，会话 `S3R2_FUSED_SMOKE_20261026T165006Z`，
  服务器本地原始证据 `g1_local_archive/gate_e_s3r2_raw_20261026/`。

## 2. 与 S3-R 的配置差异（仅两项 env，其余逐字一致）

```bash
export MF_SDMA_ORCH_DEPLOY_HELPER=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64/orch_deploy_helper
export MF_LOG_LEVEL=1   # INFO 可见性尽力项（实测未生效：级别设定在 adapter 初始化晚于编排器初始化）
```

机制：`FindDeployHelper()` **先查 env**（X_OK 即采纳），dladdr 查找被短路——私有目录装载
不再影响助手发现；Worker 进程零 KFC 命令，探测+部署全部在 fork+exec 助手子进程（全新 ACL
上下文，非零退出无害）。其余（LD_PRELOAD 修复版 1dc46d7d、隔离 worktree、V2 SO/KVER11/
Chunk1024/Slots4、设备 2,3、模型、90 秒、bench 25 请求）与 S3-R 完全一致。

发射前准入清单 7 项全 PASS（设备健康/无残留、生产位 88d39285×2 与 helper 849f93c0 不变、
修复版在位、worktree porcelain=0、CPU 预检仅私有副本映射、监护 v2 模式自测 7/7+8/8——
详见 `evidence/gate_e_s3r2_20261026/s3r2_06_precheck.txt`）。

## 3. 运行结果（observed）

### 3.1 部署链：助手模式成功（核心验证点）

serve.log 恰 4 条 `[helper]` 行、**零** `falling back to in-process`、**零** Worker 侧
`kfc launch failed`：

```
[helper] kernel already present: /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/libmf_sdma_orch_v11.so
[helper] kfc sync=507018 cmd='test -f /usr/lib64/.../libmf_sdma_orch_v11.so'
[helper] kernel missing, deploying: /usr/lib64/.../libmf_sdma_orch_v11.so
[helper] deployed and verified: /usr/lib64/.../libmf_sdma_orch_v11.so size=13960
```

- 两次助手调用（每 Rank 一次，flock 串行）：一次 miss→deploy→verified（**rc=0**），一次
  probe 命中 already present（**rc=3**，共享设备文件系统传播，与 G1-D dev0 部署/dev1 命中
  同型）；`size=13960` 与宿主 `libmf_sdma_orch.so` 精确尺寸一致（verify 的 size-match 真实有效）。
- `[helper] kfc sync=507018 cmd='test -f …'` 为**助手子进程内的预期探测缺失同步码**——
  与 S3-R 中 Worker 进程内的 507018 发射失败性质完全不同（前者全新进程可容忍，后者打死
  torch_npu 上下文）。监护 v2 的语境判别设计正确：预期行零误触，全程零触发。
- **行序注记**：无时间戳的 `[helper]` 行在 serve.log 中 "already present" 块先于部署块出现；
  按时序逻辑（S3-R 已证 15:04 时设备侧文件缺失 + 助手 flock 串行化），命中必然后于部署，
  跨进程 stdio 落盘顺序不可作为时序依据，判定为并发写伪影，不影响任何结论。
- 门侧 INFO（`deploy helper rc: …`）未出现（见 §2 注），模式判定以 helper printf 行 +
  失败签名零出现 + 下游全通为准。

### 3.2 融合路径与负载验收

- 双 Rank `[mf310p-v2] M=4096 within [2048,4096]: V2 MM+MemFabric (chunk=1024)` 路由 ✓
- 双 Rank `first fused call verified bit-exact vs stock (M=4096, aligned=True)` ✓
- bench：**Successful requests 25 / Failed 0**，Total token throughput 390.56 tok/s，
  Mean TPOT 247.20 ms（90 秒冒烟以功能性为准，非性能基线）
- **507014 = 0、507899 = 0、Worker 死亡 = 0**；监护 v2 全程零触发，launcher/forensic 正常退出
  （`S3R2_FORENSIC_RC=0`）
- maps：全进程（main/EngineCore/双 Worker）仅映射私有修复版
  `libmf_hybm_core.so`（1dc46d7d），EngineCore 另见隔离树 `libmf310p_v2_trial_device.so`——
  SO 隔离合同保持

### 3.3 初筛 UNKNOWN 的人工裁决（4 处模式命中全良性）

| 命中 | 实际内容 | 裁决 |
|---|---|---|
| acl_errors serve=1 | 助手子进程预期探测缺失同步码 `kfc sync=507018` | 预期（设计内） |
| bench 致命字样 2 | bench 命名空间 `ready_check_timeout_sec=0` 与 `Failed requests: 0` 计数器字样 | 良性（值=0） |
| runner 异常字样 2 | `requests_completed errors: 25 0`（0 错误）等 | 良性（值=0） |

人工复审判定：**PASS**（25/25 完成、双 Rank bit-exact、目标故障零出现）。

## 4. 根因实验证实（S3-R vs S3-R2 单变量对照）

| | S3-R（16:50 无 env） | S3-R2（仅增 helper env） |
|---|---|---|
| 助手发现 | dladdr 落空 → 回退进程内 | env 命中 → 助手模式 |
| Worker KFC | 首探执行→上下文死→级联 507018 | 零 KFC 命令 |
| 部署 | 失败（clean residue failed） | 成功（verified size=13960） |
| 首调 | 未到达 | 双 Rank bit-exact |
| 请求 | 0 完成 | 25/25 |

单变量（显式助手路径 env）翻转全部结局 → 审计根因判定（`ed63bf32e`）**实验证实**。

## 5. 恢复核验（PASS）

生产 `libmf_hybm_core.so` 88d39285（双位置）与 helper 849f93c0 逐字节不变；修复版副本
1dc46d7d 完好；设备进程表空、端口 8000/8581 回收、卡 16384 健康 OK、worktree porcelain=0；
本轮**零系统变更**（仅 env 与私有目录副本），无需回滚动作。

## 6. 遗留与建议

1. **Gate E 507014 仍未复现**：90 秒冒烟证明融合路径在 2,3 功能性全绿，但目标故障的复现
   需要 S4/S5 级真实浸泡（长时+真实流量），未在本轮授权范围。
2. **后续 Gate E 轮次建议标配** `MF_SDMA_ORCH_DEPLOY_HELPER` 显式路径（私有目录/异位装载
   场景的必要配套），并保留修复版 SO（507899 回归已二次验证）。
3. **INFO 可见性改进项**（移交 MemFabric 参考）：`MF_LOG_LEVEL` 消费点在
   `TransferAdapterPy::Initialize`（晚于编排器初始化），建议前移至 hybm 入口读取，使
   `deploy helper rc` 等关键 INFO 在首个编排器初始化即可见。
4. 助手 stdout 建议加 pid/devId 前缀与时间戳，消除跨进程行序歧义（本轮 §3.1 行序注记）。

## 7. 证据

- 分支库：`evidence/gate_e_s3r2_20261026/`（6 件 + manifest，全部脱敏）
- 服务器本地：`g1_local_archive/gate_e_s3r2_raw_20261026/`（完整原始件，含 maps 连续采样）
- 会话目录：`…/gate_e_forensics/S3R2_FUSED_SMOKE_20261026T165006Z/`（serve/bench/runner/samples/evidence）

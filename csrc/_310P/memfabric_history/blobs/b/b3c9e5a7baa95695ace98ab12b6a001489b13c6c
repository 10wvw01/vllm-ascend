# Gate E S3-R2 复核回应：三项离线核验增补与 S4 放行判定（2026-10-26，310P3）

> 复核意见：①报告称"进程内 KFC 回退 0 次"但证据文件 count=1；②"跨 Die KFC 文件系统共享"
> 结论未经验证；③未记录 S3-R2 实际负载耗时。本增补逐项回查**原始** serve.log/runner.log，
> 给出结论与修正，并附 S4 放行判定。全程零 NPU 操作。

## 1. "falling back count=1" 悖论回查（原始 serve.log 逐模式复核）

| 模式 | 命中数 | 命中内容 |
|---|---|---|
| 宽松 `falling back`（证据文件所用） | **1** | serve.log L421：`[sampler.py:54] Triton not available, falling back to vLLM default penalty implementation` |
| 精确 `falling back to in-process KFC deploy` | **0** | — |
| 精确 `deploy helper not found`（RunDeployHelper 回退前 INFO） | **0** | — |

**结论：无真实进程内 KFC 回退。** 证据文件 `s3r2_01` 的断言块使用了宽松模式 `falling back`，
过匹配了 vLLM Python 采样器在 NPU 上无 Triton 时的 penalty 实现回退警告（与 KFC/MemFabric
部署链无关、亦为 S0–S3/H1 各轮共有背景现象）。报告正文"零 `falling back to in-process`"
的措辞是精确模式，其结论成立。**根因是断言块模式精度不足，而非存在隐藏回退**——与初筛
UNKNOWN 的 4 处良性命中同类（宽松模式过匹配），已在本增补证据中以精确模式复核归零。
若后续任一轮出现真实回退，其特征是：库侧 INFO "deploy helper not found; falling back to
in-process KFC deploy" + Worker 侧 `kfc launch failed` 级联（S3-R 实录），不可能仅凭
宽松模式单独出现。

## 2. "跨 Die 共享设备文件系统"结论纠正（降级为未验证的机制推断）

**本轮直接观测到的事实**（不依赖机制假设）：
1. 两次助手调用（每 Rank 一次，dst 路径相同），经宿主侧 flock（`/tmp/.mf_sdma_orch_deploy.lock`）
   串行执行；
2. 助手甲：探测缺失 → 部署 → `deployed and verified size=13960`（rc=0）；
3. 助手乙（后获得锁）：探测**命中** → `already present`（rc=3）。

**原报告的越界表述**：`rc=3，共享设备文件系统传播`——把"两 Die 共享
`/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/` 同一后端存储"当作既证事实写出。实际本轮
**未做任何跨 Die 文件系统同一性直接测试**，该机制系由 10-19 G1-D 验证（另一张卡上
dev0 部署/dev1 命中同型现象）外推的推断。

**修正后的表述**：助手乙探测命中与"两 Die AICPU 实例共享该路径后端存储"的假设**一致**
（且有 10-19 同型先例支持），但该机制**本轮未验证**；不排除其他使乙探测命中的机制。
时序推断（部署先于命中）不依赖该机制，依据为：S3-R 已证 15:04 设备侧文件缺失 + flock
串行化保证乙的探测后于甲的部署完成。

**对 S3-R2 结论的影响：零。** 每 Die 的运行时正确性独立成立——各 Worker 对自己的 Die 独立
执行 `TryLoadKernel`（mode=0 加载 + 首调 bit-exact 验证于本 Die 数据面），与文件系统是否
共享无关；rc=3 本身是乙进程内直接观测的探测结果，非推断。行序伪影注记维持（无时间戳的
`[helper]` 行跨进程落盘顺序不可作时序依据）。

## 3. S3-R2 实际负载耗时（原始日志时间线）

| 阶段 | 窗口（UTC） | 耗时 |
|---|---|---|
| 会话总窗（runner 起止） | 16:50:11 → 16:58:26 | **495 s** |
| serve 启动 → healthy（含模型装载/编译） | 16:50:11 → 16:53:27 | **196 s** |
| 首次 V2 初始化（helper 部署链 + warmup） | ~16:53:40 → 16:54:48 | ~68 s |
| **实际负载（bench rep1 起 → 结果落盘）** | 16:53:31 → 16:58:19 | **288 s**（tqdm 请求循环 4:38） |
| teardown/汇总 | 16:58:19 → 16:58:26 | 7 s |

要点：`GATE_E_DURATION_S=90` 为 eager 脚本负载窗参数，但单轮 bench（25 请求、并发 10）
自然耗时 288 s 超窗，runner 按脚本语义等待 rep 完成后结束（"load done after 1 reps"）——
即 S3-R2 的实际持续负载为 **288 秒**，非 90 秒。25/25 请求完成、0 错误。

**附带裁决**（补充 runner 计数器的最后一项）：`deadlock_hits=1` = serve.log L416
`shm_broadcast.py:802` "No available shared memory broadcast block found in 60 seconds…
(e.g. compilation…)"——首调 warmup 静默期 vLLM 良性 INFO（"hanging" 字样被 eager 脚本
`grep -ciE "deadlock|hang|watchdog"` 计数），与设备/融合路径无关。至此 S3-R2 全部
模式计数器（初筛 4 项 + 本项）均已逐条裁决为良性，人工复核维持 **PASS**。

## 4. S4 放行判定

- 精确模式复核：进程内 KFC 回退 = **0**（§1）；无任何未解释的回退信号。
- S3-R2 全部验收（helper 模式、双 Rank bit-exact、25/25、零 507014/507899、恢复核验）维持成立。
- **判定：放行 S4**——单次 620 秒 Gate E 融合 Eager 复现，沿用 S3-R2 冻结配置（修复版
  Host SO + 显式真实 orch_deploy_helper + KVER11 + 同模型/参数），仅
  `GATE_E_DURATION_S=620`（即 gate_e_eager.sh 的原 G3 复现工况默认值：25 prompts/轮、
  并发 10、滚动 rep 至窗口耗尽，预计 3 rep、峰值实际负载 ≈ 840 s）。
- 约束：单次执行、不重试、不换卡、不跑 Stock；G0（独占/健康/版本/网络/恢复）发射前重做；
  guardian v2 语境判别（helper 探测缺失 507018 = 预期 vs Worker 发射失败 = 致命）沿用；
  验收以实际 V2 命中/首调 bit-exact/有效请求数/620 s 持续负载证据/设备日志为准，不凭退出码。

## 5. 证据

- `evidence/gate_e_s3r2_20261026/s3r2_07_verification_addendum.txt`：§1–§3 的原始日志复核
  表（宽松/精确模式计数、命中行全文、时间线提取命令输出）
- 原始件（服务器本地）：`g1_local_archive/gate_e_s3r2_raw_20261026/`（未脱敏全集）

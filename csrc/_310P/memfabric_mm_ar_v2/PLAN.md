# memfabric_mm_ar_v2 研发与验收计划（2026-10-17）

> V1 代码冻结。2026-10-17 已验证的 T0 报告：[20261017实机测验及修复.md](20261017实机测验及修复.md)。
> 2026-10-17/18 T1（小 M 原始 NZ + 直接 ND）完成首轮实机验证与修复：[T1_NZ_DIRECT_REPORT.md](T1_NZ_DIRECT_REPORT.md)。
> T1 细节见 [NZ_DIRECT_TRIAL.md](NZ_DIRECT_TRIAL.md) / [DESIGN.md](DESIGN.md)。

| 阶段 | 验收内容 | 当前状态 |
|---|---|---|
| G0 | IBSet/IBWait 官方 2/8 核同步、跨调用复用 | **T0 实机 PASS（回归复测 PASS）** |
| G1 | T0 多核 M-split 串行 FP16 bit-exact | **T0 实机 PASS（回归复测 PASS）** |
| G2 | T0 串行/交叠，M=2048/4096/8192 | **T0 实机 PASS（回归复测 PASS，18 配置）** |
| T1 | 小 M 原始 NZ + 直接 ND + 1/2/4/8 MM/ADD | **首轮实机验证 PASS（修复 basicM 越界 + 首启 MTE3 丢弃两处缺陷后）** |
| G3 | Graph、任意 M、复用压力、完整模型和正式路由 | **未完成** |

## T1 / P0：先证实官方 Matmul 子块能力

1. CANN 9.1 `bisheng --npu-arch=dav-2002` 编译 20 个 `T×mmCores` 原型模板；如出现静态 tiling / SetSingleShape 不支持，优先针对 API 合同改正，不绕过验证。
2. 单卡先检查 `M=1,16,17,255,256`，核数 1/2/4/8 的 `W(NZ)` 子矩阵寻址是否与 `F.linear` 一致；特别检查相邻 N 片 512B 行边界、`SetOrgShape` 的 B/C pitch。
3. G0 双核、8 核 IB probe 保持 PASS，再执行两卡单 Kernel 完整 MM→signal→wait→Add→quiet→DONE。
4. 小 M 首轮只做 M={1,2,4,8,16,32,64,128,255,256}，MM 核数 {1,2,4,8}×Add 核数 {1,2,4,8}，所有组合 bit-exact、协议状态零。逐次校验。
5. 追加随机 30 个 M、动态输出指针、同权重与不同权重指针、千次重复；特别检查 1 核 MM+8 核 Add 时无需错误额外 MM 参与。
6. 同环境分别记录 Stock、V1 N-split、V2 NZ Direct 的首调用、中位数/P95；测试真实 Host 耗时（不误把单 Kernel 当成无 Host 开销），profiling 分解 MM、IB、SDMA、Add、波级 DONE。
7. 重新回归 T0 原有 G1/G2。将结果保存 `T1_NZ_DIRECT_REPORT.md`，记录编译/数值/性能、BUG 根因、修复和回归。

**T1 首轮执行结果（2026-10-17/18）：步骤 1-5、7 全部完成 PASS；步骤 6 完成 Stock/V1/V2 三方对比（见报告）。** 修复两项缺陷：`DirectNsplit` basicM=256 的 UB 工作区越界（MTE on-chip read OOB）与 T0 kernel 首启/新 y 首写 MTE3 丢弃的守护策略修正；另发现带池 `aclrtSynchronizeDevice` 随机 0-15s 的代价与一次偶发跨 rank 超时，均已记录于报告遗留问题。

## G3 本轮加固入口（2026-10-18 审视补充）

执行顺序与证据要求见 [G3_HARDENING_PLAN.md](G3_HARDENING_PLAN.md)：先修正版 T1 stress，再隔离 MTE3 首写丢失与 KFC 部署，再测真实 HCCL/MemFabric 共存及设备 profiler。旧 T1 报告的 stress 实际只覆盖 4 个 M，不是声称的随机 30 个；更正见报告第 9 节。新压力脚本尚待实机复测，不能提前标为 PASS。

## 工程加固仍需并行跟进

- 首启/新输出缓冲 MTE3 丢弃的硅级机制仍未有官方解释；T0 kernel 路径只能用 `aclrtSynchronizeDevice` 治愈（带池随机 0-15s），T1 kernel 路径 launch-stream 同步即可治愈。Graph 不可据此宣称支持。
- T0 dev0/1 KFC 初始化曾依赖异常部署绕过，需在健康设备重跑。
- 双卡测试程序已归档提交（本目录）；原始运行日志在服务器 `/tmp/opencode/v2/`。
- 禁止 V1 serving 与 V2 trial 共享单池，禁止未经序列化的同池多 Stream 调用。

## T2：原生 MM + MemFabric AR（不融合）对照，待实机

新增独立实验性 AR-only C ABI `mf310p_v2_trial_run_ar_only`、设备 Kernel 和三方案 `bench_ar_only.py`。与 T0/T1 的 MM+AR 融合实现并存，但**不修改 V1 或默认模型路由**。明确比较：

- A：原生 FP16 MM + FP16 HCCL AllReduce（唯一性能收益基线）；
- B：相同原生 FP16 MM + **完整计入 D2D staging** + MemFabric SDMA+FP16 Add（MM 不进入融合 Kernel）；
- C：V2 已有融合 MM + MemFabric AR。

本实验须先验证 B 的 bit-exact / status / MTE3 首写，再测 M=1..8192、chunk 256/512/1024、Add 核数 1/2/4/8 与 max-rank median/P95。当前**未在 310P3 编译或执行**，不能宣称 B 有收益。详见 [T2_AR_ONLY_TRIAL.md](T2_AR_ONLY_TRIAL.md)。

## G3 V2 大 M 模型接入（2026-10-18 源码已写，**待实机**）

本期单用户最大输入 4096 tokens、输出测试长度 2048，服务调度参数
`--max-num-batched-tokens 4096`。因此单次 Forward 的 M 以 4096 为上限；
M 代表合并请求后的当次 token 总行数，不是单请求输入长度。
本期只需处理 2048～4096 内非整 Chunk 尾行，不进行 M>8192 多段；
超出 M=4096 明确拒绝并提示配置错误，原 T0 8192 行独立实测结论不变。

已新增独立 V2 Torch Op/路由与 TP=2 回归脚本，固定 **M<2048=Stock MM+HCCL，M>=2048=V2 T0 融合**。新代码会在池创建前预分配持久输出槽，第一次大 M 时执行所有槽位的真实融合预热；继续保留 T0 首写全设备同步守护，仅尝试把开销转移到初始化阶段。未证明 MTE3 首写已经根治，KFC 正常部署和 Graph 大 M 均未验收。

主文档：[V2_G3_QWEN_E2E_TRIAL.md](V2_G3_QWEN_E2E_TRIAL.md)；门槛依次为构建→2047/2048 路由与 HCCL/MF 同池交替→3000/4095 动态尾行→真实 Qwen 40 层 Eager/Graph→Stock/V2 三会话吞吐/TTFT/TPOT→10min 长稳。大 M Graph 捕获不支持：仅 FULL_DECODE_ONLY，capture sizes 必须 <2048。

**收益判据仍然只有 Stock**：单层大 M 快于 Stock 不等价于真实 Qwen 端到端收益。严禁在 P0 同步风险/首写数据破坏尚未排除时将本阶段标成 PASS。

## 下一阶段 G3

根据实测 M 阈值路由 Stock / T1 / T0；进一步研究 M/N 二维切分、Add 核数、M>8192 多波、动态尾 M、真实 Qwen3.6-35B-A3B-w8a8 的 TP2 40 层 Graph / ITL / TTFT / 吞吐与长稳。

**所有性能选择必须以 310P3 实机测试为依据。**

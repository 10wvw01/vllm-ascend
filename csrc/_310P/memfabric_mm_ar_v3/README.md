# 310P3 V3：MM + SDMA FP16 ADD AllReduce

开发分支：`work/310p-mm-ar-v3-sdma-add-20261029`，从项目主线 `54d5518d` 派生。
本工程使用公共 `smem_shm_sdma_signal_at(..., ADD_FP16)` 完成求和，设备核没有本地 Vector Add。
生产/Serving 默认仍为 Stock，本工程不会修改 Model Runner、DFlash 或现有 V2 路由。

## 实现与适用范围

- TP=2、310P3/dav-2002、FP16、K=N=2048、权重原生 NZ(format29)、ND 输出。
- `V3Session.run`：同一 Kernel 完成 MM、ADD Reduce-Scatter、COPY All-Gather、MTE 输出搬运。
- `V3Session.run(..., partial=True)`：预计算部分结果的独立通信协议测试。
- `V3Session.probe_add`：独立验证 FP16 ADD 的真实数值，不以完成戳冒充正确性。
- group=256/512/1024/2048；2048 组仍使用两次已验证的1024子MM，每核basicM不扩大。
- 同时实现串行和一组前瞻流水；流水将下一组MM与上一组Reduce交叠。
- `run_exact` 覆盖1～4096的M，通过显式输入stage补齐；默认快测只计时M2048/4096。
- 只有Core0调用signal/wait/quiet；全部调用保留IB、DCCI、首写保护、quiet/DONE。
- 仅单设备上下文、单流、推理模式；调用方维持Tensor生命周期；Graph未准入。

## 为什么采用两阶段分片

每个组按M行连续分为A/B：Rank0拥有A，Rank1拥有B。
两端都先将本地MM结果写入自己的完整工作区，随后：

1. 本地各核完成MM与缓存维护，汇总IB后交换READY。
2. Rank0将其B以ADD发送给Rank1，Rank1将其A以ADD发送给Rank0。
3. 收齐本端owner分片的Reduce邮件后，owner分片已经是最终求和结果。
4. 各端将自己的owner分片以COPY发送回对端，获得完整结果。
5. 各核从工作区搬运到独立输出Tensor，汇总完成后quiet/DONE，才允许复用。

Reduce中发送源和被累加区互不相交。收到Reduce邮件也证明对应的对端发送源已消费，
因此后续Gather可以覆盖那个对端源。Gather源保持到quiet；输出搬运及DONE也必须完成。
每Rank的数据发送量仍是半份ADD加半份COPY，合计一份完整结果，控制邮件另计。
不能直接做双端全量原地ADD，否则可能修改对方尚在读取的发送源。

## 依赖与构建

固定MemFabric源码：`10wvw01/memfabric_hybrid`分支`fix/310p-v3-orch-deploy-helper-20261029`
（`db38d75e8485d31451302219eb7a1bfeb1dd107c`，基于同步锁`b9d46f97`），KVER28。
该修复分支按研发侧指令移植生产验证过的`orch_deploy_helper` fork+exec部署机制并强制独占使用
（torch_npu宿主进程内KFC命令会永久打死设备上下文；助手缺失/失败即报错，无进程内回退），
ADD/KVER28/SetAccess修复不变。旧KVER11安装包和旧V2嵌入头不适用于V3。
公共头、Host库、设备so、launch JSON和部署助手必须同批。
构建脚本校验源码HEAD/工作区、公共头SHA、JSON类型/KVER、助手存在/可执行位/SHA，
冻结所有安装SO与V3源码SHA。它只在指定目录生成私有V3 so，不覆盖共享安装、不预部署AICPU。

上游GitCode已同步到`ba3e7f42`，GitHub镜像提交为`b9d46f97`，两边完整tree一致。
新增SetAccess物理设备号修复，详见[UPSTREAM_SYNC.json](UPSTREAM_SYNC.json)。
服务器完整执行顺序、两仓分支及私有部署约定见[OPENCODE_EXECUTION.md](OPENCODE_EXECUTION.md)。

先在新的隔离MemFabric克隆中构建并准备完整私有SDK：

```bash
python csrc/_310P/memfabric_mm_ar_v3/tools/prepare_sdk.py \
  --source /absolute/private/memfabric-source \
  --out /absolute/private/memfabric-sdk \
  --cann /usr/local/Ascend/cann-9.1.0 --jobs 8
```

此脚本调用上游wrapper并显式收集Host库、公共头、AICPU SO/CUST JSON/launch JSON。
它不执行run安装器：当前上游run打包器没有复制AICPU目录，单独安装run包不足以构成完整SDK。
不修改共享安装或Python环境，不在构建阶段预部署设备。

在V3工作树根目录：

```bash
python -m pytest -q csrc/_310P/memfabric_mm_ar_v3/test --import-mode=importlib
python csrc/_310P/memfabric_mm_ar_v3/tools/build.py \
  --memfabric-source /absolute/private/memfabric-source \
  --memfabric-home /absolute/private/memfabric-sdk \
  --cann /usr/local/Ascend/cann-9.1.0 \
  --out /absolute/private/v3-build-001
```

`--out`必须是新目录。编译失败先保留`build_*.log`并停止当前轮；OpenCode可定位并修复执行中的BUG，
提交新版本后在新目录重编，职责和证据要求见[OPENCODE_EXECUTION.md](OPENCODE_EXECUTION.md)。
不把语法检查当作BiSheng PASS。
新HEAD必须重新构建；构建manifest与启动时的源码/依赖身份不一致会被拒绝。

## 给OpenCode的一次快速验证

用户已要求：只测V3，复用历史V2和真实NZ Stock性能数据，不重新运行旧性能臂。
本入口始终不初始化HCCL、不启动Serving，不运行十四场Confirm或TTFT。
建池前只计算数值参考：本地FP16 MM、CPU Gloo求和。这些参考不采性能数据。

取得健康且连续独占的完整双Die窗口后，选择实际卡号；下例仅展示38144(dev4/5)，
不能因示例卡号而在被占用的卡上启动。不要杀其他租户，也不要自动换卡重试。

```bash
python csrc/_310P/memfabric_mm_ar_v3/tools/quick.py start \
  --manifest /absolute/private/v3-build-001/manifest.json \
  --card 38144 --devices 4,5 \
  --out /absolute/private/v3-quick-001 \
  --lease-note '填写实际独占窗口或协调记录'
python csrc/_310P/memfabric_mm_ar_v3/tools/quick.py status \
  --out /absolute/private/v3-quick-001
```

`start`创建独立session，立即返回PID；OpenCode用`status`轮询，不用前台工具超时杀torchrun。
默认设备阶段最多600秒，运行中每2秒巡检占卡/健康，异常只清理自己创建的进程组。
脚本不会获取排他租约；实际独占窗口仍由现场保障。
为避开历史SetAccess逻辑/物理号分裂，子进程不设置`ASCEND_RT_VISIBLE_DEVICES`，
每Rank直接选物理DeviceID，并经CANN API核验；父Shell环境不变。

一次会话自动依次执行：

| 阶段 | 默认工作量 | 失败行为 |
|---|---|---|
| CPU/G0 | CPU合同、构建/依赖SHA、双Die健康/空闲、实际物理映射 | STOP |
| V3-0 | 官方IB64轮；4KB和1MiB双向FP16 ADD数值 | STOP，不重发ADD |
| V3-1 | 两组长、串行/流水、交替输入；额外24轮256组跨1024邮件回绕 | STOP |
| V3-2 | M2048/4096双组长、两执行模式的融合正确性；M2051尾块 | STOP |
| V3-3短测 | 仅V3流水：M2048/4096×group1024/2048×前后两轮，每轮4预热+30样本 | 保存全部样本 |
| 离线报告 | 双Rank逐样本取max后统计p50/p90/p99，再读取历史对照 | 不调度任何旧性能臂 |

默认固定1MiB分包，因为FP16 ADD大包能力须独立验证；不能把V2的4MiB COPY证据当ADD证据。
4MiB ADD/COPY包长API已实现，但不在第一次快测中自动扩大变量。
冷首写同步、首次KFC自动部署和数值校验不进入热态计时；快测耗时受冷启动及编译影响。

## 输出与结论

- `result.json`：退出码及PASS_QUICK_SCREEN_ONLY/STOP。
- `gate_*_rank*.json`、`stop_rank*.json`：逐关原始结果与失败现场。
- `rank0.json`、`rank1.json`：配置、实际装载SO、门禁、每个性能样本。
- `timing_*.json`：逐形状即时保存，后续失败不丢已有证据。
- `historical_comparison.json`、`REPORT.md`：V3相对历史D4/E1/E3/E4/真实NZ Stock的差值。
- `console.log`、`lease_guard.jsonl`、`lease_guard_stop.npu.txt`：完整执行及占卡取证。

历史基线从本仓归档JSON读取，并记录源文件SHA；D4使用主线性能表中的R2 Screen记录。
历史`SCREEN_ONLY`和`INCONCLUSIVE_DRIFT`标签原样保留，不把它们升级为正式结论。
对每个V3配置同时给出前后两轮漂移；超过5%不作为方向判断。
正的“时延缩短”表示比该历史值快，但跨版本/卡/时间的差异不构成同批因果证明。
接近历史噪声范围的结果应写暂不能确定，不自动补跑Stock/V2。

## 已知边界

- 当前开发环境无CANN/NPU：CPU测试与Host公共头语法检查不是设备编译或实测。
- KVER28源码仍保留占位SQE自拷贝，而其提交说明称已替换；build/报告显式记录该差异。
  快测允许观察新算法，但不能据此签署MemFabric长稳或生产发布结论；异常必须停止留证。
- ADD不可幂等重试；超时、重复/错代次邮件、错算或507014/507899必须双Rank停止。
- 同一Session不并发、不跨流、不支持训练或Graph；禁止将可复用通信工作区当输出返回。
- 模型40站点、DFlash、Graph和长稳另按项目门禁验收，本轮不运行这些阶段。

源码与两轮静态审查见[REVIEW.md](REVIEW.md)。项目状态归主线V2目录中的控制台/ADR/总账统一维护。

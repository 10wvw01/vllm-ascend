# R2 310P3 V2 静态GO后实机交接Runbook（草案；当前禁止执行）

> 当前 **STATIC NO-GO**；本文件仅规定未来OpenCode在用户单独授权后需要做的检查。ChatGPT研发侧不登录310P3、不安装Triton、不替换服务器SO、不修改DFlash模型配置、不自动启动推理。

## 进入现场前硬条件

- PR#7仍DRAFT直到全部T01～T25、ABI-01～15具有源码、CPU合同、真实接线、设备算子静审、两轮独立审查证据；\`python tools/310p_mm_ar/r2_static_admission.py --check\`成功；生成\`R2_STATIC_REVIEW.md\`最终PASS且候选Worker fail-closed仅在证据齐全后有明确安全替换提交。
- 固定基线：\`vllm-project/vllm@6e448d0ea9bf3d88d898b65449ca6dc2aec170ac\`、310P3单卡双Die、TP=2、PP=1、Eager、DFlash K=7，Draft配置5×sliding+1×full、sliding_window=4096，Target Qwen3.6 W8A8/GDN/Mamba。模型目录必须从现场实际路径验证，禁止修改权重/config。
- 与已归档R1-B现场证据比对；R1-B G0 PASS不能继承到新工作树。Stock+DFlash V1历史失败不能假设已修好；新V2必须按R2-B Target Stock、R2-C Stock+DFlash、R2-D融合逐阶段验证。

## G0只读取证与SO身份

1. 现场记录UTC+08时间、hostname、\`git rev-parse HEAD\`、\`git status --porcelain\`、\`git remote -v\`、pip freeze中vllm/torch/torch_npu/transformers及CANN/HCCL版本；确认\`vllm.__file__\`与源码SHA对应。不得把Python包版本字符串当成Git身份。
2. 双Rank分别记录NPU-SMI设备映射、物理device-id/die、可见卡列表、独占占用者、启动用户/权限；已有其他进程占用即NO-GO。严禁强制杀死不属于测试的进程。
3. 对实际将被加载的Host SO、Device SO、\`KVER11\`及MemFabric/算子依赖分别执行\`readlink -f\`、\`sha256sum\`，记录路径与指纹、加载映射与历史G0差异；发现指纹不符立即停测，不自行替换SO。
4. 验证实验环境没有伪造Triton或篡改\`HAS_TRITON\`，不打开Graph、不关闭数值/首写门禁、不修改DFlash配置。保存G0 JSON、双Rank原始日志和失败栈。

## 实机阶段（必须用户明确授权、逐阶段停机门禁）

- **R2-B**：V2 Target Stock Eager，TP2双Rank加载、最小prefill/decode、GDN/Mamba、request abort/reuse、长上下文、对比CPU/Stock数值；任一Rank异常即停。
- **R2-C**：V2 Stock+DFlash K7 Eager，5 sliding+1 full组、不同kernel blocksize、非因果mask、采样greedy与temperature>0概率正确性、拒绝率、mixed prefill/Decode；K7输入/采样/slot/输出与冻结gold对照。
- **R2-D**：只有R2-B/C稳定PASS才比较融合MM+MemFabric AR与同配置Stock MM+HCCL AR，记录实际融合调用次数、物理M、TTFT/TPOT/吞吐及错误/异常；不能把融合与Stock混算或误报Graph收益。
- 每阶段提供精确启动命令、环境、双Rank stdout/stderr、JSON指标、SO SHA、trace、失败原样日志与rollback证明；无证据不得升级PASS。

## 回滚与止损

保留独立的原始Stock服务工作树、可验证Host/Device SO指纹和原配置。遇到数据不一致、死锁、507014、首写静默错误、Rank不一致、非法设备访问、Triton调用或版本漂移立即停止；先保全日志/进程/设备状态，再按原始配置恢复，不自动覆盖SO。回滚后核对Stock服务恢复与双Rank释放，任何未知项记UNKNOWN而不是PASS。

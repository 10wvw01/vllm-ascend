# V2 同步与异步性能专项：两轮静态审视（2026-10-21）

> 审视范围：GitHub 源码、T2/G3 历史报告及新实验代码；尚未在 310P3 上编译/执行，也没有运行 CPU-only 测试。**静态审视不等于实机 PASS**。

## 一、审视基线

- V2 `src/op_host/mf310p_v2_trial_adapter.cpp`：`trial_create` 只建池一次，`trial_run` 不重新建池，只执行状态清理及融合 Kernel 发射。首个符号/新输出地址可能触发第一次 `aclrtSynchronizeDevice` 守护。
- V2 `src/torch_adapter/mf310p_v2_serve.cpp`：持久 Handle、输入暂存和输出池复用。
- 原 T2 `test/bench_ar_only.py`：单次创建通信资源，Shape 预热后使用 enqueue→sync 稳态计时。
- Stock `vllm_ascend/ops/linear.py`：`unquantized_gemm` 的实际实现为 `F.linear`。

## 二、Round 1：数值与同步公平性

| 检查 | 结果 |
|---|---|
| 数学口径 | Stock FP16 NZ MM+HCCL FP16 SUM；V2 相同形状/精度，结果和 FP16 MM→FP32 HCCL→FP16 参考逐元素精确对照 |
| 资源 | Stock Process Group 一次创建，V2 trial pool 一次创建；同 Shape 计时不重建 |
| 输入 | 40 组独立输入和权重，随机种子跨 Rank 与 layer 不同，两组 arm 种子完全相同 |
| 初始化 | 数据、权重和 V2 输出槽在 MemFabric 建池前分配；输出新地址首写提前预热和检查 |
| Stock async | HCCL Work 对象与 MM 输出保留到 Work.wait + NPU Stream synchronize |
| V2 async | 每窗口最多使用同等数量独立输出槽，单 stream 入队，只有窗口末完成后复用 |
| 运行时边界 | M=2048/3072/4096 且 %chunk=0；不混入 tail input stage；V2 overlap=1 |

**实质性修复 #1：** 首版 V2 `execute_window` 在正式计时中无条件做协议状态 D2H 检查，Stock 无对应开销，会制造伪负收益。现已仅在计时前后的正确性/预热阶段做 status 检查，正式计时不做 D2H、Profiler 或逐元素对比。

## 三、Round 2：版本溯源与结果解释

| 检查 | 结果 |
|---|---|
| 两仓库基线 | V2 独立 perf worktree；MemFabric 维持现有镜像修复线 KVER11，禁止切到原始 wgm-dev-310p |
| 来源锁定 | 在 TP2 Worker 内校验导入的 extension 真实路径位于 perf worktree，杜绝旧 site-packages 假装新版本 |
| Host 资源 | Runner 校验可用的系统/包内 Host SO SHA 一致，校验 KVER11 JSON 并归档 |
| 交叉顺序 | 同 M Stock→V2→V2→Stock 两对测试，第二对内部窗口顺序反转；进程之间预留 45 秒资源清退冷却 |
| 统计 | 分 Rank 输出，使用较慢 Rank 中位数近似；区别 max(rank median) 与 median(per-repeat max) |
| 核心指标 | `Gain(W4)-Gain(W1)`。双方都减少同步，只有比较相对优势变化才有意义 |
| 停机 | Bit-exact 不符、status 非零、Rank 缺失、非零退出、900s 超时都不能继续下一 Gate |
| 目标范围 | 40 次独立调用的连续入队只是压力包络，绝不是模型40层真实依赖图；无 Graph/Serving 结论 |

**实质性修复 #2：** 只检查当前 Git 分支无法防止 Python 加载另一份旧编译扩展。现已增加环境变量 `V2_SYNC_ASYNC_REPO_ROOT` 对 extension 实际路径的强校验，并限制源码工作树必须无跟踪文件修改。

**实质性加固 #3：** 新执行器记录 Host 两位置 SO 的 SHA256，不一致立即停止；不会自动安装或覆盖设备库。

## 四、现阶段明确无法替代的实机验收

- CANN 9.1.0/Bisheng/torch_npu 实际 ABI 可用、`build_ext --inplace` 构建正常、SO 来源正确；
- CPU-only `test_sync_async_summary.py` 与 `py_compile` 检查通过；
- M=4096、steps=8、W1/W2/W4 的两 Rank Smoke 正确、status=0；
- 完整 40 调用测试的稳态收益/退化趋势、两 Rank 时间差和不同进程热态漂移；
- 本测试在计时窗口内不逐调用抓数值，只有计时前后完整逐窗口校验；低概率瞬态旧数据仍需后续专项排查；
- 模型级 Prefill/Decode、实际40层数据依赖、Graph 与 Gate E 稳定性仍未覆盖，不得推断线上可用性。

**结论：** 静态准备已完成两轮，修改并消除了三处主要方法论/代码问题，但没有宣称实机测试已经通过。由 OpenCode 先完成 G0/G1，再开始 40 步 ABBA 性能采样。

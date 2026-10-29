# 310P3 HCCL 缓存与异步正确性快速探针（2026-10-21）

> 目前仅完成 GitHub 测试代码准备；未在 310P3 实机执行，不能宣布缓存已生效。

## 重要边界

1. 缓存 HCCL 的通信域、任务编排/展开结果或资源，不等于缓存 AllReduce 的旧数据；每次 AR 仍必须使用本次输入。
2. 官方 CANN 9.1 Atlas 300I Duo 对 HCCL_OP_EXPANSION_MODE 列出 HOST（默认）与 AI_CPU。没有列出 AICPU_CacheDisable，因此不能在 310P3 上贸然使用该开关作为“缓存开/关”。
3. 即使相同 Shape 首次执行慢、热态更快，也可能是首次分配、链路初始化、连接建立或设备热态变化；不能仅凭加速就证明 AI CPU cache 命中。
4. HOST 与 AI_CPU 的延迟差包含展开算法/调度位置变化，**不是严格的缓存消融实验**。第一轮只测现场当前实际设置的模式。

官方依据：[CANN 9.1 HCCL_OP_EXPANSION_MODE 产品支持范围](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_env/HCCL_OP_EXPANSION_MODE.md)，[ProcessGroupHCCL 异步调用源码](https://gitee.com/ascend/pytorch/blob/master/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp)。

## 实机执行步骤（先验证设备独占）

在新 worktree 检出 diag/310p3-hccl-cache-probe-20261021 分支；只使用 HCCL，不编译/运行/安装 V2 或 MemFabric，不修改正式分支、原 v11 库和 KVER。

```bash
set -o pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1
env | grep '^HCCL_OP_EXPANSION_MODE=' || echo 'HCCL_OP_EXPANSION_MODE=<unset/default>'
npu-smi info | head -50

timeout --signal=TERM --kill-after=15s 240s \
  torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/test/hccl_hot_cache_probe.py \
  2>&1 | tee /tmp/hccl_probe_1.log

# 同一环境重新启动独立进程，观察首调用和稳态的一致性
timeout --signal=TERM --kill-after=15s 240s \
  torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/test/hccl_hot_cache_probe.py \
  2>&1 | tee /tmp/hccl_probe_2.log
```

脚本要求 WORLD_SIZE=2，每 Rank 对相同顺序执行通信；使用当前计算流的 Work.wait 建立通信结果可见性依赖，最终做 NPU 同步后验证。两 Rank 都必须输出 phase=complete,result=PASS，否则不能认为执行成功。原始完整日志仅在服务器归档。

## 测试矩阵

| 阶段 | 操作 | 关注 |
|---|---|---|
| single | M=2048/3072/4096，首调用、6次预热、24次热态 | 首次 vs 热态 P50/P95，不能直接归因缓存 |
| fixed_same_shape_independent_buffers | 40个独立 Buffer，同 Shape M=3072，多轮输入值变化 | 热态顺序 AR 成本、同 Shape 资源复用可能性 |
| mixed_shapes_similar_total_bytes | 40个交替 Shape，合计字节量接近固定组 | 动态 Shape 的综合成本；不等于Cache失效因果 |
| async_dependency_chain | 上游异步写入、HCCL async_op、work.wait、后继读 | 当前 Stream 正确建立 HCCL 依赖和数据正确性 |

计时采用 host perf_counter，HCCL enqueue 到 Work.wait 和 torch.npu.synchronize 完成；输入填充在计时前预先完成。所有 Shape 和 Rank 的输入带不同数值，FP16严格相等校验，缓存不能偷偷重放旧数据。该测试不能模拟所有跨多流生命周期竞态。

**判据**：若首次明显慢而多轮热态稳定快，可判定“存在冷/热态性能差异”；若40个同Shape、不同地址/数值的全部结果正确，可判定“合法时序下资源复用未复用旧数据”。没有同平台受支持的纯 Cache 开关或命中/展开计数，AI CPU 缓存贡献仍标记 UNKNOWN。

## OpenCode 交付

- 两次新进程的原始日志、退出码、Rank0/1的单Shape first/median/p95、固定/混合40次的 batch+enqueue、中间数值验证；
- 当前 HCCL_OP_EXPANSION_MODE、设备/CANN/torch_npu 版本及其它HCCL配置；
- 是否有不同 Rank 的通信错序或数据校验失败；
- 若出现异常，保全上下文立即停止，不切换 MemFabric 分支，不修改任何通信协议。

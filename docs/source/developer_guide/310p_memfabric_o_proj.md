# 310P3 TP=2 Qwen o_proj + 独立 MemFabric MC² 算子

使用与实机 Gate：
[310p_memfabric_o_proj_usage.md](../user_guide/feature_guide/310p_memfabric_o_proj_usage.md)

独立算子仓：`10wvw01/memfabric_matmul_allreduce`。

## 1. 最终边界

```text
vllm-ascend
  ├─ 判断模型/层/shape/TP/阈值
  ├─ 检查外部 OPP + runtime ABI
  ├─ torch thin adapter
  └─ ACLNN call
          |
          v
memfabric_matmul_allreduce custom OPP (.run)
  ├─ ACLNN two-phase API
  ├─ persistent runtime / arena / graph lifecycle
  ├─ 8-core cooperative FP16 MatMul
  ├─ MemFabric public signal/wait/quiet
  └─ FP16 TP=2 SUM
```

vLLM 不再编译 AscendC MemFabric kernel，不包含 MemFabric lifecycle/arena/protocol，也不直接链接 `libmf_smem.so`。

只有外部算子可见且 runtime ABI 匹配时，目标 `RowParallelLinear` 才允许 `reduce_results=False`；否则完整回退 stock matmul + HCCL。

## 2. V1 算子合同

```text
hardware = Ascend 310P3 / dav-2002
tp = 2
reduce = SUM
dtype = FP16
K_global = 4096
K_local = 2048
N = 2048
baseM = 256
baseN = 256
baseK = 64
blockDim = 8
batch_m = 256*q, q in {1,2,4}
```

8 个 AI Core 均参与 MM，显式按 M 分区；core0 完成自己的计算并确认所有 ready generation 后，是唯一 data `signal()` owner。

## 3. 已迁移且保持不变的数据面

runtime 保留 pre-migration v8 的 lookahead=1：

```text
gate(wave)
P0
P1 -> W0 -> A0
P2 -> W1 -> A1
...
Wlast -> Alast
quiet()
ack()
```

目标 overlap：

```text
SDMA(n)   || MM(n+1)
SDMA(n+1) || wait/reduce(n)
```

每封 data mail 严格校验 `status/dst/len/imm`。尾批仍使用 full `batch_m` scratch zero+valid-row copy，最后只 reduce valid rows。

Eager ready generation 从 `0x40000000` 单调递增，避免每波 32 KiB ready-cell memset；Graph 使用稳定的 `batch+1` generation 并保持图内控制状态清理。

## 4. Graph / lifecycle

外部算子拥有进程级 MemFabric context。Graph capture 前必须已经完成：context、scratch、protocol/credit、producer/wait/add/protocol binary warmup。

capture 内不得 create/malloc/host barrier/lazy warmup。普通 eager 固定一个 execution stream；Graph-used context 保守保持 process lifetime。

## 5. vLLM 代码地图

```text
vllm_ascend/_310p/ops/memfabric_o_proj.py
  模型/层 eligibility、MIN_M、fallback、调用 external op

vllm_ascend/_310p/quantization/modelslim_config.py
  eligible FLOAT o_proj 的 linear-method routing

csrc/_310P/memfabric_matmul_allreduce_adapter.cpp
  torch -> dynamic ACLNN thin adapter
```

以下内容已从 vLLM 移出：

```text
custom_memfabric_o_proj/*.asc
MemFabric host adapter/runtime
cmake/memfabric_310p.cmake
libmf_smem direct link
```

## 6. 独立算子 API

外部 OPP 暴露：

```text
aclnnMemFabricMatmulAllReduceGetWorkspaceSize(...)
aclnnMemFabricMatmulAllReduce(...)
mfmc2RuntimeAbiVersion()
mfmc2RuntimeShutdown()
mfmc2RuntimeDebugSnapshot(...)
```

vLLM 通过已有 `GetOpApiFuncAddr`/`EXEC_NPU_CMD` 动态调用，不增加单独 wheel，也不修改 torch_npu/op-plugin。

## 7. 配置所有权

vLLM 只保留 routing 配置：

```text
VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ
VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_BATCH_BASEM_COUNT
VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_MIN_M
VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_WARMUP_FALLBACK
VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TRACE
```

MemFabric runtime 配置属于独立算子：

```text
MFMC2_STORE_URL
MFMC2_LOCAL_BYTES
```

## 8. 验收原则

迁移完成不等于性能结论已经重测。独立仓保存 pre-migration v8 基线：q=1/2/4 correctness bit-exact；ACL Graph bit-exact；q=2 层级 crossover 约 M=2048；M=8192 fused 5.258ms vs stock 6.760ms（-22.2%）。

新结构必须在目标 310P3 上重新通过：安装检查 → 独立 correctness → 1000 次稳定性 → 独立性能/Profiler → vLLM fallback → Qwen eager → ACL Graph → 10min，才能宣布迁移验收完成。

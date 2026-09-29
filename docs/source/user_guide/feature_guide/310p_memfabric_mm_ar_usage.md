# 310P3 Qwen3.6 mm_ar + MemFabric ABI v7 使用与实机验证

> 适用：Ascend 310P3 单卡双 die、TP=2、
> `Eco-Tech/Qwen3.6-35B-A3B-w8a8` full-attention `self_attn.o_proj` 与
> GDN (linear-attention) `linear_attn.out_proj`（两层形状合同一致：
> K_global=4096 / K_local=2048 / N=2048，均未量化）。

设计：[310p_memfabric_mm_ar.md](../../developer_guide/310p_memfabric_mm_ar.md)

Gate：[310p_memfabric_mm_ar_development_plan.md](../../developer_guide/310p_memfabric_mm_ar_development_plan.md)

## 1. 编译

先按 `memfabric_hybrid:wgm-dev-310p` 当前说明安装，并确认：

```bash
echo "$MEMFABRIC_HYBRID_HOME_PATH"
test -n "$MEMFABRIC_HYBRID_HOME_PATH"
```

然后：

```bash
export SOC_VERSION=ascend310p3
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS=8
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1

python3 -m pip install -v -e . --no-build-isolation --no-deps
```

feature-on 会从当前源码用 bisheng `--npu-arch=dav-2002` 编译
`memfabric310p_device.asc`，并链接 public `libmf_smem.so`。

## 2. Runtime 参数

当前 R7 部署约束下先固定：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
```

推荐起始配置：

```bash
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1
export VLLM_ASCEND_310P_MEMFABRIC_STORE_URL=tcp://127.0.0.1:8581

# 默认即 96 MiB；arena 由预算推导，上限 8192 rows。
export VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES=$((96 * 1024 * 1024))

# q=2 -> baseM=256, batch_m=512, 当前形状下 batch payload=2 MiB。
# 合法值仅 1/2/4。
export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_BATCH_BASEM_COUNT=2

# 无 M 阈值：启动选择融合路径后所有 M 均走融合（尾批补零到 batch_m，
# producer scratch 仅分配时清零一次，尾批只拷贝有效行）。

# 服务 bring-up 推荐：profile/dummy-run 先走 stock。
export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_WARMUP_FALLBACK=1

# 诊断时开启。
export VLLM_ASCEND_310P_MEMFABRIC_MM_AR_TRACE=1
```

ABI v7 不再使用 `VLLM_ASCEND_310P_MEMFABRIC_MM_AR_TILE_M`。

## 3. 编译后检查

```bash
python3 - <<'PY'
import torch
import vllm_ascend.vllm_ascend_C  # noqa

for op in [
    "memfabric_mm_ar_allreduce",
    "memfabric_mm_ar_shutdown",
    "memfabric_mm_ar_debug_snapshot",
]:
    print(op, hasattr(torch.ops._C_ascend, op))
PY
```

预期均为 `True`。

再检查动态依赖：

```bash
python3 - <<'PY'
import vllm_ascend.vllm_ascend_C as C
print(C.__file__)
PY

ldd /path/to/vllm_ascend_C*.so | grep -E 'mf_smem|mf310p'
```

## 4. 单层 correctness

先跑默认 q=2（含全 M 融合后的小 M 梯度与 tail/boundary）：

```bash
torchrun --standalone --nproc-per-node=2 \
  benchmarks/scripts/bench_310p_memfabric_mm_ar_layer.py \
  --batch-basem-count 2 \
  --rows 1 16 32 64 255 256 257 511 512 513 1024 2048 4096 6144 8192 \
  --repeat 20
```

再补 q=1/4：

```bash
for q in 1 4; do
  torchrun --standalone --nproc-per-node=2 \
    benchmarks/scripts/bench_310p_memfabric_mm_ar_layer.py \
    --batch-basem-count "$q" \
    --rows 1 16 32 64 255 256 257 511 512 513 1024 2048 4096 \
    --repeat 20
done
```

检查：

- 两 rank bit-exact；
- 1/16/32/64 小 M（尾批补零路径）与 255/257、511/513 tail/boundary；
- multiple batches/waves；
- protocol status 始终 OK；
- 无 hang/trap。

失败后不要继续复用 worker。进程仍可响应时可读：

```python
torch.ops._C_ascend.memfabric_mm_ar_debug_snapshot()
```

## 5. Protocol status

```text
1 READY_TIMEOUT
2 SIGNAL_FAILED
3 QUIET_FAILED
4 WAIT_FAILED
5 MAIL_MISMATCH
6 CREDIT_MISMATCH
7 ACK_FAILED
```

出现非 0 后保存双方日志并重启两个 TP worker。

## 6. Long-run / coexistence

correctness 通过后：

- representative M repeat >=1000；
- mixed-M；
- >60s；
- 10min；
- destroy/recreate；
- HCCL barrier/all_reduce 共存；
- stream/device sync 行为。

这些结论必须绑定当前 MemFabric/CANN/Driver/Firmware/代码 SHA。

## 7. Profiler / overlap

```bash
python3 benchmarks/scripts/analyze_310p_memfabric_overlap.py \
  /path/to/task_time.csv \
  --rows 8192 \
  --batch-basem-count 2 \
  --bandwidth-gbps 20
```

最终必须在 CANN timeline 直接确认：

```text
SDMA(batch n)   || MM(batch n+1)
SDMA(batch n+1) || reduce(batch n)
```

同时单独比较 cooperative producer 的 MM 时间与 stock `F.linear/aclnnMm`；
P0 目标是 median 不劣于 stock 约 5%。

## 8. Qwen eager

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL=/models/Qwen3.6-35B-A3B-w8a8

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3.6-35b-a3b-w8a8 \
  --tensor-parallel-size 2 \
  --quantization ascend \
  --dtype float16 \
  --trust-remote-code \
  --enforce-eager
```

确认 full-attention o_proj（10 层）与 GDN linear_attn.out_proj（30 层）均
命中，且没有 duplicate generic allreduce。

## 9. ACL Graph

Eager 通过后再去掉 `--enforce-eager`。

顺序：

1. eager/profile 完成 context/scratch/protocol/kernel warmup；
2. minimal fused-op capture/replay；
3. repeated replay；
4. 不同 graph bucket；
5. Qwen ACL Graph；
6. repeated requests。

capture 内不允许 create/malloc/host barrier/lazy kernel warmup。

## 10. 性能 A/B/C

```text
A: stock o_proj + HCCL
B: ABI v6 7+1 baseline
C: ABI v7 8-core batch pipeline
```

记录 single-layer、local MM、TTFT、prefill tokens/s、TPOT/ITL、output tokens/s、
peak memory、CANN timeline。

不要把 v6 的历史性能数字写成 v7 结果。v7 的性能结论以当前 commit 实机 A/B/C
为准。

## 11. 已知外部约束

- R6：偶发约 20s engine freeze 仍需 MemFabric/运行时侧继续定位；未解除前不能
  判定生产可用。
- R7：当前先使用 `ASCEND_RT_VISIBLE_DEVICES=0,1`；其它物理 device pair
  待上游 deviceId 授权语义修复/确认后放开。

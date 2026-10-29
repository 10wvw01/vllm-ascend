# 310P3 Qwen3.6 o_proj + 独立 MemFabric MC² 算子实机验证

适用：Ascend 310P3 单卡双 die、TP=2、`Eco-Tech/Qwen3.6-35B-A3B-w8a8` full-attention `self_attn.o_proj`。

## 1. 先构建/安装独立算子

独立仓：`10wvw01/memfabric_matmul_allreduce`。

```bash
cd /path/to/memfabric_matmul_allreduce
source /usr/local/Ascend/cann/set_env.sh
export MEMFABRIC_HYBRID_HOME_PATH=/path/to/installed/memfabric_hybrid
export MAX_JOBS=8
./build.sh

./build_out/custom_opp_*.run --install-path=/opt/memfabric_mc2
source /opt/memfabric_mc2/vendors/memfabric_mc2/bin/set_env.bash
```

安装后先做 ELF/API 自检：

```bash
python3 tests/st/check_install.py
```

再做**不依赖 vLLM**的数据路径与最终 correctness：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MFMC2_STORE_URL=tcp://127.0.0.1:8581
export MFMC2_LOCAL_BYTES=$((96 * 1024 * 1024))

# source rank 非零 / peer rank 全零，双向隔离验证 local MM 与 peer SDMA payload。
python3 tests/st/test_data_path_accuracy.py

# 两 rank 均非零，q=1/2/4 完整矩阵 bit-exact。
python3 tests/st/test_two_rank_accuracy.py

# protocol/arena/generation 重用稳定性。
python3 tests/st/test_repeated_stability.py --q 2 --iterations 1000
```

correctness oracle 不使用 HCCL；HCCL 仅用于性能 baseline。

然后做 standalone head-to-head：

```bash
python3 tests/perf/bench_two_rank.py --q 2 --warmup 20 --iters 100
```

在以上步骤通过前，不进入 vLLM E2E。

> `.run` 已打包 public `libmf_smem.so`；但 310P MemFabric 的系统级 AICPU/orchestrator 前置仍必须按 `memfabric_hybrid:wgm-dev-310p` 的目标机要求部署，并由上述 ST 最终验证。

## 2. 构建 vllm-ascend

vLLM 已不再编译/链接 MemFabric kernel/runtime，因此构建时不需要 `MEMFABRIC_HYBRID_HOME_PATH`。

```bash
cd /path/to/vllm-ascend
export SOC_VERSION=ascend310p3
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export MAX_JOBS=8
python3 -m pip install -v -e . --no-build-isolation --no-deps
```

启动 worker **之前**必须已经 source 独立 OPP 的 `set_env.bash`，因为 custom-op 搜索路径在 C++ 扩展加载时确定。

## 3. Runtime 参数

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1

# external operator-owned runtime
export MFMC2_STORE_URL=tcp://127.0.0.1:8581
export MFMC2_LOCAL_BYTES=$((96 * 1024 * 1024))

# vLLM routing policy
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_BATCH_BASEM_COUNT=2
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_MIN_M=4096

# Graph-safe default: do not skip the fused eager warmup/profile opportunity.
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_WARMUP_FALLBACK=0
```

`q=2 -> batch_m=512 -> N=2048/FP16 payload=2MiB`。合法 q 仅 1/2/4。

`WARMUP_FALLBACK=1` 仅作为 eager 调试/规避初始化 dummy-run 的临时开关；它可能把 MemFabric context 初始化推迟到首次真实请求。**进入 ACL Graph 验证前必须使用 0 并重启 worker，确保至少一次 fused eager 调用已经完成 context/scratch/protocol/kernel warmup。**

旧 `VLLM_ASCEND_310P_MEMFABRIC_STORE_URL` / `...LOCAL_BYTES` 仅由 external runtime ABI 1 作为兼容别名读取；新脚本统一使用 `MFMC2_*`。

## 4. vLLM 可见性/安全回退检查

```bash
python3 - <<'PY'
import torch
import vllm_ascend.vllm_ascend_C  # noqa

for op in [
    "memfabric_matmul_allreduce",
    "memfabric_matmul_allreduce_available",
    "memfabric_matmul_allreduce_shutdown",
    "memfabric_matmul_allreduce_debug_snapshot",
]:
    print(op, hasattr(torch.ops._C_ascend, op))
print("external OPP available:", torch.ops._C_ascend.memfabric_matmul_allreduce_available())
PY
```

预期最后一项为 `True`。

再做负向验证：不 source/remove external OPP 后启动同一 vLLM build，`available()` 必须为 False，目标层不得关闭 `reduce_results`，stock matmul + HCCL 正常工作。

## 5. 独立算子稳定性

correctness 后继续：

- q=1/q=4 分别新进程重复；
- diverged peer GVA；
- >60s；
- 10min；
- HCCL barrier/all_reduce 共存。

任一 protocol/device fault 后双方 worker 全部重启，不复用 poisoned runtime。

## 6. Profiler / 性能 Gate

独立 benchmark 至少覆盖：

```text
M=512,1024,2048,4096,6144,8192
```

基线：stock FP16 NZ linear + HCCL all-reduce。benchmark 的 stock/fused 两侧都在 timed call 内承担 output allocation，避免预分配造成不公平收益。

迁移前 v8 q=2 实测接受目标：

```text
M=2048 约持平
M=4096 -17.0%
M=6144 -24.2%
M=8192 -22.2% (5.258ms vs 6.760ms)
```

这些是**迁移前基线**，不是新结构已通过的结果。

CANN timeline 必须重新确认：

```text
producer(n+1) enqueue before wait/reduce(n)
SDMA(n) || MM(n+1)
```

并确认 eager 路径没有重新出现每波 32KiB ready-cell memset。

## 7. Qwen eager

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
MODEL=/home/models/Qwen3.6-35B-A3B-w8a8

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

检查：

- 仅 full-attention `o_proj` 命中；
- linear-attention/GDN 不命中；
- `M < MIN_M` 走 stock + HCCL；
- `M >= MIN_M` 走 external MC²；
- 命中后没有 duplicate generic HCCL all-reduce；
- feature-off 或 OPP unavailable 均安全 fallback。

至少发起一个 `M >= MIN_M` 的真实 fused eager 调用并确认成功，才能进入 Graph 测试。

## 8. ACL Graph

Eager 通过后去掉 `--enforce-eager`，保持 `WARMUP_FALLBACK=0`。顺序：

1. eager/profile 完成 external runtime context/scratch/protocol/kernel warmup；
2. minimal fused-op capture/replay；
3. M=2048 repeated replay（如路由阈值需要，单算子测试直接调用）；
4. M=8192 repeated replay；
5. replay 后继续 eager；
6. Qwen ACL Graph repeated requests。

capture 内不允许 external runtime 首次 create/malloc/host barrier/lazy warmup。

## 9. 最终实机 Gate

```text
custom OPP build
-> install check
-> isolated local-MM/peer-payload accuracy
-> q=2 correctness
-> q=1/4
-> repeated 1000
-> standalone perf
-> profiler
-> vLLM OPP-missing fallback
-> Qwen eager + fused eager warmup
-> ACL Graph
-> 10min
```

只有全部通过并记录 CANN/Driver/Firmware、memfabric_hybrid SHA、operator SHA、vllm-ascend SHA 后，才判定独立迁移验收完成。

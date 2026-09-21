# 310P3 Qwen3.6 o_proj + MemFabric 使用与实机验证

> 适用：Ascend 310P3 单卡双 die、TP=2、
> `Eco-Tech/Qwen3.6-35B-A3B-w8a8` full-attention `self_attn.o_proj`。
>
> 当前实现：FP16 local o_proj + MemFabric public SHM/SDMA exchange +
> local FP16 reduction。

设计说明：[310p_memfabric_o_proj.md](../../developer_guide/310p_memfabric_o_proj.md)

当前 Gate：[310p_memfabric_o_proj_development_plan.md](../../developer_guide/310p_memfabric_o_proj_development_plan.md)

## 1. 依赖

```text
CANN / Driver / Firmware
      |
      +-- PyTorch + torch-npu
      +-- matching vLLM
      +-- installed memfabric_hybrid:wgm-dev-310p
      |      public host/device headers
      |      libmf_smem.so
      |
      +-- vLLM-Ascend
             csrc/_310P/custom_memfabric_o_proj/
```

MemFabric 是外部黑盒依赖。本功能不要求用户配置其 mailbox、ring、AICPU
orchestrator 或内部库路径。

## 2. 准备 MemFabric 安装环境

按 `memfabric_hybrid:wgm-dev-310p` 当前仓库说明构建/安装，并 source 安装包
提供的环境脚本。

至少确认：

```bash
echo "$MEMFABRIC_HYBRID_HOME_PATH"
test -n "$MEMFABRIC_HYBRID_HOME_PATH"
```

vLLM CMake 会在该安装根下寻找：

- public `smem_shm.h`；
- public `smem_shm_aicore_base_sdma.h`；
- `libmf_smem.so`。

## 3. 编译 vLLM-Ascend

```bash
export SOC_VERSION=ascend310p3
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS=8

export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1

python3 -m pip install -v -e . --no-build-isolation --no-deps
```

feature-on 时 CMake：

1. 验证 `ascend310p*`；
2. 查 MemFabric public headers 与 `libmf_smem.so`；
3. bisheng `--npu-arch=dav-2002` 编译
   `csrc/_310P/custom_memfabric_o_proj/memfabric310p_device.asc`；
4. 生成/链接 `libmf310p_device.so`；
5. 编译 public API adapter 到 `vllm_ascend_C`。

device library 始终由当前 `.asc` 源码构建。

## 4. Runtime 参数

```bash
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M=32
export VLLM_ASCEND_310P_MEMFABRIC_STORE_URL=tcp://127.0.0.1:8581
export VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES=$((32 * 1024 * 1024))
```

`tile_m` 必须是 [16,4096] 内 2 的幂。

## 5. 编译后检查

### Op 注册

```bash
python3 - <<'PY'
import torch
import vllm_ascend.vllm_ascend_C  # noqa

for op in [
    "memfabric_direct_o_proj_allreduce",
    "memfabric_o_proj_shutdown",
    "memfabric_o_proj_debug_snapshot",
]:
    print(op, hasattr(torch.ops._C_ascend, op))
PY
```

预期均为 `True`。

### 动态依赖

```bash
python3 - <<'PY'
import vllm_ascend.vllm_ascend_C as C
print(C.__file__)
PY

ldd /path/to/vllm_ascend_C*.so | grep -E 'mf_smem|mf310p'
```

vLLM 显式依赖 public `libmf_smem.so`；MemFabric 的内部 runtime 依赖由其
安装包自身负责。

## 6. 单层 correctness

先不要启动 35B 模型：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M=32

torchrun --standalone --nproc-per-node=2   benchmarks/scripts/bench_310p_memfabric_o_proj_layer.py   --rows 1 8 32 33 64 128 512 2048 4096   --repeat 20
```

重点：

- M=33 tail；
- M=2048 为 64 chunks；
- M=4096 multiple waves；
- 两 rank result；
- repeated calls；
- 无 protocol trap。

如失败，在进程仍健康时读取：

```python
torch.ops._C_ascend.memfabric_o_proj_debug_snapshot()
```

snapshot 只包含 vLLM 应用层 layout/arena/protocol status。

## 7. Protocol status

```text
1 READY_TIMEOUT
2 SIGNAL_FAILED
3 QUIET_FAILED
4 WAIT_FAILED
5 MAIL_MISMATCH
6 CREDIT_MISMATCH
7 ACK_FAILED
```

出现协议错误后不要继续复用该 worker/context；保存日志并重启两 TP worker。

## 8. Long-run 与平台共存测试

correctness 通过后：

- repeat >=1000；
- mixed M；
- >60s；
- 10min。

并单独验证当前环境上的：

- stream sync；
- device sync；
- HCCL barrier；
- HCCL all_reduce；
- destroy/recreate。

这些行为必须以**当前 MemFabric + 当前 CANN 实测**为准。

## 9. Profiler / overlap

取得 CANN `task_time.csv` 后：

```bash
python3 benchmarks/scripts/analyze_310p_memfabric_overlap.py   /path/to/task_time.csv   --rows 2048   --tile-m 32
```

目标是证明 producer MM workers 仍在执行时，前面已经提交的 chunk 正在进行
SDMA，即 `MM(later) || SDMA(earlier)`。

## 10. Qwen eager

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL=/models/Qwen3.6-35B-A3B-w8a8

vllm serve "$MODEL"   --host 0.0.0.0   --port 8000   --served-model-name qwen3.6-35b-a3b-w8a8   --tensor-parallel-size 2   --quantization ascend   --dtype float16   --trust-remote-code   --enforce-eager
```

检查：

- full-attention o_proj 命中；
- linear-attention/GDN 不命中；
- 无 duplicate generic allreduce；
- prefill/decode 正确；
- repeated request 稳定。

## 11. ACL Graph

Eager 通过后去掉 `--enforce-eager`。

当前合同：

- context/scratch/fixed credit 必须在 capture 前初始化；
- 所需 producer M bucket 必须在 capture 前 warm；
- capture side stream 由 torch-npu 管理；
- capture 内不允许 lazy create、malloc、warmup sync；
- fixed credit 不使用 host wave index。

先做 minimal fused-op capture/replay，再做 Qwen Graph。

## 12. Baseline / 性能

feature-off baseline：

```bash
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=0
```

feature gate 影响 C++ build，baseline/fused 建议保存独立 build，并固定相同模型、
CANN、TP、dtype、输入与调度参数。

记录：

- single-layer latency；
- TTFT；
- prefill latency/tokens/s；
- decode TPOT/ITL；
- output tokens/s；
- peak memory；
- CANN overlap timeline。

## 13. 常见问题

### CMake 找不到 MemFabric

确认：

```bash
echo "$MEMFABRIC_HYBRID_HOME_PATH"
```

并检查安装包包含 public headers 和 `libmf_smem.so`。

### feature 开启但模型没有命中

检查：

```text
qwen3_5_moe_text
full_attention
*.self_attn.o_proj
TP=2
FP16
global K=4096
local K=2048
N=2048
unquantized target layer
```

### Graph capture 报未初始化/未 warm

先完成正常 eager/profile warmup，再 capture。不要在 Graph 内首次创建
MemFabric runtime 或首次 warm producer bucket。

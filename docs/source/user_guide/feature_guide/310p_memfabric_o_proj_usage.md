# 310P3 TP=2 FP16 o_proj + MemFabric 构建与运行手册

> 这是当前融合需求的**操作手册**，只保留可执行步骤和当前平台限制。  
> 当前状态/下一步任务见
> [AI-native 开发状态与执行计划](../../developer_guide/310p_memfabric_o_proj_development_plan.md)。

## 1. 支持范围

当前只支持：

```text
Ascend 310P3 single-card dual-die
TP=2
Eco-Tech/Qwen3.6-35B-A3B-w8a8
full_attention self_attn.o_proj only
unquantized FP16 runtime path
local K=2048, N=2048
custom wgm-dev-310p MemFabric V5
```

分支名中的 `w8a8` 是历史遗留，不代表当前 fused kernel 是 INT8/W8A8。

## 2. 依赖

已验证开发栈：

- CANN 9.1.0；
- PyTorch 2.10.0；
- torch-npu 2.10.0.post4；
- compatible vLLM；
- cmake >= 3.26；
- Ascend 310P3 driver/firmware；
- customized `wgm-dev-310p` MemFabric V5。

使用属于同一物理 310P3 卡的两个健康 die。设备号以当前服务器
`npu-smi info` 为准，不在文档中硬编码。

## 3. 安装定制 MemFabric V5

```bash
git clone <your-memfabric-repo> memfabric_hybrid
cd memfabric_hybrid
git checkout <validated-wgm-dev-310p-commit>

cmake -B build -DXPU_TYPE=NPU -DBUILD_PYTHON=OFF
cmake --build build -j

mkdir -p /opt/memfabric-wgm-dev-310p-v5
cmake --install build --prefix /opt/memfabric-wgm-dev-310p-v5
```

安装后至少应存在：

```text
$MF_ROOT/smem/include/host/smem.h
$MF_ROOT/smem/include/host/smem_shm.h
$MF_ROOT/smem/include/device/smem_shm_aicore_base_sdma.h
$MF_ROOT/smem/lib64/libmf_smem.so
$MF_ROOT/hybm/lib64/libmf_hybm_core.so
$MF_ROOT/acc_links/lib64/libacc_tcp_net.so
$MF_ROOT/hybm/aicpu_kernel/libmf_sdma_orch_v6.json
```

目标服务器当前 V5 自动部署通道不可依赖；AICPU epoch kernel 修改后需要按平台
实际部署方式重新安装到设备 CP1 可搜索路径。任何 KVER 改动都要重新验证 json 与
设备侧 so 一致。

## 4. 构建 vLLM-Ascend

```bash
export SOC_VERSION=ascend310p3
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export COMPILE_CUSTOM_KERNELS=1
export MAX_JOBS=8

export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1

export MF_ROOT=/opt/memfabric-wgm-dev-310p-v5
export VLLM_ASCEND_310P_MEMFABRIC_ROOT="$MF_ROOT"
export VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES="$MF_ROOT/smem/lib64/libmf_smem.so;$MF_ROOT/hybm/lib64/libmf_hybm_core.so;$MF_ROOT/acc_links/lib64/libacc_tcp_net.so"

python3 -m pip install -v -e . --no-build-isolation --no-deps
```

`cmake/memfabric_310p.cmake` 会自动使用
`bisheng --npu-arch=dav-2002` 编译
`csrc/memfabric_o_proj/external/memfabric310p_device.asc`。

feature-on 配置缺少指定 MemFabric root/libs/json 时应 fail fast。

## 5. 运行时环境

```bash
export ASCEND_RT_VISIBLE_DEVICES=<die0>,<die1>
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M=32
export VLLM_ASCEND_310P_MEMFABRIC_STORE_URL=tcp://127.0.0.1:8581
export VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES=$((32 * 1024 * 1024))
export MF_SDMA_ORCH_JSON=$MF_ROOT/hybm/aicpu_kernel/libmf_sdma_orch_v6.json
```

`tile_m` 必须是 [16,4096] 内 2 的幂。当前实机优化基线是 32。

## 6. Build 后 smoke check

### 6.1 op registration

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

三项都应为 `True`。

### 6.2 dynamic link

```bash
python3 -c "import vllm_ascend.vllm_ascend_C as C; print(C.__file__)"
ldd /path/to/vllm_ascend_C*.so | sort
```

确认能解析：

- `libmf_smem.so`
- `libmf_hybm_core.so`
- `libacc_tcp_net.so`
- `libmf310p_device.so`

## 7. L0：source regression

```bash
pytest -q tests/ut/_310p/test_memfabric_o_proj_source.py
```

这是结构性回归，不替代 310P 实机测试。

## 8. L2：单层 310P correctness + latency

先跑这个，不要直接启动 35B：

```bash
torchrun --standalone --nproc-per-node=2   benchmarks/scripts/bench_310p_memfabric_o_proj_layer.py   --rows 1 8 32 33 64 128 512 2048 4096   --repeat 20
```

当前基线应全部 PASS，并覆盖 tail、64-chunk 单 wave 和 4096 multi-wave。

注意 benchmark 的 HCCL reference 全部发生在 MemFabric pool 创建之前。

## 9. L3：V5 overlap 证据

分析工具：

```bash
python3 benchmarks/scripts/analyze_310p_memfabric_overlap.py   /path/to/task_time.csv   --rows 2048   --tile-m 32   --chunk-kib 128   --bandwidth-gbps 20
```

当前平台上采集 profiler 时必须遵守常驻 epoch 限制。推荐流程：

1. burn/建池在 profiler window 外；
2. profiler window 只包 steady fused calls；
3. rank rendezvous 不使用 HCCL；
4. 设备 timeline 先完整 flush；
5. `task_time.csv` 在独立健康进程离线 analyse。

只有当 analyzer 与原始 task-time 都可复核时，才记录 overlap DONE。

## 10. 当前 V5 平台限制

在 epoch 生命周期根治前必须牢记：

- pool 存活后禁止随意执行 device-wide sync；
- torch_npu HCCL collective 可能隐式等待 device；
- 当前服务器存在约 25 秒 epoch launch-timeout 风险；
- shutdown 后不要继续执行设备工作。

因此这些限制当前属于**生产 blocker**，不是“使用技巧”。后续应完成
AI-native 状态文档中的 MF-001/MF-002，而不是永久依赖 benchmark workaround。

## 11. Qwen3.6 eager 验收

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL=/models/Qwen3.6-35B-A3B-w8a8

vllm serve "$MODEL"   --host 0.0.0.0   --port 8000   --served-model-name qwen3.6-35b-a3b-w8a8   --tensor-parallel-size 2   --quantization ascend   --dtype float16   --trust-remote-code   --enforce-eager   --max-model-len 4096   --max-num-seqs 1   --gpu-memory-utilization 0.90
```

当前最终 HEAD 仍需要重新做这项 E2E；旧 commit 的成功结果不算当前验收。

检查：

- 日志出现 MemFabric o_proj enable；
- full-attention o_proj 命中；
- linear-attention/GDN 不命中；
- 不发生第二次 generic allreduce；
- 多轮生成结果稳定。

## 12. ACL Graph

单层/当前 HEAD eager 通过后才移除 `--enforce-eager`。

推荐顺序：

```text
minimal fused-op capture/replay
-> M=1/32/33/2048/4096
-> repeated replay
-> full Qwen3.6 graph
```

Graph 失败时先最小化为单 fused op case，不要立刻改 producer 算法。

## 13. Baseline

feature-off baseline 使用同一模型、TP、dtype、输入和服务参数。

关闭融合后需要使用对应 feature-off build，避免当前进程误加载上一版 extension。

必须保存 baseline/fused 的具体 commit/build 信息。

## 14. 故障定位

### fused call 卡住

先调用：

```text
torch.ops._C_ascend.memfabric_o_proj_debug_snapshot()
```

判断：

- req head/tail 是否推进；
- quiet/arrival stamp 是否推进；
- data 是否落到 recv；
- ack/gate mail 是否符合 wave generation。

如果 mailbox 已完成而宿主仍卡住，优先检查 device-wide synchronize。

### 没有命中融合

逐项检查：

```text
feature gate
model_type
prefix
layer_types[N]
TP=2
FP16
unquantized route
4096 -> local 2048 -> output 2048
```

### sdma orchestration not ready

检查 `MF_SDMA_ORCH_JSON`、设备侧 epoch kernel 部署和 KVER/json 是否匹配。

## 15. 每次硬件迭代必须记录

```text
Task ID:
vLLM-Ascend commit:
MemFabric commit:
AICPU kernel KVER:
CANN:
torch-npu:
device pair:
command:
result:
latency/perf:
profiler artifact:
known blocker:
```

状态只回填到 AI-native 开发状态文档，不再创建新的临时 handoff 文档。

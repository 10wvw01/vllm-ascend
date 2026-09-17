# 310P3 TP=2 W8A8 o_proj + MemFabric AllReduce fusion

This document tracks the intentionally narrow fused path for Qwen3.6-35B-A3B-W8A8 full-attention `o_proj` on one Ascend 310P3 card (two dies, TP=2).

## Fixed contract

- Hardware: Ascend 310P3, one card / two dies.
- Tensor parallel size: exactly 2.
- Model: Qwen3.5/3.6 MoE text trunk, full-attention `self_attn.o_proj` only.
- Full-attention geometry: 16 heads x 256 head-dim = 4096 input channels; hidden size = 2048.
- Per TP rank: `A[M, 2048] x W[2048, 2048] -> Y_local[M, 2048]`.
- Quantization: static W8A8. Existing 310P weight layout, dequant scale and rank-0-only quant bias semantics must be preserved.
- Communication: customized `wgm-dev-310p` MemFabric Hybrid SDMA path; HCCL/MC2 is not used by the fused operator.

## MemFabric build/install choice

For this project, **do not build or install the official/upstream MemFabric**. Build and install the customized `wgm-dev-310p` MemFabric instead, and use that installation as the MemFabric implementation consumed by vLLM-Ascend.

```text
wgm-dev-310p MemFabric source
        |
        | build + install
        v
310P customized MemFabric installation
        |
        | headers + libraries
        v
vLLM-Ascend 310P fused o_proj
```

There is no separately deployed vLLM adapter shared library. The small `mf310p_*` adapter is repo-owned source compiled directly into `vllm_ascend_C`; it only isolates branch-specific MemFabric calls from the rest of vLLM-Ascend.

The current build contract is:

```bash
# Build/install wgm-dev-310p MemFabric first.
export VLLM_ASCEND_310P_MEMFABRIC_ROOT=/path/to/wgm-dev-310p/install

# Use the actual libraries produced/required by that customized build.
export VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES='/abs/path/libA.so;/abs/path/libB.so'

# Until the exact wgm-dev-310p .asc build command is imported into this repo,
# compile this source with the same customized toolchain used by example 08:
#   csrc/memfabric_o_proj/external/memfabric310p_device.asc
export VLLM_ASCEND_310P_MEMFABRIC_DEVICE_OBJECT=/abs/path/memfabric310p_device.o

export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M=32
```

When the feature is enabled, CMake fails during configure if these customized build inputs are missing. Header discovery uses `NO_DEFAULT_PATH`, so a system/upstream MemFabric cannot be selected accidentally.

## Integration point

The normal vLLM `RowParallelLinear` path is:

```text
W8A8 local matmul -> tensor_model_parallel_all_reduce
```

For this feature, `process_weights_after_loading()` enables the special path only when all of these are true:

- model type is `qwen3_5_moe_text`;
- layer prefix is `*.layers.N.self_attn.o_proj`;
- `config.layer_types[N] == "full_attention"`;
- TP=2 and the Qwen3.6-35B-A3B dimensions match;
- static 310P W8A8 scheme is active;
- output dtype is BF16.

The target layer then sets `reduce_results=False`, because the MemFabric path returns an already-reduced TP=2 result. Linear-attention/GDN layers use `linear_attn` rather than `self_attn` and are not selected.

## Phase-1 staged overlap

The current bring-up implementation deliberately reuses the known-correct 310P `torch_npu.npu_quant_matmul` and overlaps earlier communication with later matmuls:

```text
MM[t] -> copy-to-send[t] -> clean/notify[t] -> SDMA[t] -> arrival[t] -> REDUCE[t]
  |
  +--------------------------------------------------------------> MM[t+1]
```

There is no communication wait between tiles. `finish()` joins only after all producer tiles in a wave have been submitted.

The default `tile_m=32` makes one BF16 `[32, 2048]` payload 128 KiB, matching the size already exercised by the supplied customized MemFabric example. A wave has at most 64 chunks; larger M values are split into waves.

This phase has two deliberate temporary costs:

1. one local copy from `npu_quant_matmul` output into symmetric `send` memory;
2. one matmul launch per M tile.

They are removed by the phase-2 producer.

## Memory layout

Do **not** reduce in-place into an MM/SDMA source while the peer may still be reading it.

Each rank has non-overlapping symmetric regions:

```text
send arena        immutable local partial results while SDMA reads
recv/final arena  peer partial results, then local reduced results
arrival flags     peer SDMA completion notification slots
workspace         customized MemFabric mailbox/AICPU orchestration workspace
```

For tile `t`:

```text
rank0.send[t] --SDMA--> rank1.recv[t]
rank1.send[t] --SDMA--> rank0.recv[t]

local final[t] = local send[t] + received peer recv[t]
```

The runtime context, symmetric pool and reduction stream are process-persistent. A partial failed wave poisons the process-local runtime; it is not silently reused because outstanding polling/SQE/flags may still exist.

## Phase-2 target

The production producer is one AscendC/CATLASS-level tiled W8A8 matmul that writes directly into the symmetric send arena:

```text
Cube MM tile[t]
    -> dequant + rank0-only quant_bias
    -> send[t]
    -> cache clean
    -> smem_shm_sdma_notify(t)
    -> continue MM[t+1] immediately
```

The TP=2 SDMA exchange and local reduction state machine stays unchanged. For small decode M, N-panel tiling should be added after the M-tiled correctness path is stable, otherwise M-only tiling may expose no useful overlap.

## Validation

`benchmarks/scripts/bench_310p_memfabric_o_proj.py` is the first hardware bring-up harness. It validates the MemFabric exchange/reduce independently of W8A8 matmul, compares the result with a TP=2 HCCL reference, and reports latency for decode/prefill-sized row counts. HCCL is reference-only; it is not part of the tested MemFabric path.

The required final validation order is:

1. customized MemFabric exchange/reduce correctness;
2. repeated-wave correctness (detect stale flags / reset races);
3. phase-1 W8A8 result vs existing `npu_quant_matmul + TP allreduce`;
4. profiler proof that MM[t+1] overlaps SDMA[t];
5. phase-2 direct producer correctness;
6. Qwen3.6 end-to-end generation correctness and performance.

## Implementation status

Completed in the feature branch:

- strict Qwen3.6 full-attention/TP=2/W8A8 eligibility;
- bypass of the generic RowParallelLinear allreduce only for eligible layers;
- direct compile/link against the explicitly selected `wgm-dev-310p` installation;
- no runtime `dlopen` and no separately deployed bridge `.so`;
- persistent MemFabric runtime state and stable symmetric arenas;
- non-aliasing send/recv layout;
- phase-1 tiled MM -> publish overlap path;
- AICore publish and correctness-baseline reduce kernels;
- runtime poison handling for failed waves;
- import-free source regressions;
- TP=2 hardware communication correctness/latency benchmark.

Needs real 310P/custom-MemFabric source or hardware verification:

- exact `wgm-dev-310p` library names/install layout;
- exact `.asc` compilation command/object format;
- wave-boundary two-rank barrier API;
- exact `smem_shm_sdma_submit` destination/flag increment semantics;
- flag value/order/reuse semantics;
- AICPU mailbox/SQE implementation details;
- vectorized/multicore BF16 reduce;
- phase-2 direct W8A8 matmul producer;
- end-to-end Qwen3.6 measurements.

## Customized MemFabric source needed to remove the remaining ABI assumptions

The supplied example proves the device-side notify/poll flow and host-side create/submit/wait flow, but it does not expose enough implementation detail to safely finalize reusable waves. The minimum additional `wgm-dev-310p` source needed is:

1. `smem_shm_aicore_sdma.h`;
2. the header that declares `smem_shm_sdma_submit`, `smem_shm_sdma_wait`, `smem_shm_sdma_get_workspace`, `smem_shm_sdma_get_result`;
3. the implementation of `smem_shm_sdma_submit` / AICPU mailbox-SQE orchestration;
4. any existing device/host control-barrier API used by the customized branch (if one exists);
5. the CMake/build/install fragment used by example 08 or by the customized library, including the `.asc` compile rule and produced library names.

Until these definitions are checked, `mf310p_prepare_wave()` is not considered final: the example itself uses `MPI_Barrier`, so a MemFabric-specific control-barrier symbol must not be assumed without source verification.

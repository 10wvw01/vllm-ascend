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

## External MemFabric dependency

The required 310P MemFabric implementation is **not** assumed to be part of the upstream/mainline MemFabric package and is **not** assumed to provide any upstream library name or ABI. It is an independently built and installed dependency from the customized `wgm-dev-310p` source tree.

vLLM-Ascend must therefore treat it as an optional external SDK. The intended build contract is:

```bash
export VLLM_ASCEND_MEMFABRIC_310P_ROOT=/path/to/custom/memfabric/install
export VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ=1
export VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M=64
```

When the fusion build is enabled, CMake will consume include/library information from that custom installation prefix (or explicit include/library overrides if the custom build layout requires them). The normal vLLM-Ascend build must remain independent of this dependency.

No library name, include directory layout, soname, or ABI will be hard-coded until the customized branch's build/install files and headers are available. The existing generic Python dependency named `memfabric_hybrid` is not considered sufficient evidence for linking this custom 310P SDMA runtime.

The runtime build must contain the customized MemFabric implementation; otherwise the custom op fails explicitly instead of silently falling back to a second all-reduce.

## Integration point

`RowParallelLinear` normally executes:

```text
W8A8 local matmul -> tensor_model_parallel_all_reduce
```

For an eligible full-attention `o_proj`, `process_weights_after_loading()` marks the layer and sets `reduce_results=False`. The W8A8 scheme then dispatches to one opaque operator:

```text
memfabric_w8a8_o_proj_allreduce(x_q, weight, deq_scale, quant_bias, tp_rank, tile_m)
```

The operator therefore owns both local matmul and TP=2 reduction. All non-target layers retain the original path.

## Pipeline

The required dependency is one-way:

```text
MM[t] -> publish[t] -> SDMA[t] -> arrival[t] -> REDUCE[t]
  |
  +----------------------------> MM[t+1]
```

`MM[t+1]` must never wait for SDMA/reduction of tile `t`. Only the final operator completion waits for all compute and communication/reduction work.

For `T = ceil(M / tile_m)` tiles, each rank performs the same symmetric algorithm:

1. Compute local W8A8 matmul tile `t` into `send[t]`.
2. Clean/publish the produced GM range and notify MemFabric mailbox slot `t`.
3. AICPU/SDMA copies `send_rank[t]` into the peer rank's `recv[t]`.
4. The peer arrival flag releases the local reduction worker for tile `t`.
5. Local reduction computes `recv[t] += send[t]` and the result remains in `recv`.
6. MM continues producing subsequent tiles independently.

This is symmetric TP=2 all-reduce: both ranks send their local contribution once and both ranks locally sum the received peer contribution.

## Memory layout

Do **not** reduce in-place into the MM producer buffer while SDMA may still read it.

Each rank owns two non-overlapping symmetric arenas:

```text
send arena        [max_M, 2048]  BF16/FP16 local contribution, SDMA source
recv/final arena  [max_M, 2048]  peer contribution, then final reduced output
control/flags      generation + arrival/reduce completion metadata
workspace          customized MemFabric SDMA mailbox/AICPU orchestration workspace
```

For tile `t`:

```text
send[t] --SDMA--> peer.recv[t]
recv[t] += send[t]
```

`send[t]` stays immutable until that invocation has completed, so the peer DMA source can never race with the local add. `recv[t]` is never written by the MM producer, so peer SDMA and local compute do not alias.

The runtime should allocate the symmetric pool and workspace once per rank/process and reuse stable addresses. Per-forward pool creation/destruction is forbidden.

## Synchronization rules

- Producer publishes a tile only after matmul output is globally visible to SDMA.
- SDMA/AICPU waits on the producer mailbox; compute does not wait on SDMA.
- Reduction waits on peer arrival for the same tile.
- Buffer reuse across invocations needs an epoch/generation discipline so stale non-zero flags cannot release a later invocation.
- The operator may return only after every tile has completed MM, SDMA arrival and local reduction on both ranks.

The exact flag-generation protocol must follow the customized 310P MemFabric branch ABI; it must not be guessed from the example's one-shot non-zero polling.

## Implementation stages

### Stage 1: protocol bring-up

- Opaque PyTorch custom op and strict eligibility checks.
- Persistent customized-MemFabric TP=2 context.
- Tile-by-tile W8A8 matmul launches into the send arena.
- Small AICore publish/reduce kernels plus AICPU/SDMA pipeline.
- Numerical comparison against the existing local W8A8 matmul + TP all-reduce path.
- Timeline/profiling proof that MM[t+1] overlaps SDMA[t].

### Stage 2: true single-kernel producer

Replace per-tile matmul launches with one AscendC tiled W8A8 matmul producer. It writes each completed M tile directly to the send arena, performs the required cache clean, and issues the customized MemFabric notify from inside the producer kernel. The communication/reduction state machine remains unchanged.

## Required customized MemFabric ABI before runtime code is finalized

The supplied 310P example establishes the intended calls (`smem_shm_sdma_notify`, `smem_shm_sdma_poll_flag`, `smem_shm_sdma_submit`, `smem_shm_sdma_wait`, symmetric SHM segments and the SDMA workspace), but the following branch-specific definitions are required to compile and to make buffer reuse correct:

- `smem_shm_aicore_sdma.h` (or the actual header declaring the notify/poll primitives and workspace constants).
- Host declarations for `smem_shm_sdma_submit`, `smem_shm_sdma_wait`, `smem_shm_sdma_get_workspace` and result APIs.
- The AICPU SDMA orchestration implementation, especially mailbox consumption, destination-offset calculation and flag write semantics.
- The customized branch's CMake/build/install files so the exact produced libraries, include directories, RPATH requirements and link dependencies are known.

Until these definitions are available, the repository intentionally contains the model-side/operator contract but not a guessed MemFabric ABI implementation.

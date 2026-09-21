# CUSTOM 310P3 MemFabric o_proj fusion

> **CUSTOMIZED 310P3 ONLY**
>
> This directory contains project-specific code for the Qwen3.6 full-attention
> `o_proj + TP=2 MemFabric reduction` path on Ascend 310P3. It is intentionally
> isolated from generic `csrc/` code.

## Scope

- Target: Ascend 310P3 / dav-2002
- Model route: Qwen3.6 full-attention `self_attn.o_proj`
- TP: 2
- Compute: FP16 local o_proj partial
- Transport: installed `wgm-dev-310p` MemFabric public SHM/SDMA API only
- MemFabric internals (mailbox/ring/orchestrator/reserved layout) are opaque

## Files

- `memfabric_o_proj_binding.cpp` — PyTorch custom-op registration
- `memfabric_o_proj_runtime.cpp` — host runtime, waves, fixed credit, graph/lifecycle
- `memfabric_o_proj_torch_adpt.h` — torch-facing declarations
- `memfabric310p_adapter_api.h` — internal vLLM adapter ABI
- `memfabric310p_adapter.cpp` — MemFabric public host API bridge
- `memfabric310p_device.asc` — 7 MM workers + 1 communication coordinator,
  public signal/wait/quiet, FP16 add and fail-stop protocol

## Boundary

Code in this directory must not directly use MemFabric request/arrival ring
layout, mailbox offsets, head/tail counters, AICPU orchestrator details, or
reserved-region internals.

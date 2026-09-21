# SPDX-License-Identifier: Apache-2.0
"""Import-free source regressions for the 310P3 MemFabric o_proj path."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "vllm_ascend" / "_310p" / "ops" / "memfabric_o_proj.py"
ENVS = ROOT / "vllm_ascend" / "envs.py"
W8A8 = ROOT / "vllm_ascend" / "_310p" / "quantization" / "methods" / "w8a8_static.py"
MODELSLIM = ROOT / "vllm_ascend" / "_310p" / "quantization" / "modelslim_config.py"
BINDING = ROOT / "csrc" / "memfabric_o_proj_binding.cpp"
RUNTIME = ROOT / "csrc" / "memfabric_o_proj_runtime.cpp"
ADAPTER_API = ROOT / "csrc" / "memfabric_o_proj" / "external" / "memfabric310p_adapter_api.h"
ADAPTER = ROOT / "csrc" / "memfabric_o_proj" / "external" / "memfabric310p_adapter.cpp"
DEVICE = ROOT / "csrc" / "memfabric_o_proj" / "external" / "memfabric310p_device.asc"
MEMFABRIC_CMAKE = ROOT / "cmake" / "memfabric_310p.cmake"


def _func(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in {path}")


def _src(node: ast.AST) -> str:
    return ast.unparse(node)


def test_eligibility_is_deliberately_narrow() -> None:
    src = _src(_func(HELPER, "should_enable_memfabric_o_proj"))
    full_attn = _src(_func(HELPER, "_is_full_attention_prefix"))

    assert "VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ" in src
    assert "_is_full_attention_prefix" in src
    assert "prefix.endswith('.self_attn.o_proj')" in full_attn
    assert "layer_types[layer_idx] == 'full_attention'" in full_attn
    assert "_EXPECTED_TP_SIZE" in src
    assert "_EXPECTED_INPUT_SIZE" in src
    assert "_EXPECTED_INPUT_SIZE_PER_PARTITION" in src
    assert "_EXPECTED_OUTPUT_SIZE" in src
    # Owner decision A (2026-09-18): unquantized o_proj; the model runs FP16
    # on 310P (no torch_dtype in the checkpoint, BF16 NZ linear unsupported).
    assert "torch.float16" in src
    assert "AscendUnquantizedLinearMethod" in src
    assert "AscendW8A8LinearMethod310" not in src


def test_configure_disables_generic_row_parallel_reduce() -> None:
    src = _src(_func(HELPER, "configure_memfabric_o_proj"))
    assert "layer.reduce_results = False" in src
    assert "RowParallelLinear(reduce_results=True)" in src


def test_w8a8_scheme_no_longer_references_memfabric() -> None:
    # Owner decision A: o_proj is unquantized; the W8A8 scheme must not
    # reference the MemFabric pipeline anymore.
    module_src = W8A8.read_text()
    assert "memfabric" not in module_src


def test_modelslim_router_dispatches_eligible_float_layer() -> None:
    src = MODELSLIM.read_text()
    assert "should_enable_memfabric_o_proj(layer)" in src
    assert "make_memfabric_o_proj_linear_method" in src
    assert "MemFabricOProjLinearMethod310" not in src  # built lazily
    # The stock path stays the fallback.
    assert "return AscendUnquantizedLinearMethod()" in src


def test_fused_op_is_the_single_python_route() -> None:
    """V5: the staged begin/publish/finish pipeline is gone; the fused op is
    the only route and the phase-2 opt-in env no longer exists."""
    src = _src(_func(HELPER, "memfabric_o_proj_allreduce"))
    assert "memfabric_direct_o_proj_allreduce" in src
    assert "layer.weight.data" in src
    assert "F.linear" not in src
    module_src = HELPER.read_text()
    for op in (
        "memfabric_o_proj_begin",
        "memfabric_o_proj_publish",
        "memfabric_o_proj_finish",
        "memfabric_o_proj_mark_failed",
    ):
        assert op not in module_src
    assert "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_DIRECT" not in ENVS.read_text()


def test_cpp_binding_registers_fused_op_and_shutdown() -> None:
    src = BINDING.read_text()
    assert "TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)" in src
    assert "memfabric_direct_o_proj_allreduce" in src
    assert "memfabric_o_proj_shutdown" in src
    assert "torch::kPrivateUse1" in src
    for op in (
        "memfabric_o_proj_begin",
        "memfabric_o_proj_publish",
        "memfabric_o_proj_finish",
        "memfabric_o_proj_mark_failed",
    ):
        assert op not in src


def test_runtime_directly_calls_internal_adapter_without_dlopen() -> None:
    src = RUNTIME.read_text()
    assert "mf310p_create(" in src
    assert "mf310p_control_barrier(" in src
    assert "mf310p_direct_producer_async(" in src
    assert "mf310p_wait_mails_async(" in src
    assert "dlopen" not in src
    assert "dlsym" not in src
    assert "VLLM_ASCEND_310P_MEMFABRIC_ADAPTER_SO" not in src
    assert "poisoned" in src


def test_custom_memfabric_headers_are_isolated_to_internal_adapter() -> None:
    runtime = RUNTIME.read_text()
    adapter = ADAPTER.read_text()
    device = DEVICE.read_text()
    # Host-side MemFabric headers are confined to the internal adapter.
    for header in ("smem.h", "smem_shm.h"):
        assert f"#include <{header}>" not in runtime
        assert f"#include <{header}>" in adapter
    # The device cooperation header pulls in AscendC (kernel_operator.h) and is
    # only compilable by the bisheng device toolchain, so it must never leak
    # into host sources. V5 renamed it to smem_shm_aicore_base_sdma.h.
    assert "#include <smem_shm_aicore_base_sdma.h>" not in runtime
    assert "#include <smem_shm_aicore_base_sdma.h>" not in adapter
    assert '#include "smem_shm_aicore_base_sdma.h"' in device
    # The V4 workspace header must not come back.
    assert "smem_shm_aicore_sdma.h" not in device


def test_build_links_only_explicit_wgm_dev_310p_install() -> None:
    src = MEMFABRIC_CMAKE.read_text()
    assert "VLLM_ASCEND_310P_MEMFABRIC_ROOT" in src
    assert "VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES" in src
    assert "VLLM_ASCEND_310P_MEMFABRIC_DEVICE_LIBRARY" in src
    assert "--npu-arch=dav-2002" in src
    assert "NO_DEFAULT_PATH" in src
    assert "memfabric310p_adapter.cpp" in src
    assert "target_link_libraries" in src
    assert "VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ" in src
    # V5 split layout: host headers under smem/include/host, device headers
    # under smem/include/device, epoch launch json under hybm/aicpu_kernel.
    assert "smem/include/host" in src
    assert "smem/include/device" in src
    assert "libmf_sdma_orch_v6.json" in src
    # MF_SDMA_ORCH_JSON must be provided explicitly (auto-discovery assumes a
    # flat lib64 layout and the kfc fallback channel is dead on the target).
    assert "MF_SDMA_ORCH_JSON" in src


def test_adapter_layout_keeps_send_and_recv_non_aliasing() -> None:
    src = ADAPTER.read_text()
    assert "ctx->layout.send_arena = own_segment" in src
    assert "ctx->layout.recv_arena = own_segment + arena_bytes" in src
    assert "ctx->layout.peer_recv_arena = peer_segment + arena_bytes" in src
    # V5: kernels need the pool base (gva), not a V4 workspace.
    assert "ctx->layout.pool_base = reinterpret_cast<uint64_t>(gva)" in src
    # V5 reserved tail is the 48 KiB mailbox-ring region.
    assert "kSdmaReservedRegionSize = 48ULL * 1024ULL" in src


def test_wave_join_is_stream_sync_then_control_barrier() -> None:
    """Cross-wave arena-reuse race regression: both ranks must finish every
    arena access (including the reduced-output add reading recv) before
    either rank's next-wave signals may overwrite the peer's recv arena."""
    runtime = RUNTIME.read_text()
    impl = runtime[runtime.index("memfabric_direct_o_proj_allreduce_impl") :]
    assert impl.index("aclrtSynchronizeStream(stream)") < impl.index("mf310p_control_barrier")
    adapter = ADAPTER.read_text()
    assert "smem_shm_control_barrier(ctx->shm)" in adapter


def test_device_pipeline_uses_signal_wait_quiet_and_host_add() -> None:
    src = DEVICE.read_text()
    # V5 mailbox-ring epoch API replaces notify/poll_flag.
    assert "smem_shm_sdma_signal_at" in src
    assert "smem_shm_sdma_wait_at" in src
    assert "smem_shm_sdma_quiet_at" in src
    assert "smem_shm_sdma_reserved" in src
    assert "smem_shm_sdma_notify" not in src
    assert "smem_shm_sdma_poll_flag" not in src
    # The reduction is the repo-owned multi-block vector-add kernel
    # (shape-agnostic; torch add_out pays a per-shape GE compile), so the V4
    # in-kernel FP16 scalar reduce helpers are gone.
    assert "Mf310pFp16ToFloat" not in src
    assert "mf310pAddKernel" in src
    assert "Add(o, va, vb, n)" in src
    assert "at::add_out(" not in RUNTIME.read_text()


def test_device_waiter_is_quiet_then_ordered_waits() -> None:
    src = DEVICE.read_text()
    waiter = src[src.index("mf310pWaitKernel") :]
    assert waiter.index("smem_shm_sdma_quiet_at") < waiter.index("smem_shm_sdma_wait_at")
    assert "SMEM_SHM_SDMA_MAIL_OK" in waiter
    assert "chunks == 0" in waiter  # side-effect-free warmup shape


def test_adapter_abi_does_not_expose_memfabric_types() -> None:
    src = ADAPTER_API.read_text()
    assert "VLLM_ASCEND_MF310P_ADAPTER_ABI_VERSION 3u" in src
    assert "mf310p_control_barrier" in src
    assert "mf310p_direct_producer_async" in src
    assert "mf310p_wait_mails_async" in src
    assert "smem_shm_t" not in src
    assert "smem_shm_config_t" not in src
    assert "mf310p_context_t" in src
    assert "mf310p_layout_t" in src
    assert "pool_base" in src
    # The V4 wave/workspace ABI is fully removed.
    for gone in (
        "mf310p_prepare_wave",
        "mf310p_submit_wave",
        "mf310p_wait_wave",
        "mf310p_publish_chunk_async",
        "mf310p_launch_reduce_consumer_async",
        "arrival_flags",
    ):
        assert gone not in src


def test_fused_runtime_keeps_wave_protocol_and_producer_order() -> None:
    src = RUNTIME.read_text()
    impl = src[src.index("memfabric_direct_o_proj_allreduce_impl") :]
    # Wave protocol: join (stream sync + control barrier) -> warmup ->
    # producer(s) -> waiter -> add_out, all on the current stream with no
    # host syncs between chunks.
    assert impl.index("mf310p_control_barrier") < impl.index("mf310p_direct_producer_async")
    assert impl.index("mf310p_direct_producer_async") < impl.index("mf310p_wait_mails_async")
    assert impl.index("mf310p_wait_mails_async") < impl.index("mf310p_add_async")
    # The add kernel is warmed with the same double-launch contract.
    assert "mf310p_warmup_add_async" in src
    # The partial tail chunk never reads beyond x: it stages into a scratch
    # copy and the untouched slot rows are zeroed deterministically.
    assert "aclrtMemcpyAsync" in impl
    assert "producer_scratch" in impl
    # First-launch warmup per M-bucket (dav-2002 silent binary-load quirk).
    assert "warm_producer_bucket_locked" in impl
    assert "mf310p_warmup_producer_async" in src
    assert "warm_waiter_locked" in impl
    assert "mf310p_warmup_waiter_async" in src
    # A failed wave poisons the runtime (restart both TP workers).
    assert "state.poisoned = true" in impl


def test_fused_device_producer_fuses_matmul_clean_signal() -> None:
    src = DEVICE.read_text()
    producer = src[src.index("Mf310pDirectProducerKernel") :]
    # The bit-exact dav-2002 matmul recipe (see probe evidence): explicit
    # runtime B transpose, UB local workspace for vec ND2NZ, static tilings
    # with the small-M l1Size override, and line-clean + signal per chunk.
    assert "mm.SetTensorB(bGm, true)" in producer
    assert "mm.SetLocalWorkspace" in producer
    assert "REGIST_MATMUL_OBJ" in producer
    assert "mm.IterateAll(cGm)" in producer
    assert "GetMatmulApiTiling" in src
    assert "128 * 1024" in src  # small-M l1Size override
    assert "kMf310pFunc = {false, true" in src  # enVecND2NZ
    assert "SINGLE_CACHE_LINE" in src  # 64B-line clean posture
    assert "Mf310pCleanRegion(slot, chunkBytes)" in producer
    assert "peerRecvArena" in producer
    # P6 multi-block producer: interleaved chunk ownership plus ordered
    # request-ring posting (global seq, capacity-checked spin) replaces the
    # single-writer signal_at call inside the kernel.
    assert "Mf310pPostSlotOrdered(" in producer
    assert "firstSeq + i" in producer
    # Ordered posting waits for both predecessor order and ring capacity.
    assert "SMEM_SHM_SDMA_RS_REQ_SLOTS" in src
    assert "SMEM_SHM_SDMA_RS_REQ_HEAD" in src
    assert "GetBlockIdx() >= blockCount" in producer
    # chunk_count == 0 is the side-effect-free warmup shape.
    assert "chunkCount == 0" in producer


def test_fused_adapter_validates_bucket_and_slot_capacity() -> None:
    api = ADAPTER_API.read_text()
    adapter = ADAPTER.read_text()
    assert "mf310p_direct_producer_async" in api
    assert "mf310p_warmup_producer_async" in api
    assert "mf310p_warmup_waiter_async" in api
    assert "is_valid_producer_bucket" in adapter
    # The kernel writes m_bucket rows into a tile_m-row slot.
    assert "kOProjWidthElems * sizeof(uint16_t) >" in adapter
    assert "a_advance_rows < m_bucket" in adapter

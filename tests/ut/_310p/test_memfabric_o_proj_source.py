# SPDX-License-Identifier: Apache-2.0
"""Import-free source regressions for the 310P3 MemFabric o_proj path."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "vllm_ascend" / "_310p" / "ops" / "memfabric_o_proj.py"
W8A8 = ROOT / "vllm_ascend" / "_310p" / "quantization" / "methods" / "w8a8_static.py"
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
    assert "torch.bfloat16" in src
    assert "AscendW8A8LinearMethod310" in src


def test_configure_disables_generic_row_parallel_reduce() -> None:
    src = _src(_func(HELPER, "configure_memfabric_o_proj"))
    assert "layer.reduce_results = False" in src
    assert "RowParallelLinear(reduce_results=True)" in src


def test_w8a8_routes_configured_layer_to_memfabric_pipeline() -> None:
    module_src = W8A8.read_text()
    assert "is_memfabric_o_proj_configured(layer)" in module_src
    assert "memfabric_w8a8_o_proj_allreduce(" in module_src
    assert "configure_memfabric_o_proj(layer)" in module_src


def test_phase1_pipeline_is_mm_then_publish_then_single_join() -> None:
    src = _src(_func(HELPER, "memfabric_w8a8_o_proj_allreduce"))
    assert "torch_npu.npu_quant_matmul" in src
    assert "send_tile.narrow(0, 0, rows).copy_(y_tile)" in src
    assert "publish(send, int(chunk_idx))" in src
    assert "finish(recv)" in src
    assert src.index("publish(send, int(chunk_idx))") < src.index("finish(recv)")
    assert "mark_failed(recv, str(exc))" in src


def test_cpp_binding_registers_staged_pipeline_and_failure_marker() -> None:
    src = BINDING.read_text()
    assert "TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)" in src
    for name in (
        "memfabric_o_proj_begin",
        "memfabric_o_proj_publish",
        "memfabric_o_proj_finish",
        "memfabric_o_proj_mark_failed",
    ):
        assert name in src
    assert "torch::kPrivateUse1" in src


def test_runtime_directly_calls_internal_adapter_without_dlopen() -> None:
    src = RUNTIME.read_text()
    assert "mf310p_create(" in src
    assert "mf310p_prepare_wave(" in src
    assert "mf310p_submit_wave(" in src
    assert "mf310p_wait_wave(" in src
    assert "dlopen" not in src
    assert "dlsym" not in src
    assert "VLLM_ASCEND_310P_MEMFABRIC_ADAPTER_SO" not in src
    assert "poisoned" in src
    assert "memfabric_o_proj_mark_failed" in src


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
    # into host sources; the workspace constants it defines reach the adapter
    # through smem_shm.h.
    assert "#include <smem_shm_aicore_sdma.h>" not in runtime
    assert "#include <smem_shm_aicore_sdma.h>" not in adapter
    assert '#include "smem_shm_aicore_sdma.h"' in device


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


def test_adapter_layout_keeps_send_and_recv_non_aliasing() -> None:
    src = ADAPTER.read_text()
    assert "ctx->layout.send_arena = own_segment" in src
    assert "ctx->layout.recv_arena = own_segment + arena_bytes" in src
    assert "ctx->layout.peer_recv_arena = peer_segment + arena_bytes" in src


def test_device_pipeline_uses_notify_poll_and_recv_inplace_reduce() -> None:
    src = DEVICE.read_text()
    assert "smem_shm_sdma_notify" in src
    assert "smem_shm_sdma_poll_flag" in src
    # Reduce is BF16 -> F32 add -> BF16 round via explicit conversion helpers.
    assert "recv[i] = Mf310pFloatToBf16(lhs + rhs)" in src
    assert "Mf310pCleanWords" in src


def test_device_reduce_is_gated_by_produced_and_arrival_flags() -> None:
    """Real-machine race regression: the reduce must not read send[c] before
    the local producer wrote it (produced gate on the mailbox slot words) and
    must invalidate send on its own core across waves (arena reuse)."""
    src = DEVICE.read_text()
    assert "produced gate" in src
    assert "arrival gate" in src
    # Polls the mailbox slot's words field (offset +8) as the produced gate.
    assert "mailbox + static_cast<uint64_t>(c) * SMEM_SHM_SDMA_WS_MAILBOX_SLOT_SIZE +" in src
    assert "sizeof(uint64_t)" in src
    # Cross-wave cache invalidation of the send chunk before reading.
    assert "Invalidate send on this core before reading" in src


def test_adapter_abi_does_not_expose_memfabric_types() -> None:
    src = ADAPTER_API.read_text()
    assert "VLLM_ASCEND_MF310P_ADAPTER_ABI_VERSION 2u" in src
    assert "mf310p_prepare_wave" in src
    assert "smem_shm_t" not in src
    assert "smem_shm_config_t" not in src
    assert "mf310p_context_t" in src
    assert "mf310p_layout_t" in src

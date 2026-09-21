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

INTERNAL_MEMFABRIC_TOKENS = (
    "SMEM_SHM_SDMA_RS_",
    "SMEM_SHM_SDMA_RESERVED",
    "SdmaReadCell",
    "SdmaWriteCell",
    "smem_shm_sdma_reserved",
    "Mf310pPostSlotOrdered",
)


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
    assert "torch.float16" in src
    assert "AscendUnquantizedLinearMethod" in src
    assert "AscendW8A8LinearMethod310" not in src


def test_configure_disables_generic_row_parallel_reduce() -> None:
    src = _src(_func(HELPER, "configure_memfabric_o_proj"))
    assert "layer.reduce_results = False" in src
    assert "RowParallelLinear(reduce_results=True)" in src


def test_w8a8_scheme_no_longer_references_memfabric() -> None:
    assert "memfabric" not in W8A8.read_text()


def test_modelslim_router_dispatches_eligible_float_layer() -> None:
    src = MODELSLIM.read_text()
    assert "should_enable_memfabric_o_proj(layer)" in src
    assert "make_memfabric_o_proj_linear_method" in src
    assert "MemFabricOProjLinearMethod310" not in src
    assert "return AscendUnquantizedLinearMethod()" in src


def test_fused_op_is_the_single_python_route() -> None:
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


def test_memfabric_internals_never_escape_into_vllm_sources() -> None:
    for path in (RUNTIME, ADAPTER_API, ADAPTER, DEVICE):
        src = path.read_text()
        for token in INTERNAL_MEMFABRIC_TOKENS:
            assert token not in src, f"{token} leaked into {path}"


def test_device_uses_only_public_memfabric_data_plane() -> None:
    src = DEVICE.read_text()
    assert '#include "smem_shm_aicore_base_sdma.h"' in src
    assert "smem_shm_sdma_signal(" in src
    assert "smem_shm_sdma_wait(" in src
    assert "smem_shm_sdma_quiet(" in src
    for private_variant in (
        "smem_shm_sdma_signal_at",
        "smem_shm_sdma_wait_at",
        "smem_shm_sdma_quiet_at",
    ):
        assert private_variant not in src


def test_phase_a_separates_multiblock_compute_from_single_publisher() -> None:
    src = DEVICE.read_text()
    producer = src[src.index("Mf310pDirectProducerKernel") :]
    publisher = src[src.index("mf310pPublishKernel") :]

    assert "GetBlockIdx()" in producer
    assert "mm.IterateAll(cGm)" in producer
    assert "Mf310pCleanRegion(slot, chunkBytes)" in producer
    assert "smem_shm_sdma_signal(" not in producer[: producer.index("mf310pPublishKernel")]

    assert "GetBlockIdx() != 0" in publisher
    assert "smem_shm_sdma_signal(gva" in publisher

    launch = src[src.index("#define MF310P_PRODUCER_LAUNCH_CASE") :]
    assert launch.index("Mf310pDirectProducerKernel") < launch.index("mf310pPublishKernel")


def test_waiter_and_wave_credit_use_public_mail_api() -> None:
    src = DEVICE.read_text()

    waiter = src[src.index("mf310pWaitKernel") : src.index("mf310pAddKernel")]
    assert waiter.index("smem_shm_sdma_quiet(gva)") < waiter.index("smem_shm_sdma_wait(gva)")
    assert "SMEM_SHM_SDMA_MAIL_OK" in waiter

    ack = src[src.index("mf310pAckKernel") : src.index("#define MF310P_PRODUCER_LAUNCH_CASE")]
    assert "smem_shm_sdma_signal(gva" in ack
    assert "smem_shm_sdma_wait(gva)" in ack


def test_adapter_abi_v5_is_application_only() -> None:
    src = ADAPTER_API.read_text()
    assert "VLLM_ASCEND_MF310P_ADAPTER_ABI_VERSION 5u" in src
    assert "mf310p_context_t" in src
    assert "mf310p_layout_t" in src
    assert "pool_base" in src
    assert "send_arena" in src
    assert "recv_arena" in src
    assert "peer_recv_arena" in src
    assert "ack_slot" in src
    assert "peer_ack_slot" in src

    for gone in (
        "sdma_workspace",
        "first_seq",
        "smem_shm_t",
        "smem_shm_config_t",
    ):
        assert gone not in src


def test_adapter_uses_public_host_readiness_without_retaining_workspace() -> None:
    src = ADAPTER.read_text()
    assert "smem_shm_sdma_get_workspace(ctx->shm) == nullptr" in src
    assert "sdma_workspace" not in src
    assert "48ULL * 1024ULL" not in src
    assert "smem_shm_get_symmetric_size" in src
    assert "ctx->layout.send_arena = own_segment" in src
    assert "ctx->layout.recv_arena = own_segment + arena_bytes" in src
    assert "ctx->layout.peer_recv_arena = peer_segment + arena_bytes" in src


def test_runtime_has_no_memfabric_sequence_tracking_or_mailbox_dump() -> None:
    src = RUNTIME.read_text()
    assert "posted_seqs" not in src
    assert "own_reserved" not in src
    assert "peer_reserved" not in src
    assert "kReqTail" not in src
    assert "kArrMail" not in src

    assert "mf310p_direct_producer_async(" in src
    assert "mf310p_wait_mails_async(" in src
    assert "mf310p_add_async(" in src
    assert "mf310p_ack_async(" in src
    assert "poisoned" in src


def test_build_consumes_current_memfabric_public_package() -> None:
    src = MEMFABRIC_CMAKE.read_text()
    assert "MEMFABRIC_HYBRID_HOME_PATH" in src
    assert "libmf_smem.so" in src
    assert "smem_shm_aicore_base_sdma.h" in src
    assert "--npu-arch=dav-2002" in src
    assert "memfabric310p_adapter.cpp" in src
    assert "VLLM_ASCEND_ENABLE_310P_MEMFABRIC_O_PROJ" in src

    for legacy in (
        "VLLM_ASCEND_310P_MEMFABRIC_ROOT",
        "VLLM_ASCEND_310P_MEMFABRIC_LIBRARIES",
        "libmf_hybm_core",
        "libacc_tcp_net",
        "libmf_sdma_orch_v6.json",
        "MF_SDMA_ORCH_JSON",
    ):
        assert legacy not in src


def test_matmul_recipe_and_add_kernel_are_preserved() -> None:
    src = DEVICE.read_text()
    assert "mm.SetTensorB(bGm, true)" in src
    assert "mm.SetLocalWorkspace" in src
    assert "REGIST_MATMUL_OBJ" in src
    assert "GetMatmulApiTiling" in src
    assert "128 * 1024" in src
    assert "kMf310pFunc = {false, true" in src
    assert "mf310pAddKernel" in src
    assert "Add(o, va, vb, n)" in src
    assert "at::add_out(" not in RUNTIME.read_text()


def test_tail_path_remains_bounds_safe() -> None:
    impl = RUNTIME.read_text()
    impl = impl[impl.index("memfabric_direct_o_proj_allreduce_impl") :]
    assert "producer_scratch" in impl
    assert "aclrtMemcpyAsync" in impl
    assert "tail scratch memset failed" in impl
    assert "tail slot pad memset failed" in impl

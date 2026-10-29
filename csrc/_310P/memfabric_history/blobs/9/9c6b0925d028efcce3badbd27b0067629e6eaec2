# SPDX-License-Identifier: Apache-2.0
"""Import-free source regressions for external 310P3 MemFabric MC2 routing."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "vllm_ascend" / "_310p" / "ops" / "memfabric_o_proj.py"
ENVS = ROOT / "vllm_ascend" / "envs.py"
MODELSLIM = ROOT / "vllm_ascend" / "_310p" / "quantization" / "modelslim_config.py"
ADAPTER = ROOT / "csrc" / "_310P" / "memfabric_matmul_allreduce_adapter.cpp"
CMAKE = ROOT / "CMakeLists.txt"
USAGE = (
    ROOT
    / "docs"
    / "source"
    / "user_guide"
    / "feature_guide"
    / "310p_memfabric_o_proj_usage.md"
)
OLD_CUSTOM_CSRC = ROOT / "csrc" / "_310P" / "custom_memfabric_o_proj"
OLD_MEMFABRIC_CMAKE = ROOT / "cmake" / "memfabric_310p.cmake"


def _func(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in {path}")


def _src(node: ast.AST) -> str:
    return ast.unparse(node)


def test_embedded_memfabric_implementation_is_removed_from_vllm() -> None:
    assert ADAPTER.is_file()
    assert not OLD_CUSTOM_CSRC.exists()
    assert not OLD_MEMFABRIC_CMAKE.exists()
    cmake = CMAKE.read_text()
    assert "memfabric_matmul_allreduce_adapter.cpp" in cmake
    assert "memfabric_310p.cmake" not in cmake
    assert "MEMFABRIC_HYBRID_HOME_PATH" not in cmake
    assert "libmf_smem" not in cmake
    assert "memfabric310p_device" not in cmake


def test_thin_adapter_uses_dynamic_acl_nn_api_only() -> None:
    src = ADAPTER.read_text()
    assert "GetOpApiFuncAddr" in src
    assert "aclnnMemFabricMatmulAllReduceGetWorkspaceSize" in src
    assert "aclnnMemFabricMatmulAllReduce" in src
    assert "mfmc2RuntimeAbiVersion" in src
    assert "kExpectedRuntimeAbi = 1u" in src
    assert "EXEC_NPU_CMD(aclnnMemFabricMatmulAllReduce" in src
    assert "MEMFABRIC_HYBRID_HOME_PATH" not in src
    assert "smem_" not in src
    assert "libmf_smem" not in src
    assert "bisheng" not in src


def test_thin_adapter_preserves_nz_weight_address_contract() -> None:
    src = ADAPTER.read_text()
    assert "weight.const_data_ptr()" in src
    assert "weight.is_contiguous()" not in src
    assert "x.is_contiguous()" not in src
    assert "queries tensor internals between phase 1 and phase 2" in src


def test_operator_availability_is_a_routing_precondition() -> None:
    src = _src(_func(HELPER, "should_enable_memfabric_o_proj"))
    available = _src(_func(HELPER, "_external_op_available"))
    assert "VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ" in src
    assert "_external_op_available()" in src
    assert "memfabric_matmul_allreduce_available" in available


def test_eligibility_remains_deliberately_narrow() -> None:
    src = _src(_func(HELPER, "should_enable_memfabric_o_proj"))
    full_attn = _src(_func(HELPER, "_is_full_attention_prefix"))
    assert "prefix.endswith('.self_attn.o_proj')" in full_attn
    assert "layer_types[layer_idx] == 'full_attention'" in full_attn
    assert "_EXPECTED_TP_SIZE" in src
    assert "_EXPECTED_INPUT_SIZE_PER_PARTITION" in src
    assert "_EXPECTED_OUTPUT_SIZE" in src
    assert "torch.float16" in src
    assert "AscendUnquantizedLinearMethod" in src


def test_reduction_is_disabled_only_after_external_op_is_available() -> None:
    configure = _src(_func(HELPER, "configure_memfabric_o_proj"))
    should = _src(_func(HELPER, "should_enable_memfabric_o_proj"))
    assert "_external_op_available()" in should
    assert "enabled = should_enable_memfabric_o_proj(layer)" in configure
    assert "if not enabled" in configure
    assert "layer.reduce_results = False" in configure
    assert configure.index("if not enabled") < configure.index("layer.reduce_results = False")


def test_python_plan_is_base_m_batch_based() -> None:
    module = HELPER.read_text()
    plan = module[module.index("class MemFabricOProjPlan") : module.index("def _external_op_available")]
    assert "_BASE_M = 256" in module
    assert "batch_basem_count" in plan
    assert "def batch_m" in plan
    assert "def batch_bytes" in plan
    assert "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_BATCH_BASEM_COUNT" in plan
    assert "q not in (1, 2, 4)" in plan
    assert "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M" not in module


def test_fused_python_route_calls_generic_external_operator() -> None:
    src = _src(_func(HELPER, "memfabric_o_proj_allreduce"))
    assert "memfabric_matmul_allreduce" in src
    assert "memfabric_direct_o_proj_allreduce" not in src
    assert "layer.weight.data" in src
    assert "plan.batch_basem_count" in src
    assert "F.linear" not in src


def test_small_m_and_warmup_fallback_keep_hccl_reduction() -> None:
    module = HELPER.read_text()
    apply_src = module[module.index("def apply(") : module.index("return MemFabricOProjLinearMethod310")]
    assert "memfabric_o_proj_warmup_fallback_active()" in apply_src
    assert "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_MIN_M" in apply_src
    assert "tensor_model_parallel_all_reduce" in apply_src
    assert "super().apply(layer, x, bias)" in apply_src


def test_modelslim_router_still_dispatches_eligible_float_layer() -> None:
    src = MODELSLIM.read_text()
    assert "should_enable_memfabric_o_proj(layer)" in src
    assert "make_memfabric_o_proj_linear_method" in src
    assert "return AscendUnquantizedLinearMethod()" in src


def test_vllm_env_owns_routing_not_memfabric_transport() -> None:
    src = ENVS.read_text()
    assert "VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ" in src
    assert "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_BATCH_BASEM_COUNT" in src
    assert "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_MIN_M" in src
    assert "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_WARMUP_FALLBACK" in src
    assert 'VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_WARMUP_FALLBACK", "0"' in src
    assert "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TRACE" in src
    assert "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_TILE_M" not in src
    assert "VLLM_ASCEND_310P_MEMFABRIC_STORE_URL" not in src
    assert "VLLM_ASCEND_310P_MEMFABRIC_LOCAL_BYTES" not in src


def test_graph_validation_uses_eager_runtime_warmup() -> None:
    usage = USAGE.read_text()
    assert "VLLM_ASCEND_310P_MEMFABRIC_O_PROJ_WARMUP_FALLBACK=0" in usage
    assert "至少一次真实 fused eager 调用" in usage
    assert "capture 内不允许 external runtime 首次 create/malloc/host barrier/lazy warmup" in usage


def test_usage_requires_external_opp_before_vllm_start() -> None:
    usage = USAGE.read_text()
    assert "source /opt/memfabric_mc2/vendors/memfabric_mc2/bin/set_env.bash" in usage
    assert "source 独立 OPP" in usage
    assert "启动 worker" in usage

# SPDX-License-Identifier: Apache-2.0
"""Import-free source regressions for the 310P3 MemFabric o_proj path."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "vllm_ascend" / "_310p" / "ops" / "memfabric_o_proj.py"
W8A8 = ROOT / "vllm_ascend" / "_310p" / "quantization" / "methods" / "w8a8_static.py"
BINDING = ROOT / "csrc" / "memfabric_o_proj_binding.cpp"


def _func(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in {path}")


def _src(node: ast.AST) -> str:
    return ast.unparse(node)


def test_eligibility_is_deliberately_narrow() -> None:
    src = _src(_func(HELPER, "should_enable_memfabric_o_proj"))

    assert "VLLM_ASCEND_310P_ENABLE_MEMFABRIC_O_PROJ" in src
    assert "prefix.endswith('.self_attn.o_proj')" in src
    assert "_EXPECTED_TP_SIZE" in src
    assert "_EXPECTED_INPUT_SIZE" in src
    assert "_EXPECTED_INPUT_SIZE_PER_PARTITION" in src
    assert "_EXPECTED_OUTPUT_SIZE" in src
    assert "AscendW8A8LinearMethod310" in src


def test_configure_disables_generic_row_parallel_reduce() -> None:
    src = _src(_func(HELPER, "configure_memfabric_o_proj"))

    assert "layer.reduce_results = False" in src
    assert "RowParallelLinear(reduce_results=True)" in src


def test_w8a8_routes_configured_layer_to_fused_op() -> None:
    module_src = W8A8.read_text()

    assert "is_memfabric_o_proj_configured(layer)" in module_src
    assert "memfabric_w8a8_o_proj_allreduce(" in module_src
    assert "configure_memfabric_o_proj(layer)" in module_src
    assert module_src.index("memfabric_w8a8_o_proj_allreduce(") < module_src.index("torch_npu.npu_quant_matmul(")


def test_310p_cpp_binding_registers_opaque_operator() -> None:
    src = BINDING.read_text()

    assert "TORCH_LIBRARY_FRAGMENT(_C_ascend, ops)" in src
    assert "memfabric_w8a8_o_proj_allreduce" in src
    assert "torch::kPrivateUse1" in src

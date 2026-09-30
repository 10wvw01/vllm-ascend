# SPDX-License-Identifier: Apache-2.0
"""Source regressions for 310P MemFabric/DFlash runner lifecycle hooks.

These tests are intentionally import-free so they can run on a host without
an NPU or torch_npu. They protect the graph lifecycle contract that profile
runs may fall back to stock matmul+HCCL, but graph pre-capture eager warmups
must execute MemFabric so graph-only execution never performs lazy setup.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HOOKS = ROOT / "vllm_ascend" / "_310p" / "ops" / "memfabric_mm_ar_runner_hooks.py"
RUNNER = ROOT / "vllm_ascend" / "_310p" / "model_runner_310p.py"


def _function(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found in {path}")


def test_310p_dummy_run_exposes_profile_and_graph_capture_phase() -> None:
    dummy = _function(RUNNER, "_dummy_run")
    args = {arg.arg for arg in dummy.args.args}
    assert "is_profile" in args
    assert "is_graph_capturing" in args


def test_warmup_fallback_is_profile_only_not_all_dummy_runs() -> None:
    install_src = ast.unparse(_function(HOOKS, "install_memfabric_mm_ar_runner_hooks"))

    # Bind through the original signature so positional and keyword callers
    # get identical phase classification.
    assert "signature(original_dummy_run)" in install_src
    assert 'bound.arguments.get("is_profile", False)' in install_src

    wrapper = _function(HOOKS, "_dummy_run_with_memfabric")
    wrapper_src = ast.unparse(wrapper)
    assert "if is_profile:" in wrapper_src
    assert "with memfabric_mm_ar_warmup_fallback():" in wrapper_src

    # The non-profile path must call the original directly. This is the graph
    # pre-capture eager-warmup path (is_graph_capturing remains False there),
    # so MemFabric lazy state is initialized before actual graph capture.
    assert wrapper_src.rstrip().endswith("return original_dummy_run(self, *args, **kwargs)")


def test_sync_hook_preserves_original_runner_contract_before_pool_start() -> None:
    install_src = ast.unparse(_function(HOOKS, "install_memfabric_mm_ar_runner_hooks"))
    assert "original_sync_device = NPUModelRunner310._sync_device" in install_src

    wrapper_src = ast.unparse(_function(HOOKS, "_sync_device_with_memfabric"))
    assert "if memfabric_mm_ar_pool_started():" in wrapper_src
    assert "torch.npu.current_stream().synchronize()" in wrapper_src
    assert "return original_sync_device(self)" in wrapper_src
    assert "torch.npu.synchronize()" not in wrapper_src

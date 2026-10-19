# SPDX-License-Identifier: Apache-2.0
"""Import-free G3 regressions for the opt-in V2 large-M Qwen serving path."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
V2PY = ROOT / "vllm_ascend/_310p/ops/memfabric_mm_ar_v2_serve.py"
V1PY = ROOT / "vllm_ascend/_310p/ops/memfabric_mm_ar.py"
ROUTER = ROOT / "vllm_ascend/_310p/quantization/modelslim_config.py"
HOOKS = ROOT / "vllm_ascend/_310p/ops/memfabric_mm_ar_runner_hooks.py"
V2SRC = ROOT / "csrc/_310P/memfabric_mm_ar_v2/src"
V2CPP = V2SRC / "torch_adapter/mf310p_v2_serve.cpp"
V2ADAPTER = V2SRC / "op_host/mf310p_v2_trial_adapter.cpp"
V2DEVICE = V2SRC / "op_kernel/mf310p_v2_trial_device.asc"
V2MANIFEST = V2SRC / "CMakeLists.txt"
V2BUILD = ROOT / "cmake/memfabric_310p_v2.cmake"


def test_v2_route_boundary_is_exact_and_uses_stock_outside_fused_interval():
    module = ast.parse(V2PY.read_text())
    func = next(node for node in module.body if isinstance(node, ast.FunctionDef)
                and node.name == "v2_threshold_route")
    namespace = {"MIN_FUSED_M": 2048, "MAX_FUSED_M": 4096}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func],
                 type_ignores=[])), str(V2PY), "exec"), namespace)
    route = namespace["v2_threshold_route"]
    assert [(m, route(m)) for m in (0, 1, 10, 1024, 2047, 2048, 2049, 3073, 4096, 4097, 5000, 8192, 8193)] == [
        (0, "stock"), (1, "stock"), (10, "stock"),
        (1024, "stock"), (2047, "stock"),
        (2048, "v2"), (2049, "v2"), (3073, "v2"), (4096, "v2"),
        (4097, "stock"), (5000, "stock"), (8192, "stock"), (8193, "stock"),
    ]
    try:
        route(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative M must fail")
    # Surprising M>4096 must be an ordinary Stock route, not an
    # exception or a hidden V2 multi-wave fallback.
    assert all(route(m) == "stock" for m in (4097, 5000, 8192, 8193))


def test_all_stock_shapes_restore_original_tp_reduce_exactly_once():
    code = V2PY.read_text()
    apply = code[code.index("        def apply("):]
    assert "v2_threshold_route(int(x.shape[0])) == \"stock\"" in apply
    assert "partial = super().apply(layer, x, bias)" in apply
    assert "return tensor_model_parallel_all_reduce(partial)" in apply
    assert 'memfabric_mm_ar_v2_allreduce' in apply
    assert "return out" in apply
    assert "tensor_model_parallel_all_reduce(out)" not in apply
    assert "MIN_FUSED_M = 2048" in code
    assert "MAX_FUSED_M = 4096" in code
    assert "tensor_model_parallel_all_reduce(out)" not in apply


def test_v1_route_is_unchanged_without_opt_in():
    router = ROUTER.read_text()
    assert 'VLLM_ASCEND_310P_MEMFABRIC_MM_AR_V2_SERVE' in router
    assert 'make_memfabric_mm_ar_v2_linear_method' in router
    assert 'make_memfabric_mm_ar_linear_method()' in router
    assert '"MemFabricMmArV2LinearMethod310"' in V1PY.read_text()
    hook = HOOKS.read_text()
    assert "memfabric_mm_ar_pool_started() or v2_pool_started()" in hook


def test_v2_native_op_is_separately_compiled_and_graph_fail_closed():
    cpp = V2CPP.read_text()
    cmake = V2BUILD.read_text()
    assert 'include("${_dir}/CMakeLists.txt")' in cmake
    manifest = V2MANIFEST.read_text()
    assert "mf310p_v2_serve.cpp" in manifest
    assert "mf310p_v2_trial_adapter.cpp" in manifest
    assert "mf310p_v2_trial_device.asc" in manifest
    assert V2DEVICE.exists()
    assert "memfabric_mm_ar_v2_allreduce(Tensor x, Tensor weight" in cpp
    assert "currentStreamCaptureStatusMayInitCtx" in cpp
    assert "FULL_DECODE_ONLY" in cpp
    assert "MTE3" in cpp
    assert "mf310p_v2_trial_run(" in cpp
    assert "mf310p_v2_trial_run_ar_only(" not in cpp


def test_pool_outputs_are_preallocated_and_live_views_not_overwritten():
    cpp = V2CPP.read_text()
    before = cpp.index("s.output_slots.reserve(")
    create = cpp.index("mf310p_v2_trial_create(")
    assert before < create
    # torch 2.10: TensorBase::storage() returns const Storage& (no
    # temporary ref), so an idle slot's StorageImpl count is exactly 1
    # (the state vector entry); any view held by model code raises it.
    assert "slot.storage().use_count() == 1" in cpp
    assert "use_count() == 2" not in cpp
    assert "V2 pooled outputs exhausted" in cpp
    assert "warm_outputs_locked(s, x, w, stream)" in cpp
    assert "check_protocol(s, \"prewarm\")" in cpp
    # The unaligned-tail staging path must also be warmed during init:
    # the cold-orchestrator incident hit the first staged wave first.
    assert "check_protocol(s, \"staged-tail prewarm\")" in cpp
    assert "s.input_stage.narrow(0, 0, tail_rows)" in cpp
    assert "std::atexit(&memfabric_mm_ar_v2_shutdown)" in cpp
    assert "output_slots.clear()" in cpp
    assert "memfabric_mm_ar_v2_shutdown()" in cpp
    adapter = V2ADAPTER.read_text()
    assert "trial_warmed_ys[64]" in adapter
    assert "trial_warmed_y_next" in adapter


def test_first_fused_call_is_numerically_verified_per_shape_class():
    code = V2PY.read_text()
    assert "_verified_aligned = False" in code
    assert "_verified_padded = False" in code
    assert "torch.equal(out, reference)" in code
    assert "first fused call verification FAILED" in code
    # Verification recomputes the stock path (native MM + HCCL AR) once
    # per shape class and must not turn into a per-call tax.
    assert code.count("needs_verify") >= 2
    assert "super().apply(layer, x, bias))" in code


def test_g3_tail_is_padded_and_multi_window_is_out_of_scope():
    cpp = V2CPP.read_text()
    assert "kMaxWindowRows = 4096" in cpp
    assert "V2 serving requires M <=4096 per Forward" in cpp
    assert "std::max<int64_t>(" in cpp
    assert "kThreshold," in cpp
    assert "s.input_stage.narrow(0, 0, rows).copy_(" in cpp
    assert "for (int64_t start = 0; start < rows; start += kMaxWindowRows)" not in cpp
    assert "return slot.narrow(0, 0, rows);" in cpp
    assert "src.narrow(0, 0, rows)" not in cpp
    assert "--max-num-batched-tokens" in cpp


def test_v2_opt_in_is_disabled_by_default():
    envs = (ROOT / "vllm_ascend/envs.py").read_text()
    assert 'VLLM_ASCEND_310P_MEMFABRIC_MM_AR_V2_SERVE' in envs
    assert 'VLLM_ASCEND_310P_MEMFABRIC_V2_CHUNK_ROWS' in envs
    assert 'VLLM_ASCEND_310P_MEMFABRIC_V2_OUTPUT_SLOTS' in envs
    assert 'os.getenv("VLLM_ASCEND_310P_MEMFABRIC_MM_AR_V2_SERVE", "0")' in envs

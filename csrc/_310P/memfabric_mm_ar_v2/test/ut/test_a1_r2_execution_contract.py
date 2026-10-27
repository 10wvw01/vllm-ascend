# SPDX-License-Identifier: Apache-2.0
"""R2实机工具CPU静态/安全门禁回归：Linux bash-n、Python AST和关键拒绝条件。"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

PERF = Path(__file__).resolve().parents[1] / "perf"
PYFILES = (
    "a1_nz_stock_profile_probe.py",
    "a1_nz_route_gate.py",
    "a1_r1_l2_budget.py",
    "a1_r1_tail_analyze.py",
    "a1_r2_diag_assess.py",
    "a1_r2_epoch_assess.py",
    "u1_analyze.py",
    "u1_arm.py",
)
SHELL = (
    "a1_r2_nz_route_run.sh", "a1_r2_diag_run.sh",
    "a1_r2_epoch_run.sh", "a1_r2_one_batch.sh",
    "a1_prefill_serve_session.sh", "u1_run.sh",
)


@pytest.mark.parametrize("name", PYFILES)
def test_r2_sources_parse_and_have_no_global_device_reset(name):
    src=(PERF/name).read_text(encoding="utf-8")
    ast.parse(src, filename=name)
    for bad in ("npu-smi reset", "mini_kfc rm", "git reset --hard",
                "os.system(", "os.replace("):
        assert bad not in src


@pytest.mark.parametrize("name", SHELL)
def test_r2_bash_syntax(name):
    p=PERF/name
    assert p.is_file()
    result=subprocess.run(["bash","-n",str(p)],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_machine_gates_are_not_all_candidate_path_sha():
    route=(PERF/"a1_r2_nz_route_run.sh").read_text()
    assert "nz_route_verified.json" in route
    assert "NZ_STOCK_ROUTE_VERIFIED" in route
    assert "U1_EXCLUSIVE_APPROVED" in route
    assert "rank${r}_m${m}_kernel_details.csv" in route
    profile=(PERF/"a1_nz_stock_profile_probe.py").read_text()
    assert "tensorboard_trace_handler" in profile
    assert "ASCEND_PROFILER_OUTPUT" in profile
    assert "kernel_details.csv" in profile
    assert "UNVERIFIED_UNTIL_FOUR_RANK_M_ROUTE_GATE" in profile


def test_epoch_budget_controls_scope_and_no_causal_overclaim():
    wrapper=(PERF/"a1_r2_epoch_run.sh").read_text()
    assert "E15A:15 E5A:5 E5B:5 E15B:15" in wrapper
    assert "MF_SDMA_ORCH_EPOCH_BUDGET_S" in wrapper
    assert "MF_KFC_BYPASS" in wrapper
    service=(PERF/"a1_prefill_serve_session.sh").read_text()
    assert "A1_R2_EPOCH_ARM" in service
    assert "epoch_budget_source" in service
    assessor=(PERF/"a1_r2_epoch_assess.py").read_text()
    assert '"actual_epoch_boundary_causality_proven": False' in assessor
    assert "FREQUENCY_SHIFT_SUPPORTS_EPOCH_HYPOTHESIS_ONLY" in assessor


def test_diag_does_not_claim_true_d4_inner_phase_time():
    runner=(PERF/"a1_r2_diag_run.sh").read_text()
    assert "formal loop4 diag0 diag4 diag5" in runner
    assert "A1_R2_SHORT_PROFILER" in runner
    analysis=(PERF/"a1_r2_diag_assess.py").read_text()
    assert '"inner_phase_status": "INNER_PHASE_UNKNOWN"' in analysis
    assert '"no_algebraic_phase_breakdown": True' in analysis


def test_batch_never_runs_l4_and_is_gated():
    runner=(PERF/"a1_r2_one_batch.sh").read_text()
    for stage in ("NZ_REAL_ROUTE","NZ_STOCK_SMOKE","NZ_STOCK_SCREEN",
                  "NZ_STOCK_CONFIRM","D4_INNER_MECHANISM","EPOCH_ABBA"):
        assert stage in runner
    assert "U1_G0_APPROVED" in runner and "U1_RESTORE_PLAN_VERIFIED" in runner
    assert "MF_SDMA_ORCH_EPOCH_BUDGET_S" in runner
    assert "L4_" not in runner
    assert "STOP_STAGE.json" in runner


def test_production_code_not_edited_for_diagnostic():
    root=PERF.parents[1]
    # Only a static scope invariant; GitHub PR diff must be independently reviewed.
    assert (root/"src/op_kernel/mf310p_v2_kernel.asc").is_file()
    assert (root/"src/op_kernel/mf310p_v2_diagnostics.asc").is_file()

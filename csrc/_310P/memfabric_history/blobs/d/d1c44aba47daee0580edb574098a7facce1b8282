# SPDX-License-Identifier: Apache-2.0
"""Seat ownership, tail self-dependency and same-batch statistics contracts."""

import importlib.util
import random
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def test_actual_leading_seat_rule(tmp_path):
    import subprocess

    kernel = (HERE.parent / "src/op_kernel/mf310p_v3_kernel.asc").read_text()
    rule = kernel[
        kernel.index("__aicore__ inline uint32_t A3LeadingSeats") : kernel.index(
            "template <const MatmulApiStaticTiling& TILING, uint32_t MM_ROWS, bool MM>\n"
            "__aicore__ inline void A3ProduceSeat"
        )
    ]
    source = tmp_path / "rule.cpp"
    source.write_text(
        "#include <cstdint>\n#include <cassert>\n#define __aicore__\n"
        + rule
        + r"""
int main(){for(unsigned bands=1;bands<=64;bands++){
assert(A3LeadingSeats(bands,false)==bands);
assert(bands-A3LeadingSeats(bands,true)==(bands+7)/8);}}
"""
    )
    subprocess.run(["g++", str(source), "-o", str(tmp_path / "rule")], check=True)
    subprocess.run([str(tmp_path / "rule")], check=True)


@pytest.mark.parametrize("tail", [False, True])
def test_random_progress_seat_coverage_and_no_self_wait(tail):
    for rows in (512, 1024, 2048, 3072, 4096):
        for h in (64, 128):
            for comm in (512, 1024, 2048):
                if rows % comm:
                    continue
                bands = rows // h
                leading = bands - ((bands + 7) // 8 if tail else 0)
                tasks = {core: list(range(core - 1, leading, 7)) for core in range(1, 8)}
                owners = {band: core for core, seq in tasks.items() for band in seq}
                owners.update({band: 0 for band in range(leading, bands)})
                assert sorted(owners) == list(range(bands))
                assert sum(map(len, tasks.values())) + (bands - leading) == bands
                for seed in range(32):
                    rng = random.Random(seed)
                    positions = {core: 0 for core in tasks}
                    done = set()
                    next_tail = leading
                    g = 0
                    for _ in range(1000):
                        end = (g + 1) * comm // h
                        # Actual scheduler completes required self-owned tail before wait.
                        if tail and next_tail < end:
                            done.add(next_tail)
                            next_tail += 1
                        elif all(b in done for b in range(g * comm // h, end)):
                            g += 1
                            if tail and next_tail < bands:
                                done.add(next_tail)
                                next_tail += 1
                            if g == rows // comm:
                                break
                        else:
                            candidates = [c for c in tasks if positions[c] < len(tasks[c])]
                            assert candidates, "scheduler waited for an unproduced self seat"
                            c = rng.choice(candidates)
                            done.add(tasks[c][positions[c]])
                            positions[c] += 1
                    assert g == rows // comm and done == set(range(bands))


def test_dedicated_scheduler_precedes_matmul_registration():
    source = (HERE.parent / "src/op_kernel/mf310p_v3_kernel.asc").read_text()
    body = source[source.index("void A3SeatKernel(") : source.index("__global__ __aicore__ void V3RawAddKernel")]
    assert body.index("if constexpr (!TAIL)") < body.index("REGIST_MATMUL_OBJ")
    assert "band = core - 1; band < leading; band += 7" in body
    assert body.index("while (nextTail < end)") < body.index("A3WaitSeats(flags, g * commRows / h, end")


def load_compare(monkeypatch):
    tools = HERE.parent / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("a3_compare", tools / "compare_a3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reports(mod, drift=False):
    result = {}
    for compute in (512, 1024):
        for label, _ in mod.ORDER:
            latency = {"a3-before": 1000, "a3-after": 1100 if drift else 1000, "a3-2": 900, "a3-tail": 950}[label]
            result[(label, compute)] = {
                "status": "HISTORICAL_COMPARISON_ONLY",
                "formal_performance_go": False,
                "paired_sessions": [
                    {"M": m, "group": g, "repeat": r, "p50_us": latency}
                    for m in (2048, 4096)
                    for g in (512, 1024, 2048)
                    for r in (0, 1)
                ],
            }
    return result


def test_comparison_brackets_and_drift(monkeypatch):
    mod = load_compare(monkeypatch)
    report = mod.summarize(reports(mod))
    assert len(report["cases"]) == 12 and not report["formal_performance_go"]
    assert all(
        x["stable_screen"] and abs(x["latency_reduction_vs_bracketed_a3"]["a3-2"] - 0.1) < 1e-8 for x in report["cases"]
    )
    assert not any(x["stable_screen"] for x in mod.summarize(reports(mod, True))["cases"])
    damaged = reports(mod)
    del damaged[("a3-2", 512)]
    with pytest.raises(ValueError):
        mod.summarize(damaged)


def test_comparison_rejects_absent_cases_and_duplicate_rounds(monkeypatch):
    mod = load_compare(monkeypatch)
    missing = reports(mod)
    for report in missing.values():
        report["paired_sessions"] = [x for x in report["paired_sessions"] if x["M"] != 2048]
    with pytest.raises(ValueError):
        mod.summarize(missing)
    duplicate = reports(mod)
    duplicate[("a3-2", 512)]["paired_sessions"][0]["repeat"] = 1
    with pytest.raises(ValueError):
        mod.summarize(duplicate)


def test_worker_passes_variant_to_session():
    source = (HERE / "quick_worker.py").read_text()
    assert 'options = {"variant": a.a3_variant} if a.a3 else {}' in source
    assert "tensor_module=module, **options" in source

# SPDX-License-Identifier: Apache-2.0
"""F2 两轮静态审查中的纯CPU测试；不声称310P3编译/实测通过。"""
from __future__ import annotations
import ast
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PERF = ROOT / "test/perf"
AN = runpy.run_path(str(PERF / "f2_sweep_analyze.py"))


def record(rank: int, m: int = 4096, chunk: int = 1024,
           overlap: int = 1, sha: str = "1" * 64) -> dict:
    return {
        "rank": rank, "M": m, "chunk_rows": chunk, "chunks": m // chunk,
        "overlap": overlap, "bit_exact": True, "status": 0,
        "profile": False, "device_so": sha, "warmup": 3, "iters": 30,
        "samples": [{"i": i, "event_us": 2000.0 + rank * 2 + i,
                     "wall_us": 2050.0 + rank * 2 + i}
                    for i in range(30)],
    }


def test_parameter_grid_is_finite_and_static():
    p = PERF / "f2_chunk_overlap_perf.py"
    source = p.read_text()
    ast.parse(source)
    assert "M_VALUES = (2048, 4096)" in source
    assert "CHUNK_VALUES = (256, 512, 1024)" in source
    for x in ("--m", "--chunk-rows", "--overlap", "--device-so-sha"):
        assert f'"{x}"' in source
    assert "mf310p_v2_trial_run(" in source
    assert "mf310p_v2_trial_check_status" in source
    assert "torch.equal(y, ref)" in source
    assert "MF310P_V2_NOWARMUP" in source
    assert "enable_timing=True" in source
    assert "--m" in (PERF / "f2_run_sweep.sh").read_text()


def test_pairing_and_rejection():
    p = AN["paired"]
    correct = p(record(0), record(1), m=4096, chunk=1024,
                overlap=1, expected_sha="1" * 64)
    assert correct["samples"] == 30
    assert correct["paired_event_p50_us"] == 2016.5
    for field, bad in (("status", 1), ("bit_exact", False), ("profile", True),
                       ("device_so", "2" * 64), ("chunks", 3), ("rank", 0)):
        r = record(1)
        r[field] = bad
        with pytest.raises(ValueError):
            p(record(0), r, m=4096, chunk=1024,
              overlap=1, expected_sha="1" * 64)
    r = record(1)
    r["samples"][0]["event_us"] = -1
    with pytest.raises(ValueError):
        p(record(0), r, m=4096, chunk=1024,
          overlap=1, expected_sha="1" * 64)


def test_protocol_kernel_not_edited_on_f2_branch():
    kernel = (ROOT / "src/op_kernel/mf310p_v2_kernel.asc").read_text()
    assert kernel.count("smem_shm_sdma_signal_at(") == 2
    assert kernel.count("smem_shm_sdma_wait_at(") == 2
    assert "kProductionAddTile = 2048" in kernel
    assert "CleanLines(ownC + start + coreOff, perCoreBytes)" in kernel
    assert "WaitAddDone(ibGm, ibUb, core)" in kernel
    assert "smem_shm_sdma_quiet_at(reserved)" in kernel

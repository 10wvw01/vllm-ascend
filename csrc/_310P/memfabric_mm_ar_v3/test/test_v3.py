# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = load("v3_model", HERE / "protocol_model.py")
history = load("v3_history", HERE / "historical.py")
adapter = load("v3_adapter", REPO / "vllm_ascend/_310p/ops/memfabric_mm_ar_v3.py")
guard = load("v3_guard", PROJECT / "tools/stage_guard.py")


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("groups,packets", [(1, 1), (2, 4), (4, 8), (16, 1)])
def test_asymmetric_in_place_interleavings(pipeline, groups, packets):
    for seed in range(64):
        model.simulate(seed, groups, packets, pipeline)


def test_tag_boundaries_and_wire_roundtrip():
    tags = set()
    for phase in range(1, 6):
        for epoch in (1, 1024, (1 << 24) - 1):
            for group in (0, 1, 63):
                for packet in range(64):
                    tag = model.tag(phase, epoch, group, packet)
                    assert tag < 1 << 40 and tag not in tags
                    wire = (0x5244 << 48) | (3 << 40) | tag
                    assert wire & ((1 << 40) - 1) == tag
                    tags.add(tag)
    for args in ((1, 0, 0, 0), (1, 1 << 24, 0, 0), (1, 1, 64, 0), (1, 1, 0, 64)):
        with pytest.raises(ValueError):
            model.tag(*args)


def test_duplicate_wrong_generation_and_wrong_address_fail():
    mail = (0x10000, 4096, model.tag(2, 10, 0, 0))
    seen = set()
    model.verify_mail(mail, mail, seen)
    with pytest.raises(ValueError):
        model.verify_mail(mail, mail, seen)
    for bad in ((0x10040, 4096, mail[2]), (mail[0], 8192, mail[2]), (mail[0], mail[1], model.tag(2, 9, 0, 0))):
        with pytest.raises(ValueError):
            model.verify_mail(bad, mail, set())


def test_unsafe_full_in_place_exchange_is_not_allreduce():
    a, b = 3, 7
    a += b
    b += a  # outgoing source has been modified by the other rank
    assert b != 3 + 7


@pytest.mark.parametrize("group", [256, 512, 1024, 2048])
def test_every_exact_m_has_bounded_padding(group):
    for m in range(1, 4097):
        padded = adapter.padded_rows(m, group)
        assert m <= padded <= 4096 and padded % group == 0 and padded - m < group
    for m in (0, -1, 4097):
        with pytest.raises(ValueError):
            adapter.padded_rows(m, group)


def sample_pair():
    ranks = []
    for rank in (0, 1):
        measurements = []
        for repeat in (0, 1):
            for m in (2048, 4096):
                for group in (1024, 2048):
                    measurements.append(
                        {
                            "M": m,
                            "group": group,
                            "pipeline": True,
                            "repeat": repeat,
                            "samples": [
                                {
                                    "i": i,
                                    "variant": i % 2,
                                    "correct": True,
                                    "event_us": 900 + 10 * rank + i,
                                    "wall_us": 940 + 10 * rank + i,
                                }
                                for i in range(4)
                            ],
                        }
                    )
        ranks.append(
            {
                "rank": rank,
                "run_id": "unit",
                "build_manifest_sha256": "a" * 64,
                "configuration": {
                    "M": [2048, 4096],
                    "groups": [1024, 2048],
                    "repeats": 2,
                    "pipeline": True,
                    "samples": 4,
                },
                "status": "PASS",
                "gates": {k: "PASS" for k in ("ib", "add", "protocol", "fused", "tail")},
                "measurements": measurements,
            }
        )
    return ranks


def test_rank_pairing_uses_per_sample_max_and_preserves_historical_limits():
    a, b = sample_pair()
    result = history.assess(a, b)
    assert result["paired_sessions"][0]["p50_us"] == 911.5
    assert not result["formal_performance_go"] and not result["same_batch_controls_run"]
    sources = history.baselines()
    assert any(x["arm"] == "stock" and x["M"] == 4096 and x["p50_us"] == 2022.553 for x in sources)
    assert any(x["arm"] == "e4t8" and x["original_status"] == "SCREEN_ONLY_NO_SIGNED_PERFORMANCE" for x in sources)


@pytest.mark.parametrize("failure", ["missing_gate", "sample_order", "nan", "bad_rank", "identity", "partial", "count"])
def test_assessor_rejects_incomplete_or_mismatched_evidence(failure):
    a, b = copy.deepcopy(sample_pair())
    if failure == "missing_gate":
        del b["gates"]["tail"]
    elif failure == "sample_order":
        b["measurements"][0]["samples"][0]["variant"] = 1
    elif failure == "nan":
        b["measurements"][0]["samples"][0]["event_us"] = float("nan")
    elif failure == "bad_rank":
        b["rank"] = 0
    elif failure == "identity":
        b["build_manifest_sha256"] = "other"
    elif failure == "partial":
        a["measurements"].pop()
        b["measurements"].pop()
    else:
        a["measurements"][0]["samples"].pop()
        b["measurements"][0]["samples"].pop()
    with pytest.raises(ValueError):
        history.assess(a, b)


def snapshot(rows=""):
    return "| 38144 310P3 | OK |\n| 38144 310P3 | OK |\nChip Process id\n" + rows


def test_guard_stops_busy_cross_card_and_extra_pids():
    idle = guard.parse_info(snapshot(), 38144)
    guard.assess(idle, stage_running=False)
    busy = guard.parse_info(snapshot("| 38144 0 | 111 |\n"), 38144)
    with pytest.raises(guard.LeaseLost):
        guard.assess(busy, stage_running=False)
    cross = guard.parse_info(snapshot("| 38144 0 | 111 |\n| 16384 0 | 111 |\n"), 38144)
    with pytest.raises(guard.LeaseLost):
        guard.assess(cross, stage_running=True)
    extra = guard.parse_info(snapshot("| 38144 0 | 111 |\n| 38144 1 | 222 |\n| 38144 0 | 333 |\n"), 38144)
    with pytest.raises(guard.LeaseLost):
        guard.assess(extra, stage_running=True)


def test_guard_runtime_stop_preserves_unrelated_process(tmp_path, monkeypatch):
    calls = [0]
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    def poll(_card):
        calls[0] += 1
        raw = snapshot() if calls[0] == 1 else snapshot("| 38144 0 | 111 |\n| 16384 0 | 111 |\n")
        return raw, guard.parse_info(raw, 38144)

    monkeypatch.setattr(guard, "sample", poll)
    try:
        rc = guard.run_stage(
            38144, tmp_path / "guard.jsonl", 0.01, [sys.executable, "-c", "import time; time.sleep(30)"], 1
        )
        assert rc == 97 and unrelated.poll() is None
        assert (tmp_path / "lease_guard_stop.npu.txt").exists()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_portable_cpp_contract(tmp_path):
    source = tmp_path / "contract.cpp"
    source.write_text(
        '#include "mf310p_v3_protocol.h"\n#include "mf310p_v3_api.h"\n'
        "static_assert(mfv3::Tag(5,mfv3::kMaxEpoch,63,63)<(1ULL<<40));\n"
        "static_assert(mfv3::PaddedRows(2051,1024)==3072);\n"
        "static_assert(mfv3::PacketCount(4ULL<<20,1<<20)==4);\n"
        "static_assert(!mfv3::ShapeValid(4097,1024));\nint main() {}\n"
    )
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(PROJECT / "src"),
            str(source),
            "-o",
            str(tmp_path / "contract"),
        ],
        check=True,
    )
    subprocess.run([str(tmp_path / "contract")], check=True)


def sdk_fixture(tmp_path):
    source = tmp_path / "source"
    files = {
        "VERSION": "1.2.0\n",
        "src/smem/include/host/smem.h": "public host\n",
        "src/smem/include/device/smem_shm_aicore_base_sdma.h": "public device\n",
        "output/smem/lib64/libmf_smem.so": "host-smem",
        "output/hybm/lib64/libmf_hybm_core.so": "host-hybm",
        "output/hybm/aicpu_kernel/libmf_sdma_orch.so": "aicpu-not-host",
        "build/src/hybm/ops/hybm_kernel/libmf_sdma_orch.so": "device-sdma",
        "build/src/hybm/ops/hybm_kernel/libmf_sdma_orch.json": "cust-json",
        "build/src/hybm/ops/hybm_kernel/libmf_sdma_orch_v28.json": "launch-json",
    }
    for name, contents in files.items():
        p = source / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(contents)
    return source


def sdk_module(monkeypatch):
    monkeypatch.syspath_prepend(str(PROJECT / "tools"))
    return load("v3_sdk", PROJECT / "tools/prepare_sdk.py")


def test_private_sdk_contains_both_host_and_exact_aicpu_triple(tmp_path, monkeypatch):
    sdk = sdk_module(monkeypatch)
    source = sdk_fixture(tmp_path)
    dest = tmp_path / "sdk"
    records = sdk.stage(source, dest)
    kernel = dest / "aarch64-linux/hybm/aicpu_kernel"
    assert {p.name for p in kernel.iterdir()} == {
        "libmf_sdma_orch.so",
        "libmf_sdma_orch.json",
        "libmf_sdma_orch_v28.json",
    }
    assert not (dest / "aarch64-linux/lib64/libmf_sdma_orch.so").exists()
    for record in records:
        assert sdk.sha(record["source"]) == sdk.sha(record["target"]) == record["sha256"]
    assert sdk.MF_BASE in (dest / "version.info").read_text()
    assert sdk.MF_UPSTREAM in (dest / "version.info").read_text()


@pytest.mark.parametrize("bad", ["missing_launch", "conflicting_host"])
def test_sdk_staging_refuses_incomplete_or_ambiguous_binaries(tmp_path, monkeypatch, bad):
    sdk = sdk_module(monkeypatch)
    source = sdk_fixture(tmp_path)
    if bad == "missing_launch":
        (source / "build/src/hybm/ops/hybm_kernel/libmf_sdma_orch_v28.json").unlink()
    else:
        p = source / "output/other/libmf_smem.so"
        p.parent.mkdir(parents=True)
        p.write_text("different-build")
    with pytest.raises((ValueError, FileNotFoundError)):
        sdk.stage(source, tmp_path / "sdk")

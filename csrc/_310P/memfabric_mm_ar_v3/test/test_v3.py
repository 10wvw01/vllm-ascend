# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import importlib.util
import json
import os
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


def test_guard_shared_card_mode_tolerates_external_pids_but_stays_recorded():
    busy = guard.parse_info(snapshot("| 38144 0 | 111 |\n| 38144 1 | 222 |\n| 16384 0 | 111 |\n"), 38144)
    checked = guard.assess(busy, stage_running=False, allow_external_target=True)
    assert checked["target_pids"] == [111, 222]
    assert checked["cross_card"] == [111]
    assert checked["target_pid_origin"] == "shared_card_mode_external_pids_tolerated"
    # Default stays exclusive: same snapshot must still fail without the flag.
    with pytest.raises(guard.LeaseLost):
        guard.assess(busy, stage_running=False)


def test_guard_shared_card_run_stage_skips_idle_preflight(tmp_path, monkeypatch):
    state = {"phase": 0}
    sequences = [
        snapshot("| 38144 0 | 111 |\n| 38144 1 | 222 |\n"),  # external pair present at preflight
        snapshot("| 38144 0 | 111 |\n| 38144 1 | 222 |\n| 38144 0 | 333 |\n| 38144 1 | 444 |\n"),  # + own ranks
        snapshot("| 38144 0 | 111 |\n| 38144 1 | 222 |\n"),  # external pair remains after exit
    ]

    def poll(_card):
        raw = sequences[min(state["phase"], len(sequences) - 1)]
        state["phase"] += 1
        return raw, guard.parse_info(raw, 38144)

    child = subprocess.Popen([sys.executable, "-c", "print('shared_stage_done')"], start_new_session=True)

    def child_poll():
        return 0 if state["phase"] >= len(sequences) else None

    monkeypatch.setattr(guard, "sample", poll)
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(child, "poll", lambda: child_poll())
    rc = guard.run_stage(
        38144, tmp_path / "guard.jsonl", 0.01, ["unused"], 5, allow_external_target=True
    )
    assert rc == 0
    events = [json.loads(line) for line in (tmp_path / "guard.jsonl").read_text().splitlines()]
    assert events[0]["lease_mode"] == "SHARED_CARD_OPERATOR_AUTHORIZED"
    assert events[1]["event"] == "PREFLIGHT_PASS"
    assert events[1]["target_pid_origin"] == "shared_card_mode_external_pids_tolerated"


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


def test_guard_deploy_window_tolerates_transient_helpers_only(tmp_path, monkeypatch):
    """声明窗口内的瞬态自有助手/编译基础设施子进程被有界容忍；持续超额或无标记超额仍STOP。"""
    state = {"phase": 0}
    (tmp_path / "deploy_window_rank0.flag").write_text('{"rank": 0}')
    sequences = [
        snapshot(),  # preflight idle
        snapshot("| 38144 0 | 111 |\n| 38144 1 | 222 |\n"),  # ranks only
        # +1 torch_npu compile-infra child surfacing before the helpers fork (V3.1 diag STOP form)
        snapshot("| 38144 0 | 111 |\n| 38144 1 | 222 |\n| 38144 1 | 333 |\n"),
        # ranks + 2 deploy helpers + 1 compile-infra child = 5 <= 2+2+2
        snapshot("| 38144 0 | 111 |\n| 38144 1 | 222 |\n| 38144 0 | 444 |\n| 38144 1 | 555 |\n| 38144 1 | 333 |\n"),
        snapshot("| 38144 0 | 111 |\n| 38144 1 | 222 |\n| 38144 0 | 444 |\n| 38144 1 | 555 |\n"),  # +2 helpers
        snapshot("| 38144 0 | 111 |\n| 38144 1 | 222 |\n"),  # healed
    ]

    def poll(_card):
        raw = sequences[min(state["phase"], len(sequences) - 1)]
        state["phase"] += 1
        return raw, guard.parse_info(raw, 38144)

    quick = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)

    def child_poll():
        return 0 if state["phase"] >= len(sequences) else None

    monkeypatch.setattr(guard, "sample", poll)
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: quick)
    monkeypatch.setattr(quick, "poll", lambda: child_poll())
    try:
        rc = guard.run_stage(38144, tmp_path / "guard.jsonl", 0.01, ["unused"], 5, deploy_flag_dir=tmp_path)
        assert rc == 0
        events = [json.loads(line)["event"] for line in (tmp_path / "guard.jsonl").read_text().splitlines()]
        assert "DEPLOY_WINDOW_TRANSIENT" in events and "DEPLOY_WINDOW_HEALED" in events
        assert "STOP" not in events
    finally:
        quick.terminate()
        quick.wait(timeout=5)


def test_guard_deploy_window_stops_seven_target_processes(tmp_path, monkeypatch):
    """声明窗口内第 7 个目标卡进程（超出 2+2+2 上限）立即 STOP，不等待持续界限。"""
    state = {"phase": 0}
    (tmp_path / "deploy_window_rank0.flag").write_text('{"rank": 0}')
    seven = snapshot(
        "| 38144 0 | 111 |\n| 38144 1 | 222 |\n| 38144 0 | 333 |\n| 38144 1 | 444 |\n"
        "| 38144 0 | 555 |\n| 38144 1 | 666 |\n| 38144 1 | 777 |\n"
    )

    def poll(_card):
        state["phase"] += 1
        raw = snapshot() if state["phase"] == 1 else seven
        return raw, guard.parse_info(raw, 38144)

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    monkeypatch.setattr(guard, "sample", poll)
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: child)
    try:
        rc = guard.run_stage(38144, tmp_path / "guard.jsonl", 0.01, ["unused"], 5, deploy_flag_dir=tmp_path)
        assert rc == 97
        events = [json.loads(line)["event"] for line in (tmp_path / "guard.jsonl").read_text().splitlines()]
        assert "STOP" in events
    finally:
        child.terminate()
        child.wait(timeout=5)


def test_guard_deploy_window_stops_persistent_and_unflagged_extras(tmp_path, monkeypatch):
    state = {"phase": 0}
    busy = snapshot("| 38144 0 | 111 |\n| 38144 1 | 222 |\n| 38144 0 | 333 |\n| 38144 1 | 444 |\n")

    def poll(_card):
        state["phase"] += 1
        raw = snapshot() if state["phase"] == 1 else busy
        return raw, guard.parse_info(raw, 38144)

    # 两个替身子进程须在补丁前创建：补丁会劫持全局 subprocess.Popen。
    children = [
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        for _ in range(2)
    ]
    holder = {"child": children[0]}
    monkeypatch.setattr(guard, "sample", poll)
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: holder["child"])
    try:
        # 无标记：立即 STOP（stop_owned 终止的是自带会话的替身子进程）。
        rc = guard.run_stage(38144, tmp_path / "g1.jsonl", 0.01, ["unused"], 5)
        assert rc == 97
        # 有标记但持续超额（超过有界时限）：仍 STOP。
        (tmp_path / "deploy_window_rank0.flag").write_text('{"rank": 0}')
        state["phase"] = 0
        holder["child"] = children[1]
        rc = guard.run_stage(
            38144,
            tmp_path / "g2.jsonl",
            0.01,
            ["unused"],
            5,
            deploy_flag_dir=tmp_path,
            deploy_persist_s=1.0,
        )
        assert rc == 97
        for name in ("g1.jsonl", "g2.jsonl"):
            events = [json.loads(line)["event"] for line in (tmp_path / name).read_text().splitlines()]
            assert "STOP" in events
    finally:
        for proc in children:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)


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


def helper_fixture(tmp_path):
    source = tmp_path / "source"
    helper = source / "tools/aicpu_kernel_deploy/orch_deploy_helper.cpp"
    helper.parent.mkdir(parents=True)
    helper.write_text("int main() { return 0; }\n")
    cann = tmp_path / "cann"
    (cann / "aarch64-linux/include/acl").mkdir(parents=True)
    (cann / "aarch64-linux/include/acl/acl.h").write_text("/* acl */\n")
    (cann / "aarch64-linux/lib64").mkdir(parents=True)
    (cann / "aarch64-linux/lib64/libascendcl.so").write_text("/* acl */\n")
    shim = tmp_path / "fake-g++"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        "target=''\n"
        "while [ $# -gt 0 ]; do\n"
        '  if [ "$1" = -o ]; then target="$2"; shift 2; continue; fi\n'
        "  shift\n"
        "done\n"
        "printf 'fake helper\\n' > \"$target\"\n"
    )
    os.chmod(shim, 0o755)
    return source, cann, shim


def test_helper_compile_command_pins_source_and_acl_paths(tmp_path, monkeypatch):
    sdk = sdk_module(monkeypatch)
    source, cann, shim = helper_fixture(tmp_path)
    monkeypatch.setattr(sdk.shutil, "which", lambda name: str(shim))
    command = sdk.helper_compile_command(source, cann)
    assert command[-5:] == ["-lascendcl", "-lruntime", "-ldl"] or command[-1] == "-ldl"
    assert str(source / "tools/aicpu_kernel_deploy/orch_deploy_helper.cpp") in command
    assert f"-I{cann / 'aarch64-linux/include'}" in command
    assert f"-Wl,-rpath,{cann / 'aarch64-linux/lib64'}" in command
    (source / "tools/aicpu_kernel_deploy/orch_deploy_helper.cpp").unlink()
    with pytest.raises(FileNotFoundError):
        sdk.helper_compile_command(source, cann)


def test_build_helper_compiles_records_and_refuses_overwrite(tmp_path, monkeypatch):
    sdk = sdk_module(monkeypatch)
    source, cann, shim = helper_fixture(tmp_path)
    monkeypatch.setattr(sdk.shutil, "which", lambda name: str(shim))
    dest = tmp_path / "sdk"
    records = []
    receipt = sdk.build_helper(source, dest, cann, records)
    staged = dest / "aarch64-linux/lib64/orch_deploy_helper"
    assert staged.is_file() and os.access(staged, os.X_OK)
    assert receipt["sha256"] == sdk.sha(staged) == records[-1]["sha256"]
    assert (dest / "orch_deploy_helper_build.log").is_file()
    with pytest.raises(FileExistsError):
        sdk.build_helper(source, dest, cann, records)

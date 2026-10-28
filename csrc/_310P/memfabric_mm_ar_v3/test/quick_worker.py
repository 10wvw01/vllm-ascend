#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One V3-only TP2 session: IB -> FP16 ADD -> protocol -> MM correctness -> short timing.

Only Gloo is initialized. Stock MM is used before pool creation for numerical
references, never as a performance arm; HCCL is never initialized.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

from historical import digest

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[2]
sys.path.insert(0, str(REPO / "vllm_ascend/_310p/ops"))
from memfabric_mm_ar_v3 import V3Session  # noqa: E402


def write(path, data):
    with path.open("x") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def loaded_identity(manifest):
    installed = manifest["dependency"]["installed_files"]
    loaded = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        path = line.split()[-1]
        if not path.startswith("/") or ".so" not in path:
            continue
        p = Path(path).resolve()
        if p.name.startswith(("libmf_", "libmf310p_v3")):
            loaded[str(p)] = digest(p)
    expected = {**installed, **manifest["artifacts"]}
    for p, sha in loaded.items():
        if expected.get(p) != sha:
            raise RuntimeError(f"unreviewed MemFabric/V3 library actually loaded: {p}")
    if str(Path(manifest["host_so"]).resolve()) not in loaded:
        raise RuntimeError("V3 Host not actually loaded")
    return loaded


def physical_device(cann, logical):
    fn = None
    for library, name in (
        ("libascendcl.so", "aclrtGetPhyDevIdByLogicDevId"),
        ("libruntime.so", "aclrtGetPhyDevIdByLogicDevId"),
        ("libruntime.so", "rtGetDevicePhyIdByIndex"),
    ):
        lib = ctypes.CDLL(str(Path(cann) / "aarch64-linux/lib64" / library))
        fn = getattr(lib, name, None)
        if fn is not None:
            break
    if fn is None:
        raise RuntimeError("no supported CANN logical->physical mapping API")
    fn.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    fn.restype = ctypes.c_int32
    physical = ctypes.c_int32(-1)
    rc = fn(logical, ctypes.byref(physical))
    if rc:
        raise RuntimeError(f"logical->physical mapping failed rc={rc}")
    return physical.value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--store-url", required=True)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--protocol-reps", type=int, default=8)
    p.add_argument("--packet-bytes", type=int, choices=[1 << 20, 2 << 20, 4 << 20], default=4 << 20)
    p.add_argument("--physical-devices", required=True)
    a = p.parse_args()
    if min(a.samples, a.warmup, a.protocol_reps) < 1:
        p.error("positive samples/warmup/protocol repetitions required")
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        p.error("two workers required")
    rank = int(os.environ["RANK"])
    selected = int(a.physical_devices.split(",")[rank])
    manifest = json.loads(a.manifest.read_text())
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import torch_npu

    if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("V3 standalone screen requires unmasked direct physical-device selection")
    torch.npu.set_device(selected)
    actual = physical_device(manifest["cann"], selected)
    if actual != selected:
        raise RuntimeError(f"CANN device{selected} maps to physical{actual}, wrong target")
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(f"npu:{selected}")
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    gates, measures = {}, []
    config = {
        "M": [2048, 4096],
        "groups": [512, 1024, 2048],
        "pipeline": True,
        "samples": a.samples,
        "warmup": a.warmup,
        "repeats": 2,
        "K": 2048,
        "N": 2048,
        "dtype": "fp16",
        "packet_bytes": a.packet_bytes,
        "baseline_execution": "NONE",
    }
    data = {
        "schema": "MF310P_V3_QUICK_RANK_V1",
        "rank": rank,
        "run_id": a.run_id,
        "build_manifest_sha256": digest(a.manifest),
        "configuration": config,
        "physical_device": actual,
        "gates": gates,
        "measurements": measures,
        "status": "RUNNING",
        "event_semantics": "stream completion window; includes host enqueue gaps",
        "runtime": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device_name": torch.npu.get_device_name(selected),
        },
    }
    session = None
    try:
        # Allocate all buffers/references BEFORE a MemFabric epoch can delay device allocation sync.
        torch.manual_seed(41029 + rank)
        weight = torch.randn(2048, 2048, device=device, dtype=torch.float16)
        weight_nz = torch_npu.npu_format_cast(weight, 29)
        xs, refs, outputs, partials, partial_refs = {}, {}, {}, {}, {}
        for m in (2048, 4096):
            xs[m], refs[m] = [], []
            for variant in (0, 1):
                torch.manual_seed(m + 17 * rank + 101 * variant)
                x = torch.randn(m, 2048, device=device, dtype=torch.float16)
                ref = F.linear(x, weight_nz).float().cpu()
                dist.all_reduce(ref)  # CPU Gloo; not timed and no HCCL group
                xs[m].append(x)
                refs[m].append(ref.half().to(device))
            outputs[m] = torch.zeros(m, 2048, device=device, dtype=torch.float16)
            row = torch.arange(m, dtype=torch.float32).remainder(97).reshape(-1, 1)
            col = torch.arange(2048, dtype=torch.float32).remainder(31).reshape(1, -1)
            pattern = row + col
            partials[m] = [(pattern + rank * 7 + v * 0.25).half().to(device) for v in (0, 1)]
            partial_refs[m] = [(pattern * 2 + 7 + v * 0.5).half().to(device) for v in (0, 1)]
        raw_buffers = {}
        for size in (4096, 1 << 20):
            raw_buffers[size] = [
                torch.full((size // 2,), val, device=device, dtype=torch.float16)
                for val in (1 + rank, 100 + 10 * rank, 0, 102 + 9 * rank)
            ]
        edge = torch.tensor(
            [1.0, -1.0, 2.0**-24, -(2.0**-24), 65504.0, -65504.0, 0.333251953125, 1.0009765625], dtype=torch.float16
        )
        src_edge = edge.roll(rank).repeat(256)
        seed_edge = edge.roll(rank + 2).repeat(256)
        peer_edge = edge.roll(1 - rank).repeat(256)
        raw_buffers["fp16_edges"] = [
            src_edge.to(device),
            seed_edge.to(device),
            torch.zeros(2048, device=device, dtype=torch.float16),
            (seed_edge.float() + peer_edge.float()).half().to(device),
        ]
        # Odd M tail, no extra Stock benchmark: reuse an independently computed numerical reference.
        tail_m = 2051
        tail_x = xs[4096][0][:tail_m]
        tail_ref = refs[4096][0][:tail_m]
        tail_stage = torch.zeros(3072, 2048, device=device, dtype=torch.float16)
        tail_output = torch.zeros_like(tail_stage)
        e0, e1 = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        dist.barrier()
        # 声明部署窗口：池创建期间库会 fork+exec orch_deploy_helper（每 Rank ≤1 个
        # 短命设备进程），守护器据此容忍有界瞬态超额；窗口结束即撤销标记。
        deploy_flag = a.out / f"deploy_window_rank{rank}.flag"
        deploy_flag.write_text(json.dumps({"rank": rank, "phase": "pool-create"}))
        try:
            session = V3Session(manifest["host_so"], rank, a.store_url, packet_bytes=a.packet_bytes)
        finally:
            deploy_flag.unlink(missing_ok=True)
        data["loaded_libraries"] = loaded_identity(manifest)

        def checked(out, ref, label):
            session.synchronize_check()
            # Compare FP16 bit patterns, not torch.equal floating point (+0/-0/NaN semantics).
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                bad = int((out.view(torch.int16) != ref.view(torch.int16)).sum().item())
                raise AssertionError(f"{label}: {bad} differing FP16 bit patterns")

        def gate(name):
            gates[name] = "PASS"
            write(a.out / f"gate_{name}_rank{rank}.json", {**data, "status": "GATE_PASS"})
            print(json.dumps({"rank": rank, "gate": name, "status": "PASS"}), flush=True)

        session.ib_probe()
        gate("ib")
        for size, (src, seed, out, ref) in raw_buffers.items():
            dist.barrier()
            session.probe_add(src, seed, out)
            checked(out, ref, f"raw ADD {size}")
        gate("add")
        # Serial and pipeline; multiple packets/groups, alternate data to expose stale accumulation.
        for group in (512, 1024, 2048):
            for pipeline in (False, True):
                for i in range(a.protocol_reps):
                    dist.barrier()
                    v = i % 2
                    session.run(partials[4096][v], None, outputs[4096], group=group, pipeline=pipeline, partial=True)
                    checked(outputs[4096], partial_refs[4096][v], "protocol")
        # Cheap explicit arrival-ring wrap: 49 outgoing mails/wave * 24 waves >1024.
        for i in range(24):
            dist.barrier()
            v = i % 2
            session.run(partials[4096][v], None, outputs[4096], group=256, partial=True)
            checked(outputs[4096], partial_refs[4096][v], "protocol ring wrap")
        gate("protocol")
        for m in (2048, 4096):
            for group in (512, 1024, 2048):
                for pipeline in (False, True):
                    dist.barrier()
                    session.run(xs[m][0], weight_nz, outputs[m], group=group, pipeline=pipeline)
                    checked(outputs[m], refs[m][0], "fused smoke")
        gate("fused")
        dist.barrier()
        tail_out = session.run_exact(tail_x, weight_nz, tail_output, input_stage=tail_stage, group=1024)
        checked(tail_out, tail_ref, "odd M padding")
        gate("tail")
        for repeat in range(2):
            order = [(m, g) for m in (2048, 4096) for g in (512, 1024, 2048)]
            if repeat:
                order.reverse()
            for m, group in order:
                samples = []
                for i in range(a.warmup + a.samples):
                    v = i % 2
                    dist.barrier()  # outside measured window
                    start = time.perf_counter_ns()
                    e0.record(session.stream)
                    session.run(xs[m][v], weight_nz, outputs[m], group=group, pipeline=True)
                    e1.record(session.stream)
                    session.stream.synchronize()
                    wall = (time.perf_counter_ns() - start) / 1000
                    event = e0.elapsed_time(e1) * 1000
                    checked(outputs[m], refs[m][v], "timed fused")
                    if i >= a.warmup:
                        samples.append(
                            {"i": i - a.warmup, "variant": v, "event_us": event, "wall_us": wall, "correct": True}
                        )
                row = {"M": m, "group": group, "pipeline": True, "repeat": repeat, "samples": samples}
                measures.append(row)
                write(a.out / f"timing_r{repeat}_m{m}_g{group}_rank{rank}.json", row)
                print(
                    json.dumps({"rank": rank, "M": m, "group": group, "repeat": repeat, "status": "PASS"}), flush=True
                )
        data["loaded_libraries_after"] = loaded_identity(manifest)
        if data["loaded_libraries"] != data["loaded_libraries_after"]:
            raise RuntimeError("loaded libraries changed")
        session.close()
        session = None
        data["status"] = "PASS"
        write(a.out / f"rank{rank}.json", data)
        dist.barrier()
        dist.destroy_process_group()
    except BaseException as exc:
        data["status"] = "STOP"
        data["error"] = repr(exc)
        write(a.out / f"stop_rank{rank}.json", data)
        # Do not synchronize/destroy a suspect device or enter another collective here.
        raise


if __name__ == "__main__":
    main()

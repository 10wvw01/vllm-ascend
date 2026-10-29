#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""L0+L1 attribution worker: differential arms plus per-wave host phase times.

Only Gloo is initialized. Arms: IB probe -> raw ADD transport (1/2/4MiB) ->
partial-protocol timing -> fused timing, all in one exclusive session.
No Stock/V2 arm runs here; attribution only, no performance GO claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[2]
sys.path.insert(0, str(REPO / "vllm_ascend/_310p/ops"))
sys.path.insert(0, str(PROJECT / "test"))
from quick_worker import loaded_identity, physical_device, write  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--store-url", required=True)
    p.add_argument("--physical-devices", required=True)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--ib-reps", type=int, default=1000)
    a = p.parse_args()
    if min(a.samples, a.warmup, a.ib_reps) < 1:
        p.error("positive samples/warmup/ib repetitions required")
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
        raise RuntimeError("attribution requires unmasked direct physical-device selection")
    torch.npu.set_device(selected)
    actual = physical_device(manifest["cann"], selected)
    if actual != selected:
        raise RuntimeError(f"CANN device{selected} maps to physical{actual}, wrong target")
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(f"npu:{selected}")
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    arms = {"ib": {}, "raw_add": {}, "partial": [], "fused": []}
    data = {
        "schema": "MF310P_V3_ATTRIBUTION_RANK_V1",
        "rank": rank,
        "run_id": a.run_id,
        "arms": arms,
        "status": "RUNNING",
        "configuration": {
            "samples": a.samples,
            "warmup": a.warmup,
            "ib_reps": a.ib_reps,
            "packet_bytes": 4 << 20,
            "K": 2048,
            "N": 2048,
            "dtype": "fp16",
            "path": "V32_PLAN_A_POOL_RETURN",
            "attribution": "L0 differential + L1 host phase times",
        },
        "physical_device": actual,
        "runtime": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device_name": torch.npu.get_device_name(selected),
        },
    }
    from memfabric_mm_ar_v32 import V32Session

    # Declare the startup+deploy window BEFORE the first device op: V32Session
    # creation forks torch_npu compile-infrastructure children and the MemFabric
    # orch_deploy_helper, exactly like the quick worker's documented ordering.
    deploy_flag = a.out / f"deploy_window_rank{rank}.flag"
    deploy_flag.write_text(json.dumps({"rank": rank, "phase": "startup+pool-create"}))
    tensor = manifest["tensor_adapter"]
    spec = importlib.util.spec_from_file_location("mf310p_v32_tensor", tensor["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    session = V32Session(manifest["host_so"], rank, a.store_url, packet_bytes=4 << 20, tensor_module=module)
    session.phase_log = []
    e0, e1 = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    try:
        data["loaded_libraries"] = loaded_identity(manifest)

        def checked(out, ref, label):
            session.synchronize_check()
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                bad = int((out.view(torch.int16) != ref.view(torch.int16)).sum().item())
                raise AssertionError(f"{label}: {bad} differing FP16 bit patterns")

        # ---- arm 0: pure inter-core IB synchronization ----
        start = time.perf_counter_ns()
        session.ib_probe(a.ib_reps)
        ib_wall_us = (time.perf_counter_ns() - start) / 1000
        arms["ib"] = {
            "reps": a.ib_reps,
            "wall_us": ib_wall_us,
            "per_round_us": ib_wall_us / a.ib_reps,
        }
        # ---- arm 1: raw SDMA ADD transport, no MM/protocol ----
        for size in (1 << 20, 2 << 20, 4 << 20):
            src = torch.full((size // 2,), float(1 + rank), device=device, dtype=torch.float16)
            seed = torch.full((size // 2,), float(100 + 10 * rank), device=device, dtype=torch.float16)
            out = torch.zeros(size // 2, device=device, dtype=torch.float16)
            ref = torch.full((size // 2,), float(102 + 9 * rank), device=device, dtype=torch.float16)
            samples = []
            for i in range(a.warmup + a.samples):
                dist.barrier()
                start = time.perf_counter_ns()
                e0.record(session.stream)
                session.probe_add(src, seed, out)
                e1.record(session.stream)
                session.stream.synchronize()
                wall = (time.perf_counter_ns() - start) / 1000
                event = e0.elapsed_time(e1) * 1000
                checked(out, ref, f"raw add {size}")
                if i >= a.warmup:
                    samples.append({"i": i - a.warmup, "event_us": event, "wall_us": wall})
            arms["raw_add"][str(size)] = {"bytes": size, "samples": samples}
            del src, seed, out, ref
        # ---- references for partial / fused arms ----
        torch.manual_seed(41029 + rank)
        weight = torch.randn(2048, 2048, device=device, dtype=torch.float16)
        weight_nz = torch_npu.npu_format_cast(weight, 29)
        xs, refs, partials, partial_refs = {}, {}, {}, {}
        for m in (2048, 4096):
            xs[m], refs[m] = [], []
            for variant in (0, 1):
                torch.manual_seed(m + 17 * rank + 101 * variant)
                x = torch.randn(m, 2048, device=device, dtype=torch.float16)
                ref = F.linear(x, weight_nz).float().cpu()
                dist.all_reduce(ref)
                xs[m].append(x)
                refs[m].append(ref.half().to(device))
            row = torch.arange(m, dtype=torch.float32).remainder(97).reshape(-1, 1)
            col = torch.arange(2048, dtype=torch.float32).remainder(31).reshape(1, -1)
            pattern = row + col
            partials[m] = [(pattern + rank * 7 + v * 0.25).half().to(device) for v in (0, 1)]
            partial_refs[m] = [(pattern * 2 + 7 + v * 0.5).half().to(device) for v in (0, 1)]
        torch.npu.synchronize()
        dist.barrier()

        def timed_arm(name, picker, checker):
            for m in (2048, 4096):
                for group in (512, 1024, 2048):
                    samples = []
                    for i in range(a.warmup + a.samples):
                        dist.barrier()
                        v = i % 2
                        n0 = len(session.phase_log)
                        start = time.perf_counter_ns()
                        e0.record(session.stream)
                        out = picker(m, group, v)
                        e1.record(session.stream)
                        session.stream.synchronize()
                        wall = (time.perf_counter_ns() - start) / 1000
                        event = e0.elapsed_time(e1) * 1000
                        checker(out, m, group, v)
                        del out
                        if i >= a.warmup:
                            phases = session.phase_log[n0:]
                            samples.append(
                                {
                                    "i": i - a.warmup,
                                    "event_us": event,
                                    "wall_us": wall,
                                    "phases": phases[0] if len(phases) == 1 else None,
                                }
                            )
                    arms[name].append({"M": m, "group": group, "samples": samples})
                    print(json.dumps({"rank": rank, "arm": name, "M": m, "group": group, "status": "PASS"}), flush=True)

        # ---- arm 2: partial-protocol (input staged; copy + R/G, no MM) ----
        timed_arm(
            "partial",
            lambda m, g, v: session.run_pool(partials[m][v], group=g, partial=True),
            lambda out, m, g, v: checked(out, partial_refs[m][v], f"partial m{m} g{g}"),
        )
        # ---- arm 3: fused (MM + R/G), same session for same-batch subtraction ----
        timed_arm(
            "fused",
            lambda m, g, v: session.run_pool(xs[m][v], weight_nz, group=g),
            lambda out, m, g, v: checked(out, refs[m][v], f"fused m{m} g{g}"),
        )
        data["loaded_libraries_after"] = loaded_identity(manifest)
        if data["loaded_libraries"] != data["loaded_libraries_after"]:
            raise RuntimeError("loaded libraries changed")
        session.close()
        session = None
        data["status"] = "PASS"
        write(a.out / f"attribution_rank{rank}.json", data)
        dist.barrier()
        dist.destroy_process_group()
    except BaseException as exc:
        data["status"] = "STOP"
        data["error"] = repr(exc)
        write(a.out / f"stop_rank{rank}.json", data)
        raise
    finally:
        deploy_flag.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

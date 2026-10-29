#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone A3 pipe-utilization capture (user-authorized diagnostic).

Not part of the contracted campaign tooling; lives outside the repo so the
frozen manifest source set is untouched. Mirrors quick_worker's proven
A3Session initialization, then profiles run_pool calls with
torch_npu.profiler + _ExperimentalConfig(Level1, AiCMetrics.PipeUtilization).
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from torch_npu.profiler import (
    AiCMetrics,
    ProfilerActivity,
    ProfilerLevel,
    _ExperimentalConfig,
    profile,
    tensorboard_trace_handler,
)

REPO = Path("/home/w00804146/310P/a3ut-20261029/vllm-ascend")
sys.path.insert(0, str(REPO / "vllm_ascend/_310p/ops"))
sys.path.insert(0, str(REPO / "csrc/_310P/memfabric_mm_ar_v3/test"))


def digest(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variant", default="a3", choices=["a3", "a3-2", "a3-tail"])
    p.add_argument("--compute-rows", type=int, choices=[512, 1024], default=1024)
    p.add_argument("--physical-devices", required=True)
    p.add_argument("--store-url", required=True)
    p.add_argument("--calls", type=int, default=10)
    a = p.parse_args()

    if os.environ.get("WORLD_SIZE") != "2" or "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("direct physical TP2 required")
    manifest = json.loads(a.manifest.read_text())
    rank = int(os.environ["RANK"])
    selected = int(a.physical_devices.split(",")[rank])
    torch.npu.set_device(selected)
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    (a.out / f"deploy_window_rank{rank}.flag").write_text("startup+pool-create")
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    stream = torch.npu.current_stream()
    cpu_group = dist.group.WORLD

    from memfabric_mm_ar_a3 import A3Session

    tensor = manifest["tensor_adapter"]
    if digest(Path(tensor["path"])) != tensor["sha256"]:
        raise RuntimeError("Storage SO identity changed")
    spec = importlib.util.spec_from_file_location("mf310p_v32_tensor", tensor["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    session = A3Session(
        manifest["host_so"], rank, a.store_url, packet_bytes=4 << 20,
        tensor_module=module, variant=a.variant,
    )
    session.ib_probe()
    src = torch.ones((512, 2048), dtype=torch.float16, device=f"npu:{selected}")
    seed = src.clone()
    out = torch.empty_like(src)
    session.probe_add(src, seed, out)
    session.synchronize_check()
    assert torch.equal(out, src + seed)
    del out, src, seed
    session.initialize()
    (a.out / f"deploy_window_rank{rank}.flag").unlink(missing_ok=True)

    torch.manual_seed(4242)
    w = torch.randn((2048, 2048), dtype=torch.float16, device=f"npu:{selected}")
    wnz = torch_npu.npu_format_cast(w, 29)
    xs, refs = {}, {}
    for m in (2048, 4096):
        xs[m], refs[m] = [], []
        for v in (0, 1):
            torch.manual_seed(m + rank * 17 + v * 1009)
            x = torch.randn((m, 2048), dtype=torch.float16, device=f"npu:{selected}")
            ref = F.linear(x, wnz).float().cpu()
            dist.all_reduce(ref, group=cpu_group)
            xs[m].append(x)
            refs[m].append(ref.to(device=f"npu:{selected}", dtype=torch.float16))
    stream.synchronize()

    # Warm every geometry outside the profile windows.
    for m in (2048, 4096):
        for comm in (512, 1024, 2048):
            for _ in range(3):
                o = session.run_pool(xs[m][0], wnz, compute_rows=a.compute_rows, comm_rows=comm)
                stream.synchronize()
                assert torch.equal(o, refs[m][0])
                del o

    results = []
    for comm in (512, 1024, 2048):
        label = f"c{comm}"
        route = a.out / f"prof-{a.variant}-cr{a.compute_rows}-{label}-rank{rank}"
        exp = _ExperimentalConfig(
            profiler_level=ProfilerLevel.Level1,
            aic_metrics=AiCMetrics.PipeUtilization,
        )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            experimental_config=exp,
            on_trace_ready=tensorboard_trace_handler(str(route)),
        ) as prof_ctx:
            for m in (2048, 4096):
                for i in range(a.calls):
                    dist.barrier(group=cpu_group)
                    o = session.run_pool(
                        xs[m][i % 2], wnz, compute_rows=a.compute_rows, comm_rows=comm
                    )
                    stream.synchronize()
                    ok = torch.equal(o, refs[m][i % 2])
                    assert ok
                    del o
                    prof_ctx.step_num += 1
        files = sorted(route.rglob("kernel_details.csv"))
        results.append({"comm": comm, "route": str(route), "csv": [str(f) for f in files]})
        print(f"[rank{rank}] {label} captured -> {files}", flush=True)

    session.close()
    dist.destroy_process_group()
    write = lambda name, obj: (a.out / name).write_text(json.dumps(obj, indent=2) + "\n")
    write(f"prof_index_rank{rank}.json", {"variant": a.variant, "compute": a.compute_rows,
                                          "calls": a.calls, "results": results,
                                          "status": "PASS"})
    print(f"[rank{rank}] DONE", flush=True)


if __name__ == "__main__":
    main()

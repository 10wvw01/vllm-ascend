#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fresh-process Stock/V2/V3/A3 full Tensor-input/output timing and A3 attribution."""

import argparse
import ctypes
import importlib.util
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

from quick_worker import digest, loaded_identity, physical_device, write
from unified_metrics import CONTRACT, STAGES, decode_trace, timed_call

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "vllm_ascend/_310p/ops"))


class V2Runner:
    """Frozen production D4 trial ABI; two persistent outputs, no fallback."""

    def __init__(self, spec, rank, url, torch, stream):
        self.lib = ctypes.CDLL(spec["library"], mode=ctypes.RTLD_LOCAL)
        self.stream, self.torch = stream, torch
        self.raw = ctypes.c_void_p(stream.npu_stream)
        self.handle = ctypes.c_void_p()
        p, q, u = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32
        signatures = {
            "create": [ctypes.c_int, ctypes.c_char_p, u, ctypes.POINTER(p)],
            "ib_probe": [p, u, p],
            "run": [p, q, q, q, u, u, p],
            "check_status": [p, ctypes.POINTER(q)],
            "destroy": [p],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, "mf310p_v2_trial_" + name)
            fn.argtypes, fn.restype = args, ctypes.c_int
        self.check(self.lib.mf310p_v2_trial_create(rank, url.encode(), 1024, ctypes.byref(self.handle)))
        self.check(self.lib.mf310p_v2_trial_ib_probe(self.handle, 64, self.raw))
        self.outputs = {
            m: [
                torch.empty((m, 2048), dtype=torch.float16, device=f"npu:{torch.npu.current_device()}")
                for _ in range(2)
            ]
            for m in (2048, 4096)
        }
        self.index = 0

    @staticmethod
    def check(rc):
        if rc:
            raise RuntimeError(f"V2 rc={rc}; stop both ranks, no uncertain retry")

    def run(self, x, w):
        out = self.outputs[x.shape[0]][self.index % 2]
        self.index += 1
        self.check(
            self.lib.mf310p_v2_trial_run(
                self.handle, x.data_ptr(), w.data_ptr(), out.data_ptr(), x.shape[0], 1, self.raw
            )
        )
        return out

    def status(self):
        value = ctypes.c_uint64()
        self.check(self.lib.mf310p_v2_trial_check_status(self.handle, ctypes.byref(value)))
        self.check(value.value)

    def close(self):
        self.stream.synchronize()
        self.check(self.lib.mf310p_v2_trial_destroy(self.handle))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", choices=["stock", "v2", "v3", "a3", "a3-2", "a3-tail"], required=True)
    p.add_argument("--compute-rows", type=int, choices=[512, 1024], default=1024)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--physical-devices", required=True)
    p.add_argument("--store-url", required=True)
    a = p.parse_args()
    if os.environ.get("WORLD_SIZE") != "2" or "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
        raise RuntimeError("direct physical TP2 required")
    manifest = json.loads(a.manifest.read_text())
    rank = int(os.environ["RANK"])
    selected = int(a.physical_devices.split(",")[rank])
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import torch_npu

    torch.npu.set_device(selected)
    if physical_device(manifest["cann"], selected) != selected:
        raise RuntimeError("physical-device identity mismatch")
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)
    # Declare before FIRST device tensor/pool operation, including Stock runtime children.
    flag = a.out / f"deploy_window_rank{rank}.flag"
    flag.write_text("startup+pool-create")
    dist.init_process_group("hccl" if a.arm == "stock" else "gloo", timeout=timedelta(seconds=120))
    cpu_group = dist.new_group(backend="gloo") if a.arm == "stock" else dist.group.WORLD
    stream = torch.npu.current_stream()
    report = {
        "contract": CONTRACT,
        "run_id": a.out.name,
        "rank": rank,
        "status": "RUNNING",
        "cases": [],
        "identity": {"manifest_sha256": digest(a.manifest), "head": manifest["head"]},
        "configuration": {
            "arm": a.arm,
            "M": [2048, 4096],
            "comm_rows": [512, 1024, 2048] if a.arm.startswith("a3") or a.arm == "v3" else [1024],
            "compute_rows": a.compute_rows,
            "samples": a.samples,
            "warmup": a.warmup,
        },
    }
    report["configuration"].update(
        K=2048,
        N=2048,
        dtype="fp16",
        input_format="ND",
        weight_format="NZ29",
        output_format="ND",
        packet_bytes=(4 << 20) if a.arm not in ("stock", "v2") else None,
        graph=False,
    )
    report["runtime"] = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device_name": torch_npu.npu.get_device_name(selected),
        "physical_device": selected,
        "cann": manifest["cann"],
    }
    session = None
    try:
        torch.manual_seed(4242)
        w = torch.randn((2048, 2048), dtype=torch.float16, device=f"npu:{selected}")
        wnz = torch_npu.npu_format_cast(w, 29)
        xs, refs = {}, {}
        for m in (2048, 4096):
            xs[m], refs[m] = [], []
            for v in (0, 1):
                torch.manual_seed(m + rank * 17 + v * 1009)
                x = torch.randn((m, 2048), dtype=torch.float16, device=f"npu:{selected}")
                # Reference local FP16 MM then cross-rank SUM, outside every timed window.
                ref = F.linear(x, wnz).float().cpu()
                dist.all_reduce(ref, group=cpu_group)
                xs[m].append(x)
                refs[m].append(ref.to(device=f"npu:{selected}", dtype=torch.float16))
        stream.synchronize()
        if a.arm == "v2":
            session = V2Runner(manifest["v2"], rank, a.store_url, torch, stream)
        elif a.arm != "stock":
            from memfabric_mm_ar_v3 import V3Session

            if a.arm.startswith("a3"):
                from memfabric_mm_ar_a3 import A3Session

                spec = manifest["tensor_adapter"]
                if digest(Path(spec["path"])) != spec["sha256"]:
                    raise RuntimeError("Storage SO identity changed")
                module_spec = importlib.util.spec_from_file_location("unified_storage", spec["path"])
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                session = A3Session(manifest["host_so"], rank, a.store_url, tensor_module=module, variant=a.arm)
            else:
                session = V3Session(manifest["host_so"], rank, a.store_url)
            session.ib_probe()
            # Existing real ADD gate before output-pool mode; no forged warmup bypass.
            src = torch.ones((512, 2048), dtype=torch.float16, device=f"npu:{selected}")
            seed = src.clone()
            out = torch.empty_like(src)
            session.probe_add(src, seed, out)
            session.synchronize_check()
            if not torch.equal(out, src + seed):
                raise AssertionError("raw ADD gate")
            del out, src, seed
            if a.arm.startswith("a3"):
                session.initialize()
        outputs = {m: [torch.empty_like(xs[m][0]) for _ in range(2)] for m in (2048, 4096)}

        def op(x, comm, v):
            # Identical Tensor->Tensor API boundary. Persistent pool allocation is allowed;
            # per-call wrap/reserve/copy is never moved outside this function.
            if a.arm == "stock":
                y = F.linear(x, wnz)
                dist.all_reduce(y)
                return y
            if a.arm == "v2":
                return session.run(x, wnz)
            if a.arm.startswith("a3"):
                return session.run_pool(x, wnz, compute_rows=a.compute_rows, comm_rows=comm)
            return session.run(x, wnz, outputs[x.shape[0]][v], group=comm)

        def checked(out, ref):
            if a.arm == "v2":
                session.status()
            elif session is not None:
                session.synchronize_check()
            if torch_npu.get_npu_format(out) != 2 or not torch.equal(out, ref):
                raise AssertionError("ND/bit-exact output gate")
            # A true downstream read is gated outside timing, not hidden output repair.
            if not torch.equal(torch.add(out, 0), ref):
                raise AssertionError("downstream ND read")

        # Warm BOTH persistent output addresses and ALL geometry before uninstrumented data.
        for m in (2048, 4096):
            for comm in report["configuration"]["comm_rows"]:
                for i in range(a.warmup):
                    dist.barrier(group=cpu_group)
                    out = op(xs[m][i % 2], comm, i % 2)
                    stream.synchronize()
                    checked(out, refs[m][i % 2])
                    del out
        report["loaded_libraries"] = loaded_identity(manifest) if a.arm not in ("stock", "v2") else []
        if a.arm == "stock":
            # One route-only capture, never included in performance samples.
            route = a.out / f"route-rank{rank}"
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(route)),
            ):
                out = op(xs[2048][0], 1024, 0)
                stream.synchronize()
                checked(out, refs[2048][0])
                del out
            files = list(route.rglob("*.csv"))
            content = "\n".join(f.read_text(errors="replace") for f in files)
            if "aclnnMatmulWeightNz" not in content or "hccl" not in content.lower():
                raise RuntimeError("native NZ MM/HCCL route unproven; preserve profiler output and STOP")
            report["route"] = {str(f.relative_to(a.out)): digest(f) for f in files}
        maps = Path("/proc/self/maps").read_text()
        if a.arm == "v2":
            spec = manifest["v2"]
            for filename, expected in {spec["library"]: spec["sha256"], **spec["dependencies"]}.items():
                if digest(Path(filename)) != expected:
                    raise RuntimeError("V2 frozen file changed")
            if str(Path(spec["library"]).resolve()) not in maps:
                raise RuntimeError("V2 actual library not mapped")
            # Every mapped MemFabric library must be one of the separately frozen V2 files.
            frozen = {str(Path(f).resolve()) for f in [spec["library"], *spec["dependencies"]]}
            for line in maps.splitlines():
                if "/libmf_" in line:
                    filename = line.split()[-1]
                    if filename not in frozen:
                        raise RuntimeError(f"unreviewed V2 mapped dependency: {filename}")
            report["loaded_libraries"] = sorted(frozen)
        elif a.arm == "stock" and any("/libmf_" in line for line in maps.splitlines()):
            raise RuntimeError("Stock process unexpectedly contains MemFabric")
        flag.unlink()
        events = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        stages = STAGES if a.arm.startswith("a3") else ("off-before", "off-after")
        last_trace_epoch = 0
        for stage in stages:
            if a.arm.startswith("a3"):
                session.set_trace(stage == "device")
                session.phase_log = [] if stage == "host" else None
            for repeat in range(2):
                cases = [(m, c) for m in (2048, 4096) for c in report["configuration"]["comm_rows"]]
                if repeat:
                    cases.reverse()
                for m, comm in cases:
                    samples = []
                    for i in range(a.warmup + a.samples):
                        dist.barrier(group=cpu_group)
                        v = i % 2
                        out, row = timed_call(
                            lambda m=m, v=v, comm=comm: op(xs[m][v], comm, v), stream, events, time.perf_counter_ns
                        )
                        checked(out, refs[m][v])
                        if stage == "device":
                            raw = session.fetch_trace()
                            if raw[4] <= last_trace_epoch:
                                raise RuntimeError("stale/repeated trace epoch")
                            last_trace_epoch = raw[4]
                            row["device"] = decode_trace(raw, m, a.compute_rows, comm)
                        if stage == "host":
                            row["host_us"] = session.phase_log.pop()
                        del out  # release is after output-ready boundary for every arm
                        if i >= a.warmup:
                            samples.append({"i": i - a.warmup, "correct": True, **row})
                    report["cases"].append(
                        {"M": m, "comm_rows": comm, "stage": stage, "repeat": repeat, "samples": samples}
                    )
                    write(a.out / f"checkpoint-rank{rank}-{stage}-r{repeat}-m{m}-c{comm}.json", report["cases"][-1])
        if session is not None:
            session.close()
        report["status"] = "PASS"
        write(a.out / f"unified_rank{rank}.json", report)
        dist.destroy_process_group()
    except BaseException as exc:
        report.update(status="STOP", error=repr(exc))
        write(a.out / f"stop_rank{rank}.json", report)
        raise  # no cleanup/destroy/retry against an uncertain pool


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A1-R2 N1 route-fix diagnostic (isolated, throwaway): determine which call
form / profiler config reproduces the model's aclnnMatmulWeightNz route.

Variants per rank (M=4096, K=N=2048, FP16, TP2 HCCL all_reduce per iter):
  V1 F.linear(x, w_nz)                          [current probe form]
  V2 torch.ops.vllm.unquantized_gemm(x, w_nz, None)  [model site form]
  V3 torch.matmul(x, w_nz.t())                  [explicit transposed-NZ matmul]
Profiler mirrors the proven L2 serve path exactly:
  torch.profiler.profile(activities=[CPU, CUDA], record_shapes=False,
  with_stack=False, on_trace_ready=tensorboard_trace_handler(dir)).
No MemFabric pool, no KFC, no shared-state mutation. G0 identity gates same
as a1_nz_stock_profile_probe.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C as extension  # noqa: F401  identity gate
import vllm_ascend.ops.linear as _lin  # noqa: F401  registers unquantized_gemm
from u1_spec import D4_DEVICE_SHA, PLUGIN_SHA, MEMFABRIC_HOST_SHA, KVER_SHA

K = N = 2048
M = 4096
VARIANTS = ("v1_linear", "v2_unquantized_gemm", "v3_matmul_t")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iters", type=int, default=4)
    a = p.parse_args()
    if not a.out.is_dir() or a.iters < 2 or a.iters > 10:
        p.error("existing dir and iters 2..10")
    for key, value in (("U1_DEVICE_SHA256", D4_DEVICE_SHA),
                       ("U1_HOST_SHA256", MEMFABRIC_HOST_SHA),
                       ("U1_PLUGIN_SHA256", PLUGIN_SHA),
                       ("U1_KVER_SHA256", KVER_SHA)):
        if __import__("os").environ.get(key) != value:
            p.error(f"identity gate {key}")
    rank = int(__import__("os").environ["RANK"])
    local = int(__import__("os").environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    device = torch.device(f"npu:{local}")
    # Mirror serve startup (vllm_ascend.worker.model_runner_v1:229 /
    # _310p/worker_310p.py:44) which a bare torchrun process does not have.
    torch.npu.config.allow_internal_format = True
    torch_npu.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group(backend="hccl")
    cpu = dist.new_group(backend="gloo")
    findings = []
    try:
        torch.manual_seed(4242)
        w = torch.randn(N, K, dtype=torch.float16, device=device)
        w_nz = torch_npu.npu_format_cast(w, 29)
        torch.manual_seed(1200 + rank * 17 + M)
        x = torch.randn(M, K, dtype=torch.float16, device=device)
        ref = F.linear(x, w).float().cpu()
        dist.all_reduce(ref, group=cpu)
        ref = ref.to(device=device, dtype=torch.float16)
        stream = torch.npu.current_stream()

        def call(variant, inp):
            if variant == "v1_linear":
                return F.linear(inp, w_nz)
            if variant == "v2_unquantized_gemm":
                return torch.ops.vllm.unquantized_gemm(inp, w_nz, None)
            return torch.matmul(inp, w_nz.t())

        for variant in VARIANTS:
            for _ in range(3):
                dist.barrier(group=cpu)
                y = call(variant, x)
                dist.all_reduce(y, op=dist.ReduceOp.SUM)
                stream.synchronize()
            if not torch.equal(y, ref):
                raise AssertionError(f"rank{rank} {variant}: not bit-exact")
            trace_dir = a.out / f"rank{rank}_{variant}_profiler"
            trace_dir.mkdir(mode=0o700, exist_ok=False)
            # Mirror TorchNPUProfilerWrapper (vllm_ascend serve) exactly:
            # torch_npu.profiler + _ExperimentalConfig; no record_shapes.
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=torch_npu.profiler.ExportType.Text,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                msprof_tx=False,
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                l2_cache=False, op_attr=False, data_simplification=True,
                record_op_args=False, gc_detect_threshold=None)
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                with_stack=False,
                experimental_config=experimental_config,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(trace_dir), worker_name=f"rank{rank}_{variant}"),
            ):
                for _ in range(a.iters):
                    dist.barrier(group=cpu)
                    y = call(variant, x)
                    dist.all_reduce(y, op=dist.ReduceOp.SUM)
                    stream.synchronize()
                    if not torch.equal(y, ref):
                        raise AssertionError(f"rank{rank} {variant}: bad")
            # CANN post-analysis may lag the export; poll before failing.
            import time as _time
            csvs = []
            for _ in range(60):
                csvs = [f for f in trace_dir.rglob("kernel_details.csv")
                        if f.is_file() and f.stat().st_size > 100
                        and "ASCEND_PROFILER_OUTPUT" in f.parts]
                if len(csvs) == 1:
                    break
                _time.sleep(2)
            if len(csvs) != 1:
                raise RuntimeError(f"rank{rank} {variant}: {len(csvs)} CSVs")
            dest = a.out / f"rank{rank}_{variant}_kernel_details.csv"
            with csvs[0].open("rb") as i, dest.open("xb") as o:
                shutil.copyfileobj(i, o)
            import csv as _csv
            rows = list(_csv.DictReader(dest.open(encoding="utf-8-sig", newline="")))
            mm = [r for r in rows if "Matmul" in r.get("Name", "")
                  or "MatMul" in r.get("Name", "")]
            findings.append({
                "variant": variant, "rank": rank, "rows": len(rows),
                "mm_names": sorted({r["Name"] for r in mm}),
                "mm_input_formats": sorted({r.get("Input Formats", "")
                                            for r in mm}),
                "mm_durations_us": [round(float(r["Duration(us)"]), 1) for r in mm],
                "allreduce_rows": sum("hcom_allReduce" in r.get("Name", "")
                                      for r in rows),
            })
        dest = a.out / f"route_diag_rank{rank}.json"
        with dest.open("x") as f:
            json.dump({"schema": "A1_R2_NZ_ROUTE_FIX_DIAG_V1",
                       "bit_exact": True, "findings": findings}, f, indent=2)
            f.write("\n")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

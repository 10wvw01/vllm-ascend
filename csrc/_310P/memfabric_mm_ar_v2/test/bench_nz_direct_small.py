# SPDX-License-Identifier: Apache-2.0
"""310P3 TP=2 experimental V2 original-NZ direct-ND small-M benchmark.

Run only with the V2 experimental build enabled; no V1 serving in parallel.

  ASCEND_RT_VISIBLE_DEVICES=0,1 \
  VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR=1 \
  VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR_V2=1 \
  torchrun --standalone --nproc-per-node=2 \
    csrc/_310P/memfabric_mm_ar_v2/test/bench_nz_direct_small.py \
    --rows 1 2 4 8 16 32 64 128 255 256 --add-cores 1 2 4 8

Each rank computes the stock FP16 MM + FP32 HCCL sum reference BEFORE
initializing MemFabric, avoiding a concurrent HCCL/MemFabric data plane.
C timings include enqueue and stream sync; no input staging, weight-slice build or blocked-output transform.
Compare V1 through the separate V1 benchmark after this process exits.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C as ascend_ext

from vllm_ascend.utils import maybe_trans_nz


def c_api():
    so = ctypes.CDLL(ascend_ext.__file__)
    U32 = ctypes.c_uint32
    U64 = ctypes.c_uint64
    P = ctypes.c_void_p
    so.mf310p_v2_trial_create.argtypes = [ctypes.c_int, ctypes.c_char_p, U32, ctypes.POINTER(P)]
    so.mf310p_v2_trial_create.restype = ctypes.c_int
    so.mf310p_v2_trial_ib_probe.argtypes = [P, U32, P]
    so.mf310p_v2_trial_ib_probe.restype = ctypes.c_int
    so.mf310p_v2_trial_run_nz_direct_small.argtypes = [P, U64, U64, U64, U32, U32, U32, P]
    so.mf310p_v2_trial_run_nz_direct_small.restype = ctypes.c_int
    so.mf310p_v2_trial_check_status.argtypes = [P, ctypes.POINTER(U64)]
    so.mf310p_v2_trial_check_status.restype = ctypes.c_int
    so.mf310p_v2_trial_destroy.argtypes = [P]
    so.mf310p_v2_trial_destroy.restype = ctypes.c_int
    return so


def check(rc: int, name: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{name}: rc={rc}")


def sync() -> None:
    torch.npu.current_stream().synchronize()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 255, 256])
    p.add_argument("--mm-cores", type=int, nargs="+", choices=(1, 2, 4, 8), default=[1, 2, 4, 8])
    p.add_argument("--add-cores", type=int, nargs="+", choices=(1, 2, 4, 8), default=[1, 2, 4, 8])
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--store-url", default="tcp://127.0.0.1:8581")
    p.add_argument("--chunk-rows", type=int, choices=(256, 512, 1024), default=256)
    a = p.parse_args()
    if any(m < 1 or m > 256 for m in a.rows) or a.repeat < 2:
        p.error("--rows must be in 1..256 and --repeat >= 2")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise RuntimeError("Exactly two ranks required")
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group("hccl")
    cpu_group = dist.new_group(backend="gloo")

    # The same non-quantized FP16 NZ weight contract as the V1 layer test.
    torch.manual_seed(123 + rank)
    weight = maybe_trans_nz(torch.randn(2048, 2048, dtype=torch.float16, device=device))
    xs, refs, stock = {}, {}, {}
    for m in a.rows:
        torch.manual_seed(1200 + rank * 17 + m)
        x = torch.randn(m, 2048, device=device, dtype=torch.float16)
        xs[m] = x
        local = F.linear(x, weight).float()
        dist.all_reduce(local)
        refs[m] = local.to(torch.float16)
        sync()
        timings = []
        for _ in range(a.repeat):
            t0 = time.perf_counter()
            y = F.linear(x, weight).float()
            dist.all_reduce(y)
            y = y.to(torch.float16)
            sync()
            timings.append((time.perf_counter() - t0) * 1000)
        stock[m] = statistics.median(timings)
    sync()
    dist.barrier(group=cpu_group)

    api = c_api()
    stream = torch.npu.current_stream()
    raw_stream = getattr(stream, "npu_stream", None)
    if raw_stream is None:
        raise RuntimeError("torch_npu stream has no npu_stream attribute")
    stream_ptr = ctypes.c_void_p(raw_stream)
    handle = ctypes.c_void_p()
    check(api.mf310p_v2_trial_create(rank, a.store_url.encode(), a.chunk_rows, ctypes.byref(handle)), "trial_create")
    if not handle.value:
        raise RuntimeError("trial_create returned a null handle")
    try:
        check(api.mf310p_v2_trial_ib_probe(handle, 1000, stream_ptr), "IB probe")
        if rank == 0:
            print("M\tMMCores\tAddCores\tstock_ms\tv2_direct_ms\tfirst_ms\tmax_diff\tPASS", flush=True)
        for m in a.rows:
            x = xs[m]
            ref = refs[m]
            for mm_cores in a.mm_cores:
                for add_cores in a.add_cores:
                    # Deliberately allocate y AFTER MemFabric creation;
                    # retain T0's first-output guard as a separate finding.
                    out = torch.empty_like(ref)
                    timings = []
                    first = 0.0
                    for k in range(a.repeat):
                        t0 = time.perf_counter()
                        check(
                            api.mf310p_v2_trial_run_nz_direct_small(
                                handle,
                                x.data_ptr(),
                                weight.data_ptr(),
                                out.data_ptr(),
                                m,
                                mm_cores,
                                add_cores,
                                stream_ptr,
                            ),
                            f"direct small M={m} MM={mm_cores} Add={add_cores}",
                        )
                        sync()
                        ms = (time.perf_counter() - t0) * 1000
                        if k == 0:
                            first = ms
                        else:
                            timings.append(ms)
                        status = ctypes.c_uint64(0xFFFFFFFFFFFFFFFF)
                        check(api.mf310p_v2_trial_check_status(handle, ctypes.byref(status)), "check_status")
                        if status.value:
                            raise AssertionError(f"protocol status={status.value} M={m} MM={mm_cores} Add={add_cores}")
                        if not torch.equal(out, ref):
                            diff = (out.float() - ref.float()).abs()
                            raise AssertionError(
                                f"bitwise mismatch M={m} MM={mm_cores} "
                                f"Add={add_cores} bad={(out != ref).sum().item()} "
                                f"max_abs={diff.max().item()}"
                            )
                    med = statistics.median(timings)
                    if rank == 0:
                        print(
                            f"{m}\t{mm_cores}\t{add_cores}\t{stock[m]:.3f}\t{med:.3f}\t{first:.3f}\t0\tPASS", flush=True
                        )
        dist.barrier(group=cpu_group)
    finally:
        # Requires the caller's stream to be drained before pool teardown.
        sync()
        check(api.mf310p_v2_trial_destroy(handle), "trial_destroy")
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

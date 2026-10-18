# SPDX-License-Identifier: Apache-2.0
"""T2 non-fused transport benchmark for Ascend 310P3 TP=2.

A = torch_npu native F.linear + FP16 HCCL all_reduce
B = the SAME F.linear + D2D stage + MemFabric signal/wait + vector Add
C = V2 fused MM + MemFabric AR (eligible shapes only; optional)

All HCCL work, including FP32 correctness references, completes BEFORE
MemFabric pool creation: co-running HCCL and MemFabric in a live serving
process is a separate G3 safety gate, NOT validated by this benchmark.

Build: both V1/V2 compilation flags enabled (see T2_AR_ONLY_TRIAL.md).
Example:
  ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
    csrc/_310P/memfabric_mm_ar_v2/bench_ar_only.py \
    --rows 1 16 64 128 255 256 512 1024 2048 4096 8192 \
    --chunk-rows 1024 --add-cores 1 2 4 8 --repeat 30 --fused
"""
from __future__ import annotations

import argparse
import atexit
import ctypes
import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C as extension

from vllm_ascend.utils import maybe_trans_nz


def bind_library():
    lib = ctypes.CDLL(extension.__file__)
    P, U32, U64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
    signatures = {
        "mf310p_v2_trial_create": ([ctypes.c_int, ctypes.c_char_p, U32, ctypes.POINTER(P)], ctypes.c_int),
        "mf310p_v2_trial_ib_probe": ([P, U32, P], ctypes.c_int),
        "mf310p_v2_trial_run_ar_only": ([P, U64, U64, U32, U32, P], ctypes.c_int),
        "mf310p_v2_trial_run_nz_direct_small": ([P, U64, U64, U64, U32, U32, U32, P], ctypes.c_int),
        "mf310p_v2_trial_run": ([P, U64, U64, U64, U32, U32, P], ctypes.c_int),
        "mf310p_v2_trial_check_status": ([P, ctypes.POINTER(U64)], ctypes.c_int),
        "mf310p_v2_trial_destroy": ([P], ctypes.c_int),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    return lib


def optional_test_kfc_bypass():
    """For the already-broken dev0/dev1 test KFC control plane only.

    This is the same explicit experiment workaround as T1; never enable
    it in production or count bypassed startup as a healthy deployment.
    """
    if os.environ.get("MF_KFC_BYPASS") != "1":
        return
    paths = []
    for index in range(16):
        path = f"/tmp/mf_kfc_{os.getpid()}_{index}.json"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.close(fd)
            paths.append(path)
        except FileExistsError:
            continue

    def cleanup():
        for path in paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
    atexit.register(cleanup)


def chk(rc, desc):
    if rc:
        raise RuntimeError(f"{desc}: rc={rc}")


def measure(fn, warmup, repeat):
    for _ in range(warmup):
        keepalive = fn()
        torch.npu.current_stream().synchronize()
        del keepalive
    samples = []
    for _ in range(repeat):
        begin = time.perf_counter()
        keepalive = fn()
        torch.npu.current_stream().synchronize()
        samples.append((time.perf_counter() - begin) * 1e3)
        del keepalive
    samples.sort()
    return {
        "median": round(statistics.median(samples), 5),
        "p95": round(samples[min(len(samples) - 1, int(0.95 * (len(samples) - 1)))], 5),
        "min": round(samples[0], 5),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", type=int,
                    default=[1, 16, 64, 128, 255, 256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--add-cores", nargs="+", type=int, choices=(1, 2, 4, 8), default=[1, 2, 4, 8])
    ap.add_argument("--chunk-rows", type=int, choices=(256, 512, 1024), default=1024)
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--fused", action="store_true",
                    help="Also measure the existing V2 small or T0 fused path when shape is eligible")
    ap.add_argument("--store-url", default=os.environ.get(
        "VLLM_ASCEND_310P_MEMFABRIC_STORE_URL", "tcp://127.0.0.1:8581"))
    a = ap.parse_args()
    if any(m < 1 or m > 8192 for m in a.rows):
        ap.error("M must be 1..8192; multi-wave not yet supported")
    if a.repeat < 5 or a.warmup < 1:
        ap.error("require repeat >= 5 and warmup >= 1")
    if len(set(a.rows)) != len(a.rows):
        ap.error("duplicate M values are not supported in this benchmark")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 2:
        raise RuntimeError("TP must be exactly 2")
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")
    cpu_group = dist.new_group(backend="gloo")
    stream = torch.npu.current_stream()
    raw_stream = ctypes.c_void_p(stream.npu_stream)

    torch.manual_seed(7865 + rank)
    weight = maybe_trans_nz(torch.randn(2048, 2048, dtype=torch.float16, device=device))
    x_map, reference, stock, mm_only = {}, {}, {}, {}

    # Stock measurement matches deployed FP16 HCCL AR (NOT FP32 AR).
    # Correctness reference is a SEPARATE FP32 TP sum -> FP16.
    # Neither HCCL measurement runs while MemFabric owns the pool.
    for m in a.rows:
        torch.manual_seed(2500 + rank * 103 + m)
        x = torch.randn(m, 2048, device=device, dtype=torch.float16)
        x_map[m] = x
        partial = F.linear(x, weight)
        ref = partial.float()
        dist.all_reduce(ref)
        reference[m] = ref.half()
        stream.synchronize()

        def native_hccl():
            local = F.linear(x, weight)
            dist.all_reduce(local)
            return local  # keep allocated tensor alive until stream sync

        stock[m] = measure(native_hccl, a.warmup, a.repeat)
        mm_only[m] = measure(lambda: F.linear(x, weight), a.warmup, a.repeat)
    torch.npu.synchronize()
    dist.barrier(group=cpu_group)

    optional_test_kfc_bypass()
    lib = bind_library()
    handle = ctypes.c_void_p()
    chk(lib.mf310p_v2_trial_create(
        rank, a.store_url.encode("utf-8"), a.chunk_rows,
        ctypes.byref(handle)), "trial_create")
    if not handle.value:
        raise RuntimeError("null MemFabric trial context")

    rows = []
    try:
        chk(lib.mf310p_v2_trial_ib_probe(handle, 1000, raw_stream), "IB probe")
        for m in a.rows:
            x = x_map[m]
            ref = reference[m]
            for add in a.add_cores:
                y = torch.empty_like(ref)

                def native_memfabric():
                    # Native framework MM, not V1/V2 custom MM.
                    partial = F.linear(x, weight)
                    chk(lib.mf310p_v2_trial_run_ar_only(
                        handle, partial.data_ptr(), y.data_ptr(), m, add, raw_stream),
                        f"AR-only m={m} add={add}")
                    # Keep the producer tensor alive until this Stream is
                    # synchronized in the caller. Do not return it to the
                    # caching allocator while the D2D is still pending.
                    return partial

                def run_and_verify():
                    partial = native_memfabric()
                    stream.synchronize()
                    status = ctypes.c_uint64(0)
                    chk(lib.mf310p_v2_trial_check_status(
                        handle, ctypes.byref(status)), "AR-only protocol status")
                    if status.value != 0:
                        raise RuntimeError(f"AR-only status={status.value} M={m} add={add}")
                    if not torch.equal(y, ref):
                        bad = int((y != ref).sum().item())
                        diff = float((y.float() - ref.float()).abs().max().item())
                        raise AssertionError(
                            f"AR-only bit-exact mismatch M={m} add={add} "
                            f"bad={bad} max_abs={diff}")
                    return partial

                # One correctness run before recording timings.
                run_and_verify()
                # Keep tensor lifetime valid across each individual sync;
                # measure() invokes the closure and synchronizes after it.
                t = measure(native_memfabric, a.warmup, a.repeat)
                run_and_verify()
                rows.append({"m": m, "method": "native_mm+memfabric_stage_ar",
                             "add": add, **t, "correct": True})

            if a.fused:
                fused_kind = "V2_T1_nz_direct" if m <= 256 else (
                    "V2_T0_overlap" if m % a.chunk_rows == 0 and m >= 2*a.chunk_rows
                    else "V2_T0_serial" if m % a.chunk_rows == 0 else None)
                if fused_kind:
                    yy = torch.empty_like(ref)

                    def fused():
                        if m <= 256:
                            add = 1 if m < 255 else 8
                            chk(lib.mf310p_v2_trial_run_nz_direct_small(
                                handle, x.data_ptr(), weight.data_ptr(), yy.data_ptr(),
                                m, 8, add, raw_stream), "V2 T1")
                        else:
                            chk(lib.mf310p_v2_trial_run(
                                handle, x.data_ptr(), weight.data_ptr(), yy.data_ptr(),
                                m, 1 if fused_kind.endswith("overlap") else 0,
                                raw_stream), "V2 T0")

                    # Warmup is not a Graph/production-ready guarantee;
                    # T0 may still require device-wide first-output sync.
                    fused()
                    stream.synchronize()
                    status = ctypes.c_uint64(0)
                    chk(lib.mf310p_v2_trial_check_status(
                        handle, ctypes.byref(status)), "fused protocol")
                    if status.value or not torch.equal(yy, ref):
                        raise AssertionError(f"fused mismatch M={m}, status={status.value}")
                    ft = measure(fused, a.warmup, a.repeat)
                    rows.append({"m": m, "method": fused_kind,
                                 "add": 1 if m < 255 else 8, **ft, "correct": True})

        dist.barrier(group=cpu_group)
    finally:
        stream.synchronize()
        chk(lib.mf310p_v2_trial_destroy(handle), "trial_destroy")

    # Both ranks report latencies; performance is gated by the slower
    # rank, not whichever rank prints to stdout.
    gathered = [None, None]
    dist.all_gather_object(gathered, {"rows": rows, "stock": stock, "mm_only": mm_only}, group=cpu_group)
    if rank == 0:
        mapping = {(r["m"], r["method"], r["add"]): r for r in gathered[1]["rows"]}
        print("M\tmethod\tadd\tstock_max_ms\tnative_mm_max_ms\tcandidate_max_ms\tcandidate_max_p95\t"
              "speedup_percent\tcorrect", flush=True)
        for rr in rows:
            key = (rr["m"], rr["method"], rr["add"])
            other = mapping[key]
            two = max(rr["median"], other["median"])
            p95 = max(rr["p95"], other["p95"])
            # The TP step completes only after the slower rank.
            stock_time = max(gathered[0]["stock"][rr["m"]]["median"],
                             gathered[1]["stock"][rr["m"]]["median"])
            mm_time = max(gathered[0]["mm_only"][rr["m"]]["median"],
                          gathered[1]["mm_only"][rr["m"]]["median"])
            gain = (stock_time - two) / stock_time * 100.0
            print(f"{rr['m']}\t{rr['method']}\t{rr['add']}\t"
                  f"{stock_time:.4f}\t{mm_time:.4f}\t{two:.4f}\t{p95:.4f}\t"
                  f"{gain:.1f}%\t{rr['correct'] and other['correct']}",
                  flush=True)
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

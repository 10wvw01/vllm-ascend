# SPDX-License-Identifier: Apache-2.0
"""T2 AR-only stress: 1000 sequential calls, 30 distinct M, rotating
x-seeds / add-cores / dynamic y pointers (reuse + fresh), bounded
inter-rank skew, per-call protocol status + bit-exact verification.

All HCCL references are computed BEFORE MemFabric pool creation.

D2D staging source: the harness stages through one PRE-WARMED, fixed
per-M staging buffer (stage.copy_(F.linear(...))) so the allocator does
not churn fresh addresses per call (each churned address forces the
load-bearing first-touch device-sync guard, ~5-15 s each, making a
1000-call sweep take hours). The REAL B path (D2D directly from a fresh
native-MM output) is covered bit-exactly by the full matrix (192 first
calls with fresh partials + guard) and by the srcsweep probe (42/42 with
guard, 42/42 wrong without guard). This stress targets protocol/chunk/
add-core/y-dynamic/skew robustness under repetition.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import ctypes
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C as extension

from vllm_ascend.utils import maybe_trans_nz

# 30 distinct M values incl. boundaries 1/255/256/257/8192 and multi-chunk
# shapes; deterministic on both ranks.
STRESS_M = [
    1,
    2,
    4,
    8,
    15,
    16,
    31,
    32,
    63,
    64,
    100,
    127,
    128,
    200,
    255,
    256,
    257,
    300,
    511,
    512,
    777,
    1024,
    1536,
    2048,
    2570,
    3000,
    4096,
    6000,
    8191,
    8192,
]
X_SEEDS = 4
Y_SLOTS = 4


def bind_library():
    lib = ctypes.CDLL(extension.__file__)
    P, U32, U64 = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
    signatures = {
        "mf310p_v2_trial_create": ([ctypes.c_int, ctypes.c_char_p, U32, ctypes.POINTER(P)], ctypes.c_int),
        "mf310p_v2_trial_ib_probe": ([P, U32, P], ctypes.c_int),
        "mf310p_v2_trial_run_ar_only": ([P, U64, U64, U32, U32, P], ctypes.c_int),
        "mf310p_v2_trial_check_status": ([P, ctypes.POINTER(U64)], ctypes.c_int),
        "mf310p_v2_trial_destroy": ([P], ctypes.c_int),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
    return lib


def optional_test_kfc_bypass():
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
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)

    atexit.register(cleanup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-rows", type=int, default=1024, choices=(256, 512, 1024))
    ap.add_argument("--repeat", type=int, default=1000)
    a = ap.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")
    cpu_group = dist.new_group(backend="gloo")
    stream = torch.npu.current_stream()
    raw_stream = ctypes.c_void_p(stream.npu_stream)

    torch.manual_seed(7865 + rank)
    weight = maybe_trans_nz(torch.randn(2048, 2048, dtype=torch.float16, device=device))

    # Pre-pool: inputs, HCCL references, per-M staging buffers (written once
    # so their pages settle before pool creation) and y slots.
    xs, refs, stages = {}, {}, {}
    for m in STRESS_M:
        for s in range(X_SEEDS):
            torch.manual_seed(6100 + rank * 31 + m * 4 + s)
            x = torch.randn(m, 2048, device=device, dtype=torch.float16)
            local = F.linear(x, weight).float()
            dist.all_reduce(local)
            xs[(m, s)] = x
            refs[(m, s)] = local.to(torch.float16)
        stage = torch.empty(m, 2048, dtype=torch.float16, device=device)
        stage.copy_(F.linear(xs[(m, 0)], weight))
        stages[m] = stage
    y_slots = [torch.empty(8192, 2048, dtype=torch.float16, device=device) for _ in range(Y_SLOTS)]
    torch.npu.synchronize()
    dist.barrier(group=cpu_group)
    if rank == 0:
        print(f"STRESS pre-pool refs ready: {len(refs)} combos", flush=True)

    optional_test_kfc_bypass()
    lib = bind_library()
    handle = ctypes.c_void_p()
    rc = lib.mf310p_v2_trial_create(
        rank,
        os.environ.get("VLLM_ASCEND_310P_MEMFABRIC_STORE_URL", "tcp://127.0.0.1:8581").encode("utf-8"),
        a.chunk_rows,
        ctypes.byref(handle),
    )
    assert rc == 0 and handle.value, f"create rc={rc}"
    assert lib.mf310p_v2_trial_ib_probe(handle, 1000, raw_stream) == 0

    adds = [1, 2, 4, 8]
    skew_ms_cycle = [0, 1, 5, 10, 50, 100]

    observed_m, observed_add, fresh_y_calls = set(), set(), 0
    t_start = time.perf_counter()
    status = ctypes.c_uint64(0)
    for i in range(a.repeat):
        m = STRESS_M[i % len(STRESS_M)]
        s = (i // 7) % X_SEEDS
        add = adds[(i // 3) % 4]
        use_fresh_y = (i % 97) == 96
        x = xs[(m, s)]
        ref = refs[(m, s)]

        # Bounded inter-rank skew: rank0 stalls its HOST before enqueue;
        # the device kernel mail-waits must tolerate this without timeout.
        if rank == 0 and i % 211 == 0:
            time.sleep(skew_ms_cycle[(i // 211) % len(skew_ms_cycle)] / 1000.0)

        if use_fresh_y:
            y = torch.empty(m, 2048, dtype=torch.float16, device=device)
            fresh_y_calls += 1
        else:
            y = y_slots[i % Y_SLOTS][:m]

        stage = stages[m]
        stage.copy_(F.linear(x, weight))
        # Platform coherence requirement (see bench_ar_only.py): the
        # staging kernel's MTE1 reads can miss the producer's
        # cache-resident stores; a stream sync between producer and
        # staging is the verified cure and part of B's real cost.
        stream.synchronize()
        rc = lib.mf310p_v2_trial_run_ar_only(handle, stage.data_ptr(), y.data_ptr(), m, add, raw_stream)
        assert rc == 0, f"call {i} rc={rc}"
        stream.synchronize()
        rc = lib.mf310p_v2_trial_check_status(handle, ctypes.byref(status))
        assert rc == 0 and status.value == 0, f"call {i} status={status.value}"
        if not torch.equal(y, ref):
            bad = int((y != ref).sum().item())
            raise AssertionError(f"stress call {i}: m={m} seed={s} add={add} fresh={use_fresh_y} bad={bad}/{y.numel()}")
        observed_m.add(m)
        observed_add.add(add)
        if (i + 1) % 100 == 0:
            print(f"STRESS {i + 1}/{a.repeat} ok elapsed={time.perf_counter() - t_start:.1f}s", flush=True)

    assert observed_m == set(STRESS_M), "M coverage incomplete"
    assert observed_add == set(adds), "add-cores coverage incomplete"
    assert fresh_y_calls > 0, "no fresh-y calls exercised"
    print(
        f"STRESS PASS: {a.repeat} calls, unique_m={len(observed_m)}, "
        f"adds={sorted(observed_add)}, fresh_y={fresh_y_calls}, "
        f"chunk={a.chunk_rows}, "
        f"elapsed={time.perf_counter() - t_start:.1f}s",
        flush=True,
    )

    dist.barrier(group=cpu_group)
    stream.synchronize()
    lib.mf310p_v2_trial_destroy(handle)
    dist.barrier(group=cpu_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""E2-P M4096 507014受控复现器：逐迭代心跳追踪，三臂隔离(e2p/e2r/e1)。
不修改任何正式代码；与e2_arm.py相同身份门禁；仅用于定位E2新协议挂死机制。"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
import vllm_ascend.vllm_ascend_C as extension

from u1_spec import D4_DEVICE_SHA, MEMFABRIC_HOST_SHA, PLUGIN_SHA, KVER_SHA

K = N = 2048


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("e2p", "e2r", "e1", "e2p2048"), required=True)
    p.add_argument("--mode", choices=("single", "sequence"), default="single")
    p.add_argument("--watcher", choices=("on", "off"), default="off")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--build-manifest", type=Path, required=True)
    p.add_argument("--store-url", required=True)
    a = p.parse_args()
    if int(os.getenv("WORLD_SIZE", "0")) != 2 or os.getenv("RANK") not in ("0", "1"):
        p.error("true TP2 two processes required")
    identity = {"U1_DEVICE_SHA256": D4_DEVICE_SHA, "U1_PLUGIN_SHA256": PLUGIN_SHA,
                "U1_HOST_SHA256": MEMFABRIC_HOST_SHA, "U1_KVER_SHA256": KVER_SHA}
    if any(os.environ.get(k) != v for k, v in identity.items()):
        p.error("G0 verified identity required")
    if sha(Path(extension.__file__)) != PLUGIN_SHA:
        p.error("plugin identity mismatch")
    meta = json.loads(a.build_manifest.read_text(encoding="utf-8"))
    for key, which in (("candidate_sha256", "candidate_so"),
                       ("private_host_sha256", "private_host_so")):
        lib = Path(meta.get(which, "/nonexistent"))
        if not lib.is_file() or sha(lib) != meta.get(key):
            p.error(f"{which} identity mismatch")
    a.meta = meta
    return a


def bind_trial(lib):
    P = ctypes.c_void_p; U = ctypes.c_uint64; I = ctypes.c_uint32
    sigs = {
        "mf310p_v2_trial_create": ([ctypes.c_int, ctypes.c_char_p, I, ctypes.POINTER(P)], ctypes.c_int),
        "mf310p_v2_trial_ib_probe": ([P, I, P], ctypes.c_int),
        "mf310p_v2_trial_run": ([P, U, U, U, I, I, P], ctypes.c_int),
        "mf310p_v2_trial_check_status": ([P, ctypes.POINTER(U)], ctypes.c_int),
        "mf310p_v2_trial_destroy": ([P], ctypes.c_int),
    }
    for name, (args, res) in sigs.items():
        f = getattr(lib, name); f.argtypes = args; f.restype = res
    return lib


def need(rc, label):
    if rc:
        raise RuntimeError(f"{label} failed: {rc}")


def main():
    a = parse()
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    torch.npu.config.allow_internal_format = True
    torch_npu.npu.set_compile_mode(jit_compile=False)
    dev = torch.device(f"npu:{local}")
    dist.init_process_group(backend="gloo")
    group = dist.group.WORLD
    stream = torch.npu.current_stream()
    raw = ctypes.c_void_p(stream.npu_stream)
    handle = ctypes.c_void_p()
    trial = bind_trial(ctypes.CDLL(str(extension.__file__)))
    a._dump = None
    kind = {"e1": 2, "e2r": 3, "e2p": 4, "e2p2048": 4}[a.arm]
    m = 2048 if a.arm == "e2p2048" else 4096
    try:
        torch.manual_seed(4242)
        w = torch.randn(N, K, dtype=torch.float16, device=dev)
        w_nz = torch_npu.npu_format_cast(w, 29)
        xs = []; refs = []
        for variant in range(2):
            torch.manual_seed(1200 + rank * 17 + m + variant * 1009)
            x = torch.randn(m, K, dtype=torch.float16, device=dev)
            local_partial = F.linear(x, w).float().cpu()
            dist.all_reduce(local_partial, group=group)
            xs.append(x)
            refs.append(local_partial.to(device=dev, dtype=torch.float16))
        output = torch.empty((m, N), dtype=torch.float16, device=dev)
        stream.synchronize()
        dist.barrier(group=group)
        t_create0 = time.time()
        need(trial.mf310p_v2_trial_create(
            rank, a.store_url.encode(), 1024, ctypes.byref(handle)), "pool create")
        need(trial.mf310p_v2_trial_ib_probe(handle, 1000, raw), "IB probe")
        stream.synchronize()
        candidate = ctypes.CDLL(str(a.meta["private_host_so"]))
        f = candidate.mf310p_e2_run_private
        f.argtypes = [ctypes.c_void_p] + [ctypes.c_uint64] * 3 + [ctypes.c_uint32] * 2 + [ctypes.c_void_p]
        f.restype = ctypes.c_int
        read_pool = getattr(candidate, "mf310p_e2_diag_read_pool", None)
        read_abs = getattr(candidate, "mf310p_e2_diag_read_abs", None)
        pool_info = getattr(candidate, "mf310p_e2_diag_pool_info", None)
        a._stop_flag = None
        if read_pool is not None and pool_info is not None:
            import threading
            read_pool.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p]
            read_pool.restype = ctypes.c_int
            read_abs.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p]
            read_abs.restype = ctypes.c_int
            pool_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
            pool_info.restype = ctypes.c_int
            info = (ctypes.c_uint64 * 6)()
            if pool_info(handle, info) == 0:
                reserved = int(info[4])
                ib_addr = int(info[5])
                print(f"POOLINFO rank={rank} reserved_off={reserved} ib={ib_addr:#x}", flush=True)
                stop_flag = [False]
                REQ_TAIL, REQ_HEAD = 0x0800, 0x0808
                ARR_HEAD = 0xAA20
                ARR_STAMP = 0x0A20
                ARR_MAIL = 0x2A20

                def u64(buf, off):
                    return int.from_bytes(bytes(buf[off:off+8]), "little")

                def dump(tag):
                    cells = {}
                    read64 = getattr(candidate, "mf310p_e2_diag_read64", None)
                    if read64 is not None:
                        sb0 = (ctypes.c_ubyte * 64)()
                        if read64(handle, sb0) == 0:
                            raw0 = bytes(sb0)
                            cells["status_err"] = int.from_bytes(raw0[0:8], "little")
                            cells["status_raw"] = raw0[8:40].hex()
                    buf = (ctypes.c_ubyte * 16)()
                    if read_pool(handle, reserved + REQ_TAIL, 16, buf) == 0:
                        cells["reqTail"] = u64(buf, 0)
                        cells["reqHead"] = u64(buf, 8)
                    buf2 = (ctypes.c_ubyte * 8)()
                    if read_pool(handle, reserved + ARR_HEAD, 8, buf2) == 0:
                        cells["arrHead"] = u64(buf2, 0)
                    ah = cells.get("arrHead", 0)
                    for name, off, length in (
                            ("nextStamp", ARR_STAMP + (ah % 1024) * 8, 8),
                            ("nextMail", ARR_MAIL + (ah % 1024) * 32, 32)):
                        b = (ctypes.c_ubyte * length)()
                        if read_pool(handle, reserved + off, length, b) == 0:
                            cells[name] = [u64(b, i) for i in range(0, length, 8)]
                    for extra in range(1, 4):
                        sb = (ctypes.c_ubyte * 8)()
                        if read_pool(handle, reserved + ARR_STAMP + ((ah + extra) % 1024) * 8, 8, sb) == 0:
                            cells[f"stamp+{extra}"] = u64(sb, 0)
                    if read_abs and ib_addr:
                        ibb = (ctypes.c_ubyte * 2560)()
                        if read_abs(handle, ib_addr, 2560, ibb) == 0:
                            raw = bytes(ibb)
                            cells["ib_nonzero"] = [
                                [i // 32, (i % 32) // 4, int.from_bytes(raw[i:i+4], "little")]
                                for i in range(0, 2560, 4)
                                if raw[i:i+4] != b"\x00\x00\x00\x00"
                            ][:40]
                    print(f"FORENSIC rank={rank} {tag} {cells}", flush=True)
                    return cells

                a._dump = dump

                def watcher():
                    last = None
                    stable = 0
                    while not stop_flag[0]:
                        time.sleep(0.05)
                        cells = {}
                        buf = (ctypes.c_ubyte * 16)()
                        if read_pool(handle, reserved + REQ_TAIL, 16, buf) == 0:
                            cells["reqTail"] = u64(buf, 0)
                            cells["reqHead"] = u64(buf, 8)
                        buf2 = (ctypes.c_ubyte * 8)()
                        if read_pool(handle, reserved + ARR_HEAD, 8, buf2) == 0:
                            cells["arrHead"] = u64(buf2, 0)
                        key = (cells.get("reqTail"), cells.get("reqHead"), cells.get("arrHead"))
                        if key != last:
                            print(f"CURSOR rank={rank} {cells}", flush=True)
                            last = key
                            stable = 0
                        else:
                            stable += 1
                            if stable in (60, 120, 200):
                                dump(f"STALL_{stable*50}ms")

                if a.watcher == "on":
                    wth = threading.Thread(target=watcher, daemon=True)
                    wth.start()
                a._stop_flag = stop_flag
        print(f"REPRO rank={rank} pool_ready kind={kind} M={m} create_s={time.time()-t_create0:.1f}", flush=True)
        if a.mode == "sequence":
            # campaign确切序列：同池先M2048(120次)再M4096(120次)，双变体双输出槽。
            torch.manual_seed(4242)
            w2 = torch.randn(N, K, dtype=torch.float16, device=dev)
            w_nz2 = torch_npu.npu_format_cast(w2, 29)
            data = {}
            for mm in (2048, 4096):
                xs2 = []; refs2 = []
                for variant in range(2):
                    torch.manual_seed(1200 + rank * 17 + mm + variant * 1009)
                    x2 = torch.randn(mm, K, dtype=torch.float16, device=dev)
                    lp = F.linear(x2, w2).float().cpu()
                    dist.all_reduce(lp, group=group)
                    xs2.append(x2); refs2.append(lp.to(device=dev, dtype=torch.float16))
                outs2 = [torch.empty((mm, N), dtype=torch.float16, device=dev) for _ in range(2)]
                data[mm] = (xs2, refs2, outs2)
            stream.synchronize()
            bad = 0
            for mm, niter in ((2048, 120), (4096, 320)):
                xs2, refs2, outs2 = data[mm]
                for i in range(niter):
                    variant = i % 2
                    time.sleep(0.25)
                    dist.barrier(group=group)
                    print(f"HB rank={rank} M={mm} i={i} phase=pre", flush=True)
                    t0 = time.perf_counter_ns()
                    try:
                        need(f(handle, xs2[variant].data_ptr(), w_nz2.data_ptr(),
                             outs2[variant].data_ptr(), mm, kind, raw), "candidate")
                        stream.synchronize()
                    except BaseException as ex:
                        print(f"HANG_CAUGHT rank={rank} M={mm} i={i} exc={type(ex).__name__}: {str(ex)[:200]}", flush=True)
                        if a._dump is not None:
                            try:
                                a._dump(f"POSTMORTEM_M{mm}_i{i}")
                            except BaseException as ex2:
                                print(f"POSTMORTEM_DUMP_FAIL rank={rank} {ex2}", flush=True)
                        bad = 1
                        raise
                    dt = (time.perf_counter_ns() - t0) / 1e6
                    status = ctypes.c_uint64()
                    need(trial.mf310p_v2_trial_check_status(handle, ctypes.byref(status)), "status")
                    if status.value or not torch.equal(outs2[variant], refs2[variant]):
                        print(f"REPRO_SEQ_FAIL rank={rank} M={mm} i={i} status={status.value}", flush=True)
                        bad += 1
                        break
                    print(f"HB rank={rank} M={mm} i={i} phase=post ms={dt:.3f}", flush=True)
                if bad:
                    break
            print(f"REPRO_SEQ_DONE rank={rank} bad={bad}", flush=True)
        else:
            bad = 0
            for i in range(a.warmup + a.iters):
                variant = i % 2
                x = xs[variant]; ref = refs[variant]
                dist.barrier(group=group)
                print(f"HB rank={rank} i={i} phase=pre", flush=True)
                t0 = time.perf_counter_ns()
                need(f(handle, x.data_ptr(), w_nz.data_ptr(), output.data_ptr(), m, kind, raw),
                     "candidate call")
                stream.synchronize()
                dt = (time.perf_counter_ns() - t0) / 1e6
                status = ctypes.c_uint64()
                need(trial.mf310p_v2_trial_check_status(handle, ctypes.byref(status)), "status read")
                if status.value:
                    print(f"REPRO_FAIL rank={rank} i={i} status={status.value}", flush=True)
                    bad += 1
                    break
                if not torch.equal(output, ref):
                    print(f"REPRO_BITEXACT_FAIL rank={rank} i={i}", flush=True)
                    bad += 1
                    break
                if i >= a.warmup:
                    print(f"HB rank={rank} i={i} phase=post ms={dt:.3f}", flush=True)
            print(f"REPRO_DONE rank={rank} bad={bad} iters={a.iters}", flush=True)
        dist.barrier(group=group)
    finally:
        if a._stop_flag is not None:
            a._stop_flag[0] = True
            time.sleep(0.1)
        if handle.value:
            try:
                stream.synchronize()
                trial.mf310p_v2_trial_destroy(handle)
            except BaseException:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

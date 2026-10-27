#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A1 MM影子实现 vs 原生NZ MM。每个arm独立torchrun进程；对称GM只在sym arms创建MF pool。"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

K = N = 2048
CHUNK = 1024
MS = (2048, 4096)
ARMS = ("native", "nd", "sym", "sym_clean")
SHADOW_SYMBOL = "mf310p_v2_mm_shadow_launch_async"
PINNED_D4 = "2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19"
PINNED_HOST = "1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30"
PINNED_KVER = "25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78"


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Layout(ctypes.Structure):
    # 精确对齐 memfabric310p_adapter_api.h 的 mf310p_layout_t
    _fields_ = [
        (name, ctypes.c_uint64) for name in (
            "pool_base", "own_segment", "peer_segment", "symmetric_size",
            "local_size", "send_arena", "recv_arena", "peer_recv_arena",
            "ack_slot", "peer_ack_slot", "arena_bytes", "batch_bytes",
            "expected_credit_dst", "expected_recv_base")
    ] + [(name, ctypes.c_uint32) for name in (
        "arena_rows", "batch_m", "max_batches", "reserved")]


class Symmetric:
    def __init__(self, rank: int, stream, url: str, libfile: Path):
        lib = ctypes.CDLL(str(libfile))
        P = ctypes.c_void_p
        lib.mf310p_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
                                      ctypes.c_uint64, ctypes.c_uint32,
                                      ctypes.c_uint32, ctypes.POINTER(P)]
        lib.mf310p_create.restype = ctypes.c_int
        lib.mf310p_control_barrier.argtypes = [P]
        lib.mf310p_control_barrier.restype = ctypes.c_int
        lib.mf310p_exchange_geometry.argtypes = [P]
        lib.mf310p_exchange_geometry.restype = ctypes.c_int
        lib.mf310p_get_layout.argtypes = [P, ctypes.POINTER(Layout)]
        lib.mf310p_get_layout.restype = ctypes.c_int
        lib.mf310p_destroy.argtypes = [P]
        lib.mf310p_destroy.restype = ctypes.c_int
        self.lib = lib
        self.ctx = P()
        rc = lib.mf310p_create(rank, 2, url.encode(), 96 * 1024 * 1024,
                               8192, CHUNK, ctypes.byref(self.ctx))
        if rc or not self.ctx.value:
            raise RuntimeError(f"MemFabric private pool create rc={rc}")
        try:
            for name in ("mf310p_control_barrier", "mf310p_exchange_geometry"):
                rc = getattr(lib, name)(self.ctx)
                if rc:
                    raise RuntimeError(f"{name} rc={rc}")
            self.layout = Layout()
            rc = lib.mf310p_get_layout(self.ctx, ctypes.byref(self.layout))
            if rc:
                raise RuntimeError(f"mf310p_get_layout rc={rc}")
            self.ptr = (self.layout.send_arena if rank == 0
                        else self.layout.recv_arena)
            if (not self.ptr or self.layout.arena_bytes < 4096 * N * 2
                    or self.layout.pool_base == 0):
                raise RuntimeError("invalid symmetric arena for M4096")
            # 同步/首写保护在采样前完成，而不是放进Event计时窗口。
            torch.npu.synchronize()
            acl = ctypes.CDLL("libascendcl.so")
            acl.aclrtSynchronizeDevice.argtypes = []
            acl.aclrtSynchronizeDevice.restype = ctypes.c_int
            rc = acl.aclrtSynchronizeDevice()
            if rc:
                raise RuntimeError(f"protected first device sync rc={rc}")
            acl.aclrtMemcpy.argtypes = [P, ctypes.c_size_t, P,
                                        ctypes.c_size_t, ctypes.c_int]
            acl.aclrtMemcpy.restype = ctypes.c_int
            self.acl = acl
        except BaseException:
            self.close()
            raise

    def read(self, m: int) -> torch.Tensor:
        size = m * N * 2
        buf = ctypes.create_string_buffer(size)
        rc = self.acl.aclrtMemcpy(buf, size, ctypes.c_void_p(self.ptr),
                                  size, 2)  # ACL_MEMCPY_DEVICE_TO_HOST
        if rc:
            raise RuntimeError(f"Symmetric D2H correctness read rc={rc}")
        array = np.frombuffer(buf, dtype=np.float16).copy().reshape((m, N))
        return torch.from_numpy(array)

    def close(self):
        if getattr(self, "ctx", None) and self.ctx.value:
            rc = self.lib.mf310p_destroy(self.ctx)
            self.ctx = ctypes.c_void_p()
            if rc:
                raise RuntimeError(f"MemFabric pool destroy rc={rc}")


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shadow-so", type=Path, required=True)
    p.add_argument("--build-manifest", type=Path, required=True)
    p.add_argument("--store-url", default="")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    a = p.parse_args()
    if not a.out.is_absolute() or not a.out.is_dir():
        p.error("existing absolute output directory required")
    if a.warmup < 20 or a.iters < 100 or a.iters > 500:
        p.error("minimum 20 warmup + 100 useful samples, max500")
    if any(a.out.glob("*rank*.json")):
        p.error("never overwrite rank results")
    if int(os.environ.get("WORLD_SIZE", "0")) != 2:
        p.error("actual TP2 two ranks required")
    if any(k in os.environ for k in (
        "MF_KFC_BYPASS", "MF310P_V2_NOWARMUP",
        "MF310P_V2_AR_ONLY_NOWARMUP", "MF_SDMA_ORCH_EPOCH_BUDGET_S")):
        p.error("do not bypass KFC, first-write or epoch safety")
    if not a.shadow_so.is_absolute() or not a.build_manifest.is_absolute():
        p.error("explicit absolute shadow SO and manifest required")
    manifest = json.loads(a.build_manifest.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "A1_MM_SHADOW_BUILD_V1"
            or manifest.get("so_sha256") != digest(a.shadow_so)):
        p.error("shadow device binary not verified against current build")
    if os.environ.get("U1_DEVICE_SHA256") != PINNED_D4:
        p.error("actual frozen D4 SHA unverified")
    if a.arm in ("sym", "sym_clean"):
        if not a.store_url.startswith("tcp://127.0.0.1:"):
            p.error("private local MemFabric store URL required")
        path = Path(os.environ.get("U1_CORE_SO_LIB64", ""))
        if not path.is_file() or digest(path) != PINNED_HOST:
            p.error("real patched private Host SO not verified")
        if os.environ.get("U1_HOST_SHA256") != PINNED_HOST or os.environ.get("U1_KVER_SHA256") != PINNED_KVER:
            p.error("real Host/KVER identity not verified")
    return a, manifest


def main():
    a, manifest = parse()
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local)
    torch.npu.config.allow_internal_format = True
    torch_npu.npu.set_compile_mode(jit_compile=False)
    device = torch.device(f"npu:{local}")
    dist.init_process_group(backend="gloo")
    mem = None
    failure = None
    try:
        stream = torch.npu.current_stream()
        ptr = ctypes.c_void_p(stream.npu_stream)
        if a.arm in ("sym", "sym_clean"):
            mem = Symmetric(rank, stream, a.store_url,
                            Path(os.environ["U1_CORE_SO_LIB64"]))
        lib = ctypes.CDLL(str(a.shadow_so))
        fn = getattr(lib, SHADOW_SYMBOL)
        fn.argtypes = [ctypes.c_uint64]*3 + [ctypes.c_uint32]*3 + [ctypes.c_void_p]
        fn.restype = ctypes.c_int
        if a.arm != "native":
            for m in MS:
                for _ in range(2):
                    rc = fn(0, 0, 0, m, 0, 0, ptr)
                    if rc:
                        raise RuntimeError(f"disabled shadow symbol warmup rc={rc}")
            stream.synchronize()
        torch.manual_seed(4242)
        w = torch.randn(N, K, dtype=torch.float16, device=device)
        w_nz = torch_npu.npu_format_cast(w, 29)
        records = []
        for m in MS:
            xs, refs = [], []
            for variant in range(2):
                torch.manual_seed(1200 + rank*17 + m + variant*1009)
                x = torch.randn(m, K, dtype=torch.float16, device=device)
                xs.append(x)
                refs.append(F.linear(x, w).cpu())
            outs = [torch.empty((m,N),dtype=torch.float16,device=device)
                    for _ in range(2)] if a.arm == "nd" else [None,None]
            stream.synchronize()
            dist.barrier()
            samples = []
            e0, e1 = (torch.npu.Event(enable_timing=True),
                      torch.npu.Event(enable_timing=True))
            count = a.warmup + a.iters
            for i in range(count):
                variant = i % 2
                x = xs[variant]
                def call():
                    if a.arm == "native":
                        return F.linear(x, w_nz)
                    c_ptr = (outs[variant].data_ptr() if a.arm == "nd" else mem.ptr)
                    mode_clean = 1 if a.arm == "sym_clean" else 0
                    rc = fn(x.data_ptr(), w_nz.data_ptr(), c_ptr, m,
                            mode_clean, 1, ptr)
                    if rc:
                        raise RuntimeError(f"shadow MM launch rc={rc}")
                    return outs[variant] if a.arm == "nd" else None
                t0=time.perf_counter_ns()
                e0.record(stream)
                got = call()
                e1.record(stream)
                stream.synchronize()
                wall=(time.perf_counter_ns()-t0)/1000.
                ev=e0.elapsed_time(e1)*1000.
                if not all(math.isfinite(x) and x>0 for x in (ev,wall)):
                    raise RuntimeError(f"M{m} rank{rank} invalid event")
                # 检查每个warmup、首末样本及约10%的measured样本；
                # 读回/torch.equal在Event窗口之外，不允许把D2H计入MM时间。
                valid = (i < a.warmup or i == count-1 or i == a.warmup or
                         (i-a.warmup) % 10 == 0)
                if valid:
                    observed = mem.read(m) if mem else got.cpu()
                    if not torch.equal(observed, refs[variant]):
                        raise AssertionError(
                            f"MM shadow bit-exact failure rank{rank} arm{a.arm} M{m} sample{i}")
                if i >= a.warmup:
                    samples.append({"i":i-a.warmup, "variant":variant,
                                    "event_us":round(ev,4), "wall_us":round(wall,4)})
            vals=[x["event_us"] for x in samples]
            records.append({"M":m,"rank":rank,"arm":a.arm,"warmup":a.warmup,
                "iters":a.iters,"bit_exact":True,"samples":samples,
                "event_p50_us":round(statistics.median(vals),3),
                "note":"NPU stream Event; not an in-fusion MM phase timer"})
            print(f"MM SHADOW {a.arm} rank{rank} M{m} correctness+samples PASS",flush=True)
        output=a.out/f"{a.arm}_rank{rank}.json"
        with output.open("x",encoding="utf-8") as f:
            json.dump({"schema":"A1_MM_SHADOW_RANK_V1",
                       "arm":a.arm,"rank":rank,"world_size":2,
                       "shadow_sha256":manifest["so_sha256"],
                       "d4_sha256":PINNED_D4,"mm_geometry":"M1024x2/4,N2048,K2048,8core",
                       "weight_layout":"NZ29","profile":False,
                       "memory":"MemFabric symmetric arena" if mem else "ordinary GM",
                       "device_span_is_complete_kernel":True,
                       "sym_pool_created":mem is not None,
                       "results":records},f,ensure_ascii=False,indent=2)
            f.write("\n")
        dist.barrier()
    except BaseException as err:
        failure=err
        raise
    finally:
        if mem is not None:
            try:
                torch.npu.synchronize()
                mem.close()
            except BaseException:
                if failure is None:
                    raise
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

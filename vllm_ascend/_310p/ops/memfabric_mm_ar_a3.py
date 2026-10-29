# SPDX-License-Identifier: Apache-2.0
"""Explicit A3: immutable full S, independent leased Y, one-stage Direct ADD.

One owning stream; a busy fixed seat is terminal and publishes first READY ready=0.
No hot Host collective or stream synchronization. Not enabled in serving.
"""

from __future__ import annotations

import ctypes
import time

try:
    from .memfabric_mm_ar_v3 import V3Session
except ImportError:
    from memfabric_mm_ar_v3 import V3Session


class A3Session(V3Session):
    _create_symbol = "mf310p_a3_create"
    MAX_PENDING = 64

    def __init__(self, *args, tensor_module, variant="a3", **kwargs):
        symbols = {"a3": "mf310p_a3_create", "a3-2": "mf310p_a3_2_create", "a3-tail": "mf310p_a3_tail_create"}
        if variant not in symbols:
            raise ValueError("unknown A3 variant")
        self.variant = variant
        self._create_symbol = symbols[variant]
        super().__init__(*args, **kwargs)
        self.tensor_module = tensor_module
        self.initialized = False
        self.inflight = []
        self.free_events = []
        self.phase_log = None
        p, u, q = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
        signatures = {
            "mf310p_a3_warm": [p, p],
            "mf310p_a3_reserve": [p, u, u, u, u, p, ctypes.POINTER(u), ctypes.POINTER(u), ctypes.POINTER(q)],
            "mf310p_a3_run": [p, q, q, u, u, u, u, u, u, u, p],
            "mf310p_v32_release": [p, u, u],
        }
        self._fn = {}
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, ctypes.c_int
            self._fn[name] = fn
        # Reused per-call output cells; every call runs under self.lock.
        self._slot_cell = ctypes.c_uint32()
        self._gen_cell = ctypes.c_uint32()
        self._addr_cell = ctypes.c_uint64()
        self._torch_npu = None

    def initialize(self):
        import torch

        with self.lock:
            self._ready()
            if self.initialized:
                return
            self.synchronize_check()
            try:
                import torch_npu

                self._torch_npu = torch_npu
                self._check(self._fn["mf310p_a3_warm"](self.handle, self.raw_stream), "A3 cold warm")
                self.free_events = [torch.npu.Event() for _ in range(self.MAX_PENDING)]
                for event in self.free_events:
                    event.record(self.stream)
                self.stream.synchronize()
                self.initialized = True
            except BaseException:
                self.failed = True
                raise

    def _reap(self):
        try:
            while self.inflight and self.inflight[0][0].query():
                event, _inputs = self.inflight.pop(0)
                self.free_events.append(event)
        except BaseException:
            self.failed = True
            raise

    def run_pool(self, x, weight_nz=None, *, compute_rows=1024, comm_rows=1024, partial=False):
        torch_npu = self._torch_npu
        with self.lock:
            stamps = [time.perf_counter_ns()] if getattr(self, "phase_log", None) is not None else None
            self._ready()
            if not self.initialized:
                raise RuntimeError("call initialize collectively after IB/ADD probes")
            if torch_npu is None:
                raise RuntimeError("A3 initialize must run before run_pool")
            if compute_rows not in (512, 1024) or comm_rows not in (512, 1024, 2048):
                raise ValueError("A3 compute_rows=512/1024, comm_rows=512/1024/2048")
            if x.ndim != 2 or x.shape[1] != 2048 or not 0 < x.shape[0] <= 4096:
                raise ValueError("A3 input must be [M<=4096,2048]")
            if x.shape[0] % compute_rows or x.shape[0] % comm_rows:
                raise ValueError("pad input explicitly, including padding in complete cost")
            self._reap()
            self._tensor(x)
            if torch_npu.get_npu_format(x) != 2:
                raise ValueError("A3 input requires native ND")
            partial_int = int(partial)
            if not partial_int:
                self._tensor(weight_nz, contiguous=False)
                if tuple(weight_nz.shape) != (2048, 2048) or torch_npu.get_npu_format(weight_nz) != 29:
                    raise ValueError("A3 weight requires native NZ")
            inputs, self.pending = self.pending, []
            slot, generation, address = self._slot_cell, self._gen_cell, self._addr_cell
            if stamps is not None:
                stamps.append(time.perf_counter_ns())
            rc = self._fn["mf310p_a3_reserve"](
                self.handle,
                x.shape[0],
                compute_rows,
                comm_rows,
                partial_int,
                self.raw_stream,
                ctypes.byref(slot),
                ctypes.byref(generation),
                ctypes.byref(address),
            )
            if stamps is not None:
                stamps.append(time.perf_counter_ns())
            if rc not in (0, -23):
                self._check(rc, "A3 reserve")
            admitted = rc == 0
            reason = None if admitted else RuntimeError("A3 fixed output seat held; stop BOTH ranks")
            token = (slot.value, generation.value)
            released = False

            def release():
                nonlocal released
                with self.lock:
                    if not released:
                        self._check(self._fn["mf310p_v32_release"](self.handle, *token), "A3 Storage release")
                        released = True

            out = None
            if admitted:
                try:
                    out = self.tensor_module.wrap(address.value, x.shape[0], self.device, self, release)
                except BaseException as exc:
                    release()
                    admitted, reason = False, exc
            if stamps is not None:
                stamps.append(time.perf_counter_ns())
            if not self.free_events:
                admitted, reason = False, RuntimeError("A3 pending input capacity exhausted; stop BOTH ranks")
            try:
                self._check(
                    self._fn["mf310p_a3_run"](
                        self.handle,
                        x.data_ptr(),
                        0 if partial_int else weight_nz.data_ptr(),
                        x.shape[0],
                        compute_rows,
                        comm_rows,
                        partial_int,
                        *token,
                        int(admitted),
                        self.raw_stream,
                    ),
                    "A3 enqueue",
                )
                if not admitted:
                    self.failed = True
                    raise RuntimeError("A3 first READY rejected locally; stop BOTH ranks") from reason
                if stamps is not None:
                    stamps.append(time.perf_counter_ns())
                event = self.free_events.pop()
                # Keep inputs BEFORE recording: record failure cannot free live input.
                self.inflight.append((event, inputs))
                event.record(self.stream)
                if stamps is not None:
                    stamps.append(time.perf_counter_ns())
                    self.phase_log.append(
                        dict(
                            zip(
                                ("validate_reap", "reserve", "wrap", "enqueue", "input_event"),
                                ((b - a) / 1000 for a, b in zip(stamps, stamps[1:])),
                            )
                        )
                    )
            except BaseException:
                self.failed = True
                self.pending.extend(inputs)  # preserve uncertain wave inputs
                raise
            return out

    def set_trace(self, enabled):
        with self.lock:
            self._ready()
            self.stream.synchronize()
            fn = self.lib.mf310p_v3_trace_control
            fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int
            self._check(fn(self.handle, int(enabled)), "A3 trace control")

    def fetch_trace(self):
        with self.lock:
            self._ready()
            self.stream.synchronize()
            data = (ctypes.c_uint64 * 1024)()
            fn = self.lib.mf310p_v3_trace_fetch
            fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32], ctypes.c_int
            self._check(fn(self.handle, data, ctypes.sizeof(data)), "A3 trace fetch")
            return list(data)

    def synchronize_check(self):
        super().synchronize_check()
        if hasattr(self, "inflight"):
            self._reap()

    def run(self, *args, **kwargs):
        raise RuntimeError("A3Session only supports run_pool")

    def close(self):
        with self.lock:
            if not self.handle.value:
                return
            self._ready()
            self.synchronize_check()
            rc = self.lib.mf310p_v3_destroy(self.handle)
            if rc == -26:
                raise RuntimeError("release all outputs and aliases on BOTH ranks before close")
            self._check(rc, "A3 close")
            self.handle = ctypes.c_void_p()
            self.outputs.clear()
            self.free_events.clear()

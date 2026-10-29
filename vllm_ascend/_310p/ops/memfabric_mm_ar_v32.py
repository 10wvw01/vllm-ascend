# SPDX-License-Identifier: Apache-2.0
"""Explicit V3.2 Plan A prototype. Owning stream only; outputs must not escape it.

The C++ Storage deleter, including aliases, owns the session and its slot lease.
No tensor-finalizer heuristic and no caching-allocator ownership of SMEM memory.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

try:
    from .memfabric_mm_ar_v3 import V3Session
except ImportError:  # isolated quick worker imports without loading vLLM
    from memfabric_mm_ar_v3 import V3Session


class V32Session(V3Session):
    _create_symbol = "mf310p_v32_create"

    def __init__(self, *args, tensor_module, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor_module = tensor_module
        p, u, q = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
        signatures = {
            "mf310p_v32_acquire": [p, u, u, u, p, ctypes.POINTER(u), ctypes.POINTER(u), ctypes.POINTER(q)],
            "mf310p_v32_release": [p, u, u],
            "mf310p_v32_prepare": [p, u, u, u, p],
            "mf310p_v32_run": [p, q, q, u, u, u, u, u, p],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, ctypes.c_int

    def run_pool(self, x, weight_nz=None, *, group=1024, partial=False):
        import torch_npu

        with self.lock:
            self._ready()
            if group not in (512, 1024, 2048) or x.ndim != 2 or x.shape[1] != 2048:
                raise ValueError("Plan A requires [M,2048], chunk512/1024/2048")
            if not 0 < x.shape[0] <= 4096 or x.shape[0] % group:
                raise ValueError("pad input explicitly; include padding cost in full timings")
            old_count = len(self.pending)
            try:
                self._tensor(x)
                if torch_npu.get_npu_format(x) != 2:
                    raise ValueError("Plan A input must have native ND format2")
                if not partial:
                    self._tensor(weight_nz, contiguous=False)
                    if tuple(weight_nz.shape) != (2048, 2048) or torch_npu.get_npu_format(weight_nz) != 29:
                        raise ValueError("native NZ weight format29 required")
            except BaseException:
                # Invalid local arguments have not entered a device wave.
                del self.pending[old_count:]
                raise
            slot, generation, ptr = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint64()
            try:
                rc = self.lib.mf310p_v32_acquire(
                    self.handle,
                    x.shape[0],
                    group,
                    int(partial),
                    self.raw_stream,
                    ctypes.byref(slot),
                    ctypes.byref(generation),
                    ctypes.byref(ptr),
                )
            except BaseException:
                self.failed = True
                raise
            if rc in (0, -23):
                # acquire drained the owning stream, including previous inputs.
                # The just-validated inputs are still needed by the NEW wave.
                del self.pending[:old_count]
            if rc == -23:
                self.pending.clear()  # no new wave started; both ranks agreed busy
                raise RuntimeError("both ranks must release an output slot, including all aliases, before retry")
            self._check(rc, "pool acquire")
            token = (slot.value, generation.value)

            released = False

            def release():
                nonlocal released
                # Idempotent at the Python boundary, including a wrap constructor
                # that failed before/after taking ownership of the callback.
                with self.lock:
                    if not released:
                        self._check(self.lib.mf310p_v32_release(self.handle, *token), "Storage release")
                        released = True

            out, wrap_error = None, None
            try:
                out = self.tensor_module.wrap(ptr.value, x.shape[0], self.device, self, release)
            except BaseException as exc:
                wrap_error = exc
                release()
            try:
                # Always enter prepare on both ranks, even if local wrapping failed.
                # No peer may launch MM/ADD before this agreement succeeds.
                rc = self.lib.mf310p_v32_prepare(self.handle, *token, int(wrap_error is None), self.raw_stream)
                self._check(rc, "output prepare")
                if wrap_error is not None:
                    raise RuntimeError("Storage wrapping failed; stop BOTH ranks") from wrap_error
                self._check(
                    self.lib.mf310p_v32_run(
                        self.handle,
                        x.data_ptr(),
                        0 if partial else weight_nz.data_ptr(),
                        x.shape[0],
                        group,
                        int(partial),
                        *token,
                        self.raw_stream,
                    ),
                    "Plan A run",
                )
            except BaseException:
                self.failed = True
                if wrap_error is not None:
                    raise RuntimeError("Storage wrapping failed; prepare aborted BOTH ranks") from wrap_error
                raise
            return out

    def run(self, *args, **kwargs):
        raise RuntimeError("V32Session uses run_pool; V3.1 protocol cannot mix in this session")

    def close(self):
        # Busy is recoverable: held Storage keeps the pool alive. Failed ADD is not.
        with self.lock:
            if not self.handle.value:
                return
            self._ready()
            try:
                self.stream.synchronize()
            except BaseException:
                self.failed = True
                raise
            try:
                rc = self.lib.mf310p_v3_destroy(self.handle)
            except BaseException:
                self.failed = True
                raise
            if rc == -26:
                raise RuntimeError("release all outputs and Storage aliases on BOTH ranks before close")
            self._check(rc, "destroy")
            self.handle = ctypes.c_void_p()
            self.outputs.clear()
            self.pending.clear()


def load_tensor_module(build_directory):
    """Explicit private build; run before timing/device admission, never in serving."""
    import torch_npu
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[3]
    source = root / "csrc/_310P/memfabric_mm_ar_v3/src/torch_adapter/pool_tensor.cpp"
    home = Path(torch_npu.__file__).resolve().parent
    return load(
        name="mf310p_v32_tensor",
        sources=[str(source)],
        extra_include_paths=[str(home / "include"), str(home / "include/third_party/hccl/inc")],
        extra_ldflags=[f"-L{home / 'lib'}", "-ltorch_npu", f"-Wl,-rpath,{home / 'lib'}"],
        build_directory=str(build_directory),
        verbose=True,
    )

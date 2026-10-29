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
            self._tensor(x)
            if torch_npu.get_npu_format(x) != 2:
                raise ValueError("Plan A input must have native ND format2")
            if not partial:
                self._tensor(weight_nz, contiguous=False)
                if tuple(weight_nz.shape) != (2048, 2048) or torch_npu.get_npu_format(weight_nz) != 29:
                    raise ValueError("native NZ weight format29 required")
            slot, generation, ptr = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint64()
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
            if rc == -23:
                raise RuntimeError("both ranks must release an output slot, including all aliases, before retry")
            self._check(rc, "pool acquire")
            token = (slot.value, generation.value)

            def release():
                # Storage destructor is host-only; reuse waits for the owning stream.
                with self.lock:
                    self._check(self.lib.mf310p_v32_release(self.handle, *token), "Storage release")

            # Wrap BEFORE enqueue: failure releases an unused slot, never an uncertain ADD.
            # Wrap owns the callback once invoked, including exceptions inside C++.
            out = self.tensor_module.wrap(ptr.value, x.shape[0], self.device, self, release)
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
            return out

    def run(self, *args, **kwargs):
        raise RuntimeError("V32Session uses run_pool; V3.1 protocol cannot mix in this session")

    def close(self):
        # Busy is recoverable: held Storage keeps the pool alive. Failed ADD is not.
        with self.lock:
            if not self.handle.value:
                return
            self._ready()
            self.stream.synchronize()
            rc = self.lib.mf310p_v3_destroy(self.handle)
            if rc == -26:
                raise RuntimeError("release all outputs and Storage aliases before close")
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

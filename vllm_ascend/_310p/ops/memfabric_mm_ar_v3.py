# SPDX-License-Identifier: Apache-2.0
"""Explicit, eager-only V3 adapter. Importing this module never enables serving.

The standalone SO avoids rebuilding/loading the old KVER11 vLLM extension.
Keep all input/output allocations alive until close; use one session per process.
"""

from __future__ import annotations

import ctypes
import threading
from pathlib import Path


def padded_rows(rows: int, group: int) -> int:
    if rows < 1 or rows > 4096 or group not in (256, 512, 1024, 2048):
        raise ValueError("V3 requires 1<=M<=4096 and group=256/512/1024/2048")
    return (rows + group - 1) // group * group


class V3Session:
    """PyTorch thin adapter to the public C ABI; no implicit HCCL or fallback."""

    def __init__(self, library: str | Path, rank: int, store_url: str, *, packet_bytes: int = 1 << 20):
        import torch

        if rank not in (0, 1):
            raise ValueError("TP=2 rank must be 0 or 1")
        self.library = Path(library).resolve(strict=True)
        self.lib = ctypes.CDLL(str(self.library), mode=ctypes.RTLD_LOCAL)
        self.handle = ctypes.c_void_p()
        self.lock = threading.RLock()
        self.failed = False
        self.outputs: dict[int, object] = {}
        self.pending: list[object] = []
        self.device = torch.npu.current_device()
        self.stream = torch.npu.current_stream()
        self.raw_stream = ctypes.c_void_p(self.stream.npu_stream)
        self._bind()
        if self.lib.mf310p_v3_abi() != 1:
            raise RuntimeError("V3 C ABI mismatch")
        self._check(
            self.lib.mf310p_v3_create(rank, store_url.encode(), packet_bytes, ctypes.byref(self.handle)), "create"
        )

    def _bind(self):
        p, u, q = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
        signatures = {
            "mf310p_v3_abi": ([], u),
            "mf310p_v3_create": ([ctypes.c_int, ctypes.c_char_p, u, ctypes.POINTER(p)], ctypes.c_int),
            "mf310p_v3_ib_probe": ([p, u, p], ctypes.c_int),
            "mf310p_v3_run": ([p, q, q, q, u, u, u, u, p], ctypes.c_int),
            "mf310p_v3_probe_add": ([p, q, q, q, u, p], ctypes.c_int),
            "mf310p_v3_status": ([p, ctypes.POINTER(q)], ctypes.c_int),
            "mf310p_v3_destroy": ([p], ctypes.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, result

    def _check(self, rc, op):
        if rc:
            self.failed = True
            raise RuntimeError(f"V3 {op}: rc={rc}; stop BOTH ranks; never retry an uncertain ADD")

    def _ready(self):
        import torch

        if not self.handle.value or self.failed:
            raise RuntimeError("V3 session closed/failed")
        if torch.npu.current_device() != self.device or torch.npu.current_stream() != self.stream:
            raise RuntimeError("V3 requires the owning device and original stream")
        capture = getattr(torch.npu, "is_current_stream_capturing", None)
        if capture is not None and capture():
            raise RuntimeError("V3 graph capture/replay is not qualified")

    def _tensor(self, tensor, *, contiguous=True, output=False):
        import torch

        if tensor.device.type != "npu" or tensor.device.index != self.device or tensor.dtype != torch.float16:
            raise ValueError("V3 buffers must be FP16 on the owning NPU")
        if contiguous and not tensor.is_contiguous():
            raise ValueError("V3 requires contiguous ND data (weight uses native NZ)")
        if tensor.requires_grad:
            raise ValueError("V3 is inference-only; detach buffers before calling")
        # Keep outputs through the session for first-write identity. Input references
        # only need to survive through stream completion, not every historical call.
        key = tensor.untyped_storage().data_ptr()
        if output:
            if key not in self.outputs and len(self.outputs) >= 64:
                raise RuntimeError("V3 output lifetime slots exhausted")
            self.outputs[key] = tensor
        else:
            self.pending.append(tensor)

    def ib_probe(self, repetitions=64):
        with self.lock:
            self._ready()
            self._check(self.lib.mf310p_v3_ib_probe(self.handle, repetitions, self.raw_stream), "IB probe")

    def probe_add(self, src, seed, out):
        with self.lock:
            self._ready()
            for t in (src, seed):
                self._tensor(t)
            self._tensor(out, output=True)
            if src.numel() != seed.numel() or src.numel() != out.numel():
                raise ValueError("ADD probe shape mismatch")
            self._check(
                self.lib.mf310p_v3_probe_add(
                    self.handle, src.data_ptr(), seed.data_ptr(), out.data_ptr(), src.numel() * 2, self.raw_stream
                ),
                "FP16 ADD probe",
            )
        return out

    def run(self, x, weight_nz, out, *, group=1024, pipeline=True, partial=False):
        import torch_npu

        with self.lock:
            self._ready()
            self._tensor(x)
            self._tensor(out, output=True)
            if x.ndim != 2 or x.shape[1] != 2048 or out.shape != x.shape:
                raise ValueError("V3 X/out must have equal [padded_M,2048] shapes")
            if padded_rows(x.shape[0], group) != x.shape[0]:
                raise ValueError("raw V3 run requires whole groups; use run_exact for padding")
            if not partial:
                self._tensor(weight_nz, contiguous=False)
                if tuple(weight_nz.shape) != (2048, 2048) or torch_npu.get_npu_format(weight_nz) != 29:
                    raise ValueError("V3 weight must be native NZ [2048,2048], format29")
            if x.untyped_storage().data_ptr() == out.untyped_storage().data_ptr():
                raise ValueError("V3 input/output storage must not alias")
            self._check(
                self.lib.mf310p_v3_run(
                    self.handle,
                    x.data_ptr(),
                    0 if partial else weight_nz.data_ptr(),
                    out.data_ptr(),
                    x.shape[0],
                    group,
                    int(pipeline),
                    int(partial),
                    self.raw_stream,
                ),
                "run",
            )
        return out

    def run_exact(self, x, weight_nz, out, *, input_stage, group=1024, pipeline=True):
        """Explicit stage/outputs supplied by the caller; padding cost belongs in full timing."""
        with self.lock:
            self._ready()
            rows = int(x.shape[0])
            padded = padded_rows(rows, group)
            if tuple(out.shape) != (padded, 2048):
                raise ValueError("run_exact requires output sized for padded M")
            if rows == padded:
                return self.run(x, weight_nz, out, group=group, pipeline=pipeline)
            self._tensor(x)
            self._tensor(input_stage)
            if tuple(input_stage.shape) != (padded, 2048):
                raise ValueError("incorrect input stage shape")
            input_stage[:rows].copy_(x)
            input_stage[rows:].zero_()
            self.run(input_stage, weight_nz, out, group=group, pipeline=pipeline)
            return out[:rows]

    def synchronize_check(self):
        with self.lock:
            self._ready()
            try:
                self.stream.synchronize()
            except BaseException:
                self.failed = True
                raise
            status = ctypes.c_uint64()
            self._check(self.lib.mf310p_v3_status(self.handle, ctypes.byref(status)), "status")
            self._check(status.value, "device protocol")
            self.pending.clear()

    def close(self):
        # A failed device wave must be left for the bounded external process guard.
        # Never perform another collective or retry from a failure cleanup path.
        if not self.handle.value:
            return
        if self.failed:
            raise RuntimeError("failed V3 session: preserve evidence and stop both worker processes")
        with self.lock:
            self.stream.synchronize()
            self._check(self.lib.mf310p_v3_destroy(self.handle), "destroy")
            self.handle = ctypes.c_void_p()
            self.outputs.clear()
            self.pending.clear()

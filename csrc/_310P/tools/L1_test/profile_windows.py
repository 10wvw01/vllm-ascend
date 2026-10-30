# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only host ranges; no hooks are installed in off/host timing paths."""
import ctypes
import threading
from contextlib import contextmanager


def label(job,rank,m,scenario,first,count):
    return f'L1/{job["id"]}/r{rank}/m{m}/{scenario}/i{first}/n{count}'


_MSTX_LOCK=threading.Lock()
_MSTX_LIB=None


def _mstx_lib():
    """Resolve the CANN MSTX C API once per process.

    torch_npu's python mstx wrapper resolves to a no-op returning 0 when an
    external msprof CLI owns collection (verified on CANN 9.1.0 /
    torch_npu 2.10.0.post4), so measured-window attribution must call
    mstxRangeStartA/mstxRangeEnd from libms_tools_ext.so directly, exactly as
    the official CANN mstx samples do under ``msprof --msproftx=on``.
    """
    global _MSTX_LIB
    with _MSTX_LOCK:
        if _MSTX_LIB is None:
            lib=ctypes.CDLL('libms_tools_ext.so')
            lib.mstxRangeStartA.argtypes=[ctypes.c_char_p,ctypes.c_void_p]
            lib.mstxRangeStartA.restype=ctypes.c_void_p
            lib.mstxRangeEnd.argtypes=[ctypes.c_void_p]
            lib.mstxRangeEnd.restype=None
            _MSTX_LIB=lib
    return _MSTX_LIB


@contextmanager
def marked_range(name, backend, torch):
    if backend=='msprof':
        lib=_mstx_lib()
        token=lib.mstxRangeStartA(name.encode(),None)  # Host range; end is after owning Stream join.
        if not token:
            raise RuntimeError('MSTX range_start failed; cannot attribute msprof capture safely')
        try:
            yield
        finally:
            lib.mstxRangeEnd(ctypes.c_void_p(token))
    elif backend=='profiler':
        with torch.autograd.profiler.record_function(name):
            yield
    else:
        raise ValueError('range markers are diagnostic-only')

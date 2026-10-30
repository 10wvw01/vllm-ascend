# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only host ranges; no hooks are installed in off/host timing paths."""
from contextlib import contextmanager


def label(job,rank,m,scenario,first,count):
    return f'L1/{job["id"]}/r{rank}/m{m}/{scenario}/i{first}/n{count}'


@contextmanager
def marked_range(name, backend, torch):
    if backend=='msprof':
        from torch_npu.npu.mstx import mstx
        token=mstx.range_start(name)  # Host range; end is after owning Stream join.
        if type(token) is not int or token<=0:
            raise RuntimeError('MSTX range_start failed; cannot attribute msprof capture safely')
        try:
            yield
        finally:
            mstx.range_end(token)
    elif backend=='profiler':
        with torch.autograd.profiler.record_function(name):
            yield
    else:
        raise ValueError('range markers are diagnostic-only')

# SPDX-License-Identifier: Apache-2.0
"""L2 trace control plumbing at the Python adapter boundary; no Torch/NPU."""

import ctypes
import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest

OPS = Path(__file__).parents[4] / "vllm_ascend/_310p/ops"


def load(monkeypatch, name):
    spec = importlib.util.spec_from_file_location(name, OPS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class Input:
    ndim = 2
    shape = (2048, 2048)

    def data_ptr(self):
        return 4096


class Output:
    def __init__(self, release):
        self.release = release

    def __del__(self):
        self.release()


class Lib:
    def __init__(self):
        self.generation = 0
        self.trace_enabled = None
        self.fetch_rc = -5
        self.runs = 0

    def mf310p_v32_wave_acquire(self, *args):
        self.generation += 1
        args[-3]._obj.value = 0
        args[-2]._obj.value = self.generation
        args[-1]._obj.value = 0x10000000
        return 0

    def mf310p_v32_acquire(self, *args):
        return self.mf310p_v32_wave_acquire(*args)

    def mf310p_v32_prepare(self, handle, slot, generation, ready, stream):
        return 0

    def mf310p_v32_run(self, *args):
        self.runs += 1
        return 0

    def mf310p_v32_release(self, handle, slot, generation):
        return 0

    def mf310p_v3_trace_control(self, handle, enable):
        self.trace_enabled = enable
        self.fetch_rc = 0 if enable else -5
        return 0

    def mf310p_v3_trace_fetch(self, handle, buffer, size):
        if self.fetch_rc:
            return self.fetch_rc
        for i in range(len(buffer)):
            buffer[i] = i
        return 0


@pytest.fixture
def session(monkeypatch):
    load(monkeypatch, "memfabric_mm_ar_v3")
    module = load(monkeypatch, "memfabric_mm_ar_v32")
    monkeypatch.setitem(sys.modules, "torch_npu", types.SimpleNamespace(get_npu_format=lambda t: 2))
    assert module.TRACE_U64 == 512
    s = module.V32Session.__new__(module.V32Session)
    s.lock = threading.RLock()
    s.failed = False
    s.handle = ctypes.c_void_p(1)
    s.raw_stream, s.device = None, 0
    s.pending, s.outputs = [], {}
    s.lib = Lib()
    s.tensor_module = types.SimpleNamespace(wrap=lambda *args: Output(args[-1]))
    s.stream = types.SimpleNamespace(synchronize=lambda: None)
    s._tensor = lambda x: s.pending.append(x)
    s._ready = lambda: None
    s.phase_log = None
    s.merged_wave = True
    return s


def test_trace_round_trip_and_run_unaffected(session):
    with pytest.raises(RuntimeError, match="trace fetch"):
        session.fetch_trace()  # buffer not allocated yet
    session.set_trace(True)
    assert session.lib.trace_enabled == 1
    out = session.run_pool(Input(), partial=True)
    assert session.lib.runs == 1
    del out
    raw = session.fetch_trace()
    assert len(raw) == 512 and raw[511] == 511
    session.set_trace(False)
    assert session.lib.trace_enabled == 0


def test_trace_fetch_bad_buffer_is_fatal(session):
    session.set_trace(True)
    session.lib.fetch_rc = -1
    with pytest.raises(RuntimeError, match="trace fetch"):
        session.fetch_trace()
    assert session.failed

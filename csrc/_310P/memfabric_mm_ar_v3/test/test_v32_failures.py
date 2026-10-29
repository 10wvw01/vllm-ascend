# SPDX-License-Identifier: Apache-2.0
"""Fault injection into the real Python adapter, with no Torch/NPU dependency."""

import ctypes
import importlib.util
import sys
import threading
import types
import weakref
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
        self.prepares = []
        self.runs = self.releases = self.destroys = 0
        self.generation = 0
        self.peer_ready = True
        self.busy = False

    def mf310p_v32_acquire(self, *args):
        self.generation += 1
        args[-3]._obj.value = 0
        args[-2]._obj.value = self.generation
        args[-1]._obj.value = 0x10000000
        return -23 if self.busy else 0

    def mf310p_v32_prepare(self, handle, slot, generation, ready, stream):
        self.prepares.append(ready)
        return 0 if ready and self.peer_ready else -27

    def mf310p_v32_release(self, *args):
        self.releases += 1
        return 0

    def mf310p_v32_run(self, *args):
        self.runs += 1
        return 0

    def mf310p_v3_destroy(self, *args):
        self.destroys += 1
        return -26 if self.busy else 0


@pytest.fixture
def session(monkeypatch):
    load(monkeypatch, "memfabric_mm_ar_v3")
    module = load(monkeypatch, "memfabric_mm_ar_v32")
    monkeypatch.setitem(sys.modules, "torch_npu", types.SimpleNamespace(get_npu_format=lambda t: 2))
    s = module.V32Session.__new__(module.V32Session)
    s.lock = threading.RLock()
    s.failed = False
    s.handle = ctypes.c_void_p(1)
    s.raw_stream, s.device = None, 0
    s.pending, s.outputs = [], {}
    s.lib = Lib()
    s.tensor_module = types.SimpleNamespace(wrap=lambda *args: Output(args[-1]))
    s.phase_log = None
    s.merged_wave = False
    s.stream = types.SimpleNamespace(synchronize=lambda: None)
    s._tensor = lambda x: s.pending.append(x)

    def ready():
        if s.failed or not s.handle.value:
            raise RuntimeError("closed/failed")

    s._ready = ready
    return s


@pytest.mark.parametrize("callback_taken", [False, True])
def test_wrap_failure_aborts_prepare_without_launch_and_releases_once(session, callback_taken):
    def broken(*args):
        if callback_taken:
            args[-1]()
        raise RuntimeError("Storage construction failed")

    session.tensor_module.wrap = broken
    with pytest.raises(RuntimeError, match="aborted BOTH ranks"):
        session.run_pool(Input(), partial=True)
    assert session.failed
    assert session.lib.prepares == [0]
    assert session.lib.runs == 0 and session.lib.releases == 1
    with pytest.raises(RuntimeError, match="closed/failed"):
        session.run_pool(Input(), partial=True)


def test_peer_wrap_failure_never_launches_local_kernel(session):
    session.lib.peer_ready = False
    with pytest.raises(RuntimeError, match="output prepare"):
        session.run_pool(Input(), partial=True)
    assert session.failed and session.lib.runs == 0
    assert session.lib.prepares == [1]


def test_completed_inputs_are_reclaimed_without_dropping_current_input(session):
    previous = None
    for _ in range(32):
        x = Input()
        out = session.run_pool(x, partial=True)
        assert len(session.pending) == 1 and session.pending[0] is x
        if previous is not None:
            assert previous() is None
        previous = weakref.ref(x)
        del x, out


def test_slot_busy_reclaims_completed_inputs_and_remains_recoverable(session):
    out = session.run_pool(Input(), partial=True)
    del out
    session.lib.busy = True
    with pytest.raises(RuntimeError, match="release an output slot"):
        session.run_pool(Input(), partial=True)
    assert not session.pending and not session.failed


def test_close_busy_is_recoverable_then_both_can_close(session):
    session.lib.busy = True
    with pytest.raises(RuntimeError, match="BOTH ranks"):
        session.close()
    assert not session.failed and session.handle.value
    session.lib.busy = False
    session.close()
    assert not session.handle.value


def test_close_device_failure_poisoned_without_collective_destroy(session):
    def broken():
        raise RuntimeError("device synchronization failed")

    session.stream.synchronize = broken
    with pytest.raises(RuntimeError, match="device synchronization failed"):
        session.close()
    assert session.failed and session.lib.destroys == 0
    with pytest.raises(RuntimeError, match="closed/failed"):
        session.close()

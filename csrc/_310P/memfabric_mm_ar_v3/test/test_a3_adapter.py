# SPDX-License-Identifier: Apache-2.0
import ctypes
import importlib.util
import sys
import threading
import types
import weakref
from pathlib import Path

import pytest

OPS = Path(__file__).parents[4] / "vllm_ascend/_310p/ops"


class Input:
    ndim, shape = 2, (2048, 2048)

    def data_ptr(self):
        return 4096


class Event:
    def __init__(self):
        self.done = False
        self.fail = False

    def record(self, stream):
        self.done = False

    def query(self):
        if self.fail:
            raise RuntimeError("device event failure")
        return self.done


class Output:
    def __init__(self, callback):
        self.callback = callback

    def __del__(self):
        self.callback()


class Lib:
    def __init__(self):
        self.epoch = 0
        self.held = {}
        self.admits = []

    def mf310p_a3_reserve(self, *args):
        slot = self.epoch % 2
        args[-3]._obj.value, args[-2]._obj.value, args[-1]._obj.value = slot, self.epoch + 1, 0x10000000
        if slot in self.held:
            return -23
        self.held[slot] = self.epoch + 1
        return 0

    def mf310p_v32_release(self, handle, slot, gen):
        assert self.held[slot] == gen
        del self.held[slot]
        return 0

    def mf310p_a3_run(self, *args):
        self.admits.append(args[-2])
        self.epoch += 1
        return 0


@pytest.fixture
def session(monkeypatch):
    for name in ["memfabric_mm_ar_v3", "memfabric_mm_ar_a3"]:
        spec = importlib.util.spec_from_file_location(name, OPS / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
    monkeypatch.setitem(sys.modules, "torch_npu", types.SimpleNamespace(get_npu_format=lambda x: 2))
    s = mod.A3Session.__new__(mod.A3Session)
    s.lock = threading.RLock()
    s.handle, s.device, s.raw_stream, s.stream = ctypes.c_void_p(1), 0, None, None
    s.failed, s.initialized = False, True
    s.pending, s.inflight = [], []
    s.free_events = [Event() for _ in range(64)]
    s.lib = Lib()
    s.tensor_module = types.SimpleNamespace(wrap=lambda *args: Output(args[-1]))
    s._tensor = lambda x: s.pending.append(x)

    def ready():
        if s.failed:
            raise RuntimeError("failed session")

    s._ready = ready
    return s


def test_input_is_held_until_nonblocking_event_completion(session):
    x = Input()
    ref = weakref.ref(x)
    out = session.run_pool(x, partial=True)
    del x, out
    assert ref() is not None
    session.inflight[0][0].done = True
    session._reap()
    assert ref() is None and not session.inflight and len(session.free_events) == 64


def test_busy_seat_submits_failure_without_overwriting_owner(session):
    first = session.run_pool(Input(), partial=True)
    second = session.run_pool(Input(), partial=True)
    with pytest.raises(RuntimeError, match="START rejected"):
        session.run_pool(Input(), partial=True)
    assert session.failed and session.lib.admits == [1, 1, 0]
    assert session.lib.held == {0: 1, 1: 2}
    del first, second


@pytest.mark.parametrize("callback_taken", [False, True])
def test_wrap_failure_submits_start_zero_and_releases_once(session, callback_taken):
    def broken(*args):
        if callback_taken:
            args[-1]()
        raise RuntimeError("wrap failed")

    session.tensor_module.wrap = broken
    with pytest.raises(RuntimeError, match="START rejected"):
        session.run_pool(Input(), partial=True)
    assert session.failed and session.lib.admits == [0] and not session.lib.held


def test_event_failure_blocks_next_launch(session):
    out = session.run_pool(Input(), partial=True)
    session.inflight[0][0].fail = True
    with pytest.raises(RuntimeError, match="device event failure"):
        session.run_pool(Input(), partial=True)
    assert session.failed and session.lib.admits == [1]
    del out


def test_bounded_input_capacity_failure_still_notifies_peer(session):
    session.free_events.clear()
    with pytest.raises(RuntimeError, match="START rejected"):
        session.run_pool(Input(), partial=True)
    assert session.failed and session.lib.admits == [0]

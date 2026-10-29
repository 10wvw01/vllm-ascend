# SPDX-License-Identifier: Apache-2.0
"""Merged-wave interface contract at the Python adapter boundary; no Torch/NPU."""

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
    def __init__(self, prepare_rcs=None, classic_rc=0):
        self.generation = 0
        self.wave_acquires = 0
        self.classic_acquires = 0
        self.prepares = []
        self.runs = 0
        self.releases = []
        self.prepare_rcs = list(prepare_rcs or [])
        self.classic_rc = classic_rc

    def _take(self, args):
        self.generation += 1
        args[-3]._obj.value = 0
        args[-2]._obj.value = self.generation
        args[-1]._obj.value = 0x10000000
        return 0

    def mf310p_v32_wave_acquire(self, *args):
        self.wave_acquires += 1
        return self._take(args)

    def mf310p_v32_acquire(self, *args):
        self.classic_acquires += 1
        return self.classic_rc if self.classic_rc else self._take(args)

    def mf310p_v32_prepare(self, handle, slot, generation, ready, stream):
        self.prepares.append((slot, generation, ready))
        return self.prepare_rcs.pop(0) if self.prepare_rcs else 0

    def mf310p_v32_run(self, *args):
        self.runs += 1
        return 0

    def mf310p_v32_release(self, handle, slot, generation):
        self.releases.append((slot, generation))
        return 0


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
    s.tensor_module = types.SimpleNamespace(wrap=lambda *args: Output(args[-1]))
    s.stream = types.SimpleNamespace(synchronize=lambda: None)
    s._tensor = lambda x: s.pending.append(x)
    s._ready = lambda: None
    s.phase_log = None
    s.merged_wave = False
    return s


def test_merged_happy_path_skips_classic_acquires(session):
    session.lib = Lib()
    session.merged_wave = True
    session.phase_log = []
    out = session.run_pool(Input(), partial=True)
    assert session.lib.wave_acquires == 1 and session.lib.classic_acquires == 0
    assert session.lib.prepares == [(0, 1, 1)]
    assert session.lib.runs == 1
    assert session.phase_log[-1]["fallback"] is False
    del out
    assert session.lib.releases == [(0, 1)]


def test_speculation_mismatch_falls_back_to_classic_intersection(session):
    session.lib = Lib(prepare_rcs=[-29, 0])
    session.merged_wave = True
    session.phase_log = []
    out = session.run_pool(Input(), partial=True)
    assert session.lib.wave_acquires == 1 and session.lib.classic_acquires == 1
    assert [p[2] for p in session.lib.prepares] == [1, 1]
    assert session.lib.releases == [(0, 1)]  # speculative slot released exactly once
    assert session.phase_log[-1]["fallback"] is True
    assert session.lib.runs == 1
    del out
    assert session.lib.releases == [(0, 1), (0, 2)]


def test_merged_wrap_failure_bans_after_zero_ready_prepare(session):
    def broken(*args):
        raise RuntimeError("Storage construction failed")

    session.lib = Lib(prepare_rcs=[-27])
    session.merged_wave = True
    session.tensor_module = types.SimpleNamespace(wrap=broken)
    with pytest.raises(RuntimeError, match="aborted BOTH ranks"):
        session.run_pool(Input(), partial=True)
    assert session.failed
    assert session.lib.prepares == [(0, 1, 0)] and session.lib.runs == 0
    assert session.lib.releases == [(0, 1)]


def test_fallback_wrap_failure_bans_and_never_launches(session):
    calls = []

    def flaky(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("second wrap failed")
        return Output(args[-1])

    session.lib = Lib(prepare_rcs=[-29, -27])
    session.merged_wave = True
    session.tensor_module = types.SimpleNamespace(wrap=flaky)
    with pytest.raises(RuntimeError, match="aborted BOTH ranks"):
        session.run_pool(Input(), partial=True)
    assert session.failed and session.lib.runs == 0
    assert [p[2] for p in session.lib.prepares] == [1, 0]


def test_fallback_busy_is_recoverable(session):
    session.lib = Lib(prepare_rcs=[-29], classic_rc=-23)
    session.merged_wave = True
    with pytest.raises(RuntimeError, match="release an output slot"):
        session.run_pool(Input(), partial=True)
    assert not session.failed and session.lib.runs == 0
    assert session.pending == []

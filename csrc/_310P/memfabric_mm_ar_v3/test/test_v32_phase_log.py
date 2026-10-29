# SPDX-License-Identifier: Apache-2.0
"""Phase-time recording contract for run_pool; no Torch/NPU dependency."""

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


class Lib:
    generation = 0

    def mf310p_v32_acquire(self, *args):
        Lib.generation += 1
        args[-3]._obj.value = 0
        args[-2]._obj.value = Lib.generation
        args[-1]._obj.value = 0x10000000
        return 0

    def mf310p_v32_prepare(self, *args):
        return 0

    def mf310p_v32_run(self, *args):
        return 0

    def mf310p_v32_release(self, *args):
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
    s.lib = Lib()
    s.tensor_module = types.SimpleNamespace(wrap=lambda *args: types.SimpleNamespace())
    s.stream = types.SimpleNamespace(synchronize=lambda: None)
    s._tensor = lambda x: s.pending.append(x)
    s._ready = lambda: None
    s.phase_log = None
    s.merged_wave = False
    return s


def test_disabled_by_default_appends_nothing(session):
    session.run_pool(Input(), partial=True)
    assert session.phase_log is None


def test_enabled_records_one_entry_with_all_phases(session):
    session.phase_log = []
    session.run_pool(Input(), partial=True)
    assert len(session.phase_log) == 1
    entry = session.phase_log[0]
    assert set(entry) == {"acquire_us", "wrap_us", "prepare_us", "run_us", "enqueue_us", "fallback"}
    assert entry["fallback"] is False
    assert all(isinstance(v, float) and v >= 0 for k, v in entry.items() if k != "fallback")
    assert entry["enqueue_us"] >= sum(entry[k] for k in ("acquire_us", "wrap_us", "prepare_us", "run_us"))


def test_busy_acquire_records_nothing(session):
    session.phase_log = []

    def busy(*args):
        return -23

    session.lib.mf310p_v32_acquire = busy
    with pytest.raises(RuntimeError, match="release an output slot"):
        session.run_pool(Input(), partial=True)
    assert session.phase_log == []

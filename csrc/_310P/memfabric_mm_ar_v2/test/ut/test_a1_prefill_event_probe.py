#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""无需torch/NPU的A1 Prefill自然同步事件计量单元测试。"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[6] / "vllm_ascend" / "_310p" / "a1_prefill_event_probe.py"
spec = importlib.util.spec_from_file_location("a1_prefill_event_probe", SOURCE)
assert spec and spec.loader
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class FakeEvent:
    instances = []
    def __init__(self, enable_timing=False):
        assert enable_timing
        self.ready = True
        FakeEvent.instances.append(self)

    def record(self, stream):
        assert stream == "fake-stream"

    def query(self):
        return self.ready

    def elapsed_time(self, other):
        assert isinstance(other, FakeEvent)
        return 1.25


class ProbeUT(unittest.TestCase):
    def setUp(self):
        FakeEvent.instances.clear()
        self.fake_torch = SimpleNamespace(
            npu=SimpleNamespace(Event=FakeEvent, current_stream=lambda: "fake-stream"))
        self.mock_torch = patch.dict(sys.modules, {"torch": self.fake_torch})
        self.mock_torch.start()
        self.addCleanup(self.mock_torch.stop)

    def test_disabled_in_normal_execution(self):
        with patch.dict(os.environ, {"A1_310P_PREFILL_EVENTS_DIR": ""}):
            with self.assertRaises(RuntimeError):
                probe.begin(SimpleNamespace(with_prefill=True),
                            SimpleNamespace(total_num_scheduled_tokens=4096))
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(probe.begin(SimpleNamespace(with_prefill=True),
                                          SimpleNamespace(total_num_scheduled_tokens=4096)))
            self.assertEqual(FakeEvent.instances, [])

    def test_non_prefill_not_captured(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {
            "A1_310P_PREFILL_EVENTS_DIR": d, "RANK": "1"
        }):
            self.assertIsNone(probe.begin(SimpleNamespace(with_prefill=False),
                                          SimpleNamespace(total_num_scheduled_tokens=1)))

    def test_emits_only_after_natural_completion_and_never_syncs(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {
            "A1_310P_PREFILL_EVENTS_DIR": d, "RANK": "1"
        }):
            runner = SimpleNamespace(with_prefill=True)
            s = probe.begin(runner, SimpleNamespace(total_num_scheduled_tokens=4096))
            assert s
            probe.end(runner, s)
            FakeEvent.instances[-1].ready = False
            probe.drain(runner)
            self.assertEqual(list(Path(d).glob("*.jsonl")), [])
            self.assertEqual(len(runner._a1_prefill_pending), 1)
            FakeEvent.instances[-1].ready = True
            probe.drain(runner)
            result_files = list(Path(d).glob("a1_prefill_rank1_pid*.jsonl"))
            self.assertEqual(len(result_files), 1)
            row = json.loads(result_files[0].read_text())
            self.assertEqual(row["forward_to_logits_device_ms"], 1.25)
            self.assertEqual(row["num_scheduled_tokens"], 4096)
            self.assertFalse(row["synchronization_inserted"])
            self.assertEqual(len(runner._a1_prefill_pending), 0)

    def test_reject_relative_or_symlinked_dir(self):
        with patch.dict(os.environ, {"A1_310P_PREFILL_EVENTS_DIR": "relative"}):
            with self.assertRaises(RuntimeError):
                probe.begin(SimpleNamespace(with_prefill=True),
                            SimpleNamespace(total_num_scheduled_tokens=4096))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "link"
            path.symlink_to(Path(d), target_is_directory=True)
            with patch.dict(os.environ, {"A1_310P_PREFILL_EVENTS_DIR": str(path)}):
                with self.assertRaises(RuntimeError):
                    probe.begin(SimpleNamespace(with_prefill=True),
                                SimpleNamespace(total_num_scheduled_tokens=4096))


if __name__ == "__main__":
    unittest.main()

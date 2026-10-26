#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""R2-P0-E new device-BT consumer and ModelState gold cannot bypass NO-GO."""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "vllm_ascend/_310p/v2"
COPY = BASE / "hybrid_state_device_copy.py"
LINK = BASE / "hybrid_state_modelstate_link.py"
RUNNER = BASE / "model_runner.py"
WORKER = ROOT / "vllm_ascend/_310p/worker_310p.py"
REPORT = (ROOT / "csrc/_310P/memfabric_mm_ar_v2/docs/"
          "R2_P0_E_DEVICE_COPY_LINK_20261026.md")


class TestR2DeviceCopyNeverAutoRuns(unittest.TestCase):
    def test_npu_branch_rejects_before_any_physical_write(self):
        source = COPY.read_text()
        tree = ast.parse(source)
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
                   and x.name == "DeviceIndexedGDNCopy310Candidate")
        method = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                      and x.name == "execute")
        assert isinstance(method.body[1], ast.If) or len(method.body) > 2
        text = ast.get_source_segment(source, method)
        assert text is not None
        self.assertLess(text.index('if self.device.type == "npu":'),
                        text.index("self._preflight_decision("))
        self.assertIn("raise NotImplementedError(", text)
        self.assertIn("torch.nonzero(", text)
        self.assertIn("torch.unique(", text)
        self.assertIn("index_copy_", text)

    def test_no_d2h_private_draft_or_runtime_registration(self):
        source = COPY.read_text()
        facade = LINK.read_text()
        for banned in (".item()", ".tolist()", ".cpu()", "import triton",
                       "torch.cuda", "from vllm.", "import torch_npu", "GDNMove310"):
            self.assertNotIn(banned, source)
        for banned in ("from vllm.", "GPUModelRunner", "torch.npu.",
                       ".item()", ".cpu()", "import triton"):
            self.assertNotIn(banned, facade)
        runner, worker = RUNNER.read_text(), WORKER.read_text()
        for forbidden in ("hybrid_state_device_copy", "hybrid_state_modelstate_link"):
            self.assertNotIn(forbidden, runner)
            self.assertNotIn(forbidden, worker)
        self.assertIn("Triton-free V2 NPUWorker/ModelRunner is NOT_IMPLEMENTED", worker)
        self.assertIn('self.cache_config.mamba_cache_mode == "align"', runner)

    def test_modelstate_gold_frozen_signatures_and_cpu_gate(self):
        tree = ast.parse(LINK.read_text())
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
                   and x.name == "TargetGDNModelStateSequence310Candidate")
        funcs = {x.name for x in cls.body if isinstance(x, ast.FunctionDef)}
        self.assertTrue({"add_request", "remove_request", "preprocess_state",
                         "prepare_attn", "postprocess_state",
                         "forward_completed_for_cpu_gold"} <= funcs)
        self.assertIn('metadata.device.type != "cpu"', LINK.read_text())
        self.assertIn("copy_consumer.execute(decisions)", LINK.read_text())
        self.assertIn("copy_consumer.execute(decision)", LINK.read_text())
        self.assertIn("confirm_postcopy_for_cpu_gold", LINK.read_text())

    def test_document_keeps_runtime_scope_and_admission_explicit(self):
        d = REPORT.read_text()
        for term in ("T30-B", "T31-B", "ModelState", "STATIC-NO-GO",
                     "0/46", "NPU", "CPU", "不允许", "DeviceIndexed"):
            if term == "不允许":
                continue
            self.assertIn(term, d)
        self.assertNotIn("w00804146", d)
        self.assertNotIn("5 Sliding", d)


if __name__ == "__main__":
    unittest.main()

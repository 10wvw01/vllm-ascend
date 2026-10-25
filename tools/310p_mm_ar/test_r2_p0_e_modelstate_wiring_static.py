#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Frozen 310P V2 R2-P0-E ModelState wiring invariants (stdlib-only).

Not a 310P3 runtime trace or an upstream GPU code-equivalence proof.
"""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
DEVICE_META = ROOT / "vllm_ascend/_310p/v2/hybrid_state_device_meta.py"
MOVER = ROOT / "vllm_ascend/_310p/v2/hybrid_state_npu.py"
RUNNER = ROOT / "vllm_ascend/_310p/v2/model_runner.py"
WORKER = ROOT / "vllm_ascend/_310p/worker_310p.py"
REPORT = (ROOT / "csrc/_310P/memfabric_mm_ar_v2/docs/"
          "R2_P0_E_DEVICE_STATE_AND_MODELSTATE_WIRING_20261026.md")


class TestR2P0EModelStateWiring(unittest.TestCase):
    def test_current_v2_worker_and_load_model_still_fail_closed(self):
        worker = WORKER.read_text()
        runner = RUNNER.read_text()
        self.assertIn("Triton-free V2 NPUWorker/ModelRunner is NOT_IMPLEMENTED", worker)
        self.assertIn("self.cache_config.mamba_cache_mode == \"align\"", runner)
        self.assertIn("self.model_config.is_hybrid", runner)
        self.assertIn("still requires Triton; no 310P kernel approved", runner)

    def test_runner_preserves_modelstate_post_update_data_dependency(self):
        source = RUNNER.read_text()
        cls = next(
            node for node in ast.parse(source).body
            if isinstance(node, ast.ClassDef)
            and node.name == "NPUModelRunner310V2Candidate"
        )
        methods = {
            item.name: item for item in cls.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue(
            {"execute_model", "load_model", "postprocess_sampled",
             "initialize_kv_cache", "prepare_inputs"} <= methods.keys()
        )
        post = ast.get_source_segment(source, methods["postprocess_sampled"])
        assert post is not None
        self.assertLess(
            post.index("post_update("), post.index("self.model_state.postprocess_state(")
        )
        execute = ast.get_source_segment(source, methods["execute_model"])
        assert execute is not None
        self.assertIn("GPUModelRunner.execute_model", execute)

    def test_no_accidental_runner_worker_or_draft_wiring(self):
        source = DEVICE_META.read_text()
        runner, worker = RUNNER.read_text(), WORKER.read_text()
        self.assertNotIn("hybrid_state_device_meta", runner)
        self.assertNotIn("hybrid_state_device_meta", worker)
        self.assertNotIn("hybrid_state_npu", runner)
        for forbidden in ("import triton", "from triton", "torch.cuda.",
                          "import torch_npu", "from vllm.", ".item()",
                          ".tolist()", ".cpu()", "private_draft"):
            self.assertNotIn(forbidden, source)

    def test_device_producer_host_consumer_abi_gap_stays_visible(self):
        meta = DEVICE_META.read_text()
        mover = MOVER.read_text()
        self.assertIn("class GDNDeviceDecision310:", meta)
        self.assertIn("torch.Tensor", meta)
        self.assertIn("class GDNMove310:", mover)
        self.assertIn("source_column: int", mover)
        self.assertIn("accepted_bias: int", mover)
        self.assertIn("block_table: torch.Tensor", mover)
        self.assertIn("CPU", mover)
        self.assertIn("index_copy_(0, index, whole)", meta)
        self.assertIn('self._phase = "PRECOPY_DECISION_PENDING"', meta)
        self.assertIn('self._phase = "POSTCOPY_DECISION_PENDING"', meta)
        self.assertIn("NPU needs device-native move consumer", meta)

    def test_review_artifact_exposes_no_go_and_explicit_gaps(self):
        s = REPORT.read_text()
        for t in ("T30-A", "T30-B", "T31-A", "T31-B",
                  "init_model_state", "post_update", "prepare_attn",
                  "STATIC-NO-GO", "0/46", "GDNMove310",
                  "validity", "128B", "NDA"):
            # NDA wording may be in references as "自研Draft私有资料未读"
            if t != "NDA":
                self.assertIn(t, s)
        self.assertNotIn("w00804146", s)
        self.assertNotIn("5 Sliding", s)


if __name__ == "__main__":
    unittest.main()

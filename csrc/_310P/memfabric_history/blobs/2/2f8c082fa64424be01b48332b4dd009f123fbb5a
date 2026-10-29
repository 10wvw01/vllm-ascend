#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static ledger checks for ONE 310P3 Target+DFlash K7 Eager workload.

Never imports model/runtime packages and never reads private Draft sources.
Reachability here is counterfactual-to-Worker guard, not on-device proof.
"""
from __future__ import annotations

import ast
import json
import pathlib
import unittest
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ROOT / "csrc/_310P/memfabric_mm_ar_v2/docs"
JSON_FILE = DOCS / "R2_TARGET_REACHABILITY_CUT_20261025.json"
MD_FILE = DOCS / "R2_TARGET_REACHABILITY_CUT_20261025.md"

COUNTS = {
    "BUSINESS_CORE_CONDITIONAL_ON_STAGE": 19,
    "FEATURE_GATED_GREEDY_CANDIDATE_ONLY": 7,
    "EXCLUDED_BY_PP1_STATIC_BRANCH": 1,
    "SHADOWED_BY_310P_OVERRIDE": 1,
    "UNRESOLVED_INHERITED_AUDIT": 1,
    "HYBRID_ALIGN_CONDITIONAL_BLOCKER": 2,
}


class TestR2TargetReachability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(JSON_FILE.read_text(encoding="utf-8"))

    def test_complete_31_item_target_only_partition(self):
        rows = self.data["entries"]
        self.assertEqual([x["id"] for x in rows],
                         [f"T{i:02d}" for i in range(1, 32)])
        self.assertEqual(dict(Counter(x["category"] for x in rows)), COUNTS)
        self.assertEqual(set(self.data["categories"].values()), set(COUNTS))
        for row in rows:
            # Source labels may be concise (e.g. a single local callsite),
            # while explanations need enough detail to be actionable.
            self.assertGreater(len(row["source"]), 8)
            for key in ("condition", "decision"):
                self.assertGreater(len(row[key]), 14)

    def test_frozen_workload_and_no_runtime_go(self):
        obj = self.data
        self.assertEqual(obj["target_id"], "Eco-Tech/Qwen3.6-35B-A3B-w8a8")
        self.assertEqual(obj["scope"], "LOCKED_TARGET_310P3_TP2_PP1_EAGER_DFLASH_K7")
        self.assertEqual(obj["upstream_vllm_commit"],
                         "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac")
        self.assertEqual(obj["absolute_stop"], "310P3_V2_WORKER_NOT_IMPLEMENTED")
        self.assertEqual(obj["evidence"], "SOURCE_STATIC_ONLY_RUNTIME_NOT_OBSERVED")
        self.assertEqual(obj["draft_private"], "NOT_READ_NOT_REQUIRED")
        self.assertEqual(obj["target_sampling_policy"], "NOT_FROZEN_BY_USER")
        self.assertEqual(obj["candidate_sampling_policy"],
                         "GREEDY_ONLY_REJECT_UNSUPPORTED")

    def test_specific_static_exclusions_not_overstated(self):
        rows = {x["id"]: x for x in self.data["entries"]}
        for item in ("T08", "T18", "T19", "T20", "T21", "T22", "T24"):
            self.assertEqual(rows[item]["category"],
                             "FEATURE_GATED_GREEDY_CANDIDATE_ONLY")
        for item in ("T30", "T31"):
            self.assertEqual(rows[item]["category"],
                             "HYBRID_ALIGN_CONDITIONAL_BLOCKER")
        self.assertEqual(rows["T10"]["category"], "EXCLUDED_BY_PP1_STATIC_BRANCH")
        self.assertEqual(rows["T12"]["category"], "SHADOWED_BY_310P_OVERRIDE")
        self.assertEqual(rows["T13"]["category"], "UNRESOLVED_INHERITED_AUDIT")
        # Despite greedy restrictions, both normal sampling and K7 verification
        # need correctly implemented non-Triton sample/verify paths.
        for item in ("T17", "T23", "T26", "T27", "T28", "T29"):
            self.assertEqual(rows[item]["category"],
                             "BUSINESS_CORE_CONDITIONAL_ON_STAGE")

    def test_readme_cross_link_scope_and_redaction(self):
        d = MD_FILE.read_text(encoding="utf-8")
        self.assertIn("R2_TARGET_ROUTE_PROBE_RESULT.md", d)
        self.assertIn("R2_TARGET_REACHABILITY_CUT_20261025.json", d)
        self.assertIn("NO-GO", d)
        self.assertIn("greedy", d)
        self.assertIn("T30/T31", d)
        self.assertIn("0/46", d)
        self.assertNotIn("w00804146", d)
        self.assertNotIn("5 Sliding", d)
        self.assertNotIn("layer_types = [", d)

    def test_actual_guard_and_candidate_conditions(self):
        worker = (ROOT / "vllm_ascend/_310p/worker_310p.py").read_text()
        runner = (ROOT / "vllm_ascend/_310p/v2/model_runner.py").read_text()
        sampling = (ROOT / "vllm_ascend/_310p/v2/sampling_scope.py").read_text()
        sampler = (ROOT / "vllm_ascend/_310p/v2/sampler.py").read_text()
        self.assertIn("Triton-free V2 NPUWorker/ModelRunner is NOT_IMPLEMENTED",
                      worker)
        self.assertIn("self.cache_config.mamba_cache_mode == \"align\"", runner)
        self.assertIn("and self.cache_config.mamba_cache_mode", runner)
        self.assertIn("raise NotImplementedError(", runner)
        self.assertIn('"temperature": 0.0', sampling)
        self.assertIn('"logprobs"', sampling)
        self.assertIn("NO_LOGPROBS", sampler)
        self.assertIn("NPUDFlashSpeculator310V2", runner)
        self.assertIn("_310p_v2_store_draft_tokens", runner)
        # AST proves scoped local methods exist, without importing torch/vLLM.
        tree = ast.parse(runner)
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "NPUModelRunner310V2Candidate"]
        self.assertEqual(len(classes), 1)
        funcs = {n.name for n in classes[0].body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertTrue({"load_model", "postprocess_sampled", "sample_tokens",
                         "initialize_kv_cache", "prepare_inputs"} <= funcs)

    def test_required_ids_in_markdown_and_json_are_consistent(self):
        markdown = MD_FILE.read_text(encoding="utf-8")
        for item in ("T08", "T10", "T12", "T13", "T17", "T23", "T30", "T31"):
            self.assertIn(item, markdown)
        # This is a classification only; even 31 fully classified rows must
        # never mutate the 46-item production static admission certificate.
        admission = (DOCS / "R2_STATIC_ADMISSION_EVIDENCE.json")
        self.assertTrue(admission.is_file())


if __name__ == "__main__":
    unittest.main()

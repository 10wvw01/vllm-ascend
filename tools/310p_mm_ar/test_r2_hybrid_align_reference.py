#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static + pure-Python tests for Target Hybrid ModelState T30/T31 design.

No GPU/NPU, model loads or private Draft reads; no runtime GO implied.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
MOD = ROOT / "tools/310p_mm_ar/r2_hybrid_align_reference.py"
SPEC = importlib.util.spec_from_file_location("r2_hybrid_align_reference", MOD)
assert SPEC is not None and SPEC.loader is not None
# Dataclasses look up modules in sys.modules. Ordinary import would be easier
# but tools/310p_mm_ar is not a Python package (directory name starts 310).
import sys
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


class TestR2HybridAlignReference(unittest.TestCase):
    def test_fresh_request_seed_and_pre_align(self):
        self.assertEqual(module.init_running_column(0, 16), -1)
        p = module.plan_pre_align(-1, 0, 8, 1, 16)
        self.assertEqual((p.previous_column, p.next_column), (-1, 0))
        self.assertFalse(p.needs_copy)
        self.assertFalse(p.reset_accepted_to_one)

    def test_request_initialization_and_reuse(self):
        self.assertEqual(module.init_running_column(1, 16), 0)
        self.assertEqual(module.init_running_column(16, 16), 0)
        self.assertEqual(module.init_running_column(17, 16), 1)
        with self.assertRaises(ValueError):
            module.init_running_column(-1, 16)

    def test_pre_boundary_migration_carries_old_accepted_bias(self):
        p = module.plan_pre_align(0, 15, 2, 4, 16)
        self.assertEqual((p.previous_column, p.next_column,
                          p.accepted_bias), (0, 1, 3))
        self.assertTrue(p.needs_copy)
        self.assertTrue(p.reset_accepted_to_one)

    def test_pre_same_block_retains_accepted(self):
        p = module.plan_pre_align(0, 8, 3, 5, 16)
        self.assertEqual(p.next_column, 0)
        self.assertEqual(p.accepted_bias, 4)
        self.assertFalse(p.needs_copy)
        self.assertFalse(p.reset_accepted_to_one)

    def test_post_does_not_copy_when_not_at_alignment_window(self):
        p = module.plan_post_align(0, 20, 2, 16)
        self.assertEqual((p.running_tokens, p.aligned_tokens), (19, 16))
        self.assertFalse(p.copy_needed)
        self.assertIsNone(p.destination_column)

    def test_post_same_block_shift_and_reset(self):
        p = module.plan_post_align(0, 18, 4, 16)
        self.assertEqual((p.running_tokens, p.aligned_tokens,
                          p.destination_column, p.accepted_bias),
                         (15, 16, 0, 1))
        self.assertTrue(p.copy_needed)
        self.assertTrue(p.reset_accepted_to_one)

    def test_post_same_block_zero_bias_is_copy_noop(self):
        p = module.plan_post_align(0, 16, 1, 16)
        self.assertEqual(p.destination_column, 0)
        self.assertFalse(p.copy_needed)
        self.assertTrue(p.reset_accepted_to_one)

    def test_post_different_block_copies_without_count_reset(self):
        p = module.plan_post_align(0, 32, 1, 16)
        self.assertEqual(p.destination_column, 1)
        self.assertTrue(p.copy_needed)
        self.assertFalse(p.reset_accepted_to_one)

    def test_gdn_conv_sd_and_ds_window_copies(self):
        for layout in ("SD", "DS"):
            with self.subTest(layout=layout):
                p = module.state_copy_decision(
                    module.StateCopyShape("CONV", layout, conv_width=10),
                    src_col=1, dst_col=2, bias=3)
                self.assertEqual((p.source_block_table_column,
                                  p.destination_block_table_column,
                                  p.source_offset, p.extent), (1, 2, 3, 7))
                self.assertEqual(p.layout, layout)

    def test_temporal_bias_changes_block_table_column_not_byte_offset(self):
        p = module.state_copy_decision(
            module.StateCopyShape("TEMPORAL"), 1, 2, 3)
        self.assertEqual((p.source_block_table_column,
                          p.destination_block_table_column,
                          p.source_offset, p.extent), (4, 2, 0, None))

    def test_invalid_state_must_not_produce_copy_plan(self):
        cases = [
            lambda: module.plan_pre_align(-2, 0, 2, 1, 16),
            lambda: module.plan_pre_align(0, 0, 0, 1, 16),
            lambda: module.plan_pre_align(0, 15, 2, 9, 16),
            lambda: module.plan_post_align(-1, 1, 8, 16),
            lambda: module.plan_post_align(0, 1, 8, 16),
            lambda: module.state_copy_decision(
                module.StateCopyShape("CONV", "SD", 4), 0, 1, 4),
            lambda: module.state_copy_decision(
                module.StateCopyShape("TEMPORAL", "SD", 1), 0, 1, 0),
        ]
        for i, f in enumerate(cases):
            with self.subTest(i=i), self.assertRaises(ValueError):
                f()

    def test_design_metadata_and_pruning_scope(self):
        docs = ROOT / "csrc/_310P/memfabric_mm_ar_v2/docs"
        md = (docs / "R2_P0_TARGET_HYBRID_MODELSTATE_DESIGN_20261025.md").read_text()
        ledger = json.loads((docs / "R2_P0_HYBRID_STATE_CONTRACT_20261025.json").read_text())
        self.assertEqual(ledger["scenario"]["tp"], 2)
        self.assertEqual(ledger["scenario"]["k"], 7)
        self.assertEqual(ledger["evidence"], "STATIC_DESIGN_ONLY")
        self.assertEqual(ledger["runtime"], "NOT_RUN")
        self.assertTrue({"preprocess_mamba_align_fused_kernel",
                         "precopy_mamba_align_fused_kernel",
                         "_scatter_num_accepted_kernel",
                         "postprocess_mamba_fused_kernel"}
                        <= set(ledger["kernel_replacements"]))
        self.assertIn("T30", md)
        self.assertIn("T31", md)
        self.assertIn("SD", md)
        self.assertIn("DS", md)
        self.assertIn("重叠", md)
        self.assertIn("overlapping_in_place_conv_window_shift_requires_memmove_safe_or_staged_copy_semantics", ledger["blocking_evidence"])
        self.assertIn("NDA", md)
        self.assertIn("STATIC-NO-GO", md)
        self.assertNotIn("w00804146", md)

    def test_source_guards_and_pinned_semantics(self):
        model = (ROOT / "vllm_ascend/_310p/v2/model_runner.py").read_text()
        worker = (ROOT / "vllm_ascend/_310p/worker_310p.py").read_text()
        self.assertIn("self.cache_config.mamba_cache_mode == \"align\"", model)
        self.assertIn("is NOT_IMPLEMENTED", worker)
        self.assertIn("self.model_state.postprocess_state(", model)


if __name__ == "__main__":
    unittest.main()

"""Contract gold for 310P3 V2 DFlash slot mappings; standard library only."""
import importlib.util
import unittest
from pathlib import Path


PATH = Path(__file__).resolve().parents[2] / "vllm_ascend/_310p/v2/reference_ops.py"
SPEC = importlib.util.spec_from_file_location("v2_reference_ops", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
import sys
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DFlashK7GoldenTests(unittest.TestCase):
    def args(self):
        return {
            "target_positions": [5, 6, 128, 129, 130],
            "target_query_start_loc": [0, 2, 5],
            "idx_mapping": [1, 0],
            "last_sampled": [42, 77],
            "next_prefill_tokens": [9, 10],
            "num_sampled": [1, 0],
            "num_rejected": [0, 1],
            "group_block_tables": [
                [[8, 9, 10, 11], [12, 13, 14, 15]],
                [[4, 5], [6, 7]],
            ],
            "group_kernel_block_sizes": [64, 128],
            "parallel_drafting_token_id": 999,
        }

    def test_k7_bonus_and_rejected_context_positions(self):
        result = MODULE.build_k7_inputs(**self.args())
        self.assertEqual(len(result.input_ids), 16)
        self.assertEqual(result.input_ids[:8], [77] + [999] * 7)
        self.assertEqual(result.input_ids[8:16], [9] + [999] * 7)
        self.assertEqual(result.query_positions[:8], list(range(7, 15)))
        self.assertEqual(result.query_positions[8:16], list(range(130, 138)))
        self.assertEqual(result.query_start_loc, [0, 8, 16])
        self.assertEqual(result.seq_lens, [15, 138])
        self.assertEqual(result.context_positions, [5, 6, 128, 129, 130])
        self.assertEqual(result.sample_indices, list(range(1, 8)) + list(range(9, 16)))
        self.assertEqual(result.sample_positions, list(range(8, 15)) + list(range(131, 138)))
        self.assertEqual(result.sample_idx_mapping, [1] * 7 + [0] * 7)

    def test_two_groups_keep_distinct_physical_addresses(self):
        result = MODULE.build_k7_inputs(**self.args())
        # group 0 uses 64-token kernel blocks, group 1 uses 128.
        self.assertEqual(result.context_slots[0][:2], [8 * 64 + 5, 8 * 64 + 6])
        self.assertEqual(result.context_slots[1][:2], [4 * 128 + 5, 4 * 128 + 6])
        self.assertEqual(result.query_slots[0][8], 14 * 64 + 2)
        self.assertEqual(result.query_slots[1][8], 7 * 128 + 2)

    def test_output_is_independent_of_input_mutation(self):
        args = self.args()
        result = MODULE.build_k7_inputs(**args)
        args["target_positions"][0] = 300
        self.assertEqual(result.context_positions[0], 5)

    def test_rejects_missing_group(self):
        args = self.args()
        args["group_block_tables"] = []
        args["group_kernel_block_sizes"] = []
        with self.assertRaisesRegex(ValueError, "missing KV groups"):
            MODULE.build_k7_inputs(**args)

    def test_rejects_unallocated_slot(self):
        args = self.args()
        args["group_block_tables"][0][1] = [1]
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            MODULE.build_k7_inputs(**args)

    def test_rejects_invalid_rejection(self):
        args = self.args()
        args["num_rejected"][0] = 2
        with self.assertRaisesRegex(ValueError, "invalid sample"):
            MODULE.build_k7_inputs(**args)

    def test_rejects_bad_group_count(self):
        args = self.args()
        args["group_block_tables"][0].pop()
        with self.assertRaisesRegex(ValueError, "group request count"):
            MODULE.build_k7_inputs(**args)

    def test_prepends_mask_token_after_bonus(self):
        args = self.args()
        args["parallel_drafting_token_id"] = 234
        self.assertEqual(MODULE.build_k7_inputs(**args).input_ids[1:8], [234] * 7)


if __name__ == "__main__":
    unittest.main()

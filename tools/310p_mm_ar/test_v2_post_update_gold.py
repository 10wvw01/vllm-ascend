"""Frozen vLLM 0.27.1 T09 state-update semantics; stdlib-only gold."""
import importlib.util
import pathlib
import random
import sys
import unittest

SRC = pathlib.Path(__file__).resolve().parents[2] / "vllm_ascend/_310p/v2/post_update_gold.py"
SPEC = importlib.util.spec_from_file_location("post_update_gold", SRC)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestPostUpdateGold(unittest.TestCase):
    def state(self):
        return MODULE.PostUpdateState(
            computed=[5, 11, 20],
            last_sampled=[4, 5, 6],
            total_len=[6, 12, 20],
            token_rows=[[0] * 32 for _ in range(3)],
            output_counts=[[0] * 8 for _ in range(3)],
        )

    def run_case(self, **kwargs):
        return MODULE.post_update_reference(state=self.state(), **kwargs)

    def test_partial_accept_bonus_and_masked_request(self):
        base = self.state()
        result = MODULE.post_update_reference(
            state=base,
            idx_mapping=[2, -1, 0],
            sampled=[[3, 4, 3, 9], [2], [1, 6]],
            num_sampled=[3, 1, 2],
            num_rejected=[1, 0, 0],
            query_start_loc=[0, 4, 7, 9],
        )
        self.assertEqual(result.computed, [7, 11, 23])
        self.assertEqual(result.last_sampled, [6, 5, 3])
        self.assertEqual(result.total_len, [8, 12, 23])
        self.assertEqual(result.token_rows[2][20:23], [3, 4, 3])
        self.assertEqual(result.token_rows[0][6:8], [1, 6])
        self.assertEqual(result.output_counts[2], [0, 0, 0, 2, 1, 0, 0, 0])
        self.assertEqual(result.output_counts[0], [0, 1, 0, 0, 0, 0, 1, 0])
        self.assertEqual(result.output_counts[1], [0] * 8)
        self.assertEqual(base.total_len, [6, 12, 20])
        self.assertEqual(base.computed, [5, 11, 20])

    def test_zero_sample_preserves_last_and_total_but_updates_computed(self):
        x = self.run_case(
            idx_mapping=[1], sampled=[[9] * 8], num_sampled=[0],
            num_rejected=[1], query_start_loc=[0, 4],
        )
        self.assertEqual(x.computed, [5, 14, 20])
        self.assertEqual(x.total_len, [6, 12, 20])
        self.assertEqual(x.last_sampled, [4, 5, 6])

    def test_no_query_start_loc_zero_delta(self):
        x = self.run_case(
            idx_mapping=[0], sampled=[[1]], num_sampled=[1],
            num_rejected=[0], query_start_loc=None,
        )
        self.assertEqual(x.computed[0], 5)
        self.assertEqual(x.total_len[0], 7)

    def test_masked_entry_does_not_validate_unused_tokens(self):
        x = self.run_case(
            idx_mapping=[-1, 1], sampled=[[999], [2]],
            num_sampled=[100, 1], num_rejected=[100, 0],
            query_start_loc=[0, 4, 5],
        )
        self.assertEqual(x.total_len, [6, 13, 20])
        self.assertEqual(x.output_counts[1][2], 1)

    def test_fail_closed_duplicate_and_invalid_state(self):
        for ids in ([0, 0], [-2], [3]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.run_case(
                    idx_mapping=ids, sampled=[[] for _ in ids],
                    num_sampled=[0] * len(ids), num_rejected=[0] * len(ids),
                    query_start_loc=[0] * (len(ids) + 1),
                )

    def test_rejects_overflow_and_no_partial_write(self):
        original = self.state()
        old = original.token_rows[2][:]
        with self.assertRaisesRegex(ValueError, "capacity"):
            MODULE.post_update_reference(
                state=original, idx_mapping=[2],
                sampled=[[1] * 13], num_sampled=[13],
                num_rejected=[0], query_start_loc=[0, 1],
            )
        self.assertEqual(original.token_rows[2], old)

    def test_rejects_vocabulary_violation(self):
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.run_case(
                idx_mapping=[2], sampled=[[9]], num_sampled=[1],
                num_rejected=[0], query_start_loc=[0, 1],
            )

    def test_rejects_bad_prefix_offsets(self):
        for offsets in ([0], [1, 4], [0, -1], [0, 1, 0]):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                self.run_case(
                    idx_mapping=[0], sampled=[[]], num_sampled=[0],
                    num_rejected=[0], query_start_loc=offsets,
                )

    def test_no_bin_count_variant(self):
        original = self.state()
        original.output_counts = None
        result = MODULE.post_update_reference(
            state=original, idx_mapping=[0], sampled=[[2, 2]],
            num_sampled=[2], num_rejected=[0], query_start_loc=[0, 2],
        )
        self.assertIsNone(result.output_counts)
        self.assertEqual(result.token_rows[0][6:8], [2, 2])

    def test_randomized_unique_request_and_conservation(self):
        rng = random.Random(271)
        for _ in range(80):
            n = rng.randint(0, 3)
            ids = rng.sample(range(3), n)
            cnt = [rng.randint(0, 7) for _ in ids]
            rejected = [rng.randint(0, 7) for _ in ids]
            lengths = [rng.randint(r, 8) for r in rejected]
            offsets = [0]
            for length in lengths:
                offsets.append(offsets[-1] + length)
            tokens = [[rng.randrange(8) for _ in range(8)] for _ in ids]
            state = self.state()
            out = MODULE.post_update_reference(
                state=state, idx_mapping=ids, sampled=tokens,
                num_sampled=cnt, num_rejected=rejected,
                query_start_loc=offsets,
            )
            for i, req in enumerate(ids):
                self.assertEqual(out.total_len[req] - state.total_len[req], cnt[i])
                self.assertEqual(out.computed[req] - state.computed[req],
                                 lengths[i] - rejected[i])
                self.assertEqual(sum(out.output_counts[req]) -
                                 sum(state.output_counts[req]), cnt[i])


if __name__ == "__main__":
    unittest.main()

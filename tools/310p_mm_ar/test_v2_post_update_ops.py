"""310P3 T09 Eager tensor differential against frozen CPU post-update gold."""
import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "vllm_ascend/_310p/v2"


def load(name):
    spec = importlib.util.spec_from_file_location(
        "_t09_" + name, SRC / (name + ".py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gold = load("post_update_gold")


class TorchPostUpdate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch absent, mandatory T09 gold still runs")
        cls.t = torch
        cls.ops = load("post_update_ops")

    def _state(self):
        t = self.t
        computed = t.tensor([5, 11, 20], dtype=t.int32)
        laststorage = t.zeros((3, 4), dtype=t.int64)
        laststorage[:, 0] = t.tensor([4, 5, 6])
        total = t.tensor([6, 12, 20], dtype=t.int32)
        rows = t.zeros((3, 32), dtype=t.int32)
        counts = t.zeros((3, 8), dtype=t.int32)
        return computed, laststorage, total, rows, counts

    def test_mixed_k7_duplicates_and_masked_rows(self):
        t = self.t
        computed, laststorage, total, rows, counts = self._state()
        idx = t.tensor([2, -1, 0], dtype=t.int32)
        tokens = t.tensor([[3, 4, 3, 1], [2, 9, 9, 9],
                           [1, 6, 9, 9]], dtype=t.int64)
        sampled = t.tensor([3, 1, 2], dtype=t.int32)
        rejected = t.tensor([1, 0, 0], dtype=t.int32)
        query = t.tensor([0, 4, 7, 9], dtype=t.int32)
        self.ops.post_update(
            idx, computed, laststorage[:, :1], counts, tokens,
            sampled, rejected, query, rows, total,
            idx_mapping_np=[2, -1, 0], last_sampled_storage=laststorage,
        )
        ref = gold.post_update_reference(
            state=gold.PostUpdateState(
                computed=[5, 11, 20], last_sampled=[4, 5, 6],
                total_len=[6, 12, 20],
                token_rows=[[0] * 32 for _ in range(3)],
                output_counts=[[0] * 8 for _ in range(3)],
            ),
            idx_mapping=[2, -1, 0],
            sampled=tokens.tolist(), num_sampled=sampled.tolist(),
            num_rejected=rejected.tolist(), query_start_loc=query.tolist(),
        )
        self.assertEqual(computed.tolist(), ref.computed)
        self.assertEqual(total.tolist(), ref.total_len)
        self.assertEqual(laststorage[:, 0].tolist(), ref.last_sampled)
        self.assertEqual(rows.tolist(), ref.token_rows)
        self.assertEqual(counts.tolist(), ref.output_counts)
        self.assertEqual(laststorage.stride(0), 4)

    def test_no_bin_counts_and_zero_samples(self):
        t = self.t
        computed, laststorage, total, rows, counts = self._state()
        self.ops.post_update(
            t.tensor([1], dtype=t.int32), computed,
            laststorage[:, :1], None,
            t.tensor([[3, 4]], dtype=t.int64),
            t.tensor([0], dtype=t.int32),
            t.tensor([1], dtype=t.int32),
            t.tensor([0, 4], dtype=t.int32),
            rows, total,
            idx_mapping_np=[1], last_sampled_storage=laststorage,
        )
        self.assertEqual(computed.tolist(), [5, 14, 20])
        self.assertEqual(total.tolist(), [6, 12, 20])
        self.assertEqual(laststorage[:, 0].tolist(), [4, 5, 6])

    def test_invalid_cpu_request_ownership_fails_before_writes(self):
        t = self.t
        for cpu_ids in ([1, 1], [3, 1], [-2, 1]):
            with self.subTest(ids=cpu_ids):
                computed, laststorage, total, rows, counts = self._state()
                with self.assertRaisesRegex(ValueError, "duplicate or invalid"):
                    self.ops.post_update(
                        t.tensor([1, 1], dtype=t.int32), computed,
                        laststorage[:, :1], counts,
                        t.zeros((2, 4), dtype=t.int64),
                        t.tensor([1, 1], dtype=t.int32),
                        t.tensor([0, 0], dtype=t.int32),
                        t.tensor([0, 2, 4], dtype=t.int32),
                        rows, total, idx_mapping_np=cpu_ids,
                        last_sampled_storage=laststorage,
                    )
                self.assertEqual(total.tolist(), [6, 12, 20])

    def test_frozen_source_310p_only(self):
        src = (SRC / "model_runner.py").read_text()
        self.assertIn("def postprocess_sampled(", src)
        self.assertIn("from .post_update_ops import post_update", src)
        self.assertNotIn("HAS_TRITON = True", src)


if __name__ == "__main__":
    unittest.main()

"""Frozen 0.27.1 T02/T04: mandatory no-Torch gold and Torch CPU adapter tests.

No vLLM/torch-npu imports; real 310P device op support remains unverified.
"""
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "vllm_ascend/_310p/v2"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SRC / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gold = load("input_gold")


class TestInputGold(unittest.TestCase):
    def test_pos_seq_lens_mixed_decode_prefill_padding(self):
        self.assertEqual(
            gold.positions_and_lengths([2, 0], [0, 1, 4], [5, 0, 10], 8, 4),
            ([10, 5, 6, 7, 0, 0, 0, 0], [11, 8, 0, 0]),
        )

    def test_expand_logits_k7_mixed_and_zero(self):
        self.assertEqual(
            gold.expanded_logits([2, 0, 1], [0, 8, 9, 12], 12, 8),
            ([2] * 8 + [0] + [1] * 3, list(range(8)) + [0] + [0, 1, 2]),
        )
        self.assertEqual(gold.expanded_logits([], [0], 0, 8), ([], []))

    def test_negative_and_invalid_cumulative(self):
        for cum in ([1, 2], [0, 4, 3], [0, 99]):
            with self.subTest(cum=cum), self.assertRaises(ValueError):
                gold.positions_and_lengths([0], cum, [2], 8, 1)
        with self.assertRaises(ValueError):
            gold.expanded_logits([0], [0, 9], 9, 8)
        with self.assertRaises(ValueError):
            gold.expanded_logits([0], [0, 4], 3, 8)
        with self.assertRaises(ValueError):
            gold.positions_and_lengths([4], [0, 1], [0], 1, 1)


class TestTorchInputAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch absent; no-Torch input gold remains mandatory")
        cls.torch = torch
        cls.ops = load("input_ops")

    def test_t02_matches_gold(self):
        t = self.torch
        idx = t.tensor([2, 0], dtype=t.int32)
        q = t.tensor([0, 1, 4], dtype=t.int32)
        computed = t.tensor([5, 0, 10], dtype=t.int32)
        pos = t.full((8,), 123, dtype=t.int64)
        seq = t.full((4,), 123, dtype=t.int32)
        self.ops.prepare_pos_seq_lens(idx, q, computed, pos, seq)
        expected = gold.positions_and_lengths([2, 0], [0, 1, 4], [5, 0, 10], 8, 4)
        self.assertEqual((pos.tolist(), seq.tolist()), expected)

    def test_t04_matches_gold(self):
        t = self.torch
        idx = t.tensor([2, 0, 1], dtype=t.int32)
        cu = t.tensor([0, 8, 9, 12], dtype=t.int32)
        a, b = self.ops.expand_idx_mapping(idx, 12, cu, 8)
        self.assertEqual(
            (a.tolist(), b.tolist()),
            gold.expanded_logits([2, 0, 1], [0, 8, 9, 12], 12, 8),
        )
        empty_a, empty_b = self.ops.expand_idx_mapping(
            t.empty(0, dtype=t.int32), 0, t.tensor([0], dtype=t.int32), 8
        )
        self.assertEqual((empty_a.numel(), empty_b.numel()), (0, 0))

    def test_invalid_shape_dtype_fail_closed(self):
        t = self.torch
        with self.assertRaises(ValueError):
            self.ops.prepare_pos_seq_lens(
                t.tensor([0], dtype=t.int64), t.tensor([0, 1], dtype=t.int32),
                t.tensor([0], dtype=t.int32),
                t.empty(1, dtype=t.int64), t.empty(1, dtype=t.int32),
            )
        with self.assertRaises(ValueError):
            self.ops.expand_idx_mapping(
                t.tensor([0], dtype=t.int32), 1,
                t.tensor([0, 1], dtype=t.int64), 8,
            )


if __name__ == "__main__":
    unittest.main()

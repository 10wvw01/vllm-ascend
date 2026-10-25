"""R2-A T01 no-Triton prefill: stdlib gold and optional Torch CPU differential."""
import importlib.util
from pathlib import Path
import random
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "vllm_ascend/_310p/v2"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SRC / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gold = load("prefill_gold")


class TestPrefillGold(unittest.TestCase):
    def test_mixed_prefill_decode_and_inactive_slot(self):
        ids = [99] * 8
        next_tokens = [[-7, -8, -9], [-7, -8, -9], [-7, -8, -9]]
        tokens = [[10, 11, 12, 13, 14, 15, 16, 17],
                  [30, 31, 32, 33, 34, 35, 36, 37],
                  [20, 21, 22, 23, 24, 25, 26, 27]]
        out, look = gold.prepare_prefill_inputs(
            ids, next_tokens, [2, 0], [0, 2, 5], tokens, [3, 0, 6], [3, 0, 1]
        )
        self.assertEqual(out, [21, 22, 99, 99, 99, 99, 99, 99])
        self.assertEqual(look, [[-7, -8, 23], [-7, -8, 24], [-7, -8, 25]])
        self.assertEqual(ids, [99] * 8)

    def test_lookahead_zero_fills_tail(self):
        out, look = gold.prepare_prefill_inputs(
            [99] * 4, [[8], [8], [8], [8]], [0], [0, 2],
            [[10, 11, 12, 13, 14, 15]], [5], [3]
        )
        self.assertEqual(out, [13, 14, 99, 99])
        self.assertEqual(look, [[0], [0], [0], [0]])

    def test_negative_contracts(self):
        args = ([0] * 4, [[0]], [0], [0, 2], [[1, 2, 3, 4]], [3], [1])
        for changes in ({3: [1, 2]}, {2: [1]}, {5: [5]},
                        {6: [5]}, {3: [0, 5]}):
            case = list(args)
            for i, value in changes.items():
                case[i] = value
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                gold.prepare_prefill_inputs(*case)

    def test_empty_batch(self):
        self.assertEqual(gold.prepare_prefill_inputs(
            [], [], [], [0], [], [], []), ([], []))


class TestTorchPrefill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch absent; stdlib T01 oracle still runs")
        cls.t, cls.ops = torch, load("input_ops")

    def compare(self, ids, next_tokens, idx, q, all_tokens, pref, computed):
        t = self.t
        expected = gold.prepare_prefill_inputs(
            ids, next_tokens, idx, q, all_tokens, pref, computed)
        ids_t = t.tensor(ids, dtype=t.int32)
        look_t = t.tensor(next_tokens, dtype=t.int32)
        self.ops.prepare_prefill_inputs(
            ids_t, look_t, t.tensor(idx, dtype=t.int32),
            t.tensor(q, dtype=t.int32), t.tensor(all_tokens, dtype=t.int32),
            t.tensor(pref, dtype=t.int32), t.tensor(computed, dtype=t.int32))
        self.assertEqual((ids_t.tolist(), look_t.tolist()), expected)

    def test_mixed_and_tail(self):
        self.compare(
            [99] * 8, [[-7, -8, -9]] * 3, [2, 0], [0, 2, 5],
            [[10, 11, 12, 13, 14, 15, 16, 17],
             [30, 31, 32, 33, 34, 35, 36, 37],
             [20, 21, 22, 23, 24, 25, 26, 27]],
            [3, 0, 6], [3, 0, 1])
        self.compare([99] * 4, [[8]] * 4, [0], [0, 2],
                     [[10, 11, 12, 13, 14, 15]], [5], [3])

    def test_randomized_differential(self):
        rng = random.Random(310001)
        for _ in range(100):
            cap, width = rng.randint(1, 5), 32
            idx = list(range(cap))
            rng.shuffle(idx)
            idx = idx[:rng.randint(1, cap)]
            pref = [rng.randint(0, width) for _ in range(cap)]
            computed = [rng.randint(0, width) for _ in range(cap)]
            q = [0]
            for state in idx:
                q.append(q[-1] + rng.randint(0, width - computed[state]))
            look = rng.randint(1, 8)
            self.compare(
                [99] * (q[-1] + 5),
                [[-7] * cap for _ in range(look)],
                idx, q,
                [[rng.randint(1, 500) for _ in range(width)] for _ in range(cap)],
                pref, computed)

    def test_dtype_shape_fail_closed(self):
        t = self.t
        with self.assertRaises(ValueError):
            self.ops.prepare_prefill_inputs(
                t.zeros(3, dtype=t.int64), t.zeros((1, 1), dtype=t.int32),
                t.tensor([0], dtype=t.int32), t.tensor([0, 1], dtype=t.int32),
                t.zeros((1, 8), dtype=t.int32), t.tensor([2], dtype=t.int32),
                t.tensor([0], dtype=t.int32))


if __name__ == "__main__":
    unittest.main()

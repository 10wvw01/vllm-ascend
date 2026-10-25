"""R2-A T03 vLLM0.27.1 no-Torch gold + Torch CPU tensor contracts."""
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


gold = load("token_gold")


class TestTokenGold(unittest.TestCase):
    def test_k7_and_prompt_tail_and_bonus_zero(self):
        base = [99] * 20
        out, indices = gold.combine_sampled_and_draft_tokens(
            base, [1, 0], [[101], [202]], [0, 8, 11],
            [20, 8], [9, 12], [[31, 32, 33], [41, 42, 43, 44, 45, 46, 47]],
            [0, 8, 10], 10,
        )
        self.assertEqual(indices, list(range(8)) + [9, 10])
        self.assertEqual(out[:8], [202, 41, 42, 43, 44, 45, 46, 47])
        self.assertEqual(out[8:11], [99] * 3)  # chunked prefill untouched
        self.assertEqual(out[11:], [99] * 9)
        out2, indices2 = gold.combine_sampled_and_draft_tokens(
            [99] * 5, [0], [[202]], [0, 5], [10], [9], [[71, 72]],
            [0, 2], 2,
        )
        self.assertEqual((out2, indices2), ([99, 99, 99, 99, 71], [3, 4]))
        out3, indices3 = gold.combine_sampled_and_draft_tokens(
            [99] * 7, [0], [[202]], [0, 7], [20], [9],
            [[71, 72, 73, 74, 75, 76, 77]], [0, 7], 7,
            num_new_sampled_tokens=0,
        )
        self.assertEqual((out3, indices3), ([71, 72, 73, 74, 75, 76, 77], list(range(7))))

    def test_invalid_and_zero(self):
        self.assertEqual(gold.combine_sampled_and_draft_tokens(
            [], [], [], [0], [], [], [], [0], 0), ([], []))
        cases = [
            ([0], [0], [[1]], [0, 1], [3], [1], [[2]], [0, 2], 2, 1),
            ([0], [0], [[1]], [0, 1], [3], [1], [[2]], [0, 1], 1, 2),
            ([0], [1], [[1]], [0, 1], [3], [1], [[2]], [0, 1], 1, 1),
        ]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                gold.combine_sampled_and_draft_tokens(*args)


class TestTorchToken(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch absent; stdlib gold still mandatory")
        cls.t, cls.ops = torch, load("token_ops")

    def run_case(self, input_ids, idx, last, q, seq, pref, draft, cu, bonus):
        t = self.t
        expected, indices = gold.combine_sampled_and_draft_tokens(
            input_ids, idx, last, q, seq, pref, draft, cu, cu[-1], bonus)
        inp = t.tensor(input_ids, dtype=t.int32)
        actual_indices = self.ops.combine_sampled_and_draft_tokens(
            inp, t.tensor(idx, dtype=t.int32), t.tensor(last, dtype=t.int64),
            t.tensor(q, dtype=t.int32), t.tensor(seq, dtype=t.int32),
            t.tensor(pref, dtype=t.int32), t.tensor(draft, dtype=t.int64),
            t.tensor(cu, dtype=t.int32), cu[-1], bonus)
        self.assertEqual((inp.tolist(), actual_indices.tolist()), (expected, indices))

    def test_k7_mixed_and_bonus_zero(self):
        self.run_case([99] * 20, [1, 0], [[101], [202]], [0, 8, 11],
                      [20, 8], [9, 12], [[31, 32, 33, 0, 0, 0, 0],
                                         [41, 42, 43, 44, 45, 46, 47]], [0, 8, 10], 1)
        self.run_case([99] * 7, [0], [[202]], [0, 7], [20], [9],
                      [[71, 72, 73, 74, 75, 76, 77]], [0, 7], 0)

    def test_randomized_differential(self):
        rng = random.Random(310003)
        for _ in range(80):
            n, width = rng.randint(1, 5), 7
            idx = list(range(n))
            rng.shuffle(idx)
            bonus = rng.randrange(2)
            q, cu = [0], [0]
            for _ in range(n):
                query = rng.randint(1, 9)
                q.append(q[-1] + query)
                cu.append(cu[-1] + rng.randint(bonus, min(query, width + bonus)))
            pref = [rng.randint(0, 20) for _ in range(n)]
            seq = [rng.randint(0, 30) for _ in range(n)]
            last = [[rng.randint(1, 500)] for _ in range(n)]
            draft = [[rng.randint(1, 500) for _ in range(width)] for _ in range(n)]
            self.run_case([99] * (q[-1] + 4), idx, last, q, seq, pref, draft, cu, bonus)

    def test_dtype_fail_closed(self):
        t = self.t
        with self.assertRaises(ValueError):
            self.ops.combine_sampled_and_draft_tokens(
                t.zeros(2, dtype=t.int64), t.tensor([0], dtype=t.int32),
                t.zeros((1, 1), dtype=t.int64), t.tensor([0, 2], dtype=t.int32),
                t.tensor([2], dtype=t.int32), t.tensor([0], dtype=t.int32),
                t.zeros((1, 7), dtype=t.int64), t.tensor([0, 1], dtype=t.int32), 1)


if __name__ == "__main__":
    unittest.main()

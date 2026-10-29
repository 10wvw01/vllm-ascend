"""R2-A T07/T08: mandatory stdlib gold and independent Torch CPU tensor tests.

No GPU/NPU execution and no production runner import. K7, prefill and mixed
requests exercise frozen vLLM 0.27.1 semantics, not device op availability.
"""
import importlib.util
import pathlib
import random
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "vllm_ascend/_310p/v2"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SRC / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gold = load("sampling_gold")


class TestSamplingGold(unittest.TestCase):
    def test_k7_chunked_prefill_and_full_decode(self):
        self.assertEqual(
            gold.sampled_and_rejected(
                [8, 5, 1, 0], [2, 13, 4, 6], [0, 8, 16, 17, 18],
                [2, 0, 3, 1], [13, 6, 5, 9],
            ),
            ([0, 5, 0, 0], [0, 3, 0, 1]),
        )
        self.assertEqual(
            gold.flattened_sampled(
                [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
                [3, 1, 0], [0, 4, 7, 9], 9,
            ),
            [10, 11, 12, 0, 20, 0, 0, 0, 0],
        )

    def test_zero_and_negative_contracts(self):
        self.assertEqual(gold.sampled_and_rejected([], [], [0], [], []), ([], []))
        self.assertEqual(gold.flattened_sampled([], [], [0], 0), [])
        for args in [
            ([2], [1], [0, 1], [0], [0]),
            ([0], [1], [1, 2], [0], [0]),
            ([0], [1], [0, 1], [-1], [0]),
            ([0], [1], [0, 1], [3], [0]),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                gold.sampled_and_rejected(*args)
        for args in [
            ([[1]], [2], [0, 1], 1),
            ([[1]], [0], [0, 2], 1),
            ([[1]], [-1], [0, 1], 1),
            ([[1]], [1], [0, 1], -1),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                gold.flattened_sampled(*args)

    def test_randomized_gold(self):
        rng = random.Random(3100271)
        for _ in range(120):
            n = rng.randrange(0, 7)
            counts = [rng.randrange(1, 9) for _ in range(n)]
            sampled = [rng.randrange(c + 1) for c in counts]
            cu = [0]
            for count in counts:
                cu.append(cu[-1] + count)
            rows = [[rng.randrange(1, 1000) for _ in range(8)] for _ in range(n)]
            flat = gold.flattened_sampled(rows, sampled, cu, cu[-1])
            for r in range(n):
                self.assertEqual(flat[cu[r]:cu[r] + sampled[r]], rows[r][:sampled[r]])
                self.assertEqual(flat[cu[r] + sampled[r]:cu[r + 1]],
                                 [0] * (counts[r] - sampled[r]))


class TestTorchSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch absent; mandatory stdlib gold still runs")
        cls.t = torch
        cls.ops = load("sampling_ops")

    def _i(self, values):
        return self.t.tensor(values, dtype=self.t.int32)

    def test_t07_matches_gold_and_mutates_input(self):
        num = self._i([8, 5, 1, 0])
        seq = self._i([2, 13, 4, 6])
        cu = self._i([0, 8, 16, 17, 18])
        idx = self._i([2, 0, 3, 1])
        pref = self._i([13, 6, 5, 9])
        returned, rejected = self.ops.get_num_sampled_and_rejected(num, seq, cu, idx, pref)
        self.assertIs(returned, num)
        self.assertEqual((num.tolist(), rejected.tolist()),
                         gold.sampled_and_rejected([8, 5, 1, 0], [2, 13, 4, 6],
                                                   [0, 8, 16, 17, 18], [2, 0, 3, 1],
                                                   [13, 6, 5, 9]))

    def test_t08_stride_k7_padding_and_zero(self):
        t = self.t
        # Noncontiguous sampled row stride is legal in upstream.
        sampled = t.arange(4 * 16, dtype=t.int64).reshape(4, 16)[:, ::2]
        counts = [8, 5, 0, 1]
        cu = [0, 8, 16, 17, 18]
        out = self.ops.flatten_sampled(sampled, self._i(counts), self._i(cu), 18)
        self.assertEqual(out.tolist(), gold.flattened_sampled(
            sampled.tolist(), counts, cu, 18))
        self.assertEqual(self.ops.flatten_sampled(
            t.empty((0, 8), dtype=t.int64), self._i([]), self._i([0]), 0
        ).numel(), 0)

    def test_randomized_cpu_differential(self):
        rng, t = random.Random(27107), self.t
        for _ in range(80):
            n = rng.randrange(1, 6)
            counts = [rng.randrange(1, 9) for _ in range(n)]
            sampled = [rng.randrange(c + 1) for c in counts]
            cu = [0]
            for c in counts:
                cu.append(cu[-1] + c)
            rows = [[rng.randrange(1, 999) for _ in range(8)] for _ in range(n)]
            out = self.ops.flatten_sampled(
                t.tensor(rows, dtype=t.int64), self._i(sampled), self._i(cu), cu[-1])
            self.assertEqual(out.tolist(), gold.flattened_sampled(rows, sampled, cu, cu[-1]))
            idx = list(range(n))
            rng.shuffle(idx)
            prefill = [rng.randrange(0, 15) for _ in range(n)]
            seq = [rng.randrange(0, 15) for _ in range(n)]
            in_sampled = self._i(sampled)
            got, rej = self.ops.get_num_sampled_and_rejected(
                in_sampled, self._i(seq), self._i(cu), self._i(idx), self._i(prefill))
            self.assertEqual((got.tolist(), rej.tolist()),
                             gold.sampled_and_rejected(sampled, seq, cu, idx, prefill))

    def test_invalid_shape_dtype_device_contract(self):
        t = self.t
        with self.assertRaises(ValueError):
            self.ops.get_num_sampled_and_rejected(
                self._i([1]), self._i([2]), self._i([0, 1]),
                t.tensor([0], dtype=t.int64), self._i([1]))
        with self.assertRaises(ValueError):
            self.ops.flatten_sampled(t.ones((1, 2)), self._i([1]), self._i([0, 1]), 1)
        with self.assertRaises(ValueError):
            self.ops.flatten_sampled(t.ones((1, 2), dtype=t.int64),
                                     self._i([1]), self._i([0, 1]), -1)


if __name__ == "__main__":
    unittest.main()

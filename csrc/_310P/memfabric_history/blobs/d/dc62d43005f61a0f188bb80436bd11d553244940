"""T17/T18 CPU greedy/temperature candidate; never claims stochastic GO."""
import ast
import importlib.util
import pathlib
import sys
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "vllm_ascend/_310p/v2/gumbel_ops.py"


class TestGreedyCandidateSource(unittest.TestCase):
    def test_no_triton_or_host_scalar_extract(self):
        src = SRC.read_text()
        self.assertNotIn("triton", src.lower().split("from __future__")[1])
        tree = ast.parse(src)
        self.assertFalse(any(isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Attribute)
                             and n.func.attr == "item"
                             for n in ast.walk(tree)))

    def test_non_greedy_fail_closed(self):
        self.assertIn("stochastic Gumbel/Philox is not implemented",
                      SRC.read_text())


class TestGreedyCandidateTorch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest("Torch absent; source contracts still run")
        spec = importlib.util.spec_from_file_location("_r2_gumbel", SRC)
        cls.ops = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.ops
        spec.loader.exec_module(cls.ops)

    def test_temperature_zero_one_two(self):
        logits = torch.tensor([[2., 4., 6.], [3., 9., 12.], [4., 2., 1.]])
        ids = torch.tensor([1, 0, 2], dtype=torch.int32)
        self.ops.apply_temperature(logits, ids, torch.tensor([0., 2., 1.]))
        self.assertEqual(logits.tolist(), [[1., 2., 3.],
                                           [3., 9., 12.], [4., 2., 1.]])

    def test_greedy(self):
        logits = torch.tensor([[1., 3., 2.], [3., 1., 2.]])
        out = self.ops.gumbel_sample(
            logits, torch.tensor([2, 0], dtype=torch.int32),
            torch.tensor([0., 0., 0.]),
            torch.zeros(3, dtype=torch.int64), torch.tensor([1, 2]), False,
        )
        self.assertEqual(out.tolist(), [1, 0])
        self.assertEqual(out.dtype, torch.int64)

    def test_stochastic_rejected_before_any_output(self):
        logits = torch.tensor([[1., 3., 2.]])
        with self.assertRaisesRegex(RuntimeError, "stochastic"):
            self.ops.gumbel_sample(
                logits, torch.tensor([0], dtype=torch.int32),
                torch.tensor([0.8]), torch.zeros(1, dtype=torch.int64),
                torch.tensor([1]), False,
            )

    def test_invalid_request_index_before_mutation(self):
        logits = torch.tensor([[1., 3., 2.]])
        with self.assertRaisesRegex(RuntimeError, "persistent"):
            self.ops.apply_temperature(
                logits, torch.tensor([3], dtype=torch.int32),
                torch.tensor([0.]),
            )
        self.assertEqual(logits.tolist(), [[1., 3., 2.]])


if __name__ == "__main__":
    unittest.main()

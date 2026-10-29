"""T23 non-Triton greedy K7 rejection differential and error contracts."""
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
SRC = ROOT / "vllm_ascend/_310p/v2/rejection_ops.py"


class TestT23Source(unittest.TestCase):
    def test_not_triton_and_no_host_scalar_extraction(self):
        tree = ast.parse(SRC.read_text())
        self.assertFalse(any(isinstance(n, ast.Call)
                             and isinstance(n.func, ast.Attribute)
                             and n.func.attr == "item" for n in ast.walk(tree)))
        self.assertNotIn("@triton.jit", SRC.read_text())
        self.assertIn("stochastic rejection is not implemented", SRC.read_text())


class TestT23Torch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest("Torch absent; source audit remains mandatory")
        spec = importlib.util.spec_from_file_location("_t23_rejection", SRC)
        cls.ops = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.ops
        spec.loader.exec_module(cls.ops)

    def fixture(self, lengths=(8, 3), mismatch=None, temp=(0., 0.)):
        n = len(lengths)
        total = sum(lengths)
        target = torch.full((total, 11), -10.)
        ids = torch.tensor([(i*3) % 11 for i in range(total)], dtype=torch.int64)
        target.scatter_(1, ids[:, None], 10.)
        draft = ids.clone()
        for i in range(total-1):
            draft[i+1] = ids[i]
        if mismatch is not None:
            req, step = mismatch
            idx = sum(lengths[:req]) + step
            draft[idx+1] = (ids[idx] + 1) % 11
        cu = torch.tensor([0] + [sum(lengths[:i+1]) for i in range(n)],
                          dtype=torch.int32)
        mapping = torch.arange(n, dtype=torch.int32)
        args = (target, None, draft, cu, torch.arange(total), mapping,
                torch.zeros(total, dtype=torch.int32),
                torch.zeros(total, dtype=torch.int32),
                torch.tensor(temp), torch.zeros(n, dtype=torch.int64), 7)
        return args, ids

    def test_all_accepted_mixed_decode_and_prefill(self):
        args, ids = self.fixture()
        out, count = self.ops.rejection_sample(*args)
        self.assertEqual(count.tolist(), [8, 3])
        self.assertEqual(out[0].tolist(), ids[:8].tolist())
        self.assertEqual(out[1, :3].tolist(), ids[8:11].tolist())
        self.assertEqual(out[1, 3:].tolist(), [0]*5)

    def test_first_mismatch_resampled_from_target_argmax(self):
        args, ids = self.fixture(mismatch=(0, 2))
        out, count = self.ops.rejection_sample(*args)
        self.assertEqual(count.tolist(), [3, 3])
        self.assertEqual(out[0, :3].tolist(), ids[:3].tolist())
        self.assertEqual(out[0, 3:].tolist(), [0]*5)

    def test_first_token_rejected(self):
        args, ids = self.fixture(mismatch=(1, 0))
        out, count = self.ops.rejection_sample(*args)
        self.assertEqual(count.tolist(), [8, 1])
        self.assertEqual(out[1, 0].item(), ids[8].item())

    def test_non_greedy_fails_closed(self):
        args, _ = self.fixture(temp=(0., 1.))
        with self.assertRaisesRegex(RuntimeError, "stochastic"):
            self.ops.rejection_sample(*args)

    def test_invalid_cumulative_logits_fails_closed(self):
        args, _ = self.fixture()
        values = list(args)
        values[3] = torch.tensor([0, 8, 20], dtype=torch.int32)
        with self.assertRaisesRegex(RuntimeError, "cumulative"):
            self.ops.rejection_sample(*values)


if __name__ == "__main__":
    unittest.main()

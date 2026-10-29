"""310P3 V2 synchronous Eager output must not enter upstream CUDA event path."""
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "vllm_ascend/_310p/v2"


class TestAsyncOutputStatic(unittest.TestCase):
    def test_copied_sample_tokens_binds_npu_snapshot(self):
        runner = (SRC / "model_runner.py").read_text()
        self.assertIn("from .async_output import NPUAsyncOutput310V2 as AsyncOutput", runner)
        self.assertNotIn("from vllm.v1.worker.gpu.async_utils import AsyncOutput", runner)
        source = (SRC / "async_output.py").read_text()
        body = source.split("class NPUAsyncOutput310V2")[1]
        self.assertNotIn("torch.cuda", body)
        self.assertNotIn("non_blocking=True", body)
        self.assertIn('to("cpu", non_blocking=False)', body)


class TestAsyncOutputCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch absent; source contract still runs")
        cls.torch = torch
        output = types.ModuleType("vllm.v1.outputs")
        output.AsyncModelRunnerOutput = type("AsyncModelRunnerOutput", (), {})
        with patch.dict(sys.modules, {
            "vllm": types.ModuleType("vllm"),
            "vllm.v1": types.ModuleType("vllm.v1"),
            "vllm.v1.outputs": output,
        }):
            spec = importlib.util.spec_from_file_location(
                "_r2_npu_async_output", SRC / "async_output.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        cls.Output = mod.NPUAsyncOutput310V2

    def test_snapshot_precedes_mutation_and_trims_k7(self):
        t = self.torch
        device = t.tensor([[10, 11, 12, 13, 14, 15, 16, 17],
                           [20, 21, 22, 23, 24, 25, 26, 27]], dtype=t.int64)
        count = t.tensor([3, 8], dtype=t.int32)
        sampler = types.SimpleNamespace(
            sampled_token_ids=device, logprobs_tensors=None, num_nans=None,
        )
        result = types.SimpleNamespace(
            req_ids=["a", "b"], prompt_logprobs_dict={}, sampled_token_ids=None,
        )
        out = self.Output(result, sampler, count, None, None)
        device.fill_(999)
        count.zero_()
        self.assertIs(out.get_output(), result)
        self.assertEqual(result.sampled_token_ids, [
            [10, 11, 12], [20, 21, 22, 23, 24, 25, 26, 27],
        ])
        with self.assertRaisesRegex(RuntimeError, "once"):
            out.get_output()

    def test_unsupported_outputs_and_bad_counts_fail_closed(self):
        t = self.torch
        sampled = types.SimpleNamespace(
            sampled_token_ids=t.zeros((1, 8), dtype=t.int64),
            logprobs_tensors=None, num_nans=None,
        )
        result = types.SimpleNamespace(
            req_ids=["a"], prompt_logprobs_dict={}, sampled_token_ids=None,
        )
        with self.assertRaisesRegex(NotImplementedError, "EP fault"):
            self.Output(result, sampled, t.tensor([1]), None, None, True)
        sampled.num_nans = t.zeros(1, dtype=t.int32)
        with self.assertRaisesRegex(NotImplementedError, "logprobs/NaN"):
            self.Output(result, sampled, t.tensor([1]), None, None)
        sampled.num_nans = None
        with self.assertRaises(ValueError):
            self.Output(result, sampled, t.tensor([1, 1]), None, None)
        out = self.Output(result, sampled, t.tensor([9]), None, None)
        with self.assertRaisesRegex(ValueError, "count"):
            out.get_output()
        self.assertIsNone(result.sampled_token_ids)


if __name__ == "__main__":
    unittest.main()

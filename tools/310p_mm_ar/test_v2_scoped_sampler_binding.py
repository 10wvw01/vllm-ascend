"""Source-only scope contracts for 310P3 V2 T07/T17/T23 sampler wiring."""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "vllm_ascend/_310p/v2/sampler.py"
RUNNER = ROOT / "vllm_ascend/_310p/v2/model_runner.py"


class TestScopedSamplers(unittest.TestCase):
    def test_frozen_methods_are_locally_rebound(self):
        src = SRC.read_text()
        tree = ast.parse(src)
        keys = {
            k.value
            for node in ast.walk(tree) if isinstance(node, ast.Dict)
            for k in node.keys if isinstance(k, ast.Constant)
        }
        self.assertIn("gumbel_sample", keys)
        self.assertIn("rejection_sample", keys)
        self.assertIn("get_num_sampled_and_rejected", keys)
        self.assertNotIn("Sampler.sample =", src)
        self.assertNotIn("RejectionSampler._verify =", src)

    def test_frozen_load_model_factory_scoped_to_310p(self):
        src = RUNNER.read_text()
        self.assertIn('"Sampler": NPUSampler310V2', src)
        self.assertIn('"RejectionSampler": NPURejectionSampler310V2', src)
        worker = (ROOT / "vllm_ascend/_310p/worker_310p.py").read_text()
        self.assertNotIn("NPUModelRunner310V2Candidate", worker)
        self.assertIn("NOT_IMPLEMENTED", worker)

    def test_unimplemented_logprobs_and_flashinfer_fail_closed(self):
        src = SRC.read_text()
        self.assertIn("self.use_flashinfer = False", src)
        self.assertIn("raise NotImplementedError(", src)
        self.assertIn("T08/T22", src)


if __name__ == "__main__":
    unittest.main()

"""Scope-limited Greedy Sampler must fail closed before Triton operations."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "vllm_ascend/_310p/v2/sampler.py"


class TestGreedySamplerGuard(unittest.TestCase):
    def test_guard_checks_frozen_upstream_processing_predicate(self):
        module = ast.parse(PATH.read_text())
        cls = next(c for c in module.body if isinstance(c, ast.ClassDef)
                   and c.name == "NPUSampler310V2")
        guard = next(f for f in cls.body if isinstance(f, ast.FunctionDef)
                     and f.name == "apply_sampling_params")
        self.assertEqual([x.arg for x in guard.args.args], [
            "self", "logits", "expanded_idx_mapping", "idx_mapping_np",
            "pos", "input_ids", "expanded_local_pos", "skip_top_k_top_p",
        ])
        self.assertTrue(any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_requires_logits_processing"
            for n in ast.walk(guard)
        ))
        self.assertTrue(any(isinstance(n, ast.Raise) for n in ast.walk(guard)))
        self.assertFalse(any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in {"apply_min_p", "apply_bad_words",
                                "apply_logit_bias", "apply_penalties"}
            for n in ast.walk(guard)
        ))

    def test_runner_load_model_binds_scope_only_sampler(self):
        runner = (ROOT / "vllm_ascend/_310p/v2/model_runner.py").read_text()
        self.assertIn('"Sampler": NPUSampler310V2', runner)
        self.assertIn('"RejectionSampler": NPURejectionSampler310V2', runner)


if __name__ == "__main__":
    unittest.main()

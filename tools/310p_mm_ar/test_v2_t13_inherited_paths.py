"""T13 inherited V2 Eager/profile and hybrid-Mamba fail-closed contract.

Stdlib-only AST tests: they cannot prove actual NPU 310P3 kernel support.
"""
import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "vllm_ascend/_310p/v2/model_runner.py"
WORKER = ROOT / "vllm_ascend/_310p/worker_310p.py"
MATRIX = ROOT / "csrc/_310P/memfabric_mm_ar_v2/docs/V2_TRITON_REPLACEMENT_MATRIX.md"

def candidate_method(name):
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
               and x.name == "NPUModelRunner310V2Candidate")
    return next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                and x.name == name)

class TestT13InheritedPaths(unittest.TestCase):
    def test_inherited_execute_model_dummy_uses_scoped_ascend_batch(self):
        m = candidate_method("execute_model")
        s = ast.unparse(m)
        self.assertIn("inspect.unwrap(GPUModelRunner.execute_model)", s)
        self.assertIn("'InputBatch': AscendInputBatch", s)
        self.assertIn("dummy_run=dummy_run", s)
        self.assertIn("is_profile=is_profile", s)
        self.assertNotIn("super().execute_model(", s)
        self.assertIn("torch.inference_mode()", s)

    def test_inherited_dummy_sampler_uses_scoped_ascend_batch(self):
        s = ast.unparse(candidate_method("_dummy_sampler_run"))
        self.assertIn("inspect.unwrap(GPUModelRunner._dummy_sampler_run)", s)
        self.assertIn("'InputBatch': AscendInputBatch", s)

    def test_hybrid_align_prohibits_model_loader_until_t30_t31(self):
        s = ast.unparse(candidate_method("load_model"))
        self.assertIn("self.model_config.is_hybrid", s)
        self.assertIn("self.cache_config.mamba_cache_mode == 'align'", s)
        self.assertIn("T30/T31", s)
        self.assertLess(s.index("T30/T31"), s.index("GPUModelRunner.load_model"))

    def test_new_model_state_triton_is_logged_and_worker_protected(self):
        s = MATRIX.read_text(encoding="utf-8")
        for token in ("| T30 |", "| T31 |",
                      "preprocess_mamba_align_fused_kernel",
                      "_scatter_num_accepted_kernel"):
            self.assertIn(token, s)
        self.assertIn("Triton-free V2 NPUWorker/ModelRunner is NOT_IMPLEMENTED",
                      WORKER.read_text(encoding="utf-8"))

if __name__ == "__main__":
    unittest.main()

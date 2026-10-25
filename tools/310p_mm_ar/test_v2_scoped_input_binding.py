"""V2 310P3 T01-T04 Eager source binding; no vllm import in CI."""
import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHILD = ROOT / "vllm_ascend/_310p/v2/model_runner.py"
ASCEND = ROOT / "vllm_ascend/worker/v2/model_runner.py"
REQUIRED = {
    "prepare_prefill_inputs",
    "prepare_pos_seq_lens",
    "expand_idx_mapping",
    "combine_sampled_and_draft_tokens",
}


class TestScopedInputBinding(unittest.TestCase):
    def test_all_four_real_gpu_triton_callers_are_rebound(self):
        tree = ast.parse(CHILD.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "NPUModelRunner310V2Candidate")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == "prepare_inputs")
        dict_literals = [n for n in ast.walk(method) if isinstance(n, ast.Dict)]
        keys = set()
        for d in dict_literals:
            keys.update(n.value for n in d.keys if isinstance(n, ast.Constant)
                        and isinstance(n.value, str))
        self.assertEqual(keys, REQUIRED)

    def test_upstream_ascend_imports_unmodified(self):
        src = ASCEND.read_text()
        self.assertIn("from vllm.v1.worker.gpu.input_batch import (", src)
        for name in REQUIRED:
            self.assertIn(name, src)

    def test_not_registered_by_310p_worker(self):
        worker = (ROOT / "vllm_ascend/_310p/worker_310p.py").read_text()
        self.assertNotIn("NPUModelRunner310V2Candidate", worker)
        self.assertIn("NOT_IMPLEMENTED", worker)


if __name__ == "__main__":
    unittest.main()

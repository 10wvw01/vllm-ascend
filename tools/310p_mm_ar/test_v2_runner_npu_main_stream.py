"""310P3 V2 class-level main_stream is NPU-native, without global patches."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "vllm_ascend/_310p/v2/model_runner.py"


class TestNPUStreamContract(unittest.TestCase):
    def test_310p_candidate_overrides_inherited_cuda_cached_property(self):
        tree = ast.parse(RUNNER.read_text())
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
                   and x.name == "NPUModelRunner310V2Candidate")
        method = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                      and x.name == "main_stream")
        self.assertTrue(any(
            isinstance(x, ast.Name) and x.id == "cached_property"
            for x in method.decorator_list
        ))
        calls = [x for x in ast.walk(method) if isinstance(x, ast.Call)]
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertIsInstance(call.func, ast.Attribute)
        self.assertEqual(call.func.attr, "current_stream")
        self.assertIsInstance(call.func.value, ast.Attribute)
        self.assertEqual(call.func.value.attr, "npu")
        self.assertEqual(len(call.args), 1)

    def test_runtime_worker_guard_unchanged(self):
        worker = (ROOT / "vllm_ascend/_310p/worker_310p.py").read_text()
        self.assertIn("Triton-free V2 NPUWorker/ModelRunner is NOT_IMPLEMENTED", worker)


if __name__ == "__main__":
    unittest.main()

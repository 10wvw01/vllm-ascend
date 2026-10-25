"""CPU static contract for 310P frozen KV orchestrator dependency swap."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
CHILD = ROOT / "vllm_ascend/_310p/v2/model_runner.py"
BLOCKS = ROOT / "vllm_ascend/_310p/v2/block_tables.py"
WORKER = ROOT / "vllm_ascend/_310p/worker_310p.py"


class TestKVScopedCtor(unittest.TestCase):
    def test_kv_injects_non_triton_table_and_eager_graph_only(self):
        source = CHILD.read_text()
        tree = ast.parse(source)
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
                   and x.name == "NPUModelRunner310V2Candidate")
        method = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                      and x.name == "initialize_kv_cache")
        keys = set()
        for node in ast.walk(method):
            if isinstance(node, ast.Dict):
                keys.update(x.value for x in node.keys
                            if isinstance(x, ast.Constant)
                            and isinstance(x.value, str))
        self.assertEqual(keys, {"BlockTables", "ModelCudaGraphManager"})
        self.assertIn("CUDAGraphMode.NONE", source)
        self.assertIn("_310p_v2_kv_poisoned = True", source)

    def test_slot_dtype_matches_real_ascend_backend(self):
        source = BLOCKS.read_text()
        self.assertIn("PAD_SLOT_ID, dtype=torch.int32, device=device", source)
        self.assertIn('slot = (blocks.long() * bs +', source)

    def test_worker_still_fail_closed(self):
        self.assertIn("NOT_IMPLEMENTED", WORKER.read_text())


if __name__ == "__main__":
    unittest.main()

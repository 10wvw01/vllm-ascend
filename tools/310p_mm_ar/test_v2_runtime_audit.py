"""CPU-only R2 audit source boundary tests; no NPU imports."""
import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "vllm_ascend/worker/v2/model_runner.py"
DFLASH = ROOT / "vllm_ascend/worker/v2/spec_decode/dflash/speculator.py"
WORKER = ROOT / "vllm_ascend/_310p/worker_310p.py"
DOCS = ROOT / "csrc/_310P/memfabric_mm_ar_v2/docs"
UPSTREAM_SHA = "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"


def calls(tree):
    return [x for x in ast.walk(tree) if isinstance(x, ast.Call)]


class AuditContracts(unittest.TestCase):
    def test_known_target_triton_symbols_still_imported(self):
        tree = ast.parse(BACKEND.read_text())
        imports = {
            alias.name for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "vllm.v1.worker.gpu.input_batch"
            for alias in node.names
        }
        self.assertEqual(
            imports,
            {
                "prepare_prefill_inputs", "prepare_pos_seq_lens",
                "combine_sampled_and_draft_tokens", "expand_idx_mapping",
            },
            "update Triton audit matrix if backend input path changes",
        )

    def test_dflash_inherits_upstream_propose_without_mutating_has_triton(self):
        src = DFLASH.read_text()
        self.assertIn("super().propose(", src)
        self.assertNotIn("HAS_TRITON = True", src)

    def test_fail_closed_remains_during_audit(self):
        src = WORKER.read_text()
        self.assertIn("Triton-free V2 NPUWorker/ModelRunner is NOT_IMPLEMENTED", src)

    def test_both_live_audit_ledgers_exist(self):
        for file in ("V2_API_ABI_MATRIX.md", "V2_TRITON_REPLACEMENT_MATRIX.md"):
            with self.subTest(file=file):
                doc = (DOCS / file).read_text()
                self.assertIn("NO", doc.upper())  # no false all-done marker
                self.assertTrue(len(doc) > 1000)

    def test_git_commit_is_pinned_in_abi_matrix(self):
        self.assertIn(UPSTREAM_SHA, (DOCS / "V2_API_ABI_MATRIX.md").read_text())

if __name__ == "__main__":
    unittest.main()

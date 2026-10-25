"""Frozen GPUModelRunner DraftTokensHandler constructor must not allocate CUDA."""
import ast
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "vllm_ascend/_310p/v2"


class TestDraftHandlerNoCUDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "_r2_npu_draft_handler", SRC / "draft_handler.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls.Handler = mod.NPUDraftTokensHandler310V2

    def test_constructor_is_private_global_override(self):
        runner = (SRC / "model_runner.py").read_text()
        self.assertIn('"DraftTokensHandler": NPUDraftTokensHandler310V2', runner)
        self.assertIn('"RequestState": NPURequestState', runner)
        handler = (SRC / "draft_handler.py").read_text()
        self.assertNotIn("torch.cuda", handler.split("class NPUDraftTokensHandler310V2:")[1])
        self.assertNotIn("import torch", handler)
        self.assertNotIn("DraftTokensHandler.__init__", runner)

    def test_greedy_k7_and_mixed_reuse(self):
        h = self.Handler("cpu")
        batch = types.SimpleNamespace(
            has_structured_output_reqs=False, req_ids=["a", "b"]
        )
        tokens = types.SimpleNamespace(ndim=2, shape=(2, 7))
        h.set_draft_tokens(batch, tokens)
        self.assertEqual(h.req_ids, ["a", "b"])
        self.assertEqual(h.num_draft_tokens, 7)
        self.assertIsNone(h.draft_tokens_np)
        output = types.ModuleType("vllm.v1.outputs")
        output.DraftTokenIds = lambda req_ids, draft_token_ids: (req_ids, draft_token_ids)
        with patch.dict(sys.modules, {
            "vllm": types.ModuleType("vllm"),
            "vllm.v1": types.ModuleType("vllm.v1"),
            "vllm.v1.outputs": output,
        }):
            self.assertEqual(h.get_draft_tokens(), (
                ["a", "b"], [[-1] * 7, [-1] * 7],
            ))
        batch.req_ids = ["c"]
        h.set_draft_tokens(batch, types.SimpleNamespace(ndim=2, shape=(1, 7)))
        self.assertEqual(h.req_ids, ["c"])

    def test_grammar_rejected_before_state_mutation(self):
        h = self.Handler("cpu")
        h.set_draft_tokens(
            types.SimpleNamespace(has_structured_output_reqs=False, req_ids=["a"]),
            types.SimpleNamespace(ndim=2, shape=(1, 7)),
        )
        with self.assertRaisesRegex(NotImplementedError, "structured draft"):
            h.set_draft_tokens(
                types.SimpleNamespace(has_structured_output_reqs=True, req_ids=["b"]),
                types.SimpleNamespace(ndim=2, shape=(1, 7)),
            )
        self.assertEqual(h.req_ids, ["a"])
        with self.assertRaises(ValueError):
            h.set_draft_tokens(
                types.SimpleNamespace(has_structured_output_reqs=False, req_ids=["x"]),
                types.SimpleNamespace(ndim=2, shape=(2, 7)),
            )
        self.assertEqual(h.req_ids, ["a"])


if __name__ == "__main__":
    unittest.main()

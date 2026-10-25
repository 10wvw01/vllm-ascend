"""Upstream vLLM 0.27.1 ↔ Ascend V2 source ABI contract (CPU/AST-only).

Avoid importing torch_npu or Triton; CI verifies the static contract and flags
version drift before entering the 310P3 machine. This is not runtime GO.
"""
import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "vllm_ascend/worker/v2/model_runner.py"
DFLASH = ROOT / "vllm_ascend/worker/v2/spec_decode/dflash/speculator.py"
UPSTREAM_SHA = "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac"


def _class(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == name)


def _method(cls, name):
    return next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == name)


class V2AbiTests(unittest.TestCase):
    def test_dflash_set_attn_matches_frozen_upstream_5_args(self):
        cls = _class(DFLASH, "AscendDFlashSpeculator")
        method = _method(cls, "set_attn")
        self.assertEqual(
            [arg.arg for arg in method.args.args],
            [
                "self", "model_state", "kv_cache_config", "block_tables",
                "target_input_buffers", "target_attn_groups",
            ],
        )
        call = next(
            x for x in ast.walk(method) if isinstance(x, ast.Call)
            and isinstance(x.func, ast.Attribute) and x.func.attr == "set_attn"
        )
        self.assertEqual(len(call.args), 5)
        self.assertEqual(len(call.keywords), 0)

    def test_removed_obsolete_dflash_metadata_wrapper(self):
        cls = _class(DFLASH, "AscendDFlashSpeculator")
        methods = [x.name for x in cls.body if isinstance(x, ast.FunctionDef)]
        self.assertNotIn("build_draft_attn_metadatas", methods)

    def test_runner_has_no_removed_upstream_postprocess_hook(self):
        cls = _class(RUNNER, "NPUModelRunner")
        methods = [x.name for x in cls.body if isinstance(x, ast.FunctionDef)]
        self.assertNotIn("postprocess", methods)
        self.assertIn("postprocess_sampled", methods)

    def test_stream_is_npu_not_cuda(self):
        cls = _class(RUNNER, "NPUModelRunner")
        method = _method(cls, "_copy_num_computed_tokens_to_cpu")
        calls = [
            x for x in ast.walk(method) if isinstance(x, ast.Call)
            and isinstance(x.func, ast.Attribute) and x.func.attr == "current_stream"
        ]
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].func.value, ast.Attribute)
        self.assertEqual(calls[0].func.value.attr, "npu")

    def test_bonus_token_count_is_forwarded_explicitly(self):
        cls = _class(RUNNER, "NPUModelRunner")
        meth = _method(cls, "prepare_inputs")
        calls = [x for x in ast.walk(meth) if isinstance(x, ast.Call)
                 and isinstance(x.func, ast.Name)
                 and x.func.id == "combine_sampled_and_draft_tokens"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 10)
        self.assertIsInstance(calls[0].args[-1], ast.Attribute)
        self.assertEqual(calls[0].args[-1].attr, "num_new_sampled_tokens_per_step")

    def test_pinned_version_is_not_runtime_proof(self):
        self.assertEqual(len(UPSTREAM_SHA), 40)


if __name__ == "__main__":
    unittest.main()

"""Project-scope governance, no torch/NPU imports and no device actions."""
import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOC = ROOT / "csrc/_310P/memfabric_mm_ar_v2/docs"
SCENARIO = DOC / "R2_TARGET_SCENARIO_310P_QWEN36_DFLASH.json"
ENTRY = ROOT / "PROJECT_310P_MEMFABRIC.md"
PLAN = DOC / "V2_RUNNER_TRITON_FREE_MASTER_PLAN_20261025.md"

class TestTargetScope(unittest.TestCase):
    def test_frozen_only_model_and_workload(self):
        scenario = json.loads(SCENARIO.read_text(encoding="utf-8"))
        self.assertEqual(scenario["target_model_id"],
                         "Eco-Tech/Qwen3.6-35B-A3B-w8a8")
        self.assertEqual(scenario["hardware"], "Ascend 310P3 one card dual-die")
        self.assertEqual(scenario["tensor_parallel_size"], 2)
        for field, expected in (("input_tokens",4096),("output_tokens",2048),
                                ("concurrent_requests",10),("total_requests",50),
                                ("num_speculative_tokens",7)):
            self.assertEqual(scenario[field], expected)
        self.assertEqual(scenario["speculative_method"], "dflash")
        self.assertTrue(scenario["stock_control_required"])
        self.assertEqual(scenario["runtime_dflash_triton_hit"], "NOT_RUN")

    def test_entry_and_execution_plan_link_model_specific_route(self):
        for path in (ENTRY, PLAN):
            content = path.read_text(encoding="utf-8")
            self.assertIn("Eco-Tech/Qwen3.6-35B-A3B-w8a8",content)
            self.assertIn("R2_TARGET_MODEL_DFLASH_ROUTE_20261025.md",content)
        src = (DOC / "R2_TARGET_MODEL_DFLASH_ROUTE_20261025.md").read_text(
            encoding="utf-8")
        for token in ("npu_copy_and_expand_dflash_inputs","_prepare_dflash_inputs_kernel",
                      "mamba_cache_mode","NOT_IMPLEMENTED","K=7"):
            self.assertIn(token,src)

if __name__=="__main__":
    unittest.main()

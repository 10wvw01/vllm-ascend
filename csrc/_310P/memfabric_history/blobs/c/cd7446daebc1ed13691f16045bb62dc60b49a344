"""T11 frozen DFlash K7 CPU tensor/gold and isolated proposer binding.

Torch tests skip in lightweight CI without Torch; separate torch2.10+cpu
job runs all tests. No vllm/torch_npu import is required by this test.
"""
import ast
import importlib.util
import pathlib
import sys
import types
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "vllm_ascend/_310p/v2"


def isolated_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestT11Binding(unittest.TestCase):
    def test_propose_rebinds_decorated_frozen_upstream(self):
        src = (SRC / "dflash_speculator.py").read_text()
        tree = ast.parse(src)
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
                   and x.name == "NPUDFlashSpeculator310V2")
        method = next(x for x in cls.body if isinstance(x, ast.FunctionDef)
                      and x.name == "propose")
        self.assertTrue(any(isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Attribute)
                            and n.func.attr == "unwrap" for n in ast.walk(method)))
        self.assertIn('"prepare_dflash_inputs": prepare_dflash_inputs', src)
        self.assertIn("build_attn_metadata_wrapper", src)
        self.assertNotIn("DFlashSpeculator.propose =", src)

    def test_runner_factory_is_candidate_only(self):
        src = (SRC / "model_runner.py").read_text()
        generic = (ROOT / "vllm_ascend/worker/v2/model_runner.py").read_text()
        worker = (ROOT / "vllm_ascend/_310p/worker_310p.py").read_text()
        self.assertIn("_310p_v2_speculator_factory", src)
        self.assertIn('getattr(', generic)
        self.assertIn('"_310p_v2_speculator_factory"', generic)
        self.assertNotIn("NPUDFlashSpeculator310V2", worker)
        self.assertIn("NOT_IMPLEMENTED", worker)

    def test_t11_no_triton_launch(self):
        src = (SRC / "dflash_input_ops.py").read_text()
        self.assertNotIn("@triton.jit", src)
        self.assertNotIn("triton.cdiv", src)
        self.assertNotIn("torch.cuda", src)
        # A documentation comment may mention .item(); reject executable
        # device-to-host scalar extraction, not explanatory prose.
        tree = ast.parse(src)
        self.assertFalse(any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "item" for n in ast.walk(tree)
        ))


class TestT11TorchCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None or np is None:
            raise unittest.SkipTest("Torch/NumPy absent; T11 stdlib ABI tests still run")
        cls.ops = isolated_module("dflash_input_ops.py", "_t11_cpu")
        cls.gold = isolated_module("reference_ops.py", "_t11_gold")

    def setup(self):
        t = torch
        nmax, max_tokens, max_model_len = 4, 64, 128
        positions = t.tensor([10, 11, 12, 40, 41], dtype=t.int64)
        batch = types.SimpleNamespace(
            num_reqs=2, positions=positions,
            query_start_loc=t.tensor([0, 3, 5], dtype=t.int32),
            query_start_loc_np=np.array([0, 3, 5], dtype=np.int32),
            idx_mapping=t.tensor([2, 0], dtype=t.int32),
            idx_mapping_np=np.array([2, 0], dtype=np.int32),
        )
        buffers = types.SimpleNamespace(
            input_ids=t.full((max_tokens,), -100, dtype=t.int32),
            positions=t.full((max_tokens,), -100, dtype=t.int64),
            query_start_loc=t.full((nmax+1,), -100, dtype=t.int32),
            seq_lens=t.full((nmax,), -100, dtype=t.int32),
        )
        return {
            "input_buffers": buffers,
            "query_slot_mapping": t.full((max_tokens,), -99, dtype=t.int32),
            "context_positions": t.full((max_tokens,), -99, dtype=t.int64),
            "context_slot_mapping": t.full((max_tokens,), -99, dtype=t.int32),
            "sample_indices": t.full((nmax*7,), -99, dtype=t.int64),
            "sample_pos": t.full((nmax*7,), -99, dtype=t.int64),
            "sample_idx_mapping": t.full((nmax*7,), -99, dtype=t.int32),
            "temperature": t.full((nmax,), -99, dtype=t.float32),
            "seeds": t.full((nmax,), -99, dtype=t.int64),
            "input_batch": batch,
            "num_sampled": t.tensor([3, 0], dtype=t.int32),
            "num_rejected": t.tensor([1, 0], dtype=t.int32),
            # Non-contiguous padded [req,1] view mirrors 310P RequestState.
            "last_sampled": t.tensor(
                [[10, 0, 0, 0], [20, 0, 0, 0],
                 [30, 0, 0, 0], [40, 0, 0, 0]], dtype=t.int64)[:, :1],
            "next_prefill_tokens": t.tensor([[55, 66, 77, 88]], dtype=t.int32),
            "input_temperature": t.tensor([0.1, 0.2, 0.3, 0.4]),
            "input_seeds": t.tensor([11, 22, 33, 44], dtype=t.int64),
            "block_size": 16,
            "block_table": t.tensor(
                [[100+i for i in range(8)], [200+i for i in range(8)]],
                dtype=t.int32),
            "parallel_drafting_token_id": 999,
            "num_query_per_req": 8, "num_speculative_steps": 7,
            "max_num_reqs": nmax, "max_num_tokens": max_tokens,
            "max_model_len": max_model_len,
        }

    def test_six_kv_groups_gold_and_padding(self):
        sizes = [16, 16, 16, 16, 16, 32]  # 5 sliding + 1 full
        tables = [
            [[(g+1)*1000 + row*100 + j for j in range(128//size)]
             for row in range(2)]
            for g, size in enumerate(sizes)
        ]
        ref = self.gold.build_k7_inputs(
            target_positions=[10, 11, 12, 40, 41],
            target_query_start_loc=[0, 3, 5],
            idx_mapping=[2, 0], last_sampled=[10, 20, 30, 40],
            next_prefill_tokens=[55, 66, 77, 88],
            num_sampled=[3, 0], num_rejected=[1, 0],
            group_block_tables=tables, group_kernel_block_sizes=sizes,
            parallel_drafting_token_id=999, max_model_len=128,
        )
        p = self.setup()
        for gid, size in enumerate(sizes):
            p["block_size"] = size
            p["block_table"] = torch.tensor(tables[gid], dtype=torch.int32)
            self.ops.prepare_dflash_inputs(**p)
            self.assertEqual(p["context_slot_mapping"][:5].tolist(),
                             ref.context_slots[gid])
            self.assertEqual(p["query_slot_mapping"][:16].tolist(),
                             ref.query_slots[gid])
            self.assertEqual(p["input_buffers"].input_ids[:16].tolist(),
                             ref.input_ids)
            self.assertEqual(p["sample_indices"][:14].tolist(),
                             ref.sample_indices)
            self.assertEqual(p["sample_pos"][:14].tolist(),
                             ref.sample_positions)
            self.assertEqual(p["sample_idx_mapping"].tolist(),
                             [2]*7 + [0]*7 + [-1]*14)
            self.assertEqual(p["query_slot_mapping"][16:].tolist(), [-1]*48)
            self.assertEqual(p["input_buffers"].query_start_loc.tolist(),
                             [0, 8, 16, 16, 16])
            self.assertEqual(p["input_buffers"].seq_lens.tolist(),
                             [20, 50, 0, 0])
            self.assertEqual(p["seeds"].tolist(), [11, -99, 33, -99])
            self.assertAlmostEqual(p["temperature"][2].item(), 0.3, places=6)

    def test_invalid_offsets_no_output_mutation(self):
        p = self.setup()
        p["input_batch"].query_start_loc_np = np.array(
            [0, 0, 5], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "scheduler"):
            self.ops.prepare_dflash_inputs(**p)
        self.assertEqual(p["input_buffers"].input_ids[0].item(), -100)

    def test_invalid_device_rejected_no_output_mutation(self):
        p = self.setup()
        p["num_rejected"][0] = 3
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            self.ops.prepare_dflash_inputs(**p)
        self.assertEqual(p["input_buffers"].input_ids[0].item(), -100)

    def test_insufficient_group_width_no_output_mutation(self):
        p = self.setup()
        p["block_table"] = torch.zeros((2, 1), dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "width"):
            self.ops.prepare_dflash_inputs(**p)
        self.assertEqual(p["input_buffers"].input_ids[0].item(), -100)

    def test_anchor_sampling_rejected(self):
        p = self.setup()
        p["sample_from_anchor"] = True
        with self.assertRaises(NotImplementedError):
            self.ops.prepare_dflash_inputs(**p)

    def test_invalid_query_position_fails_before_output_write(self):
        p = self.setup()
        p["input_batch"].positions[-1] = 127
        with self.assertRaisesRegex(RuntimeError, "query position"):
            self.ops.prepare_dflash_inputs(**p)
        self.assertEqual(p["input_buffers"].input_ids[0].item(), -100)


if __name__ == "__main__":
    unittest.main()

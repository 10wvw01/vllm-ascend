"""CPU Torch contracts for candidate 310P V2 T05/T06/T14/T15 adapters.

This tests semantic tensor behavior, NOT NPU op availability or runner wiring.
The GitHub lightweight gate has no torch: tests skip cleanly there, while the
pure-Python staged-write golden remains mandatory.
"""
import importlib.util
import pathlib
import sys
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "vllm_ascend/_310p/v2"


def load_isolated_modules():
    name = "_r2_cpu_isolated"
    package = types.ModuleType(name)
    package.__path__ = [str(SRC)]
    sys.modules[name] = package
    for part in ("staged_buffers", "request_state", "block_tables"):
        fullname = name + "." + part
        spec = importlib.util.spec_from_file_location(fullname, SRC / (part + ".py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fullname] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name + ".request_state"], sys.modules[name + ".block_tables"]


class TestCandidateAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import numpy
        except ImportError:
            raise unittest.SkipTest("torch/numpy absent; mandatory pure Python gold still runs")
        cls.torch = torch
        cls.request_module, cls.block_module = load_isolated_modules()

    def test_request_add_apply_release_reuse(self):
        t = self.torch
        s = self.request_module.NPURequestState(
            2, 16, 32, 7, 32000, "cpu"
        )
        s.add_request("a", 3, [11, 12, 13], 0, 10)
        idx = s.req_id_to_index["a"]
        self.assertEqual(s.num_reqs, 1)
        self.assertEqual(s.all_token_ids.gpu[idx, :3].tolist(), [0, 0, 0])
        s.apply_staged_writes()
        self.assertEqual(s.all_token_ids.gpu[idx, :3].tolist(), [11, 12, 13])
        self.assertEqual(s.prefill_len.gpu[idx].item(), 3)
        self.assertEqual(s.total_len.gpu[idx].item(), 3)
        self.assertEqual(s._last_sampled_storage.stride(0), 4)
        self.assertEqual(s._draft_storage.stride(0), 8)
        s._last_sampled_storage[idx].fill_(42)
        s._draft_storage[idx].fill_(99)
        s.next_prefill_tokens[:, idx].fill_(66)
        self.assertEqual(s.remove_request("a"), idx)
        s.add_request("b", 2, [21, 22], 1, 10)
        self.assertEqual(s.req_id_to_index["b"], idx)
        self.assertEqual(s.last_sampled_tokens[idx, 0].item(), 0)
        self.assertEqual(s.draft_tokens[idx].tolist(), [0] * 7)
        self.assertEqual(s.next_prefill_tokens[:, idx].tolist(), [0])
        s.apply_staged_writes()
        self.assertEqual(s.all_token_ids.gpu[idx, :2].tolist(), [21, 22])
        self.assertEqual(s.all_token_ids.gpu[idx, 2:].tolist(), [0] * 14)
        self.assertIsNone(s.remove_request("missing"))

    def test_request_invalid_is_transactional(self):
        s = self.request_module.NPURequestState(1, 8, 16, 7, 32000, "cpu")
        for args in [
            ("bad", 3, [1, 2], 0, 1),
            ("bad", 2, [1, 2], 3, 1),
            ("bad", 2, [1, 2], 0, 7),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                s.add_request(*args)
        self.assertEqual(s.num_reqs, 0)
        self.assertEqual(len(s.free_indices), 1)

    def test_pending_request_release_is_fail_closed(self):
        s = self.request_module.NPURequestState(1, 8, 16, 7, 32000, "cpu")
        s.add_request("a", 2, [1, 2], 0, 3)
        with self.assertRaisesRegex(RuntimeError, "pending token writes"):
            s.remove_request("a")
        self.assertEqual(s.num_reqs, 1)
        self.assertEqual(s.req_id_to_index["a"], 0)
        s.apply_staged_writes()
        self.assertEqual(s.remove_request("a"), 0)

    def test_multigroup_physical_block_and_padding(self):
        t = self.torch
        b = self.block_module.NPUBlockTables(
            block_sizes=[64, 128],
            kernel_block_sizes=[32, 128],
            max_num_reqs=3,
            max_num_batched_tokens=16,
            max_num_blocks_per_group=[4, 4],
            device="cpu",
        )
        b.append_block_ids(2, ([3, 4], [8]), overwrite=False)
        b.apply_staged_writes()
        self.assertEqual(b.block_tables[0].gpu[2, :4].tolist(), [6, 7, 8, 9])
        self.assertEqual(b.block_tables[1].gpu[2, 0].item(), 8)
        idx = t.tensor([2], dtype=t.int32)
        tables = b.gather_block_tables(idx, 2)
        self.assertEqual(tables[0][0, :4].tolist(), [6, 7, 8, 9])
        self.assertEqual(tables[0][1].count_nonzero().item(), 0)
        slots = b.compute_slot_mappings(
            idx,
            t.tensor([0, 3], dtype=t.int32),
            t.tensor([0, 31, 32, 0, 0], dtype=t.int32),
            num_tokens_padded=5,
        )
        self.assertEqual(slots.shape, (2, 5))
        self.assertEqual(slots[0].tolist(), [192, 223, 224, -1, -1])
        self.assertEqual(slots[1].tolist(), [1024, 1055, 1056, -1, -1])
        # Upstream DFlash reads the persistent slot_mappings attribute,
        # not the fresh return tensor. The Eager adapter must write both.
        self.assertEqual(b.slot_mappings[:, :5].tolist(), slots.tolist())
        self.assertEqual(slots.data_ptr(), b.slot_mappings.data_ptr())
        # 0.27.1 passes positions for real tokens, not padded tokens.
        short = b.compute_slot_mappings(
            idx, t.tensor([0, 3], dtype=t.int32),
            t.tensor([0, 31, 32], dtype=t.int32),
            num_tokens_padded=5,
        )
        self.assertEqual(short.tolist(), slots.tolist())
        self.assertEqual(short.data_ptr(), b.slot_mappings.data_ptr())
        empty = b.compute_slot_mappings(
            t.empty(0, dtype=t.int32),
            t.tensor([0], dtype=t.int32),
            t.empty(0, dtype=t.int32),
            num_tokens_padded=5,
        )
        self.assertEqual(empty.tolist(), [[-1] * 5, [-1] * 5])
        self.assertEqual(empty.data_ptr(), b.slot_mappings.data_ptr())
        b.append_block_ids(2, ([10], [11]), overwrite=True)
        b.apply_staged_writes()
        self.assertEqual(b.num_blocks.gpu[:, 2].tolist(), [2, 1])
        # Reused slot still has old tail in source; gather must mask it.
        reused = b.gather_block_tables(idx, 2)
        self.assertEqual(reused[0][0, :4].tolist(), [20, 21, 0, 0])
        self.assertEqual(reused[1][0, :2].tolist(), [11, 0])
        with self.assertRaises(NotImplementedError):
            b.gather_block_tables(idx, 2, out=(tables[0], tables[1]))

    def test_failed_kv_commit_blocks_future_gather_and_slot(self):
        from unittest.mock import patch

        t = self.torch
        b = self.block_module.NPUBlockTables(
            [64, 128], 2, 16, [2, 2], "cpu", [32, 128]
        )
        b.append_block_ids(0, ([1], [2]), False)
        with patch.object(
            b.block_tables[1], "apply_write",
            side_effect=RuntimeError("second group H2D failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "second group H2D failed"):
                b.apply_staged_writes()
        self.assertTrue(b._poisoned)
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            b.gather_block_tables(t.tensor([0], dtype=t.int32), 1)
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            b.compute_slot_mappings(
                t.tensor([0], dtype=t.int32),
                t.tensor([0, 1], dtype=t.int32),
                t.tensor([0], dtype=t.int32), 1,
            )
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            b.append_block_ids(0, ([3], [4]), False)

    def test_invalid_group_is_atomic_and_cp_rejected(self):
        b = self.block_module.NPUBlockTables(
            [64, 128], 2, 16, [2, 2], "cpu", [32, 128]
        )
        with self.assertRaises(ValueError):
            b.append_block_ids(0, ([1], [1, 2, 3]), False)
        self.assertEqual(b.num_blocks.np[:, 0].tolist(), [0, 0])
        self.assertTrue(all(x.plan.pending_count == 0 for x in b.block_tables))
        with self.assertRaises(NotImplementedError):
            self.block_module.NPUBlockTables(
                [64], 2, 16, [2], "cpu", [64], cp_size=2
            )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""310P3 GDN Whole-block Torch-NPU 候选的无设备 CPU gold。

轻量治理 CI 可跳过 torch；CPU-Torch CI 必须执行完整数据路径。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vllm_ascend/_310p/v2/hybrid_state_npu.py"


@unittest.skipIf(torch is None, "CPU Torch required")
class TestNPUHybridCandidateCPUContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest("CPU Torch not installed")
        spec = importlib.util.spec_from_file_location(
            "hybrid_state_npu_unregistered", SOURCE)
        assert spec is not None and spec.loader is not None
        cls.c = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.c
        spec.loader.exec_module(cls.c)

    def make(self, *, ds=False, groups=1, rows=1):
        c = self.c
        cache_groups = []
        for g in range(groups):
            # 每块物理数据>=32B、整块对齐；Conv与Temporal可不同dtype。
            conv = (torch.arange(12 * 10 * 8, dtype=torch.float32)
                    .reshape(12, 10, 8) + g * 1400).half()
            if ds:
                conv = conv.permute(0, 2, 1).contiguous()
            temporal = (torch.arange(12 * 4 * 8, dtype=torch.float32)
                        .reshape(12, 4, 8) + g * 100)
            bt = torch.tensor([[2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                               for _ in range(rows)], dtype=torch.int32)
            if g:
                bt[0] = torch.tensor(
                    [5, 3, 2, 0, 1, 4, 6, 7, 8, 9, 10, 11],
                    dtype=torch.int32)
            cache_groups.append(c.GDNDeviceGroup310(
                group_id=1 + 3 * g,
                block_table=bt,
                layers=(c.GDNDeviceLayer310(conv, temporal,
                                           "DS" if ds else "SD"),),
            ))
        return c.GDNWholeBlockCopy310Candidate(cache_groups), cache_groups

    def test_cross_block_precopy_real_tensor_multigroup(self):
        owner, groups = self.make(groups=2)
        original = [(g.layers[0].conv.clone(), g.layers[0].temporal.clone())
                    for g in groups]
        moved = owner.execute([self.c.GDNMove310(0, 0, 1, 3)])
        self.assertEqual(moved, 4)
        self.assertFalse(owner.device_pending)
        for g, (conv_old, temporal_old) in zip(groups, original):
            src, dest = int(g.block_table[0, 0]), int(g.block_table[0, 1])
            t_src = int(g.block_table[0, 3])
            layer = g.layers[0]
            torch.testing.assert_close(
                layer.conv[dest, :7], conv_old[src, 3:], rtol=0, atol=0)
            torch.testing.assert_close(
                layer.conv[dest, 7:], conv_old[dest, 7:], rtol=0, atol=0)
            torch.testing.assert_close(
                layer.temporal[dest], temporal_old[t_src], rtol=0, atol=0)

    def test_same_physical_block_overlap_sd_uses_snapshot(self):
        owner, groups = self.make()
        t = groups[0].layers[0].conv
        old = t.clone()
        physical = int(groups[0].block_table[0, 0])
        self.assertEqual(owner.execute([self.c.GDNMove310(0, 0, 0, 1)]), 2)
        torch.testing.assert_close(t[physical, :9], old[physical, 1:],
                                   rtol=0, atol=0)
        torch.testing.assert_close(t[physical, 9:], old[physical, 9:],
                                   rtol=0, atol=0)

    def test_same_physical_block_overlap_ds_uses_snapshot(self):
        owner, groups = self.make(ds=True)
        t = groups[0].layers[0].conv
        old = t.clone()
        physical = int(groups[0].block_table[0, 0])
        owner.execute([self.c.GDNMove310(0, 0, 0, 2)])
        torch.testing.assert_close(t[physical, :, :8], old[physical, :, 2:],
                                   rtol=0, atol=0)
        torch.testing.assert_close(t[physical, :, 8:], old[physical, :, 8:],
                                   rtol=0, atol=0)

    def test_noop_does_not_write_when_same_block_zero_bias(self):
        owner, groups = self.make()
        conv, temp = groups[0].layers[0].conv, groups[0].layers[0].temporal
        a, b = conv.clone(), temp.clone()
        self.assertEqual(owner.execute([self.c.GDNMove310(0, 0, 0, 0)]), 0)
        torch.testing.assert_close(conv, a)
        torch.testing.assert_close(temp, b)

    def test_two_batch_rows_independent_physical_destinations(self):
        owner, gs = self.make(rows=2)
        gs[0].block_table[1] = torch.tensor(
            [8, 9, 10, 11, 4, 5, 6, 7, 2, 3, 0, 1],
            dtype=torch.int32)
        before = gs[0].layers[0].conv.clone()
        owner.execute([self.c.GDNMove310(0, 0, 1, 2),
                       self.c.GDNMove310(1, 0, 1, 3)])
        block = gs[0].layers[0].conv
        torch.testing.assert_close(block[0, :8], before[2, 2:],
                                   rtol=0, atol=0)
        torch.testing.assert_close(block[9, :7], before[8, 3:],
                                   rtol=0, atol=0)

    def test_duplicate_physical_destination_reject_before_copy(self):
        owner, gs = self.make(rows=2)
        before = gs[0].layers[0].conv.clone()
        with self.assertRaisesRegex(ValueError, "同一物理 Conv"):
            owner.execute([self.c.GDNMove310(0, 0, 1, 1),
                           self.c.GDNMove310(1, 0, 1, 2)])
        self.assertTrue(owner.poisoned)
        torch.testing.assert_close(gs[0].layers[0].conv, before)

    def test_second_group_bad_block_preflight_preserves_first(self):
        owner, gs = self.make(groups=2)
        before = gs[0].layers[0].conv.clone()
        gs[1].block_table[0, 1] = -1
        with self.assertRaises(ValueError):
            owner.execute([self.c.GDNMove310(0, 0, 1, 3)])
        self.assertTrue(owner.poisoned)
        torch.testing.assert_close(gs[0].layers[0].conv, before)

    def test_partial_commit_failure_poison_and_no_retry(self):
        owner, gs = self.make(groups=2)
        owner._test_fail_after_write = 1
        with self.assertRaisesRegex(RuntimeError, "模拟 NPU"):
            owner.execute([self.c.GDNMove310(0, 0, 1, 3)])
        self.assertTrue(owner.poisoned)
        with self.assertRaises(RuntimeError):
            owner.execute([])

    def test_cpu_alignment_and_dtype_guard(self):
        c = self.c
        base = torch.zeros((3, 10, 3), dtype=torch.float16)
        # 10x3x2 = 60B，禁止误认为物理整行满足32B。
        temporal = torch.zeros((3, 4, 8), dtype=torch.float32)
        group = c.GDNDeviceGroup310(0, torch.zeros((1, 3), dtype=torch.int32),
                                    (c.GDNDeviceLayer310(base, temporal),))
        with self.assertRaises(NotImplementedError):
            c.GDNWholeBlockCopy310Candidate([group])

    def test_cpu_missing_torch_npu_is_explicit_no_go(self):
        if hasattr(torch, "npu"):
            self.skipTest("torch_npu is installed in this test environment")
        owner, groups = self.make()
        with self.assertRaises(NotImplementedError):
            self.c.GDNWholeBlockCopy310Candidate(groups, device="npu")
        with self.assertRaises(NotImplementedError):
            self.c.PaddedInt32State310Candidate(8, device="npu")

    def test_padded_int32_same_32b_row_merges_and_preserves_neighbours(self):
        state = self.c.PaddedInt32State310Candidate(13)
        state.scatter_authoritative_counts([1, 2, 7, 8, 12], [4, 7, 0, 8, 2])
        expected = torch.tensor([
            [1, 4, 7, 1, 1, 1, 1, 1],
            [8, 1, 1, 1, 2, 1, 1, 1]], dtype=torch.int32)
        torch.testing.assert_close(state.data, expected)
        state.scatter_authoritative_counts([2, 3], [5, 2])
        self.assertEqual(int(state.data[0, 2]), 5)
        self.assertEqual(int(state.data[0, 3]), 2)
        self.assertEqual(int(state.data[0, 1]), 4)

    def test_padded_int32_invalid_batch_prevents_partial_update(self):
        state = self.c.PaddedInt32State310Candidate(10)
        old = state.data.clone()
        with self.assertRaises(ValueError):
            state.scatter_authoritative_counts([1, 3], [4, 9])
        torch.testing.assert_close(state.data, old)
        with self.assertRaises(ValueError):
            state.scatter_authoritative_counts([2, 2], [1, 3])
        torch.testing.assert_close(state.data, old)

    def test_cpu_confirm_completion_is_safe_and_no_pending_state(self):
        owner, _ = self.make()
        self.assertFalse(owner.device_pending)
        # Temporal BT[src_col+bias] == BT[dst_col], so only Conv writes.
        self.assertEqual(owner.execute([self.c.GDNMove310(0, 0, 1, 1)]), 1)
        self.assertFalse(owner.device_pending)
        owner.confirm_completed()
        self.assertFalse(owner.device_pending)
        state = self.c.PaddedInt32State310Candidate(9)
        state.scatter_authoritative_counts([0, 8], [8, 2])
        self.assertFalse(state.device_pending)
        state.confirm_completed()
        self.assertFalse(state.device_pending)

    def test_unsupported_device_and_no_runtime_global_patch(self):
        src = SOURCE.read_text(encoding="utf-8")
        runner = (ROOT / "vllm_ascend/_310p/v2/model_runner.py").read_text()
        worker = (ROOT / "vllm_ascend/_310p/worker_310p.py").read_text()
        self.assertIn("NOT_IMPLEMENTED", worker)
        self.assertIn('self.cache_config.mamba_cache_mode == "align"', runner)
        for term in ("import triton", "from triton", "torch.cuda",
                     "from vllm.", "import torch_npu", "raw_pointer"):
            self.assertNotIn(term, src)
        self.assertIn("index_select", src)
        self.assertIn("index_copy_", src)
        self.assertIn("device_pending", src)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent CPU Tensor gold / fault cases for unregistered R2-P0-E.

This file is skipped on the lightweight (no-Torch) project-control job,
and executed in the pinned CPU Torch job. Never touches an NPU.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vllm_ascend/_310p/v2/hybrid_state.py"


@unittest.skipIf(torch is None, "CPU Torch required for GDN Tensor gold")
class TestTargetHybridTensorState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest("CPU Torch not installed")
        spec = importlib.util.spec_from_file_location(
            "target_hybrid_state_310_unregistered", SOURCE
        )
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    def create(self, groups=1, rows=1, *, ds=False):
        c = self.module
        gs = []
        for g in range(groups):
            conv = torch.arange(10 * 10 * 3, dtype=torch.float32).reshape(10, 10, 3)
            conv = (conv + g * 400).to(torch.float16)
            if ds:
                conv = conv.permute(0, 2, 1).contiguous()
            temporal = (torch.arange(10 * 2 * 3, dtype=torch.float32)
                        .reshape(10, 2, 3) + g * 1000)
            physical = ([2, 0, 1, 3, 4, 5, 6, 7, 8, 9]
                        if g == 0 else [5, 1, 3, 2, 4, 0, 6, 7, 8, 9])
            bt = torch.tensor([physical[:] for _ in range(rows)],
                              dtype=torch.int32)
            group = c.GDNStateGroup310(
                group_id=2 + g * 3,
                block_table=bt,
                layers=(c.GDNStateLayer310(
                    conv=conv, temporal=temporal,
                    conv_layout="DS" if ds else "SD"),),
            )
            gs.append(group)
        owner = c.HybridAlignTensor310Candidate(
            block_size=16, max_reqs=10, tp=2, k=7)
        owner.bind_groups(gs)
        return owner, gs

    def test_pre_boundary_preserves_accepted_bias_and_copies_both_states(self):
        owner, gs = self.create(groups=2)
        owner.add_request(3, 10)
        owner.postprocess_state([3], [4], [12])   # running 9: no postcopy
        self.assertEqual(owner.snapshot(3), (0, 4))
        before = [(g.layers[0].conv.clone(), g.layers[0].temporal.clone())
                  for g in gs]
        owner.preprocess_state([3], [15], [2])
        self.assertEqual(owner.snapshot(3), (1, 1))
        for g, (conv_before, temporal_before) in zip(gs, before):
            with self.subTest(group=g.group_id):
                source = int(g.block_table[0, 0])
                dest = int(g.block_table[0, 1])
                temporal_source = int(g.block_table[0, 3])
                torch.testing.assert_close(
                    g.layers[0].conv[dest, :7, :],
                    conv_before[source, 3:, :], rtol=0, atol=0)
                torch.testing.assert_close(
                    g.layers[0].temporal[dest], temporal_before[temporal_source],
                    rtol=0, atol=0)

    def test_pre_same_block_no_copy_with_state_advancement(self):
        owner, gs = self.create()
        owner.add_request(0, 8)
        before = gs[0].layers[0].conv.clone()
        owner.preprocess_state([0], [8], [3])
        self.assertEqual(owner.snapshot(0), (0, 1))
        torch.testing.assert_close(gs[0].layers[0].conv, before)

    def test_post_same_block_shift_requires_memmove_semantics_sd(self):
        owner, gs = self.create()
        owner.add_request(0, 10)
        layer = gs[0].layers[0]
        conv_old, temporal_old = layer.conv.clone(), layer.temporal.clone()
        owner.postprocess_state([0], [4], [18])  # running15, aligned16, bias1
        dest = int(gs[0].block_table[0, 0])
        t_src = int(gs[0].block_table[0, 1])
        torch.testing.assert_close(
            layer.conv[dest, :9, :], conv_old[dest, 1:, :],
            rtol=0, atol=0)
        torch.testing.assert_close(layer.conv[dest, 9:, :],
                                   conv_old[dest, 9:, :], rtol=0, atol=0)
        torch.testing.assert_close(layer.temporal[dest], temporal_old[t_src],
                                   rtol=0, atol=0)
        self.assertEqual(owner.snapshot(0), (0, 1))

    def test_post_same_block_shift_ds_noncontiguous_axis(self):
        owner, gs = self.create(ds=True)
        owner.add_request(0, 10)
        layer = gs[0].layers[0]
        conv_old = layer.conv.clone()
        owner.postprocess_state([0], [4], [18])
        block = int(gs[0].block_table[0, 0])
        torch.testing.assert_close(
            layer.conv[block, :, :9], conv_old[block, :, 1:],
            rtol=0, atol=0)
        torch.testing.assert_close(
            layer.conv[block, :, 9:], conv_old[block, :, 9:],
            rtol=0, atol=0)

    def test_post_same_block_noop_zero_bias_resets_accepted(self):
        owner, gs = self.create()
        owner.add_request(0, 10)
        before = gs[0].layers[0].conv.clone()
        owner.postprocess_state([0], [1], [16])
        self.assertEqual(owner.snapshot(0), (0, 1))
        torch.testing.assert_close(gs[0].layers[0].conv, before)

    def test_post_cross_block_state_copy_keeps_accepted(self):
        owner, gs = self.create()
        owner.add_request(0, 10)
        layer = gs[0].layers[0]
        old = layer.conv.clone()
        owner.postprocess_state([0], [1], [32])
        a, b = int(gs[0].block_table[0, 0]), int(gs[0].block_table[0, 1])
        torch.testing.assert_close(layer.conv[b], old[a], rtol=0, atol=0)
        self.assertEqual(owner.snapshot(0), (0, 1))

    def test_chunked_prefill_zero_is_normalized(self):
        owner, _ = self.create()
        owner.add_request(0, 10)
        owner.postprocess_state([0], [0], [11])
        self.assertEqual(owner.snapshot(0), (0, 1))

    def test_scatter_is_by_persistent_slot_not_batch_row(self):
        owner, _ = self.create(rows=2)
        owner.add_request(7, 10)
        owner.add_request(2, 10)
        owner.postprocess_state([7, 2], [3, 2], [13, 12])
        self.assertEqual(owner.snapshot(7), (0, 3))
        self.assertEqual(owner.snapshot(2), (0, 2))

    def test_two_active_requests_use_separate_batch_rows_not_req_slots(self):
        owner, gs = self.create(rows=2)
        gs[0].block_table[1] = torch.tensor(
            [6, 7, 8, 9, 4, 5, 2, 3, 0, 1], dtype=torch.int32
        )
        owner.add_request(7, 10)
        owner.add_request(2, 10)
        owner.postprocess_state([7, 2], [4, 3], [12, 13])
        original = gs[0].layers[0].conv.clone()
        temporal = gs[0].layers[0].temporal.clone()
        # req_slot=2 is row0, req_slot=7 is row1, not row[req_slot].
        owner.preprocess_state([2, 7], [15, 15], [2, 2])
        self.assertEqual(owner.snapshot(2), (1, 1))
        self.assertEqual(owner.snapshot(7), (1, 1))
        # Request 2: old accepted=3 -> shift 2; row0 blocks 2->0.
        torch.testing.assert_close(gs[0].layers[0].conv[0, :8],
                                   original[2, 2:], rtol=0, atol=0)
        # Request 7: old accepted=4 -> shift 3; row1 blocks 6->7.
        torch.testing.assert_close(gs[0].layers[0].conv[7, :7],
                                   original[6, 3:], rtol=0, atol=0)
        torch.testing.assert_close(gs[0].layers[0].temporal[0],
                                   temporal[1], rtol=0, atol=0)
        torch.testing.assert_close(gs[0].layers[0].temporal[7],
                                   temporal[9], rtol=0, atol=0)

    def test_second_kv_group_preflight_failure_does_not_touch_first(self):
        owner, gs = self.create(groups=2)
        owner.add_request(0, 10)
        owner.postprocess_state([0], [4], [12])
        conv0 = gs[0].layers[0].conv.clone()
        temp0 = gs[0].layers[0].temporal.clone()
        conv1 = gs[1].layers[0].conv.clone()
        # Second group has invalid destination col1; first group must
        # remain intact even though its valid copies were planned first.
        gs[1].block_table[0, 1] = -1
        with self.assertRaises(ValueError):
            owner.preprocess_state([0], [15], [2])
        self.assertTrue(owner.poisoned)
        torch.testing.assert_close(gs[0].layers[0].conv, conv0)
        torch.testing.assert_close(gs[0].layers[0].temporal, temp0)
        torch.testing.assert_close(gs[1].layers[0].conv, conv1)

    def test_remove_then_reuse_slot_starts_neutral(self):
        owner, _ = self.create()
        owner.add_request(0, 10)
        owner.postprocess_state([0], [3], [13])
        owner.remove_request(0)
        owner.add_request(0, 0)
        self.assertEqual(owner.snapshot(0), (-1, 1))

    def test_invalid_logical_or_physical_block_no_partial_write(self):
        owner, gs = self.create()
        owner.add_request(0, 10)
        layer = gs[0].layers[0]
        before = layer.conv.clone()
        gs[0].block_table[0, 1] = -1
        with self.assertRaises(ValueError):
            owner.preprocess_state([0], [15], [2])
        self.assertTrue(owner.poisoned)
        torch.testing.assert_close(layer.conv, before)
        with self.assertRaises(RuntimeError):
            owner.add_request(1, 0)

    def test_multi_group_mid_commit_fault_poison_and_no_retry(self):
        owner, gs = self.create(groups=2)
        owner.add_request(0, 10)
        owner.postprocess_state([0], [4], [12])
        owner._test_fail_after_copy = 1
        with self.assertRaisesRegex(RuntimeError, "injected"):
            owner.preprocess_state([0], [15], [2])
        self.assertTrue(owner.poisoned)
        with self.assertRaises(RuntimeError):
            owner.postprocess_state([0], [1], [17])
        with self.assertRaises(RuntimeError):
            owner.remove_request(0)

    def test_duplicate_destination_blocks_fail_before_copy(self):
        owner, gs = self.create(rows=2)
        owner.add_request(0, 10)
        owner.add_request(1, 10)
        old = gs[0].layers[0].conv.clone()
        with self.assertRaisesRegex(ValueError, "aliased GDN destination"):
            owner.preprocess_state([0, 1], [15, 15], [2, 2])
        self.assertTrue(owner.poisoned)
        torch.testing.assert_close(gs[0].layers[0].conv, old)

    def test_unsupported_or_aliased_storage_fails_closed(self):
        c = self.module
        owner, gs = self.create()
        with self.assertRaises(RuntimeError):
            owner.bind_groups(gs)
        with self.assertRaises(NotImplementedError):
            c.HybridAlignTensor310Candidate(block_size=16, max_reqs=4, tp=1)
        assert torch is not None
        bogus = c.GDNStateGroup310(1, torch.empty((1, 4), device="meta",
                                                   dtype=torch.int32),
                                     gs[0].layers)
        new = c.HybridAlignTensor310Candidate(block_size=16, max_reqs=2)
        with self.assertRaises(NotImplementedError):
            new.bind_groups([bogus])

    def test_unregistered_worker_guards_unchanged_and_no_cuda_triton_imports(self):
        worker = (ROOT / "vllm_ascend/_310p/worker_310p.py").read_text()
        runner = (ROOT / "vllm_ascend/_310p/v2/model_runner.py").read_text()
        src = SOURCE.read_text()
        self.assertIn("NOT_IMPLEMENTED", worker)
        self.assertIn('self.cache_config.mamba_cache_mode == "align"', runner)
        for import_line in ("import triton", "from triton", "import torch_npu",
                            "torch.cuda.", "MambaSpecDecodeGPUContext"):
            self.assertNotIn(import_line, src)


if __name__ == "__main__":
    unittest.main()

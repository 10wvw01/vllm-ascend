#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU tensor contracts for unregistered T30-A/T31-A device-owner candidate."""
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
SOURCE = ROOT / "vllm_ascend/_310p/v2/hybrid_state_device_meta.py"


@unittest.skipIf(torch is None, "CPU Torch required")
class TestGDNDeviceMetadata310(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest("Torch CPU missing")
        spec = importlib.util.spec_from_file_location(
            "unregistered_gdn_device_meta_310", SOURCE)
        assert spec is not None and spec.loader is not None
        cls.mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.mod
        spec.loader.exec_module(cls.mod)

    def fresh(self, cap=13, b=16):
        return self.mod.GDNDeviceMetadata310Candidate(
            max_reqs=cap, block_size=b, device="cpu", k=7, tp=2)

    def test_new_request_seed_and_tail_padded_storage(self):
        o = self.fresh()
        self.assertEqual(tuple(o._storage.shape), (2, 8, 4))
        o.seed_requests([0, 9], [0, 17])
        self.assertEqual(o.snapshot_for_cpu_test(0), (-1, 1, -1, 0))
        self.assertEqual(o.snapshot_for_cpu_test(9), (1, 1, -1, 0))
        self.assertEqual(o._storage.reshape(-1, 4)[8].tolist(), [-1, 1, -1, 0])

    def test_pre_boundary_uses_device_computed_and_previous_accepted(self):
        o = self.fresh()
        o.seed_requests([3], [10])
        idx = torch.tensor([3], dtype=torch.int32)
        c = torch.zeros(13, dtype=torch.int32)
        c[3] = 15
        # CPU test may emulate a previous sampled count with one step first.
        o.preprocess_device([3], idx, c, torch.tensor([0, 2], dtype=torch.int32))
        d = o.snapshot_for_cpu_test(3)
        self.assertEqual(d, (1, 1, 0, 0))
        self.assertEqual(o.phase, "PRECOPY_DECISION_PENDING")
        with self.assertRaises(RuntimeError):
            o.prepare_attn_payload([3], [7], [8])
        o.confirm_precopy_for_cpu_gold()
        meta = o.prepare_attn_payload([3], [7], [8])
        self.assertEqual(meta.num_accepted_tokens.tolist(), [1])
        self.assertEqual(meta.num_decode_draft_tokens_cpu.tolist(), [7])

    def test_pre_keeps_accepted_until_block_boundary(self):
        o = self.fresh()
        o.seed_requests([1], [10])
        idx = torch.tensor([1], dtype=torch.int32)
        c = torch.zeros(13, dtype=torch.int32)
        c[1] = 10
        d = o.preprocess_device([1], idx, c,
                                torch.tensor([0, 3], dtype=torch.int32))
        self.assertEqual(d.needs_copy.tolist(), [False])
        self.assertEqual(o.snapshot_for_cpu_test(1)[:2], (0, 1))

    def test_post_k7_accept_and_same_block_shift_and_reset(self):
        o = self.fresh()
        o.seed_requests([2], [10])
        idx = torch.tensor([2], dtype=torch.int32)
        c = torch.zeros(13, dtype=torch.int32)
        c[2] = 10
        o.preprocess_device([2], idx, c,
                            torch.tensor([0, 1], dtype=torch.int32))
        o.confirm_precopy_for_cpu_gold()
        c[2] = 18
        d = o.postprocess_device(
            [2], idx, torch.tensor([4], dtype=torch.int32), c)
        self.assertEqual((d.source_column.tolist(), d.destination_column.tolist(),
                          d.accepted_bias.tolist(), d.needs_copy.tolist()),
                         ([0], [0], [1], [True]))
        self.assertEqual(o.snapshot_for_cpu_test(2)[1], 1)
        with self.assertRaises(RuntimeError):
            o.preprocess_device([2], idx, c,
                                torch.tensor([0, 1], dtype=torch.int32))
        o.confirm_postcopy_for_cpu_gold()
        self.assertEqual(o.phase, "IDLE")

    def test_post_unaligned_no_copy_keeps_accepted(self):
        o = self.fresh()
        o.seed_requests([7], [10])
        idx = torch.tensor([7], dtype=torch.int32)
        c = torch.zeros(13, dtype=torch.int32)
        c[7] = 10
        o.preprocess_device([7], idx, c,
                            torch.tensor([0, 1], dtype=torch.int32))
        o.confirm_precopy_for_cpu_gold()
        c[7] = 20
        d = o.postprocess_device(
            [7], idx, torch.tensor([2], dtype=torch.int32), c)
        self.assertEqual(d.needs_copy.tolist(), [False])
        self.assertEqual(o.snapshot_for_cpu_test(7)[1], 2)

    def test_mixed_gdn_decode_sentinel_and_k7(self):
        o = self.fresh()
        o.seed_requests([1, 9, 12], [0, 10, 10])
        idx = torch.tensor([1, 9, 12], dtype=torch.int32)
        c = torch.zeros(13, dtype=torch.int32)
        c[1] = 0
        c[9] = c[12] = 10
        o.preprocess_device([1, 9, 12], idx, c,
                            torch.tensor([0, 8, 16, 19], dtype=torch.int32))
        o.confirm_precopy_for_cpu_gold()
        payload = o.prepare_attn_payload(
            [1, 9, 12], [0, 7, 2], [8, 8, 3])
        self.assertEqual(payload.num_decode_draft_tokens_cpu.tolist(),
                         [-1, 7, 2])
        self.assertEqual(payload.num_accepted_tokens.tolist(), [1, 1, 1])

    def test_two_slots_in_same_128b_row_preserve_neighbours(self):
        o = self.fresh()
        o.seed_requests([1, 3, 11], [5, 10, 17])
        old = o._storage.clone()
        idx = torch.tensor([3, 11], dtype=torch.int32)
        computed = torch.zeros(13, dtype=torch.int32)
        computed[3], computed[11] = 15, 17
        o.preprocess_device(
            [3, 11], idx, computed, torch.tensor([0, 2, 4], dtype=torch.int32))
        self.assertEqual(o.snapshot_for_cpu_test(1), (0, 1, -1, 0))
        torch.testing.assert_close(o._storage[0, 1], old[0, 1])
        self.assertEqual(o.snapshot_for_cpu_test(3)[:2], (1, 1))
        self.assertEqual(o.snapshot_for_cpu_test(11)[:2], (1, 1))

    def test_device_idx_mapping_mismatch_fails_without_write(self):
        o = self.fresh()
        o.seed_requests([3], [10])
        old = o._storage.clone()
        c = torch.zeros(13, dtype=torch.int32)
        c[3] = 12
        with self.assertRaises(ValueError):
            o.preprocess_device(
                [3], torch.tensor([4], dtype=torch.int32), c,
                torch.tensor([0, 1], dtype=torch.int32))
        torch.testing.assert_close(o._storage, old)
        self.assertEqual(o.phase, "IDLE")

    def test_bad_device_sampled_count_fails_without_write(self):
        o = self.fresh()
        o.seed_requests([3], [10])
        idx = torch.tensor([3], dtype=torch.int32)
        c = torch.zeros(13, dtype=torch.int32)
        c[3] = 10
        o.preprocess_device([3], idx, c,
                            torch.tensor([0, 1], dtype=torch.int32))
        o.confirm_precopy_for_cpu_gold()
        old = o._storage.clone()
        c[3] = 18
        with self.assertRaises(ValueError):
            o.postprocess_device(
                [3], idx, torch.tensor([9], dtype=torch.int32), c)
        torch.testing.assert_close(o._storage, old)

    def test_release_reuse_resets_all_metadata_and_no_midbatch_release(self):
        o = self.fresh()
        o.seed_requests([4], [17])
        idx = torch.tensor([4], dtype=torch.int32)
        c = torch.zeros(13, dtype=torch.int32)
        c[4] = 17
        o.preprocess_device([4], idx, c,
                            torch.tensor([0, 1], dtype=torch.int32))
        with self.assertRaises(RuntimeError):
            o.release_requests([4])
        o.confirm_precopy_for_cpu_gold()
        o.postprocess_device([4], idx, torch.tensor([1], dtype=torch.int32), c)
        o.confirm_postcopy_for_cpu_gold()
        o.release_requests([4])
        o.seed_requests([4], [0])
        self.assertEqual(o.snapshot_for_cpu_test(4), (-1, 1, -1, 0))

    def test_reject_wrong_tp_k_npu_and_unsupported_tensor_abi(self):
        with self.assertRaises(ValueError):
            self.mod.GDNDeviceMetadata310Candidate(
                max_reqs=8, block_size=16, tp=1)
        if not hasattr(torch, "npu"):
            with self.assertRaises(NotImplementedError):
                self.mod.GDNDeviceMetadata310Candidate(
                    max_reqs=8, block_size=16, device="npu")
        o = self.fresh()
        o.seed_requests([0], [0])
        with self.assertRaises(ValueError):
            o.preprocess_device(
                [0], torch.tensor([0], dtype=torch.int64),
                torch.zeros(13, dtype=torch.int32),
                torch.tensor([0, 1], dtype=torch.int32))

    def test_no_host_scalar_sync_cuda_or_modelstate_registration(self):
        source = SOURCE.read_text()
        for term in ("import triton", "torch.cuda", ".item()", "from vllm.",
                     "import torch_npu", "GPUModelRunner", "MambaSpecDecodeGPUContext"):
            self.assertNotIn(term, source)
        worker = (ROOT / "vllm_ascend/_310p/worker_310p.py").read_text()
        runner = (ROOT / "vllm_ascend/_310p/v2/model_runner.py").read_text()
        self.assertIn("NOT_IMPLEMENTED", worker)
        self.assertIn('self.cache_config.mamba_cache_mode == "align"', runner)
        self.assertIn("confirm_precopy_for_cpu_gold", source)
        self.assertIn("confirm_postcopy_for_cpu_gold", source)

if __name__ == "__main__":
    unittest.main()

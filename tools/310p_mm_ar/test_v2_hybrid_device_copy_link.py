#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Target Qwen3.5-MoE GDN device decision -> BT -> state-copy CPU Gold.

Imports isolated, unregistered files. No model/draft/torch_npu/NPU runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "vllm_ascend/_310p/v2"


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_DIR / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass(frozen=True)
class Request:
    num_computed_tokens: int


@dataclass(frozen=True)
class Batch:
    idx_mapping_np: tuple[int, ...]
    idx_mapping: object
    query_start_loc: object
    num_draft_tokens_per_req: tuple[int, ...]
    num_scheduled_tokens: tuple[int, ...]
    @property
    def num_reqs(self):
        return len(self.idx_mapping_np)


@unittest.skipIf(torch is None, "Torch CPU required")
class TestGDNDeviceNativeConsumer310(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch is None:
            raise unittest.SkipTest("Torch CPU missing")
        cls.meta_mod = load("r2_p0e_gdn_device_meta_gold",
                            "hybrid_state_device_meta.py")
        cls.copy_mod = load("r2_p0e_gdn_device_copy_gold",
                            "hybrid_state_device_copy.py")
        cls.link_mod = load("r2_p0e_gdn_modelstate_link_gold",
                            "hybrid_state_modelstate_link.py")

    def make(self, groups=2, rows=1, ds=False):
        c = self.copy_mod
        all_groups = []
        for g in range(groups):
            conv = (torch.arange(12 * 10 * 8, dtype=torch.float32)
                    .reshape(12, 10, 8) + g * 1000).to(torch.float16)
            if ds:
                conv = conv.permute(0, 2, 1).contiguous()
            temporal = (torch.arange(12 * 4 * 8, dtype=torch.float32)
                        .reshape(12, 4, 8) + g * 2000)
            seq = ([2, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                   if g == 0 else [5, 1, 3, 2, 4, 0, 6, 7, 8, 9, 10, 11])
            table = torch.tensor([seq[:] for _ in range(rows)], dtype=torch.int32)
            all_groups.append(c.GDNIndexedGroup310(
                g, table, (c.GDNIndexedLayer310(
                    conv, temporal, "DS" if ds else "SD"),)
            ))
        copier = c.DeviceIndexedGDNCopy310Candidate(all_groups)
        meta = self.meta_mod.GDNDeviceMetadata310Candidate(
            max_reqs=16, block_size=16)
        link = self.link_mod.TargetGDNModelStateSequence310Candidate(meta, copier)
        return link, meta, copier, all_groups

    @staticmethod
    def batch(slots, qlens, drafts=None):
        drafts = drafts if drafts is not None else [0] * len(slots)
        query = [0]
        for q in qlens:
            query.append(query[-1] + q)
        return Batch(tuple(slots), torch.tensor(slots, dtype=torch.int32),
                     torch.tensor(query, dtype=torch.int32),
                     tuple(drafts), tuple(qlens))

    @staticmethod
    def computed(slots_vals):
        t = torch.zeros(16, dtype=torch.int32)
        for slot, val in slots_vals:
            t[slot] = val
        return t

    def test_full_pre_post_chain_old_accepted_cross_group_and_postalign(self):
        link, meta, copier, groups = self.make(groups=2)
        link.add_request(3, Request(10))
        tables = tuple(g.block_table for g in groups)
        b0 = self.batch([3], [1])
        comp = self.computed([(3, 10)])
        link.preprocess_state(b0, tables, None, comp)
        self.assertEqual(meta.phase, "READY_FOR_ATTENTION")
        attn = link.prepare_attn(b0, "NONE", tables, None, None, None)
        self.assertEqual(attn.num_decode_draft_tokens_cpu.tolist(), [-1])
        link.forward_completed_for_cpu_gold()
        comp[3] = 12
        link.postprocess_state(
            b0.idx_mapping, torch.tensor([4], dtype=torch.int32), comp)
        self.assertEqual(meta.snapshot_for_cpu_test(3)[:2], (0, 4))
        old = [(g.layers[0].conv.clone(), g.layers[0].temporal.clone())
               for g in groups]
        b1 = self.batch([3], [2], [1])
        comp[3] = 15
        link.preprocess_state(b1, tables, None, comp)
        self.assertEqual(meta.snapshot_for_cpu_test(3)[:2], (1, 1))
        for g, (old_conv, old_temp) in zip(groups, old):
            layer = g.layers[0]
            src, dst = int(g.block_table[0, 0]), int(g.block_table[0, 1])
            temporal_src = int(g.block_table[0, 3])
            torch.testing.assert_close(
                layer.conv[dst, :7], old_conv[src, 3:], atol=0, rtol=0)
            torch.testing.assert_close(
                layer.conv[dst, 7:], old_conv[dst, 7:], atol=0, rtol=0)
            torch.testing.assert_close(
                layer.temporal[dst], old_temp[temporal_src], atol=0, rtol=0)
        payload = link.prepare_attn(b1, "EAGER", tables, None, None, None)
        # d=1, scheduled=2 means real speculative decode row.
        self.assertEqual(payload.num_decode_draft_tokens_cpu.tolist(), [1])
        link.forward_completed_for_cpu_gold()
        prev = [(g.layers[0].conv.clone(), g.layers[0].temporal.clone())
                for g in groups]
        comp[3] = 18
        link.postprocess_state(
            b1.idx_mapping, torch.tensor([4], dtype=torch.int32), comp)
        self.assertEqual(meta.phase, "IDLE")
        self.assertEqual(meta.snapshot_for_cpu_test(3)[1], 1)
        for g, (old_conv, old_temp) in zip(groups, prev):
            layer = g.layers[0]
            src, dst = int(g.block_table[0, 1]), int(g.block_table[0, 0])
            temporal_src = int(g.block_table[0, 2])
            torch.testing.assert_close(layer.conv[dst, :9],
                                       old_conv[src, 1:], atol=0, rtol=0)
            torch.testing.assert_close(layer.temporal[dst],
                                       old_temp[temporal_src], atol=0, rtol=0)
        link.remove_request(3)
        link.add_request(3, Request(0))
        self.assertEqual(meta.snapshot_for_cpu_test(3), (-1, 1, -1, 0))

    def test_multirow_request_order_and_layout_ds(self):
        link, meta, copier, groups = self.make(groups=1, rows=2, ds=True)
        groups[0].block_table[1] = torch.tensor(
            [8, 9, 10, 11, 4, 5, 6, 7, 2, 3, 0, 1], dtype=torch.int32)
        link.add_request(7, Request(10))
        link.add_request(2, Request(10))
        table = (groups[0].block_table,)
        before = groups[0].layers[0].conv.clone()
        comp = self.computed([(2, 15), (7, 15)])
        batch = self.batch([2, 7], [2, 2])
        link.preprocess_state(batch, table, None, comp)
        conv = groups[0].layers[0].conv
        torch.testing.assert_close(conv[0], before[2], atol=0, rtol=0)
        torch.testing.assert_close(conv[9], before[8], atol=0, rtol=0)
        link.prepare_attn(batch, "NONE", table, None, None, None)
        link.forward_completed_for_cpu_gold()
        comp[2], comp[7] = 20, 20
        link.postprocess_state(
            batch.idx_mapping, torch.tensor([2, 2], dtype=torch.int32), comp)
        self.assertEqual(meta.phase, "IDLE")

    def test_no_copy_for_fresh_request_and_non_spec_sentinel(self):
        link, meta, copier, groups = self.make(groups=1)
        link.add_request(1, Request(0))
        old = groups[0].layers[0].conv.clone()
        b = self.batch([1], [8])
        link.preprocess_state(b, (groups[0].block_table,), None,
                              self.computed([(1, 0)]))
        torch.testing.assert_close(groups[0].layers[0].conv, old)
        payload = link.prepare_attn(
            b, "NONE", (groups[0].block_table,), None, None, None)
        self.assertEqual(payload.num_decode_draft_tokens_cpu.tolist(), [-1])
        link.forward_completed_for_cpu_gold()
        link.postprocess_state(
            b.idx_mapping, 0, self.computed([(1, 8)]))
        self.assertEqual(meta.snapshot_for_cpu_test(1)[1], 1)

    def test_duplicate_destinations_rejected_before_any_write_and_poison(self):
        link, meta, copier, groups = self.make(groups=1, rows=2)
        link.add_request(2, Request(10))
        link.add_request(7, Request(10))
        old = groups[0].layers[0].conv.clone()
        b = self.batch([2, 7], [2, 2])
        with self.assertRaisesRegex(ValueError, "multiple requests"):
            link.preprocess_state(
                b, (groups[0].block_table,), None,
                self.computed([(2, 15), (7, 15)]))
        torch.testing.assert_close(groups[0].layers[0].conv, old)
        self.assertTrue(copier.poisoned)
        with self.assertRaises(RuntimeError):
            link.remove_request(2)

    def test_bad_device_blocktable_after_stage_poison_before_commit(self):
        link, meta, copier, groups = self.make(groups=2)
        link.add_request(1, Request(10))
        original = groups[0].layers[0].conv.clone()
        groups[1].block_table[0, 1] = -1
        with self.assertRaises(ValueError):
            link.preprocess_state(
                self.batch([1], [2]), tuple(g.block_table for g in groups),
                None, self.computed([(1, 15)]))
        self.assertTrue(copier.poisoned)
        torch.testing.assert_close(groups[0].layers[0].conv, original)

    def test_partial_copy_fault_poison_no_retry_or_slot_reuse(self):
        link, meta, copier, groups = self.make(groups=2)
        link.add_request(1, Request(10))
        copier._test_fail_after_write = 1
        with self.assertRaisesRegex(RuntimeError, "injected"):
            link.preprocess_state(
                self.batch([1], [2]), tuple(g.block_table for g in groups),
                None, self.computed([(1, 15)]))
        self.assertTrue(copier.poisoned)
        with self.assertRaises(RuntimeError):
            link.add_request(4, Request(0))

    def test_cannot_skip_precopy_forward_or_accept_wrong_blocktable(self):
        link, meta, copier, groups = self.make(groups=1)
        link.add_request(1, Request(10))
        b = self.batch([1], [1])
        with self.assertRaises(RuntimeError):
            link.prepare_attn(b, "NONE", (groups[0].block_table,), None,
                              None, None)
        with self.assertRaises(RuntimeError):
            link.forward_completed_for_cpu_gold()
        other = groups[0].block_table.clone()
        with self.assertRaisesRegex(ValueError, "same authoritative tensor"):
            link.preprocess_state(
                b, (other,), None, self.computed([(1, 10)]))

    def test_npu_remains_unregistered_and_device_consumer_fails_closed(self):
        src = (MODULE_DIR / "hybrid_state_device_copy.py").read_text()
        adapter = (MODULE_DIR / "hybrid_state_modelstate_link.py").read_text()
        runner = (MODULE_DIR / "model_runner.py").read_text()
        worker = (ROOT / "vllm_ascend/_310p/worker_310p.py").read_text()
        self.assertIn("NOT_IMPLEMENTED", worker)
        self.assertIn('self.cache_config.mamba_cache_mode == "align"', runner)
        self.assertNotIn("hybrid_state_device_copy", runner + worker)
        self.assertNotIn("hybrid_state_modelstate_link", runner + worker)
        self.assertIn("if self.device.type == \"npu\":", src)
        self.assertIn("raise NotImplementedError", src)
        for banned in (".item()", ".tolist()", ".cpu()", "torch.cuda",
                       "import triton", "import torch_npu", "GDNMove310"):
            self.assertNotIn(banned, src)
        for banned in ("from vllm.", "GPUModelRunner", "MambaSpecDecodeGPUContext"):
            self.assertNotIn(banned, adapter)


if __name__ == "__main__":
    unittest.main()

"""Frozen 310P3 K7 full-row draft write vs legacy seven-wide view."""
import importlib.util
import pathlib
import unittest


P=pathlib.Path(__file__).resolve().parents[2]/"vllm_ascend/_310p/v2/draft_storage.py"


class TestDraftK7(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import numpy
        except ImportError:
            raise unittest.SkipTest("torch/numpy absent; pure AST gate still runs")
        spec=importlib.util.spec_from_file_location("_r2_draftstorage",P)
        obj=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(obj)
        cls.m=obj
        cls.t=torch

    def test_full_row_write_and_slot_reuse(self):
        t=self.t
        storage=t.full((4,8),-7,dtype=t.int64)
        view=storage[:,:7]
        self.m.store_padded_draft_tokens(
            storage, view, t.tensor([3,1],dtype=t.int32),
            [3,1],t.tensor([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]],dtype=t.int64),
        )
        self.assertEqual(storage[3].tolist(),[1,2,3,4,5,6,7,0])
        self.assertEqual(storage[1].tolist(),[8,9,10,11,12,13,14,0])
        self.assertEqual(storage[0].tolist(),[-7]*8)
        self.assertEqual(view[3].tolist(),[1,2,3,4,5,6,7])
        self.m.store_padded_draft_tokens(
            storage,view,t.tensor([3],dtype=t.int32),[3],
            t.tensor([[19,18,17,16,15,14,13]],dtype=t.int64),
        )
        self.assertEqual(storage[3].tolist(),[19,18,17,16,15,14,13,0])
        self.assertEqual(storage.stride(0)*storage.element_size(),64)

    def test_duplicate_request_fails_before_write(self):
        t=self.t
        s=t.zeros((3,8),dtype=t.int64)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            self.m.store_padded_draft_tokens(s,s[:,:7],
                t.tensor([1,1],dtype=t.int32),[1,1],
                t.ones((2,7),dtype=t.int64))
        self.assertEqual(s.count_nonzero().item(),0)

    def test_short_or_detached_view_fails_closed(self):
        t=self.t
        s=t.zeros((3,8),dtype=t.int64)
        with self.assertRaisesRegex(ValueError,"shape"):
            self.m.store_padded_draft_tokens(
                s,s[:,:7],t.tensor([1],dtype=t.int32),[1],
                t.ones((1,6),dtype=t.int64))
        with self.assertRaisesRegex(ValueError,"alias"):
            self.m.store_padded_draft_tokens(
                s,s[:,:7].clone(),t.tensor([1],dtype=t.int32),[1],
                t.ones((1,7),dtype=t.int64))

    def test_mismatched_device_mapping_fails_before_writes(self):
        t=self.t
        s=t.full((3,8),-1,dtype=t.int64)
        with self.assertRaises(Exception):
            self.m.store_padded_draft_tokens(
                s,s[:,:7],t.tensor([1],dtype=t.int32),[2],
                t.ones((1,7),dtype=t.int64))
        self.assertEqual(s[1].tolist(),[-1]*8)
        self.assertEqual(s[2].tolist(),[-1]*8)


class TestDraftStatic(unittest.TestCase):
    def test_real_inherited_sample_override_no_short_slice_assign(self):
        source=(P.parent/"model_runner.py").read_text()
        self.assertIn("def sample_tokens(",source)
        self.assertIn("self._310p_v2_store_draft_tokens(input_batch, draft_tokens)",source)
        self.assertNotIn("self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens",source)


if __name__ == "__main__":
    unittest.main()

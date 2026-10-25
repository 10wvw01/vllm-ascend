"""R2 T14/T15 staged-write semantics (CPU-only, no vLLM/torch-npu)."""
import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE = ROOT / "vllm_ascend/_310p/v2/staged_buffers.py"
spec = importlib.util.spec_from_file_location("310p_v2_staged_buffers", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Plan = module.StagedWritePlan


class TestWritePlan(unittest.TestCase):
    def test_1d_scalar_staging_last_wins(self):
        p = Plan(9)
        self.assertEqual(p.padded_cols, 16)
        p.stage_write_elem(3, 17)
        p.stage_write_elem(3, 29)
        self.assertEqual(p.data[0][3], 0)
        self.assertEqual(p.commit(), (0,))
        self.assertEqual(p.data[0][3], 29)
        self.assertEqual(p.commit(), ())

    def test_2d_multi_group_independent_block_width(self):
        small = Plan((3, 5))
        large = Plan((3, 13))
        self.assertEqual((small.padded_cols, large.padded_cols), (8, 16))
        small.stage_write(2, 0, [10, 11, 12])
        large.stage_write(1, 11, [80, 81])
        self.assertEqual(small.commit(), (2,))
        self.assertEqual(large.commit(), (1,))
        self.assertEqual(small.data[2][:5], [10, 11, 12, 0, 0])
        self.assertEqual(large.data[1][11:13], [80, 81])
        self.assertEqual(small.data[1], [0] * 8)

    def test_overlap_order_and_no_cross_row_overwrite(self):
        p = Plan((2, 9))
        p.stage_write(0, 7, [3, 4])
        p.stage_write(0, 8, [5])
        p.stage_write(1, 0, [6])
        self.assertEqual(p.commit(), (0, 1))
        self.assertEqual(p.data[0][7:9], [3, 5])
        self.assertEqual(p.data[1][0], 6)
        self.assertEqual(p.data[0][9:], [0] * 7)

    def test_reject_invalid_before_mutation(self):
        p = Plan((2, 4))
        for args in [(-1, 0, [1]), (2, 0, [1]), (0, -1, [1]),
                     (0, 3, [1, 2]), (0, 5, [])]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                p.stage_write(*args)
        self.assertEqual(p.pending_count, 0)
        self.assertEqual(p.commit(), ())
        self.assertEqual(p.data, [[0] * 8 for _ in range(2)])

    def test_abort_pending_without_committing(self):
        p = Plan((1, 8))
        p.stage_write(0, 0, [1, 2])
        p.clear_staged_writes()
        self.assertEqual(p.commit(), ())
        self.assertEqual(p.data[0][:2], [0, 0])

    def test_invalid_shape_dtype_and_vector_offset(self):
        for size in (0, (-1, 3), (1, 0), (1, 2, 3)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                Plan(size)
        with self.assertRaises(ValueError):
            Plan(8, itemsize=2)
        with self.assertRaises(ValueError):
            Plan(8).stage_write(1, 1, [2])

    def test_64bit_alignment(self):
        p = Plan((2, 5), itemsize=8)
        self.assertEqual(p.padded_cols, 8)
        p.stage_write(0, 4, [2**40])
        p.commit()
        self.assertEqual(p.data[0][4], 2**40)


class TestTorchBackends(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch not installed; plan gold still runs")
        cls.torch = torch

    def test_device_backed_write_and_request_index(self):
        torch = self.torch
        t = module.NPUStagedWriteTensor((3, 5), torch.int32, "cpu", uva_instead_of_gpu=True)
        self.assertEqual(t.gpu.shape, (3, 5))
        self.assertEqual(t.gpu.stride(0), 8)
        self.assertFalse(t.uva_instead_of_gpu)
        t.stage_write(1, 3, [17, 19])
        self.assertEqual(t.gpu[1, 3].item(), 0)
        t.apply_write()
        self.assertEqual(t.gpu[1].tolist(), [0, 0, 0, 17, 19])
        t.stage_write(1, 4, [21])
        t.apply_write()
        self.assertEqual(t.gpu[1, 3:].tolist(), [17, 21])

    def test_vector_staging_preserves_device_owned_other_requests(self):
        torch = self.torch
        t = module.NPUStagedWriteTensor(16, torch.int32, "cpu")
        # Simulate sampling/post_update having advanced live request #3.
        t.gpu[3] = 37
        t.stage_write_elem(8, 5)  # newly added request
        t.apply_write()
        self.assertEqual(t.gpu[3].item(), 37)
        self.assertEqual(t.gpu[8].item(), 5)
        t.gpu[8] = 9
        t.stage_write_elem(3, 38)
        t.apply_write()
        self.assertEqual(t.gpu[8].item(), 9)
        self.assertEqual(t.gpu[3].item(), 38)

    def test_backed_copy_and_multi_group_fused(self):
        torch = self.torch
        b = module.NPUBackedTensor((2, 3), torch.int32, "cpu")
        b.np[1, 2] = 99
        self.assertEqual(b.gpu[1, 2].item(), 0)
        b.copy_to_uva()
        self.assertEqual(b.gpu[1, 2].item(), 99)
        a = module.NPUStagedWriteTensor((2, 3), torch.int32, "cpu")
        c = module.NPUStagedWriteTensor((2, 9), torch.int32, "cpu")
        a.stage_write(0, 0, [5])
        c.stage_write(1, 8, [7])
        module.NPUFusedStagedWriter("cpu", 4).apply([a, c])
        self.assertEqual(a.gpu[0, 0].item(), 5)
        self.assertEqual(c.gpu[1, 8].item(), 7)
        with self.assertRaises(TypeError):
            module.NPUFusedStagedWriter("cpu", 4).apply([object()])


if __name__ == "__main__":
    unittest.main()

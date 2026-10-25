"""Mandatory no-torch T05/T06 multi-KV group and slot CPU golden."""
import importlib.util
import pathlib
import unittest

SRC = pathlib.Path(__file__).resolve().parents[2] / "vllm_ascend/_310p/v2/block_table_gold.py"
spec = importlib.util.spec_from_file_location("r2_block_table_gold", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
Gold = mod.BlockTablesGold


class TestMultiGroupBlockTableGold(unittest.TestCase):
    def setUp(self):
        self.b = Gold([64, 128], [32, 128], [4, 4], 3)

    def test_expansion_and_group_specific_slots(self):
        self.b.append(2, ([3, 4], [8]))
        self.assertEqual(self.b.tables[0][2][:4], [6, 7, 8, 9])
        self.assertEqual(self.b.tables[1][2][0], 8)
        self.assertEqual(self.b.slots([2], [0, 3], [0, 31, 32, 0, 0], 5),
                         [[192, 223, 224, -1, -1],
                          [1024, 1055, 1056, -1, -1]])

    def test_padded_gather_clears_stale_rows(self):
        self.b.append(2, ([3, 4], [8]))
        tables = self.b.gather([2], 3)
        self.assertEqual(tables[0][0][:4], [6, 7, 8, 9])
        self.assertEqual(tables[0][1], [0] * 8)
        self.assertEqual(tables[1][2], [0] * 4)

    def test_request_reuse_overwrite_avoids_stale_tail(self):
        self.b.append(2, ([3, 4], [8]))
        self.b.append(2, ([10], [11]), overwrite=True)
        self.assertEqual([group[2] for group in self.b.counts], [2, 1])
        self.assertEqual(self.b.gather([2], 1)[0][0], [20, 21] + [0] * 6)
        self.assertEqual(self.b.slots([2], [0, 2], [0, 64], 2),
                         [[640, -1], [1408, 1472]])

    def test_capacity_failure_is_atomic(self):
        self.b.append(2, ([3], [8]))
        before = [list(group[2]) for group in self.b.tables]
        counts = [group[2] for group in self.b.counts]
        with self.assertRaises(ValueError):
            self.b.append(2, ([5], [1, 2, 3, 4]))
        self.assertEqual([group[2] for group in self.b.tables], before)
        self.assertEqual([group[2] for group in self.b.counts], counts)

    def test_unallocated_and_negative_positions_are_pad(self):
        self.b.append(1, ([3], [8]))
        self.assertEqual(self.b.slots([1], [0, 3], [-1, 63, 128], 3),
                         [[-1, -1, -1], [-1, 1087, -1]])

    def test_nonmonotonic_offsets_and_bad_indices_rejected(self):
        self.b.append(0, ([1], [2]))
        for offsets in ([1, 2], [0, 3, 2]):
            with self.subTest(offsets=offsets), self.assertRaises(ValueError):
                self.b.slots([0], offsets, [0, 1, 2], 3)
        with self.assertRaises(ValueError):
            self.b.gather([3], 1)
        with self.assertRaises(ValueError):
            Gold([64], [48], [2], 1)

    def test_zero_length_req_and_padding(self):
        self.b.append(0, ([1], [2]))
        self.b.append(1, ([3], [4]))
        self.assertEqual(self.b.slots([0, 1], [0, 0, 1], [0, 0], 2),
                         [[96, -1], [512, -1]])


if __name__ == "__main__":
    unittest.main()

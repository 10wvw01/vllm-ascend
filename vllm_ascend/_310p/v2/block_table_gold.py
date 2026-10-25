"""Pure-Python T05/T06 multi-group V2 BlockTables golden.

No torch dependency: mandatory lightweight CI validates exact physical/kernel
block expansion, request index mapping, padded gather, slot PAD and reuse.
"""
from __future__ import annotations

PAD_SLOT_ID = -1


def expand_physical_block_ids(ids: list[int], ratio: int) -> list[int]:
    if ratio <= 0 or any(not isinstance(x, int) or x < 0 for x in ids):
        raise ValueError("invalid block id or expansion ratio")
    return [x * ratio + k for x in ids for k in range(ratio)]


class BlockTablesGold:
    def __init__(
        self,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_blocks: list[int],
        max_reqs: int,
    ):
        if not block_sizes or not (
            len(block_sizes) == len(kernel_block_sizes) == len(max_blocks)
        ):
            raise ValueError("group count mismatch")
        if max_reqs <= 0 or any(
            b <= 0 or k <= 0 or b % k or n <= 0
            for b, k, n in zip(block_sizes, kernel_block_sizes, max_blocks)
        ):
            raise ValueError("invalid group block size/capacity")
        self.kernel_block_sizes = kernel_block_sizes
        self.ratios = [b // k for b, k in zip(block_sizes, kernel_block_sizes)]
        self.tables = [
            [[0] * (n * ratio) for _ in range(max_reqs)]
            for n, ratio in zip(max_blocks, self.ratios)
        ]
        self.counts = [[0] * max_reqs for _ in block_sizes]
        self.max_reqs = max_reqs

    def append(self, req_idx: int, ids_per_group: tuple[list[int], ...], overwrite=False):
        if not 0 <= req_idx < self.max_reqs:
            raise ValueError("bad request index")
        if len(ids_per_group) != len(self.tables):
            raise ValueError("group count mismatch")
        staged = []
        for g, ids in enumerate(ids_per_group):
            expanded = expand_physical_block_ids(ids, self.ratios[g])
            start = 0 if overwrite else self.counts[g][req_idx]
            if start + len(expanded) > len(self.tables[g][req_idx]):
                raise ValueError("group capacity exceeded")
            staged.append((start, expanded))
        for g, (start, expanded) in enumerate(staged):
            self.tables[g][req_idx][start:start + len(expanded)] = expanded
            self.counts[g][req_idx] = start + len(expanded)

    def gather(self, idx_mapping: list[int], num_reqs_padded: int) -> list[list[list[int]]]:
        if not len(idx_mapping) <= num_reqs_padded <= self.max_reqs:
            raise ValueError("invalid padding")
        if any(not 0 <= i < self.max_reqs for i in idx_mapping):
            raise ValueError("invalid persistent request index")
        result = []
        for g, table in enumerate(self.tables):
            rows = []
            for idx in idx_mapping:
                # Triton copies only the allocated prefix; zero the remainder
                # in this stricter gold so no stale tail becomes valid KV.
                n = self.counts[g][idx]
                rows.append(table[idx][:n] + [0] * (len(table[idx]) - n))
            rows += [[0] * len(table[0]) for _ in range(num_reqs_padded - len(rows))]
            result.append(rows)
        return result

    def slots(
        self,
        idx_mapping: list[int],
        query_start_loc: list[int],
        positions: list[int],
        num_tokens_padded: int,
    ) -> list[list[int]]:
        if len(query_start_loc) != len(idx_mapping) + 1:
            raise ValueError("query offsets mismatch")
        if not query_start_loc or query_start_loc[0] != 0:
            raise ValueError("query offsets must start at zero")
        if any(a > b for a, b in zip(query_start_loc, query_start_loc[1:])):
            raise ValueError("query offsets must be nondecreasing")
        if not 0 <= query_start_loc[-1] <= len(positions) <= num_tokens_padded:
            raise ValueError("invalid token span")
        if any(not 0 <= i < self.max_reqs for i in idx_mapping):
            raise ValueError("invalid persistent request index")
        result = [[PAD_SLOT_ID] * num_tokens_padded for _ in self.tables]
        for batch_idx, req_idx in enumerate(idx_mapping):
            for t in range(query_start_loc[batch_idx], query_start_loc[batch_idx + 1]):
                pos = positions[t]
                if pos < 0:
                    continue
                for g, block_size in enumerate(self.kernel_block_sizes):
                    logical, offset = divmod(pos, block_size)
                    if logical >= self.counts[g][req_idx]:
                        continue
                    block = self.tables[g][req_idx][logical]
                    result[g][t] = block * block_size + offset
        return result

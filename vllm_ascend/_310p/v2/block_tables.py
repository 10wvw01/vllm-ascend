"""Candidate 310P3 V2 Eager BlockTables (vLLM 0.27.1 ABI).

No GPU BlockTables constructor, UVA pool, Triton, CUDA or raw-pointer
dereference. This is NOT wired into NPUWorker310 / GPUModelRunner; the
310P-specific model/attention lifecycle and NPU op coverage remain NO-GO.
"""
from __future__ import annotations

from collections.abc import Iterable

from .block_table_gold import expand_physical_block_ids
from .staged_buffers import NPUBackedTensor, NPUFusedStagedWriter, NPUStagedWriteTensor

PAD_SLOT_ID = -1


class NPUBlockTables:
    def __init__(
        self,
        block_sizes: list[int],
        max_num_reqs: int,
        max_num_batched_tokens: int,
        max_num_blocks_per_group: list[int],
        device,
        kernel_block_sizes: list[int],
        cp_size: int = 1,
        cp_rank: int = 0,
        cp_interleave: int = 1,
    ):
        import torch

        if cp_size != 1 or cp_rank != 0 or cp_interleave != 1:
            raise NotImplementedError("310P3 V2 CP is not supported")
        if (not block_sizes or len(block_sizes) != len(kernel_block_sizes)
                or len(block_sizes) != len(max_num_blocks_per_group)):
            raise ValueError("KV group configuration mismatch")
        if max_num_reqs <= 0 or max_num_batched_tokens <= 0:
            raise ValueError("invalid KV capacity")
        if any(b <= 0 or k <= 0 or b % k or n <= 0 for b, k, n in zip(
                block_sizes, kernel_block_sizes, max_num_blocks_per_group)):
            raise ValueError("invalid physical/kernel block size")
        self.block_sizes = block_sizes
        self.kernel_block_sizes = kernel_block_sizes
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.device = device
        self.cp_size, self.cp_rank, self.cp_interleave = cp_size, cp_rank, cp_interleave
        self.num_kv_cache_groups = len(block_sizes)
        self.blocks_per_kv_block = [
            b // k for b, k in zip(block_sizes, kernel_block_sizes)
        ]
        self.block_tables = [
            NPUStagedWriteTensor(
                (max_num_reqs, n * ratio), torch.int32, device
            )
            for n, ratio in zip(max_num_blocks_per_group, self.blocks_per_kv_block)
        ]
        self.num_blocks = NPUBackedTensor(
            (self.num_kv_cache_groups, max_num_reqs), torch.int32, device
        )
        self.fused_writer = (
            NPUFusedStagedWriter(device, self.num_kv_cache_groups * max_num_reqs)
            if self.num_kv_cache_groups > 1 else None
        )
        self._input_storage = [
            torch.zeros_like(bt._storage) for bt in self.block_tables
        ]
        self.input_block_tables = [
            storage[:, :bt.plan.cols]
            for storage, bt in zip(self._input_storage, self.block_tables)
        ]
        self.slot_mappings = torch.full(
            (self.num_kv_cache_groups, max_num_batched_tokens),
            PAD_SLOT_ID, dtype=torch.int64, device=device
        )
        self.init_block_table_layout_tensors()

    def _make_ptr_tensor(self, x: Iterable) -> object:
        import torch
        # Compatibility metadata only. No pointer-kernel is launched here.
        return torch.tensor(
            [t.data_ptr() for t in x], dtype=torch.uint64, device=self.device
        )

    def init_block_table_layout_tensors(self) -> None:
        import torch
        self.block_table_ptrs = self._make_ptr_tensor(
            [b.gpu for b in self.block_tables]
        )
        self.block_table_strides = torch.tensor(
            [b.gpu.stride(0) for b in self.block_tables],
            dtype=torch.int64, device=self.device
        )
        self.block_sizes_tensor = torch.tensor(
            self.kernel_block_sizes, dtype=torch.int32, device=self.device
        )
        self.input_block_table_ptrs = self._make_ptr_tensor(
            self.input_block_tables
        )

    def append_block_ids(
        self,
        req_index: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        if not 0 <= req_index < self.max_num_reqs:
            raise ValueError("request index out of bounds")
        if len(new_block_ids) != self.num_kv_cache_groups:
            raise ValueError("missing KV group block ids")
        # Prevalidate all groups before mutating any staged write or counts.
        pending = []
        for g, ids in enumerate(new_block_ids):
            if any(not isinstance(b, int) or b < 0 for b in ids):
                raise ValueError("invalid physical block id")
            start = 0 if overwrite else int(self.num_blocks.np[g, req_index])
            ratio = self.blocks_per_kv_block[g]
            expanded = expand_physical_block_ids(ids, ratio)
            if start + len(expanded) > self.block_tables[g].plan.cols:
                raise ValueError("KV block table capacity exceeded")
            pending.append((g, start, expanded))
        for g, start, expanded in pending:
            self.block_tables[g].stage_write(req_index, start, expanded)
            self.num_blocks.np[g, req_index] = start + len(expanded)

    def apply_staged_writes(self) -> None:
        if self.fused_writer is None:
            self.block_tables[0].apply_write()
        else:
            self.fused_writer.apply(
                self.block_tables, self.block_table_ptrs, self.block_table_strides
            )
        self.num_blocks.copy_to_uva()

    def gather_block_tables(
        self,
        idx_mapping,
        num_reqs_padded: int,
        out=None,
        out_ptrs=None,
    ) -> tuple:
        import torch
        if out is not None or out_ptrs is not None:
            raise NotImplementedError("310P3 external DFlash gather buffers unreviewed")
        num_reqs = idx_mapping.numel()
        if not 0 <= num_reqs <= num_reqs_padded <= self.max_num_reqs:
            raise ValueError("invalid padded request count")
        if idx_mapping.device != self.block_tables[0].gpu.device:
            raise ValueError("idx_mapping must be on same NPU device")
        indices = idx_mapping.to(dtype=torch.int64)
        result = []
        for g, (src, dest, logical) in enumerate(zip(
            self.block_tables, self._input_storage, self.input_block_tables
        )):
            # Full padded rows: avoid 310P3 unaligned short D2D writes.
            # Mask stale blocks beyond num_blocks: overwrite/reused requests
            # must not expose an old KV allocation through the gather result.
            dest.zero_()
            if num_reqs:
                selected = torch.index_select(src._storage, 0, indices)
                counts = self.num_blocks.gpu[g].index_select(0, indices)
                columns = torch.arange(
                    src.plan.padded_cols, device=idx_mapping.device
                )
                selected = torch.where(
                    columns[None, :] < counts[:, None],
                    selected, torch.zeros_like(selected)
                )
                dest[:num_reqs].copy_(selected)
            result.append(logical[:num_reqs_padded])
        return tuple(result)

    def get_dummy_block_tables(self, num_reqs: int) -> tuple:
        if not 0 <= num_reqs <= self.max_num_reqs:
            raise ValueError("invalid dummy request count")
        for storage in self._input_storage:
            storage.zero_()
        return tuple(t[:num_reqs] for t in self.input_block_tables)

    def compute_slot_mappings(
        self,
        idx_mapping,
        query_start_loc,
        positions,
        num_tokens_padded: int,
        out=None,
    ):
        import torch
        if out is not None:
            raise NotImplementedError("310P3 external DFlash slot output unreviewed")
        if not 0 <= num_tokens_padded <= self.max_num_batched_tokens:
            raise ValueError("invalid padded token count")
        num_reqs = idx_mapping.numel()
        if query_start_loc.numel() != num_reqs + 1:
            raise ValueError("invalid query_start_loc shape")
        if positions.numel() < num_tokens_padded:
            raise ValueError("positions shorter than padded token count")
        if any(t.device != idx_mapping.device for t in (query_start_loc, positions)):
            raise ValueError("all slot inputs must be on the same device")
        # searchsorted works entirely on the accelerator. Unlike repeat_interleave
        # with tensor repeat counts, it does not need a CPU shape synchronization.
        offsets = torch.arange(
            num_tokens_padded, dtype=torch.int32, device=idx_mapping.device
        )
        boundaries = query_start_loc[1:].contiguous()
        req_rows = torch.searchsorted(boundaries, offsets, right=True)
        in_batch = req_rows < num_reqs
        req_rows = req_rows.clamp(max=max(num_reqs - 1, 0)).long()
        if num_reqs == 0:
            return torch.full(
                (self.num_kv_cache_groups, num_tokens_padded),
                PAD_SLOT_ID, dtype=torch.int64, device=idx_mapping.device
            )
        req_idx = idx_mapping.index_select(0, req_rows).long()
        pos = positions[:num_tokens_padded].to(torch.int32)
        valid = in_batch & (pos >= 0)
        slots = []
        for g, bt in enumerate(self.block_tables):
            bs = self.kernel_block_sizes[g]
            block_idx = torch.div(pos.clamp(min=0), bs, rounding_mode="floor")
            counts = self.num_blocks.gpu[g].index_select(0, req_idx)
            valid_g = valid & (block_idx < counts)
            # Clamp before gather so padding/unallocated positions cannot OOB.
            block_idx = block_idx.clamp(max=bt.plan.cols - 1).long()
            flat_idx = req_idx * bt._storage.stride(0) + block_idx
            blocks = bt._storage.reshape(-1).index_select(0, flat_idx)
            slot = blocks.long() * bs + (pos.long() % bs)
            slots.append(torch.where(
                valid_g, slot, torch.full_like(slot, PAD_SLOT_ID)
            ))
        return torch.stack(slots, dim=0)

    def get_dummy_slot_mappings(self, num_tokens: int):
        if not 0 <= num_tokens <= self.max_num_batched_tokens:
            raise ValueError("invalid dummy token count")
        self.slot_mappings.fill_(PAD_SLOT_ID)
        return self.slot_mappings[:, :num_tokens]

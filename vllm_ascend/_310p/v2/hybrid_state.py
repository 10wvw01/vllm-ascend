"""310P3 target-scoped GDN align tensor-copy candidate (UNREGISTERED).

This is an executable CPU-Torch state owner/copy prototype, NOT a drop-in
vLLM ModelState. The real 310P/NPU device path is deliberately blocked until
its small-write, torch-npu, stream and physical-KV contracts are approved.

Only frozen Qwen3.5 GDN (conv SD/DS + temporal), TP=2 and DFlash K=7.
No CUDA, Triton, UVA, private Draft classes, model loading or global patches.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch


@dataclass(frozen=True)
class GDNStateLayer310:
    conv: torch.Tensor
    temporal: torch.Tensor
    conv_layout: str = "SD"


@dataclass(frozen=True)
class GDNStateGroup310:
    """One Mamba KV group; block table rows are CURRENT BATCH order."""

    group_id: int
    block_table: torch.Tensor  # int32 [active_batch, logical_columns]
    layers: tuple[GDNStateLayer310, ...]


@dataclass
class _Request310:
    running_col: int
    num_accepted: int = 1


@dataclass(frozen=True)
class _Copy310:
    destination: torch.Tensor
    source_snapshot: torch.Tensor


class HybridAlignTensor310Candidate:
    """CPU-validatable T30/T31 state transitions, never registered in Worker.

    The pre/post copy *execution* uses real Torch Tensor slices, including
    overlapping self copies, as opposed to the pure arithmetic Oracle.
    NPU is intentionally rejected: even numerically correct Torch operations
    may violate 310P3 sub-32-byte write and asynchronous stream contracts.
    """

    def __init__(self, *, block_size: int, max_reqs: int, tp: int = 2, k: int = 7):
        if tp != 2 or k != 7:
            raise NotImplementedError("target-only TP2 / DFlash K7")
        if not isinstance(block_size, int) or block_size <= 0 or max_reqs <= 0:
            raise ValueError("invalid target Mamba block configuration")
        self.block_size = block_size
        self.max_reqs = max_reqs
        self._requests: dict[int, _Request310] = {}
        self._poisoned = False
        self._groups: tuple[GDNStateGroup310, ...] | None = None
        # Only for fault-injection unit tests; not a runtime configuration.
        self._test_fail_after_copy: int | None = None

    def _assert_healthy(self) -> None:
        if self._poisoned:
            raise RuntimeError("310P3 GDN state is poisoned; discard owner")

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def bind_groups(self, groups: Sequence[GDNStateGroup310]) -> None:
        """Bind already-allocated real layer caches and per-Mamba-group BTs."""
        self._assert_healthy()
        if self._groups is not None:
            raise RuntimeError("cannot rebind active GDN physical caches")
        if not groups:
            raise ValueError("missing Mamba KV groups")
        seen_ids: set[int] = set()
        seen_ptrs: set[int] = set()
        for group in groups:
            if group.group_id < 0 or group.group_id in seen_ids:
                raise ValueError("duplicate or invalid Mamba KV group")
            seen_ids.add(group.group_id)
            table = group.block_table
            if (table.device.type != "cpu" or table.dtype != torch.int32
                    or table.ndim != 2 or table.shape[1] < 1):
                raise NotImplementedError(
                    "CPU-only physical BT candidate; 310P3 NPU requires ABI GO"
                )
            if not group.layers:
                raise ValueError("empty GDN group")
            for layer in group.layers:
                conv, temporal = layer.conv, layer.temporal
                if (conv.device.type != "cpu" or temporal.device.type != "cpu"
                        or conv.ndim != 3 or temporal.ndim < 2
                        or conv.shape[0] < 1 or temporal.shape[0] < 1
                        or layer.conv_layout not in ("SD", "DS")
                        or conv.dtype not in (torch.float16, torch.bfloat16, torch.float32)
                        or temporal.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
                    raise NotImplementedError("unapproved GDN state layout/dtype/device")
                width = conv.shape[1] if layer.conv_layout == "SD" else conv.shape[2]
                if width <= 7:
                    # Copy bias is <=7 for K7; there must be an actual window.
                    raise ValueError("GDN conv width must support K7 acceptance")
                for state in (conv, temporal):
                    # Sharing physical storage across layers/state types is not
                    # audited and can create writes that invalidate snapshots.
                    ptr = state.untyped_storage().data_ptr()
                    if ptr in seen_ptrs:
                        raise ValueError("aliased state storage is not supported")
                    seen_ptrs.add(ptr)
        self._groups = tuple(groups)

    def add_request(self, slot: int, num_computed_tokens: int) -> None:
        self._assert_healthy()
        if (not isinstance(slot, int) or not 0 <= slot < self.max_reqs
                or slot in self._requests or not isinstance(num_computed_tokens, int)
                or num_computed_tokens < 0):
            raise ValueError("invalid/reused live GDN request slot")
        self._requests[slot] = _Request310(
            (num_computed_tokens - 1) // self.block_size
        )

    def remove_request(self, slot: int) -> None:
        self._assert_healthy()
        if slot not in self._requests:
            raise KeyError("request slot not live")
        # CPU prototype is synchronous; a real NPU owner MUST wait for all
        # queued per-group copy events before releasing/reusing this slot.
        del self._requests[slot]

    def snapshot(self, slot: int) -> tuple[int, int]:
        """Audit-only stable (running_col, accepted) state for CPU tests."""
        self._assert_healthy()
        state = self._requests[slot]
        return state.running_col, state.num_accepted

    def _rows(self, slots: Sequence[int], *inputs: Sequence[int]) -> None:
        self._assert_healthy()
        if self._groups is None:
            raise RuntimeError("GDN physical groups are not bound")
        if not slots or len(set(slots)) != len(slots):
            raise ValueError("empty/duplicate active GDN request slots")
        if any(len(x) != len(slots) for x in inputs):
            raise ValueError("GDN batch metadata shape mismatch")
        if any(slot not in self._requests for slot in slots):
            raise ValueError("unregistered GDN request slot")
        if any(g.block_table.shape[0] < len(slots) for g in self._groups):
            raise ValueError("Mamba block table missing active batch rows")

    @staticmethod
    def _physical(table: torch.Tensor, row: int, col: int,
                  capacity: int) -> int:
        if col < 0 or col >= table.shape[1]:
            raise ValueError("GDN logical block table column out of range")
        block_id = int(table[row, col])
        if not 0 <= block_id < capacity:
            raise ValueError("GDN physical block id out of range")
        return block_id

    def _copy_plan(self, row: int, src_col: int, dst_col: int,
                   bias: int) -> list[_Copy310]:
        """Snapshot ALL sources before ANY target write: memmove-safe."""
        if src_col < 0 or dst_col < 0 or not 0 <= bias <= 7:
            raise ValueError("invalid source/destination or K7 shift")
        assert self._groups is not None
        out: list[_Copy310] = []
        for group in self._groups:
            for layer in group.layers:
                conv, temporal = layer.conv, layer.temporal
                width = conv.shape[1] if layer.conv_layout == "SD" else conv.shape[2]
                if bias >= width:
                    raise ValueError("GDN conv window too short for accepted bias")
                src = self._physical(group.block_table, row, src_col, conv.shape[0])
                dst = self._physical(group.block_table, row, dst_col, conv.shape[0])
                if layer.conv_layout == "SD":
                    source_view = conv[src, bias:, :]
                    dest_view = conv[dst, :width - bias, :]
                else:
                    source_view = conv[src, :, bias:]
                    dest_view = conv[dst, :, :width - bias]
                out.append(_Copy310(dest_view, source_view.clone()))
                temporal_src = self._physical(
                    group.block_table, row, src_col + bias, temporal.shape[0]
                )
                temporal_dst = self._physical(
                    group.block_table, row, dst_col, temporal.shape[0]
                )
                out.append(_Copy310(
                    temporal[temporal_dst], temporal[temporal_src].clone()
                ))
        return out

    def _commit_copies(self, writes: list[_Copy310]) -> None:
        # Physical GDN output blocks may not be shared by active requests.
        # Reject destination collisions before writing any source snapshot.
        dest_ptrs = [w.destination.data_ptr() for w in writes]
        if len(dest_ptrs) != len(set(dest_ptrs)):
            raise ValueError("aliased GDN destination blocks across active requests")
        try:
            for i, work in enumerate(writes):
                if self._test_fail_after_copy == i:
                    raise RuntimeError("injected GDN partial write failure")
                work.destination.copy_(work.source_snapshot)
        except Exception:
            # No in-place retry after a partial multi-layer/group commit.
            self._poisoned = True
            raise

    def preprocess_state(self, slots: Sequence[int],
                         num_computed_tokens: Sequence[int],
                         query_lens: Sequence[int]) -> None:
        """T30: advance running columns then pre-copy at block boundaries.

        Caller MUST ensure block-table staged writes are complete. This CPU
        prototype does not implement any NPU stream/event coordination.
        """
        self._rows(slots, num_computed_tokens, query_lens)
        stages: list[tuple[int, int, int]] = []
        writes: list[_Copy310] = []
        try:
            for row, (slot, computed, query_len) in enumerate(
                    zip(slots, num_computed_tokens, query_lens)):
                if (not isinstance(computed, int) or computed < 0
                        or not isinstance(query_len, int) or query_len <= 0):
                    raise ValueError("invalid GDN input token counts")
                request = self._requests[slot]
                previous, accepted = request.running_col, request.num_accepted
                nxt = (computed + query_len + self.block_size - 1) // self.block_size - 1
                crossed = previous >= 0 and previous != nxt
                if crossed:
                    writes.extend(self._copy_plan(row, previous, nxt, accepted - 1))
                stages.append((slot, nxt, 1 if crossed else accepted))
            self._commit_copies(writes)
            for slot, new_col, new_accepted in stages:
                self._requests[slot] = _Request310(new_col, new_accepted)
        except Exception:
            self._poisoned = True
            raise

    def postprocess_state(self, slots: Sequence[int],
                          sampled_counts: Sequence[int],
                          computed_after_update: Sequence[int]) -> None:
        """T31: sample-count scatter + aligned GDN post-copy.

        computed_after_update MUST already include frozen post_update changes.
        """
        self._rows(slots, sampled_counts, computed_after_update)
        stages: list[tuple[int, int]] = []
        writes: list[_Copy310] = []
        try:
            for row, (slot, sampled, computed) in enumerate(
                    zip(slots, sampled_counts, computed_after_update)):
                if (not isinstance(sampled, int) or not 0 <= sampled <= 8
                        or not isinstance(computed, int) or computed < 0):
                    raise ValueError("invalid GDN accepted/computed counts")
                accepted = max(sampled, 1)  # chunked prefill zero -> neutral
                running = computed - accepted + 1
                if running < 0:
                    raise ValueError("negative effective GDN running position")
                aligned = computed // self.block_size * self.block_size
                next_accepted = accepted
                if aligned >= running:
                    dest = aligned // self.block_size - 1
                    bias = aligned - running
                    src = self._requests[slot].running_col
                    if not 0 <= bias <= 7 or src < 0 or dest < 0:
                        raise ValueError("invalid GDN postalign copy state")
                    if src != dest or bias != 0:
                        writes.extend(self._copy_plan(row, src, dest, bias))
                    if src == dest:
                        next_accepted = 1
                stages.append((slot, next_accepted))
            self._commit_copies(writes)
            for slot, accepted in stages:
                self._requests[slot].num_accepted = accepted
        except Exception:
            self._poisoned = True
            raise

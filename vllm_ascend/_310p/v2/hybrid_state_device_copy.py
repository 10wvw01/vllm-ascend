"""310P3 Target GDN device-indexed whole-block copy consumer (UNREGISTERED).

T30/T31 device decisions go directly to same-device Mamba group BlockTables.
CPU Torch executes the entire tensor path, including overlapping snapshots,
and provides a gold reference. NPU execution is deliberately denied until
device-side failure gating, 32-byte writes and event ownership are proven.
No per-request host scalar bridge, CUDA, Triton or private Draft.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GDNIndexedGroup310:
    group_id: int
    block_table: torch.Tensor  # device int32 [active_batch, logical_cols]
    layers: tuple["GDNIndexedLayer310", ...]


@dataclass(frozen=True)
class GDNIndexedLayer310:
    conv: torch.Tensor        # [physical_blocks, state_len, dim] SD
    temporal: torch.Tensor    # [physical_blocks, ...] independent dtype
    conv_layout: str = "SD"   # or DS [physical_blocks, dim, state_len]


@dataclass(frozen=True)
class _Write310:
    storage: torch.Tensor
    indices: torch.Tensor
    replacement: torch.Tensor


class DeviceIndexedGDNCopy310Candidate:
    """No scalar-host bridge: tensor indices and per-group BT stay on device.

    Actual NPU dispatch is blocked until the device can fail safely for bad
    block-ids/destination collisions and stream-write hazards. This protects
    Target Serving even when the same Torch kernel happens to be available.
    """

    def __init__(
        self, groups: Sequence[GDNIndexedGroup310],
        *, device: str | torch.device = "cpu", tp: int = 2, k: int = 7,
    ) -> None:
        if tp != 2 or k != 7:
            raise NotImplementedError("Target-only GDN TP2/DFlash K7")
        if isinstance(device, str) and device.startswith("npu") and not hasattr(torch, "npu"):
            raise NotImplementedError("torch_npu not registered")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "npu"):
            raise NotImplementedError("unsupported device")
        self.poisoned = False
        self._test_fail_after_write: int | None = None
        if not groups:
            raise ValueError("no Mamba KV groups supplied")
        seen_group: set[int] = set()
        storages: set[int] = set()
        validated: list[GDNIndexedGroup310] = []
        for group in groups:
            if group.group_id < 0 or group.group_id in seen_group:
                raise ValueError("invalid or repeated Mamba KV group")
            seen_group.add(group.group_id)
            table = group.block_table
            if (table.device != self.device or table.dtype != torch.int32
                    or table.ndim != 2 or not table.is_contiguous()
                    or table.shape[0] == 0 or table.shape[1] == 0):
                raise ValueError("BlockTable must be device-owned int32 [batch,cols]")
            if not group.layers:
                raise ValueError("Mamba KV group without states")
            for layer in group.layers:
                if layer.conv_layout not in ("SD", "DS"):
                    raise ValueError("invalid Conv layout")
                if layer.conv.ndim != 3 or layer.temporal.ndim < 2:
                    raise ValueError("invalid Conv/Temporal cache rank")
                width = layer.conv.shape[1 if layer.conv_layout == "SD" else 2]
                if width <= 7:
                    raise ValueError("Conv window too short for K7")
                for state in (layer.conv, layer.temporal):
                    if state.device != self.device or not state.is_contiguous():
                        raise ValueError("state device/storage layout mismatch")
                    if state.shape[0] < 1 or state.dtype not in (
                        torch.float16, torch.float32, torch.bfloat16
                    ):
                        raise ValueError("invalid GDN physical cache or dtype")
                    # This protects against sub-32B *requested* writes, but
                    # cannot establish actual torch_npu hardware write granularity.
                    row_bytes = state[0].numel() * state.element_size()
                    if (row_bytes < 32 or row_bytes % 32 != 0
                            or state.stride(0) * state.element_size() % 32 != 0
                            or state.data_ptr() % 32 != 0):
                        raise NotImplementedError("unaligned 32B whole-state row")
                    ptr = state.untyped_storage().data_ptr()
                    if ptr in storages:
                        raise ValueError("aliased caches between Mamba state layers")
                    storages.add(ptr)
            validated.append(group)
        self.groups = tuple(validated)

    def _check(self) -> None:
        if self.poisoned:
            raise RuntimeError("GDN copy owner poisoned")

    def _preflight_decision(self, decision) -> int:
        if getattr(decision, "stage", None) not in ("T30", "T31"):
            raise ValueError("invalid GDN stage")
        vectors = [
            decision.source_column, decision.destination_column,
            decision.accepted_bias, decision.needs_copy, decision.validity,
        ]
        n = vectors[0].numel()
        if n <= 0 or any(t.device != self.device or t.shape != (n,)
                         for t in vectors):
            raise ValueError("decision vectors must be same-device [batch]")
        if any(t.dtype != torch.int32 for t in vectors[:3]):
            raise ValueError("block columns and bias require int32")
        if vectors[3].dtype != torch.bool or vectors[4].dtype != torch.bool:
            raise ValueError("mask and validity must be bool tensors")
        for group in self.groups:
            if group.block_table.shape[0] < n:
                raise ValueError("Mamba group lacks active batch rows")
        return n

    def execute(self, decision) -> int:
        """Execute T30/T31 with device BT and device-owned decision tensors.

        All source data snapshots and destination collision checks are formed
        before writing any persistent state. CPU numeric guard is fail-closed.
        NPU copy remains intentionally NOT IMPLEMENTED: no device-side
        validity/failure kernel, safe dynamic compaction, or event proof yet.
        """
        self._check()
        if self.device.type == "npu":
            raise NotImplementedError(
                "NPU copy requires audited device validity/collision gate, "
                "32B physical writes, event ownership and dynamic compaction"
            )
        try:
            n = self._preflight_decision(decision)
            if not bool(torch.all(decision.validity)):
                raise ValueError("invalid GDN device decision")
            active_mask = decision.needs_copy
            # torch.nonzero has dynamic output length. CPU gold only:
            # do NOT reuse this for NPU where it could hide a host sync.
            active = torch.nonzero(active_mask).reshape(-1)
            if active.numel() == 0:
                return 0
            src_col = decision.source_column.index_select(0, active).long()
            dst_col = decision.destination_column.index_select(0, active).long()
            bias = decision.accepted_bias.index_select(0, active).long()
            batch_rows = active.long()
            if not bool(torch.all(
                (src_col >= 0) & (dst_col >= 0)
                & (bias >= 0) & (bias <= 7)
            )):
                raise ValueError("active GDN src/dst/bias out of bounds")

            writes: list[_Write310] = []
            for group in self.groups:
                table = group.block_table
                cols = table.shape[1]
                in_bounds = ((src_col < cols) & (dst_col < cols)
                             & ((src_col + bias) < cols))
                if not bool(torch.all(in_bounds)):
                    raise ValueError("Mamba block-table column overflow")

                def table_select(column: torch.Tensor) -> torch.Tensor:
                    # Device int32 BT is addressed with device indices,
                    # never converted to a Python per-request move.
                    return table[batch_rows, column].long()

                src_blocks = table_select(src_col)
                dst_blocks = table_select(dst_col)
                temporal_src_blocks = table_select(src_col + bias)
                for layer in group.layers:
                    conv, temporal = layer.conv, layer.temporal
                    valid_ids = (
                        (src_blocks < conv.shape[0]) & (src_blocks >= 0)
                        & (dst_blocks < conv.shape[0]) & (dst_blocks >= 0)
                        & (temporal_src_blocks < temporal.shape[0])
                        & (temporal_src_blocks >= 0)
                        & (dst_blocks < temporal.shape[0])
                    )
                    if not bool(torch.all(valid_ids)):
                        raise ValueError("invalid Mamba physical block id")
                    # Both state types independently use their group's
                    # destination BT column. No two active requests may write
                    # the same physical target within a tensor.
                    if torch.unique(dst_blocks).numel() != dst_blocks.numel():
                        raise ValueError("multiple requests target same GDN block")

                    # Even if no copy is needed for a specific state, creating
                    # every source snapshot before writing preserves alias safety.
                    old_conv_src = conv.index_select(0, src_blocks).clone()
                    old_conv_dst = conv.index_select(0, dst_blocks).clone()
                    width = conv.shape[1 if layer.conv_layout == "SD" else 2]
                    nactive = active.numel()
                    pos = torch.arange(width, dtype=torch.long, device=self.device)
                    pos = pos.unsqueeze(0).expand(nactive, width)
                    from_pos = (pos + bias[:, None]).clamp(max=width - 1)
                    copy_mask = pos + bias[:, None] < width
                    if layer.conv_layout == "SD":
                        dim = conv.shape[2]
                        expanded = from_pos[:, :, None].expand(-1, -1, dim)
                        sliding = torch.gather(old_conv_src, 1, expanded)
                        merged = torch.where(
                            copy_mask[:, :, None], sliding, old_conv_dst
                        )
                    else:
                        dim = conv.shape[1]
                        expanded = from_pos[:, None, :].expand(-1, dim, -1)
                        sliding = torch.gather(old_conv_src, 2, expanded)
                        merged = torch.where(
                            copy_mask[:, None, :], sliding, old_conv_dst
                        )
                    # full-block writes for the entire active set; a no-op
                    # shift with bias=0/src=dst still reproduces old values.
                    writes.append(_Write310(conv, dst_blocks, merged.contiguous()))
                    temporal_src = temporal.index_select(
                        0, temporal_src_blocks).clone()
                    writes.append(_Write310(
                        temporal, dst_blocks, temporal_src
                    ))
            for i, work in enumerate(writes):
                if self._test_fail_after_write == i:
                    raise RuntimeError("injected inter-group GDN commit failure")
                work.storage.index_copy_(0, work.indices, work.replacement)
            return len(writes)
        except Exception:
            self.poisoned = True
            raise

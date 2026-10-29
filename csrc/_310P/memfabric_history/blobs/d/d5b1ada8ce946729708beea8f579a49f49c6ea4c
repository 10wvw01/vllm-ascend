"""310P3 Qwen3.5 GDN T30-A/T31-A *device-owned* metadata candidate.

Unregistered, diagnostic-only. Uses only Torch tensor arithmetic and whole
128-byte physical metadata rows. No host scalar readback, Triton or vLLM.
Real NPU op coverage, scheduler ordering and asynchronous ownership are NO-GO.

Contract: active request slots originate from a trusted CPU scheduler snapshot;
the actual token counts/accepted counts/index mapping remain device tensors.
The computed device move decisions CANNOT be consumed by hybrid_state_npu.py's
CPU-GDNMove interface without an unapproved D2H synchronization. On NPU, the
step therefore stays blocked after producing the device decision.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GDNDeviceDecision310:
    source_column: torch.Tensor
    destination_column: torch.Tensor
    accepted_bias: torch.Tensor
    needs_copy: torch.Tensor
    validity: torch.Tensor
    stage: str


@dataclass(frozen=True)
class GDNHybridAttnPayload310:
    num_accepted_tokens: torch.Tensor
    num_decode_draft_tokens_cpu: torch.Tensor


class GDNDeviceMetadata310Candidate:
    """T30-A/T31-A device tensor calculations, 128B row commit, no runtime wire.

    Four int32s per slot: [running_col, accepted, pre_src, pre_bias].
    Eight slots share one 128B contiguous physical row, updated as a unit.
    Device output decisions require a device-native physical copy consumer.
    """

    def __init__(
        self, *, max_reqs: int, block_size: int, k: int = 7, tp: int = 2,
        device: torch.device | str = "cpu",
        allow_npu_diagnostic: bool = False, execution_stream=None,
    ):
        if tp != 2 or k != 7 or max_reqs <= 0 or block_size <= 0:
            raise ValueError("Qwen3.5 GDN requires TP2, K7 and valid capacities")
        if isinstance(device, str) and device.startswith("npu") and not hasattr(torch, "npu"):
            raise NotImplementedError("torch_npu not available")
        self.device = torch.device(device)
        self._stream = execution_stream
        if self.device.type not in ("cpu", "npu"):
            raise NotImplementedError("unsupported device")
        if self.device.type == "npu":
            if not allow_npu_diagnostic or execution_stream is None:
                raise NotImplementedError("NPU diagnostic requires explicit stream approval")
            if not hasattr(torch, "npu"):
                raise NotImplementedError("torch_npu has not registered")
            self._require_stream()
        self.max_reqs = max_reqs
        self.block_size = block_size
        # Materialize the FULL physical row at allocation time. Do not
        # initialize individual persistent int32 fields with 4-byte writes.
        neutral = torch.tensor([-1, 1, -1, 0],
                               dtype=torch.int32, device=self.device)
        self._storage = neutral.view(1, 1, 4).expand(
            (max_reqs + 7) // 8, 8, 4).contiguous().clone()
        self._live: set[int] = set()
        self._phase = "IDLE"
        self._active_slots: tuple[int, ...] = ()
        self._device_pending = False
        self._poisoned = False

    @property
    def device_pending(self) -> bool:
        return self._device_pending

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def _require_stream(self) -> None:
        if self.device.type == "npu":
            current = torch.npu.current_stream(self.device)
            if (getattr(current, "npu_stream", None) is None
                    or current.npu_stream != getattr(self._stream, "npu_stream", None)):
                raise RuntimeError("GDN metadata stream mismatch")

    def _ready(self) -> None:
        if self._poisoned:
            raise RuntimeError("GDN metadata poisoned; discard owner")
        if self._device_pending:
            raise RuntimeError("prior GDN metadata command not yet synchronized")
        self._require_stream()

    def _slots(self, slots: Sequence[int], *, require_live=True) -> tuple[int, ...]:
        result = tuple(slots)
        if (not result or len(result) != len(set(result))
                or any(not isinstance(s, int) or not 0 <= s < self.max_reqs
                       for s in result)):
            raise ValueError("duplicate, empty or invalid request slots")
        if require_live and any(s not in self._live for s in result):
            raise ValueError("request state not initialized")
        return result

    def _indices(self, slots: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(slots, dtype=torch.int64, device=self.device)

    def _read(self, slots: tuple[int, ...]) -> torch.Tensor:
        return self._storage.reshape(-1, 4).index_select(
            0, self._indices(slots))

    def _write_full_rows(
        self, slots: tuple[int, ...], new_values: torch.Tensor
    ) -> None:
        if (new_values.dtype != torch.int32 or new_values.device != self.device
                or new_values.shape != (len(slots), 4)):
            raise ValueError("GDN metadata values must be int32 [num_reqs,4]")
        # All snapshots are materialized BEFORE first persistent metadata write.
        groups: dict[int, list[tuple[int, int]]] = {}
        for i, slot in enumerate(slots):
            groups.setdefault(slot // 8, []).append((slot % 8, i))
        writes: list[tuple[torch.Tensor, torch.Tensor]] = []
        selectors = torch.arange(8, dtype=torch.int64, device=self.device)
        for row, members in groups.items():
            row_index = torch.tensor([row], dtype=torch.int64, device=self.device)
            whole = self._storage.index_select(0, row_index)
            for offset, batch_index in members:
                value = new_values[batch_index:batch_index + 1].unsqueeze(1)
                whole = torch.where(
                    (selectors == offset)[None, :, None], value, whole
                )
            writes.append((row_index, whole))
        try:
            for index, whole in writes:
                self._storage.index_copy_(0, index, whole)
            if writes and self.device.type == "npu":
                self._device_pending = True
        except Exception:
            self._poisoned = True
            raise

    def confirm_completed(self) -> None:
        if self._poisoned:
            raise RuntimeError("GDN metadata poisoned")
        if self.device.type == "npu" and self._device_pending:
            try:
                self._stream.synchronize()
            except Exception:
                self._poisoned = True
                raise
            self._device_pending = False

    def seed_requests(self, slots: Sequence[int], computed_tokens: Sequence[int]) -> None:
        self._ready()
        if self._phase != "IDLE":
            raise RuntimeError("cannot seed request while batch is active")
        slots = self._slots(slots, require_live=False)
        if (len(computed_tokens) != len(slots)
                or any(s in self._live for s in slots)
                or any(not isinstance(n, int) or n < 0 for n in computed_tokens)):
            raise ValueError("invalid request initialization")
        vals = torch.tensor(
            [[(n - 1) // self.block_size, 1, -1, 0] for n in computed_tokens],
            dtype=torch.int32, device=self.device,
        )
        self._write_full_rows(slots, vals)
        self._live.update(slots)

    def release_requests(self, slots: Sequence[int]) -> None:
        self._ready()
        if self._phase != "IDLE":
            raise RuntimeError("cannot release in-flight GDN state")
        slots = self._slots(slots)
        neutral = torch.tensor(
            [[-1, 1, -1, 0] for _ in slots],
            dtype=torch.int32, device=self.device,
        )
        self._write_full_rows(slots, neutral)
        self._live.difference_update(slots)

    def _tensor(self, x: torch.Tensor, shape: tuple[int, ...]) -> None:
        if (x.device != self.device or x.dtype != torch.int32
                or tuple(x.shape) != shape):
            raise ValueError("device-owned int32 tensor ABI mismatch")

    def _validate_device(self, valid: torch.Tensor) -> None:
        # CPU test can enforce numeric constraints; NPU never reads data to
        # host. A device-side validity gate before copy/forward is still P0.
        if self.device.type == "cpu" and not bool(torch.all(valid)):
            raise ValueError("invalid dynamic GDN token/slot metadata")

    def preprocess_device(
        self, slots: Sequence[int], idx_mapping: torch.Tensor,
        num_computed_tokens: torch.Tensor, query_start_loc: torch.Tensor,
    ) -> GDNDeviceDecision310:
        self._ready()
        if self._phase != "IDLE":
            raise RuntimeError("preprocess must start in IDLE")
        slots = self._slots(slots)
        n = len(slots)
        self._tensor(idx_mapping, (n,))
        self._tensor(num_computed_tokens, (self.max_reqs,))
        self._tensor(query_start_loc, (n + 1,))
        mapping = self._indices(slots)
        before = self._read(slots)
        old_col, accepted = before[:, 0], before[:, 1]
        computed = num_computed_tokens.index_select(0, mapping)
        lengths = query_start_loc[1:] - query_start_loc[:-1]
        after = computed + lengths
        dst = torch.div(after + self.block_size - 1, self.block_size,
                        rounding_mode="floor") - 1
        crossed = (old_col >= 0) & (old_col != dst)
        valid = ((idx_mapping == mapping.to(torch.int32))
                 & (accepted >= 1) & (accepted <= 8)
                 & (computed >= 0) & (lengths > 0) & (dst >= 0))
        self._validate_device(valid)
        bias = torch.clamp(accepted - 1, min=0)
        new_accepted = torch.where(crossed, torch.ones_like(accepted), accepted)
        new_values = torch.stack(
            (dst, new_accepted, old_col, bias), dim=1).to(torch.int32)
        self._write_full_rows(slots, new_values)
        self._active_slots = slots
        self._phase = "PRECOPY_DECISION_PENDING"
        return GDNDeviceDecision310(
            old_col, dst, bias, crossed, valid, "T30"
        )

    def confirm_precopy_for_cpu_gold(self) -> None:
        """CPU-only test boundary, NOT a NPU production copy acknowledgement."""
        self._ready()
        if self.device.type != "cpu" or self._phase != "PRECOPY_DECISION_PENDING":
            raise NotImplementedError("NPU needs device-native move consumer and fence")
        self._phase = "READY_FOR_ATTENTION"

    def prepare_attn_payload(
        self, slots: Sequence[int], draft_counts: Sequence[int],
        scheduled_tokens: Sequence[int],
    ) -> GDNHybridAttnPayload310:
        self._ready()
        slots = self._slots(slots)
        if self._phase != "READY_FOR_ATTENTION" or slots != self._active_slots:
            raise RuntimeError("precopy must complete before GDN prepare_attn")
        if (len(draft_counts) != len(slots)
                or len(scheduled_tokens) != len(slots)
                or any(not isinstance(d, int) or d < 0 or d > 7
                       or not isinstance(s, int) or s <= 0
                       for d, s in zip(draft_counts, scheduled_tokens))):
            raise ValueError("invalid scheduler DFlash K7 batch metadata")
        # Frozen MambaHybridAttnMetadata sentinel: >=0 is speculative decode.
        counts = torch.tensor(
            [d if d > 0 and s == d + 1 else -1
             for d, s in zip(draft_counts, scheduled_tokens)],
            dtype=torch.int32, device="cpu"
        )
        accepted = self._read(slots)[:, 1].contiguous()
        return GDNHybridAttnPayload310(accepted, counts)

    def postprocess_device(
        self, slots: Sequence[int], idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor, computed_after_update: torch.Tensor,
    ) -> GDNDeviceDecision310:
        self._ready()
        slots = self._slots(slots)
        if self._phase != "READY_FOR_ATTENTION" or slots != self._active_slots:
            raise RuntimeError("postprocess must follow pre-copy and forward")
        n = len(slots)
        self._tensor(idx_mapping, (n,))
        self._tensor(num_sampled, (n,))
        self._tensor(computed_after_update, (self.max_reqs,))
        indices = self._indices(slots)
        computed = computed_after_update.index_select(0, indices)
        state = self._read(slots)
        col = state[:, 0]
        accepted = torch.maximum(num_sampled, torch.ones_like(num_sampled))
        running = computed - accepted + 1
        aligned = torch.div(computed, self.block_size,
                            rounding_mode="floor") * self.block_size
        eligible = aligned >= running
        dest = torch.div(aligned, self.block_size, rounding_mode="floor") - 1
        bias = aligned - running
        needs = eligible & ((col != dest) | (bias != 0))
        valid = ((idx_mapping == indices.to(torch.int32))
                 & (num_sampled >= 0) & (num_sampled <= 8)
                 & (computed >= 0) & (running >= 0)
                 & ((~eligible) | ((col >= 0) & (dest >= 0)
                                    & (bias >= 0) & (bias <= 7))))
        self._validate_device(valid)
        next_accepted = torch.where(
            eligible & (col == dest),
            torch.ones_like(accepted), accepted
        )
        new_values = torch.stack(
            (col, next_accepted, state[:, 2], state[:, 3]), dim=1).to(torch.int32)
        self._write_full_rows(slots, new_values)
        self._phase = "POSTCOPY_DECISION_PENDING"
        return GDNDeviceDecision310(
            col, dest, bias, needs, valid, "T31"
        )

    def confirm_postcopy_for_cpu_gold(self) -> None:
        self._ready()
        if self.device.type != "cpu" or self._phase != "POSTCOPY_DECISION_PENDING":
            raise NotImplementedError("NPU needs device-native move consumer and fence")
        self._active_slots = ()
        self._phase = "IDLE"

    def snapshot_for_cpu_test(self, slot: int) -> tuple[int, int, int, int]:
        self._ready()
        if self.device.type != "cpu" or slot not in self._live:
            raise NotImplementedError("device scalar readback forbidden")
        return tuple(int(v) for v in self._storage.reshape(-1, 4)[slot])

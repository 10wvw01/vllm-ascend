"""Target-only unregistered ModelState ABI sequence adapter (CPU GOLD ONLY).

Frozen signature mirror:
  add_request(req_index, new_req_data)
  preprocess_state(input_batch, block_tables, kv_cache_config, num_computed)
  prepare_attn(input_batch, cudagraph_mode, block_tables, slot_mappings,
               attn_groups, kv_cache_config, for_capture=False)
  postprocess_state(idx_mapping, num_sampled, num_computed_after_post_update)

NOT a replacement for ModelState: it returns GDN-only metadata, does not
build actual attention backend metadata or execute Target forward. 310P NPU
still needs device invalidity/overlap-safe copy protocol and event ownership.
"""
from __future__ import annotations

from collections.abc import Sequence
import torch


class TargetGDNModelStateSequence310Candidate:
    """Exercise whole T30/prepare-attn/T31 ordering without Runner registration."""

    def __init__(self, metadata, copy_consumer):
        if metadata.device.type != "cpu" or copy_consumer.device.type != "cpu":
            raise NotImplementedError(
                "unregistered ModelState sequence is CPU gold only; "
                "NPU event and validity gates are not approved"
            )
        self.metadata = metadata
        self.copy_consumer = copy_consumer
        self._failed = False
        self._active_slots: tuple[int, ...] = ()
        self._attn_ready = False
        self._forward_complete = False

    def _healthy(self):
        if self._failed or self.metadata.poisoned or self.copy_consumer.poisoned:
            raise RuntimeError("Target GDN ModelState sequence poisoned")

    @staticmethod
    def _slots(input_batch) -> tuple[int, ...]:
        # idx_mapping_np is the *scheduler-controlled row->slot* binding;
        # actual device idx_mapping is independently compared by metadata.
        return tuple(int(s) for s in input_batch.idx_mapping_np)

    def _check_block_tables(self, block_tables: Sequence[torch.Tensor]) -> None:
        if len(block_tables) != len(self.copy_consumer.groups):
            raise ValueError("ModelState did not receive every Mamba KV group")
        for supplied, group in zip(block_tables, self.copy_consumer.groups):
            if (supplied.device != group.block_table.device
                    or supplied.data_ptr() != group.block_table.data_ptr()
                    or supplied.shape != group.block_table.shape):
                raise ValueError(
                    "ModelState BT must be same authoritative tensor for group"
                )

    def add_request(self, req_index: int, new_req_data) -> None:
        self._healthy()
        if self._active_slots:
            raise RuntimeError("cannot add request during active GDN batch")
        self.metadata.seed_requests([req_index], [new_req_data.num_computed_tokens])

    def remove_request(self, req_index: int) -> None:
        self._healthy()
        if self._active_slots:
            raise RuntimeError("cannot remove request during active GDN batch")
        self.metadata.release_requests([req_index])

    def preprocess_state(
        self, input_batch, block_tables, kv_cache_config,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        self._healthy()
        if self._active_slots:
            raise RuntimeError("previous GDN batch unfinished")
        self._check_block_tables(block_tables)
        slots = self._slots(input_batch)
        if len(slots) != input_batch.num_reqs:
            raise ValueError("active batch count mismatch")
        try:
            decisions = self.metadata.preprocess_device(
                slots, input_batch.idx_mapping,
                num_computed_tokens, input_batch.query_start_loc
            )
            self.copy_consumer.execute(decisions)
            self.metadata.confirm_precopy_for_cpu_gold()
            self._active_slots = slots
            self._attn_ready = False
            self._forward_complete = False
        except Exception:
            self._failed = True
            raise

    def prepare_attn(
        self, input_batch, cudagraph_mode, block_tables, slot_mappings,
        attn_groups, kv_cache_config, for_capture=False,
    ):
        """GDN-only payload; no actual build_attn_metadata backend call."""
        self._healthy()
        if for_capture or str(cudagraph_mode).upper() not in ("NONE", "EAGER"):
            raise NotImplementedError("Target CPU gold only supports Eager")
        self._check_block_tables(block_tables)
        if not self._active_slots or self._slots(input_batch) != self._active_slots:
            raise RuntimeError("GDN preprocess must complete before prepare_attn")
        result = self.metadata.prepare_attn_payload(
            self._active_slots, tuple(input_batch.num_draft_tokens_per_req),
            tuple(input_batch.num_scheduled_tokens),
        )
        self._attn_ready = True
        return result

    def forward_completed_for_cpu_gold(self) -> None:
        """Explicit test-only forward ordering marker; never an NPU event."""
        self._healthy()
        if not self._attn_ready or not self._active_slots:
            raise RuntimeError("attention metadata not prepared")
        self._forward_complete = True

    def postprocess_state(
        self, idx_mapping: torch.Tensor, num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        self._healthy()
        if not self._forward_complete:
            raise RuntimeError("postprocess requires completed forward")
        if num_computed_tokens is None:
            raise ValueError("computed_after_post_update required")
        if isinstance(num_sampled, int):
            if num_sampled != 0:
                raise ValueError("integer num_sampled is only neutral prefill=0")
            num_sampled = torch.zeros(
                len(self._active_slots), dtype=torch.int32,
                device=idx_mapping.device,
            )
        try:
            decision = self.metadata.postprocess_device(
                self._active_slots, idx_mapping, num_sampled,
                num_computed_tokens,
            )
            self.copy_consumer.execute(decision)
            self.metadata.confirm_postcopy_for_cpu_gold()
            self._active_slots = ()
            self._attn_ready = False
            self._forward_complete = False
        except Exception:
            self._failed = True
            raise

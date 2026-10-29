#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pure-stdlib *decision reference* for Qwen3.5 GDN/Mamba align under K7.

Design oracle ONLY: models upstream Triton decision arithmetic and copy
metadata; never copies tensors, executes NPU ops, imports model/Draft code,
or substitutes for frozen vLLM's real ModelState.
"""
from __future__ import annotations

from dataclasses import dataclass

MAX_K7_OUTPUTS = 8


@dataclass(frozen=True)
class PreAlignDecision:
    previous_column: int
    next_column: int
    accepted_bias: int
    reset_accepted_to_one: bool
    needs_copy: bool


@dataclass(frozen=True)
class PostAlignDecision:
    running_tokens: int
    aligned_tokens: int
    source_column: int
    destination_column: int | None
    accepted_bias: int | None
    copy_needed: bool
    reset_accepted_to_one: bool


@dataclass(frozen=True)
class StateCopyShape:
    """Copy contract only; no layer dimensions, allocation, or tensors."""

    kind: str  # CONV or TEMPORAL, from public GDN copy functions
    conv_layout: str = "SD"  # SD or DS in frozen VLLM_SSM_CONV_STATE_LAYOUT
    conv_width: int = 0


@dataclass(frozen=True)
class StateCopyDecision:
    source_block_table_column: int
    destination_block_table_column: int
    source_offset: int
    extent: int | None
    layout: str


def _check_block_size(block_size: int) -> None:
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("invalid GDN align block size")


def _check_accepted(accepted: int) -> None:
    # Only the K7 verification subset; chunked prefill's integer 0 is
    # normalized by ModelState.postprocess_state to 1, before this oracle.
    if not isinstance(accepted, int) or not 1 <= accepted <= MAX_K7_OUTPUTS:
        raise ValueError("K7 accepted count must be 1..8 after normalization")


def init_running_column(computed_tokens: int, block_size: int) -> int:
    """Frozen MambaHybridModelState.add_request seed, including -1 for fresh."""
    _check_block_size(block_size)
    if not isinstance(computed_tokens, int) or computed_tokens < 0:
        raise ValueError("negative or invalid computed count")
    return (computed_tokens - 1) // block_size


def plan_pre_align(
    running_column: int,
    num_computed_tokens: int,
    query_len: int,
    num_accepted_tokens: int,
    block_size: int,
) -> PreAlignDecision:
    """Frozen preprocess_mamba_align_fused_kernel request decision.

    pre-copy source/offset comes from old state_idx and accepted count;
    destination is ceil((computed + query_len)/B)-1.
    """
    _check_block_size(block_size)
    _check_accepted(num_accepted_tokens)
    if (not isinstance(running_column, int) or running_column < -1
            or not isinstance(num_computed_tokens, int)
            or num_computed_tokens < 0
            or not isinstance(query_len, int) or query_len <= 0):
        raise ValueError("invalid request/batch state")
    after = num_computed_tokens + query_len
    next_column = (after + block_size - 1) // block_size - 1
    crossed = running_column >= 0 and running_column != next_column
    return PreAlignDecision(
        previous_column=running_column,
        next_column=next_column,
        accepted_bias=max(num_accepted_tokens - 1, 0),
        reset_accepted_to_one=crossed,
        needs_copy=crossed,
    )


def plan_post_align(
    running_column: int,
    computed_after_post_update: int,
    num_accepted_tokens: int,
    block_size: int,
) -> PostAlignDecision:
    """Frozen V2 postprocess_mamba_fused_kernel PRECOMPUTED_NEW_COMPUTED case.

    Upstream first writes num_sampled into accepted tensor, then postprocess.
    A plan does not mutate the tensor and does not authorize a device copy.
    """
    _check_block_size(block_size)
    _check_accepted(num_accepted_tokens)
    if (not isinstance(running_column, int) or running_column < -1
            or not isinstance(computed_after_post_update, int)
            or computed_after_post_update < 0):
        raise ValueError("invalid computed count or state column")
    running_tokens = computed_after_post_update - num_accepted_tokens + 1
    if running_tokens < 0:
        raise ValueError("inconsistent post-update token accounting")
    aligned = computed_after_post_update // block_size * block_size
    needs_copy = aligned >= running_tokens
    if not needs_copy:
        return PostAlignDecision(
            running_tokens, aligned, running_column, None, None, False, False
        )
    dest = aligned // block_size - 1
    bias = aligned - running_tokens
    if running_column < 0 or dest < 0:
        # Actual frozen kernel has no checked failure for bad BT indices.
        # Device adapter MUST reject this inconsistent request earlier.
        raise ValueError("invalid source/destination for aligned copy")
    if bias < 0 or bias > num_accepted_tokens - 1:
        raise ValueError("inconsistent K7 accepted bias")
    reset_accepted = running_column == dest
    return PostAlignDecision(
        running_tokens, aligned, running_column, dest, bias,
        running_column != dest or bias != 0, reset_accepted
    )


def state_copy_decision(
    shape: StateCopyShape, src_col: int, dst_col: int, bias: int,
) -> StateCopyDecision:
    """Mirror _copy_mamba_state_block's *logical* SD/DS/temporal indexing.

    Physical group-specific block table, tensor base/stride/dtype and a
    potential 310P3 small-write-safe implementation remain separate contracts.
    """
    if (src_col < 0 or dst_col < 0 or bias < 0
            or shape.kind not in ("CONV", "TEMPORAL")
            or shape.conv_layout not in ("SD", "DS")):
        raise ValueError("invalid state copy metadata")
    if shape.kind == "CONV":
        if shape.conv_width <= 0 or bias >= shape.conv_width:
            raise ValueError("invalid convolution window shift")
        return StateCopyDecision(src_col, dst_col, bias,
                                 shape.conv_width - bias, shape.conv_layout)
    if shape.conv_width != 0:
        raise ValueError("temporal state has no convolution width")
    # Temporal: accepted bias advances the source *block-table column*.
    return StateCopyDecision(src_col + bias, dst_col, 0, None, "TEMPORAL")

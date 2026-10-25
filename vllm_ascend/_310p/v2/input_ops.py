"""310P3 V2 Eager candidates for frozen vLLM 0.27.1 T01/T02/T04.

No Triton, CUDA, UVA, or host-side .item()/tolist() synchronization.
These are NOT wired into the 310P Worker/Runner. Device operator support,
stream ordering and 310P3 small-write behavior remain STATIC NO-GO gates.
"""
from __future__ import annotations


def prepare_pos_seq_lens(
    idx_mapping, query_start_loc, num_computed_tokens, pos, seq_lens
) -> None:
    """0.27.1 input_batch.prepare_pos_seq_lens signature and tensor layout.

    Use int32 arithmetic before casting positions to int64: 310P3 int64 Add
    cannot be assumed available. Padded positions and seq lengths are zero.
    """
    import torch

    n = idx_mapping.numel()
    if (idx_mapping.ndim != 1 or query_start_loc.ndim != 1
            or query_start_loc.numel() != n + 1
            or pos.ndim != 1 or seq_lens.ndim != 1
            or seq_lens.numel() < n):
        raise ValueError("invalid 310P V2 position/sequence shapes")
    if any(x.device != pos.device for x in (
        idx_mapping, query_start_loc, num_computed_tokens, seq_lens
    )):
        raise ValueError("T02 inputs must share one device")
    if any(x.dtype != torch.int32 for x in (
        idx_mapping, query_start_loc, num_computed_tokens
    )):
        raise ValueError("T02 indices/counts must be int32")
    if pos.dtype != torch.int64 or seq_lens.dtype != torch.int32:
        raise ValueError("T02 output dtype mismatch")
    seq_lens.zero_()
    pos.zero_()
    if n == 0:
        return
    idx = idx_mapping.to(torch.int64)
    computed = num_computed_tokens.index_select(0, idx)
    lengths = query_start_loc[1:] - query_start_loc[:-1]
    seq_lens[:n].copy_(computed + lengths)
    offsets = torch.arange(pos.numel(), dtype=torch.int32, device=pos.device)
    req_rows = torch.searchsorted(query_start_loc[1:].contiguous(), offsets, right=True)
    valid = (req_rows < n) & (offsets < query_start_loc[-1])
    safe_rows = req_rows.clamp(max=n - 1).to(torch.int64)
    req_computed = computed.index_select(0, safe_rows)
    starts = query_start_loc[:-1].index_select(0, safe_rows)
    values = req_computed + (offsets - starts)
    pos.copy_(torch.where(valid, values, torch.zeros_like(values)).to(torch.int64))


def expand_idx_mapping(
    idx_mapping, total_num_logits: int, cu_num_logits, max_expand_len: int
):
    """0.27.1 T04 expand_idx_mapping: request row and per-request logit offset.

    A pure device searchsorted maps the flat logits to request rows; no
    repeat_interleave dynamic output size or CPU count synchronization.
    """
    import torch

    n = idx_mapping.numel()
    if (idx_mapping.ndim != 1 or cu_num_logits.ndim != 1
            or cu_num_logits.numel() != n + 1
            or total_num_logits < 0 or max_expand_len < 0
            or (total_num_logits > 0 and n == 0)):
        raise ValueError("invalid 310P V2 logits mapping capacity")
    if idx_mapping.device != cu_num_logits.device:
        raise ValueError("T04 tensors must share one device")
    if idx_mapping.dtype != torch.int32 or cu_num_logits.dtype != torch.int32:
        raise ValueError("T04 indices/counts must be int32")
    if total_num_logits == 0:
        return (
            idx_mapping.new_empty(0),
            torch.empty(0, dtype=torch.int32, device=idx_mapping.device),
        )
    offsets = torch.arange(
        total_num_logits, dtype=torch.int32, device=idx_mapping.device
    )
    req_rows = torch.searchsorted(
        cu_num_logits[1:].contiguous(), offsets, right=True
    )
    safe_rows = req_rows.clamp(max=n - 1).to(torch.int64)
    starts = cu_num_logits[:-1].index_select(0, safe_rows)
    # Malformed cu_num_logits must not cause an out-of-range gather. The
    # Runner still needs an explicit contract that the final sum equals
    # total_num_logits; no GPU .item() is allowed to "validate" this here.
    valid = (req_rows < n) & (offsets < cu_num_logits[-1])
    mapped = idx_mapping.index_select(0, safe_rows)
    return (
        torch.where(valid, mapped, torch.full_like(mapped, -1)),
        torch.where(valid, offsets - starts, torch.zeros_like(offsets)),
    )


def prepare_prefill_inputs(
    input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
    all_token_ids, prefill_len, num_computed_tokens,
) -> None:
    """Frozen vLLM 0.27.1 T01 ABI: prefill tokens and lookahead.

    Candidate only. CPU scheduler must validate unique request-state indices,
    nondecreasing query_start_loc, and computed + query_len <= token capacity.
    No device scalar extraction; writes are ordered only by the caller's
    stream. Full input copy and index_copy_ require 310P3 device review.
    """
    import torch

    n = idx_mapping.numel()
    if (idx_mapping.ndim != 1 or input_ids.ndim != 1
            or next_prefill_tokens.ndim != 2 or all_token_ids.ndim != 2
            or prefill_len.ndim != 1 or num_computed_tokens.ndim != 1
            or query_start_loc.ndim != 1 or query_start_loc.numel() != n + 1
            or all_token_ids.shape[0] != prefill_len.numel()
            or prefill_len.numel() != num_computed_tokens.numel()
            or next_prefill_tokens.shape[1] != all_token_ids.shape[0]
            or (n and all_token_ids.shape[1] == 0)):
        raise ValueError("invalid T01 prefill input shapes")
    if any(x.device != input_ids.device for x in (
        next_prefill_tokens, idx_mapping, query_start_loc,
        all_token_ids, prefill_len, num_computed_tokens,
    )):
        raise ValueError("T01 tensors must share one device")
    if any(x.dtype != torch.int32 for x in (
        input_ids, next_prefill_tokens, idx_mapping, query_start_loc,
        all_token_ids, prefill_len, num_computed_tokens,
    )):
        raise ValueError("T01 tensors must be int32")
    if n == 0:
        return

    # Only rows with computed < prefill_len are modified. A mixed batch may
    # contain decode requests; their existing input_ids/lookahead are retained.
    state_idx = idx_mapping.to(torch.int64)
    computed = num_computed_tokens.index_select(0, state_idx)
    prefill = prefill_len.index_select(0, state_idx)
    is_prefill = computed < prefill
    offsets = torch.arange(input_ids.numel(), dtype=torch.int32, device=input_ids.device)
    rows = torch.searchsorted(query_start_loc[1:].contiguous(), offsets, right=True)
    safe_rows = rows.clamp(max=n - 1).to(torch.int64)
    state_for_token = state_idx.index_select(0, safe_rows)
    starts = query_start_loc[:-1].index_select(0, safe_rows)
    computed_for_token = computed.index_select(0, safe_rows)
    prefill_for_token = is_prefill.index_select(0, safe_rows)
    token_pos = computed_for_token + offsets - starts
    width = all_token_ids.shape[1]
    flat_pos = state_for_token * width + token_pos.clamp(0, width - 1).to(torch.int64)
    tokens = all_token_ids.contiguous().reshape(-1).index_select(0, flat_pos)
    valid = (rows < n) & (offsets < query_start_loc[-1]) & prefill_for_token
    input_ids.copy_(torch.where(valid, tokens, input_ids))

    # Layout is [lookahead, req_state], not [req_state, lookahead].
    # Keep inactive/non-prefill state columns unchanged. The original Triton
    # kernel stores zero for lookahead positions beyond prefill_len.
    lookahead = next_prefill_tokens.shape[0]
    if lookahead:
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        look_offsets = torch.arange(
            lookahead, dtype=torch.int32, device=input_ids.device
        )
        positions = computed[:, None] + query_lens[:, None] + look_offsets[None, :]
        selected = all_token_ids.index_select(0, state_idx)
        looked = selected.gather(1, positions.clamp(0, width - 1).to(torch.int64))
        values = torch.where(
            positions < prefill[:, None], looked, torch.zeros_like(looked)
        ).transpose(0, 1)
        previous = next_prefill_tokens.index_select(1, state_idx)
        updated = torch.where(is_prefill[None, :], values, previous)
        next_prefill_tokens.index_copy_(1, state_idx, updated)

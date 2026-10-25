"""310P3 V2 Eager candidates for frozen vLLM 0.27.1 T02/T04.

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

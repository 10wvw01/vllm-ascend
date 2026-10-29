"""T03 vLLM 0.27.1 sampled+draft input assembly Eager *candidate*.

Not production-wired. This uses full input_ids device copy to avoid tiny
scattered writes, but actual 310P3 stream/operator support and CPU-scheduler
cumulative-length invariants remain NO-GO. Never install/pretend Triton.
"""
from __future__ import annotations


def combine_sampled_and_draft_tokens(
    input_ids, idx_mapping, last_sampled_tokens, query_start_loc, seq_lens,
    prefill_len, draft_tokens, cu_num_logits, num_logits: int,
    num_new_sampled_tokens: int = 1,
):
    """Frozen 0.27.1 T03 ABI, including bonus=0/1 and prompt-tail protection.

    Scheduler must prevalidate *values* of cumulative lengths, request indices,
    and num_logits against query capacity before uploading int32 metadata.
    This function validates static shapes/dtypes without device scalar sync.
    """
    import torch

    n = idx_mapping.numel()
    if (num_new_sampled_tokens not in (0, 1) or not isinstance(num_logits, int)
            or num_logits < 0 or idx_mapping.ndim != 1 or input_ids.ndim != 1
            or query_start_loc.ndim != 1 or cu_num_logits.ndim != 1
            or seq_lens.ndim != 1 or prefill_len.ndim != 1
            or draft_tokens.ndim != 2 or last_sampled_tokens.ndim != 2
            or last_sampled_tokens.shape[1] < 1
            or query_start_loc.numel() != n + 1 or cu_num_logits.numel() != n + 1
            or seq_lens.numel() < n or prefill_len.numel() != draft_tokens.shape[0]
            or prefill_len.numel() != last_sampled_tokens.shape[0]):
        raise ValueError("invalid T03 token input shapes")
    if any(x.device != input_ids.device for x in (
        idx_mapping, last_sampled_tokens, query_start_loc, seq_lens,
        prefill_len, draft_tokens, cu_num_logits,
    )):
        raise ValueError("T03 tensors must share a device")
    if (input_ids.dtype != torch.int32
            or any(x.dtype != torch.int32 for x in (
                idx_mapping, query_start_loc, seq_lens, prefill_len, cu_num_logits
            ))
            or last_sampled_tokens.dtype != torch.int64
            or draft_tokens.dtype != torch.int64):
        raise ValueError("invalid T03 token input dtype")
    if num_logits == 0:
        return torch.empty(0, dtype=torch.int64, device=input_ids.device)
    if n == 0:
        raise ValueError("nonempty logits require active requests")
    req_state_idx = idx_mapping.to(torch.int64)
    counts = cu_num_logits[1:] - cu_num_logits[:-1]
    # Fixed output shape: avoid device-scalar host synchronization.
    logit_offsets = torch.arange(num_logits, dtype=torch.int32, device=input_ids.device)
    logit_rows = torch.searchsorted(cu_num_logits[1:].contiguous(), logit_offsets, right=True)
    safe_logit_rows = logit_rows.clamp(max=n - 1).to(torch.int64)
    starts = cu_num_logits[:-1].index_select(0, safe_logit_rows)
    ends = query_start_loc[1:].index_select(0, safe_logit_rows)
    logits_per_row = counts.index_select(0, safe_logit_rows)
    logits_indices = (ends - logits_per_row + logit_offsets - starts).to(torch.int64)

    # Map all physical input slots, including padding, without dynamic-sized
    # repeat_interleave or a CPU .item() from query_start_loc[-1].
    offsets = torch.arange(input_ids.numel(), dtype=torch.int32, device=input_ids.device)
    rows = torch.searchsorted(query_start_loc[1:].contiguous(), offsets, right=True)
    safe_rows = rows.clamp(max=n - 1).to(torch.int64)
    req_idx = req_state_idx.index_select(0, safe_rows)
    row_ends = query_start_loc[1:].index_select(0, safe_rows)
    row_counts = counts.index_select(0, safe_rows)
    draft_count = row_counts - num_new_sampled_tokens
    seq = seq_lens.index_select(0, safe_rows)
    pref = prefill_len.index_select(0, req_idx)
    is_decode = seq > pref
    valid = (rows < n) & (offsets < query_start_loc[-1])
    logits_start = row_ends - row_counts
    bonus_mask = (valid & is_decode & (num_new_sampled_tokens == 1)
                  & (offsets == logits_start) & (seq - row_counts >= pref))
    draft_start = row_ends - draft_count
    draft_mask = (valid & is_decode & (draft_count > 0)
                  & (offsets >= draft_start) & (offsets < row_ends))
    last = last_sampled_tokens[:, 0].index_select(0, req_idx).to(torch.int32)
    width = draft_tokens.shape[1]
    if width:
        local_draft_pos = (offsets - draft_start).clamp(min=0, max=width - 1)
        flat_idx = req_idx * width + local_draft_pos.to(torch.int64)
        draft = draft_tokens.contiguous().reshape(-1).index_select(0, flat_idx).to(torch.int32)
    else:
        draft = torch.zeros_like(input_ids)
    updated = torch.where(draft_mask, draft, input_ids)
    updated = torch.where(bonus_mask, last, updated)
    input_ids.copy_(updated)
    return logits_indices

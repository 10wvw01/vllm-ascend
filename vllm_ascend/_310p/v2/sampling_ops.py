"""Triton-free *candidates* for vLLM 0.27.1 T07/T08 on 310P3.

NOT wired into GPUModelRunner, Sampler, or RejectionSampler. In particular,
RejectionSampler._get_logprobs_tensors binds its module-level Triton kernel;
adding a same-named helper does NOT replace that call. Device searchsorted,
index_select, index_copy/stream behavior and prevalidated scheduler counts
still need NPU review before any runner can use these functions.
"""
from __future__ import annotations


def get_num_sampled_and_rejected(
    num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len
):
    """T07 ABI: mutate num_sampled in place, return it and num_rejected.

    The caller must validate *values* (nonnegative counts, valid req IDs,
    cumulative lengths, sampled<=logits) on the CPU scheduler side before H2D.
    Checking device scalar values here would introduce a forced host sync.
    """
    import torch

    n = idx_mapping.numel()
    tensors = (num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len)
    if (any(x.ndim != 1 or x.dtype != torch.int32 for x in tensors)
            or num_sampled.numel() != n or seq_lens.numel() != n
            or cu_num_logits.numel() != n + 1):
        raise ValueError("T07 requires 1-D int32 tensors and matching request counts")
    if any(x.device != idx_mapping.device for x in tensors):
        raise ValueError("T07 tensors must share a device")
    if n == 0:
        return num_sampled, torch.empty_like(num_sampled)
    indices = idx_mapping.to(torch.int64)
    chunked = seq_lens < prefill_len.index_select(0, indices)
    sampled = torch.where(chunked, torch.zeros_like(num_sampled), num_sampled)
    rejected = torch.where(
        chunked, torch.zeros_like(sampled),
        (cu_num_logits[1:] - cu_num_logits[:-1]) - sampled,
    )
    num_sampled.copy_(sampled)
    return num_sampled, rejected


def flatten_sampled(sampled, num_sampled, cu_num_logits, num_logits: int):
    """T08 candidate: flatten accepted prefixes for RejectionSampler logprobs.

    Nonaccepted positions stay zero exactly as in upstream's zero-initialized
    flat_sampled. Output length is the known host integer num_logits, never a
    data-dependent device scalar; no Triton, .item() or .tolist().
    """
    import torch

    if (sampled.ndim != 2 or num_sampled.ndim != 1 or cu_num_logits.ndim != 1
            or sampled.shape[0] != num_sampled.numel()
            or cu_num_logits.numel() != num_sampled.numel() + 1
            or not isinstance(num_logits, int) or num_logits < 0):
        raise ValueError("invalid T08 sampled/logit shapes")
    if (sampled.dtype not in (torch.int32, torch.int64)
            or num_sampled.dtype != torch.int32 or cu_num_logits.dtype != torch.int32):
        raise ValueError("T08 requires integer sampled IDs and int32 counts")
    if any(x.device != sampled.device for x in (num_sampled, cu_num_logits)):
        raise ValueError("T08 tensors must share a device")
    n, width = sampled.shape
    if num_logits == 0:
        return sampled.new_zeros(0)
    if n == 0 or width == 0:
        raise ValueError("nonempty logits require nonempty sampled matrix")
    offsets = torch.arange(num_logits, dtype=torch.int32, device=sampled.device)
    rows = torch.searchsorted(cu_num_logits[1:].contiguous(), offsets, right=True)
    safe_rows = rows.clamp(max=n - 1).to(torch.int64)
    starts = cu_num_logits[:-1].index_select(0, safe_rows)
    local_pos = offsets - starts
    safe_cols = local_pos.clamp(min=0, max=width - 1).to(torch.int64)
    # Respect sampled row stride by materializing contiguous storage; the
    # upstream Triton kernel reads sampled_stride explicitly.
    indices = safe_rows * width + safe_cols
    values = sampled.contiguous().reshape(-1).index_select(0, indices)
    accepted = num_sampled.index_select(0, safe_rows)
    valid = ((rows < n) & (offsets < cu_num_logits[-1])
             & (local_pos >= 0) & (local_pos < accepted)
             & (local_pos < width))
    return torch.where(valid, values, torch.zeros_like(values))

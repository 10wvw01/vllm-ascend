"""310P3 V2 T09 post-update Eager candidate, pinned to vLLM 0.27.1.

No Triton, UVA or CUDA; all sampled-token data is kept on the accelerator.
CPU request indexes are only used to prevalidate the scheduler's active slot
set, never to synchronize GPU count tensors. Full owned token rows and padded
last-token rows are copied rather than issuing scalar device writes.

IMPORTANT: Dynamic GPU num_sampled/token capacity, torch-npu gather/scatter,
stream visibility and 310P3 small-width 1D index_copy_ are NOT certified.
This module remains behind NPUWorker310's NOT_IMPLEMENTED guard.
"""
from __future__ import annotations


def post_update(
    idx_mapping,
    num_computed_tokens,
    last_sampled_tokens,
    output_bin_counts,
    sampled_tokens,
    num_sampled,
    num_rejected,
    query_start_loc,
    all_token_ids,
    total_len,
    *,
    idx_mapping_np,
    last_sampled_storage,
) -> None:
    """Mirror frozen vLLM GPU V2 post_update on CPU-Torch and future NPU.

    The existing upstream ten positional arguments are preserved. Two
    keyword-only 310P arguments enforce CPU ownership and padded
    last-sampled storage; they must be supplied by NPUModelRunner310V2.
    """
    import numpy as np
    import torch

    n = int(idx_mapping.numel())
    if (idx_mapping.ndim != 1 or sampled_tokens.ndim != 2
            or sampled_tokens.shape[0] != n
            or num_sampled.shape != (n,)
            or num_rejected.shape != (n,)
            or query_start_loc is not None
            and (query_start_loc.ndim != 1 or query_start_loc.numel() != n + 1)
            or all_token_ids.ndim != 2
            or last_sampled_tokens.ndim != 2 or last_sampled_tokens.shape[1] != 1
            or last_sampled_storage.ndim != 2
            or last_sampled_storage.shape[0] != last_sampled_tokens.shape[0]
            or last_sampled_storage.shape[1] * last_sampled_storage.element_size() < 32
            or total_len.ndim != 1
            or num_computed_tokens.ndim != 1
            or total_len.numel() != num_computed_tokens.numel()
            or all_token_ids.shape[0] != total_len.numel()
            or sampled_tokens.shape[1] < 1):
        raise ValueError("invalid 310P T09 state/update shapes")
    if not np.asarray(idx_mapping_np).shape == (n,):
        raise ValueError("CPU scheduler index mapping mismatch")
    if (list(idx_mapping_np) != list(map(int, idx_mapping_np))):
        raise ValueError("non-integral request indices")
    active_rows = [i for i, s in enumerate(idx_mapping_np) if int(s) >= 0]
    active_ids = [int(idx_mapping_np[i]) for i in active_rows]
    if (len(active_ids) != len(set(active_ids))
            or any(s < -1 or s >= total_len.numel() for s in idx_mapping_np)):
        raise ValueError("duplicate or invalid persistent request-state index")
    tensors = (
        idx_mapping, num_computed_tokens, last_sampled_tokens, sampled_tokens,
        num_sampled, num_rejected, all_token_ids, total_len,
        last_sampled_storage,
    )
    if query_start_loc is not None:
        tensors += (query_start_loc,)
    if output_bin_counts is not None:
        tensors += (output_bin_counts,)
    if any(x.device != sampled_tokens.device for x in tensors):
        raise ValueError("all post-update tensors must share a device")
    if (idx_mapping.dtype != torch.int32
            or num_computed_tokens.dtype != torch.int32
            or total_len.dtype != torch.int32
            or all_token_ids.dtype != torch.int32
            or last_sampled_tokens.dtype != torch.int64
            or last_sampled_storage.dtype != torch.int64
            or sampled_tokens.dtype != torch.int64
            or num_sampled.dtype != torch.int32
            or num_rejected.dtype != torch.int32):
        raise ValueError("post-update dtype ABI mismatch")
    if output_bin_counts is not None:
        if (output_bin_counts.ndim != 2 or
                output_bin_counts.shape[0] != total_len.numel()
                or output_bin_counts.dtype != torch.int32):
            raise ValueError("invalid output penalty counts")
    if not active_rows:
        return

    device = sampled_tokens.device
    req_rows = torch.tensor(active_rows, dtype=torch.int64, device=device)
    req_states = torch.tensor(active_ids, dtype=torch.int64, device=device)
    # The CPU scheduler owns row identity, but the frozen GPU V2 call also
    # passes a device idx_mapping. Check they agree BEFORE any state mutation;
    # otherwise a stale/reused request could write another request's tokens.
    torch._assert_async(
        torch.all(idx_mapping.index_select(0, req_rows).to(torch.int64) == req_states),
        "310P T09 scheduler/device request mapping divergence",
    )
    sampled = sampled_tokens.index_select(0, req_rows)
    count = num_sampled.index_select(0, req_rows)
    rejected = num_rejected.index_select(0, req_rows)
    width = sampled.shape[1]
    token_cols = torch.arange(width, dtype=torch.int32, device=device)[None, :]
    accepted = token_cols < count[:, None]
    old_total = total_len.index_select(0, req_states)
    old_computed = num_computed_tokens.index_select(0, req_states)
    if query_start_loc is None:
        query_lens = torch.zeros_like(count)
    else:
        query_lens = (query_start_loc[1:] - query_start_loc[:-1]).index_select(
            0, req_rows
        )
    # Device-resident assertions prevent silent clamp/overwrite of adjacent
    # request rows when speculative counts, sampled IDs or KV positions are
    # corrupt. NPU _assert_async support is still an explicit static blocker.
    torch._assert_async(
        torch.all((count >= 0) & (count <= width) & (rejected >= 0)
                  & (query_lens >= 0)
                  & (old_total >= 0)
                  & (old_total + count <= all_token_ids.shape[1])
                  & (old_computed + query_lens - rejected >= 0)
                  & (old_computed + query_lens - rejected <= all_token_ids.shape[1])),
        "310P T09 invalid sampled/rejected count or state capacity",
    )
    torch._assert_async(
        torch.all((~accepted) | (sampled >= 0)),
        "310P T09 invalid accepted token ID",
    )
    if output_bin_counts is not None:
        torch._assert_async(
            torch.all((~accepted) |
                      ((sampled >= 0) & (sampled < output_bin_counts.shape[1]))),
            "310P T09 accepted token exceeds penalty vocabulary",
        )

    # Entire per-request token rows: no 4-byte NPU slice writes or dynamic
    # GPU-to-CPU count synchronization. The caller owns exclusive live slots.
    rows = all_token_ids.index_select(0, req_states).clone()
    columns = old_total[:, None] + token_cols
    columns = columns.clamp(0, all_token_ids.shape[1] - 1).to(torch.int64)
    old_values = rows.gather(1, columns)
    new_values = torch.where(accepted, sampled.to(torch.int32), old_values)
    rows.scatter_(1, columns, new_values)
    all_token_ids.index_copy_(0, req_states, rows)

    next_total = old_total + count
    total_len.index_copy_(0, req_states, next_total)
    if query_start_loc is None:
        computed_delta = -rejected
    else:
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        computed_delta = query_lens.index_select(0, req_rows) - rejected
    num_computed_tokens.index_copy_(0, req_states, old_computed + computed_delta)

    previous_last = last_sampled_storage.index_select(0, req_states).clone()
    last_index = (count - 1).clamp(min=0).to(torch.int64)[:, None]
    accepted_last = sampled.gather(1, last_index).reshape(-1)
    previous_last[:, 0] = torch.where(
        count > 0, accepted_last, previous_last[:, 0]
    )
    # 310P RequestState stores padded 32-byte token rows. Updating the full
    # backing row also updates the [:, :1] last_sampled_tokens view.
    last_sampled_storage.index_copy_(0, req_states, previous_last)

    if output_bin_counts is not None:
        penalties = output_bin_counts.index_select(0, req_states).clone()
        if penalties.shape[1] == 0:
            raise ValueError("empty vocabulary counts")
        # Masked rejected tokens can be -1; their delta is zero.
        ids = sampled.clamp(0, penalties.shape[1] - 1)
        penalties.scatter_add_(1, ids, accepted.to(penalties.dtype))
        output_bin_counts.index_copy_(0, req_states, penalties)

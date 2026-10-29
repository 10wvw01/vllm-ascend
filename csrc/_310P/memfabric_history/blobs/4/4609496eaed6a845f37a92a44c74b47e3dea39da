"""310P3 V2 T23 greedy-only K7 rejection Eager candidate, NOT wired.

No Triton, no stochastic approximation. For temperature>0 fail closed;
a true Philox/probabilistic rejection implementation is required before GO.
"""
from __future__ import annotations


def rejection_sample(
    target_logits, draft_logits, draft_sampled, cu_num_logits, pos,
    idx_mapping, expanded_idx_mapping, expanded_local_pos, temperature,
    seed, num_speculative_steps, synthetic_conditional_rates=None,
    use_fp64=False, use_block_verification=False,
):
    import torch

    if (num_speculative_steps != 7 or synthetic_conditional_rates is not None
            or use_fp64 or use_block_verification):
        raise NotImplementedError("310P3 T23 only supports standard greedy K7")
    n = idx_mapping.numel()
    if (target_logits.ndim != 2 or target_logits.shape[0] <= 0
            or n <= 0 or cu_num_logits.shape != (n + 1,)
            or draft_sampled.shape != (target_logits.shape[0],)
            or pos.shape != (target_logits.shape[0],)
            or expanded_idx_mapping.shape != (target_logits.shape[0],)
            or expanded_local_pos.shape != (target_logits.shape[0],)
            or idx_mapping.dtype != torch.int32
            or cu_num_logits.dtype != torch.int32
            or temperature.ndim != 1 or seed.ndim != 1):
        raise ValueError("310P3 T23 ABI/shape mismatch")
    dev = target_logits.device
    tensors = (draft_sampled, cu_num_logits, pos, idx_mapping,
               expanded_idx_mapping, expanded_local_pos, temperature, seed)
    if draft_logits is not None:
        tensors += (draft_logits,)
    if any(x.device != dev for x in tensors):
        raise ValueError("310P3 T23 device mismatch")
    torch._assert_async(torch.all((idx_mapping >= 0) &
                                  (idx_mapping < temperature.numel())),
                        "310P3 T23 invalid req state")
    per_req_temp = temperature.index_select(0, idx_mapping.to(torch.int64))
    torch._assert_async(torch.all(per_req_temp == 0),
                        "310P3 T23 stochastic rejection is not implemented")
    lengths = cu_num_logits[1:] - cu_num_logits[:-1]
    torch._assert_async(torch.all((lengths > 0) & (lengths <= 8)) &
                        (cu_num_logits[0] == 0) &
                        (cu_num_logits[-1] == target_logits.shape[0]),
                        "310P3 T23 invalid cumulative logits")
    offsets = torch.arange(8, device=dev, dtype=torch.int64)
    starts = cu_num_logits[:-1].to(torch.int64)
    positions = starts[:, None] + offsets[None, :]
    safe = positions.clamp(0, target_logits.shape[0] - 1)
    target_ids = target_logits.argmax(dim=-1).index_select(
        0, safe.reshape(-1)).reshape(n, 8).to(torch.int64)
    # The draft for step i lives at the NEXT logits position in the
    # flattened input_batch; the bonus position has no next draft token.
    next_ids = draft_sampled.index_select(
        0, (positions + 1).clamp(0, draft_sampled.numel() - 1).reshape(-1)
    ).reshape(n, 8)
    valid_draft = offsets[None, :] < lengths[:, None] - 1
    mismatch = valid_draft & (target_ids != next_ids)
    first_bad = torch.where(mismatch, offsets[None, :],
                            torch.full_like(positions, 8)).min(dim=1).values
    accepted_draft = torch.minimum(first_bad, lengths.to(torch.int64) - 1)
    count = (accepted_draft + 1).to(torch.int32)
    # Frozen greedy path returns target argmax for accepted prefix and first
    # mismatch/bonus. Zero the undefined tail deterministically.
    sampled = torch.where(offsets[None, :] < count[:, None],
                          target_ids, torch.zeros_like(target_ids))
    return sampled, count

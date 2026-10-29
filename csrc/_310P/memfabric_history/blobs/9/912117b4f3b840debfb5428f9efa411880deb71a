"""Frozen vLLM 0.27.1 DFlash T11 prepare Eager candidate for 310P3.

One call per draft KV group; only per-group slot buffers differ. The
original positional ABI is preserved. CPU-tested Torch candidate, NOT a
certified 310P3 runtime. NPU support for _assert_async, gather, full-buffer
D2D copies and stream ownership remains a static-admission blocker.
"""
from __future__ import annotations


def prepare_dflash_inputs(
    input_buffers, query_slot_mapping, context_positions,
    context_slot_mapping, sample_indices, sample_pos, sample_idx_mapping,
    temperature, seeds, input_batch, num_sampled, num_rejected,
    last_sampled, next_prefill_tokens, input_temperature, input_seeds,
    block_table, block_size: int, parallel_drafting_token_id: int,
    num_query_per_req: int, num_speculative_steps: int, max_num_reqs: int,
    max_num_tokens: int, max_model_len: int, sample_from_anchor: bool = False,
) -> None:
    """Construct K7 anchor/mask and context slots without Triton or UVA."""
    import numpy as np
    import torch

    if sample_from_anchor:
        raise NotImplementedError("310P3 T11 supports DFlash, not DSpark anchor sampling")
    if (num_speculative_steps != 7 or num_query_per_req != 8
            or max_model_len <= 0 or block_size <= 0
            or max_num_reqs <= 0 or max_num_tokens <= 0):
        raise ValueError("unsupported 310P3 DFlash K7 geometry")
    n = int(input_batch.num_reqs)
    if not (0 < n <= max_num_reqs and n * num_query_per_req <= max_num_tokens):
        raise ValueError("invalid DFlash request/token capacity")
    offsets = np.asarray(input_batch.query_start_loc_np)
    req_ids = np.asarray(input_batch.idx_mapping_np)
    if (offsets.shape != (n + 1,) or req_ids.shape != (n,)
            or not np.issubdtype(offsets.dtype, np.integer)
            or not np.issubdtype(req_ids.dtype, np.integer)
            or offsets[0] != 0 or np.any(np.diff(offsets) <= 0)
            or offsets[-1] > max_num_tokens
            or np.any(req_ids < 0) or np.any(req_ids >= max_num_reqs)
            or len(set(req_ids.tolist())) != n):
        raise ValueError("invalid DFlash CPU scheduler request layout")
    nctx = int(offsets[-1])
    if (block_table.ndim != 2 or block_table.shape[0] < n
            or block_table.shape[1] < (max_model_len + block_size - 1) // block_size):
        raise ValueError("DFlash group block table has insufficient width")

    device = input_batch.positions.device
    tensors = (
        input_buffers.input_ids, input_buffers.positions,
        input_buffers.query_start_loc, input_buffers.seq_lens,
        query_slot_mapping, context_positions, context_slot_mapping,
        sample_indices, sample_pos, sample_idx_mapping,
        temperature, seeds, input_batch.positions, input_batch.query_start_loc,
        input_batch.idx_mapping, num_sampled, num_rejected, last_sampled,
        next_prefill_tokens, input_temperature, input_seeds, block_table,
    )
    if any(t.device != device for t in tensors):
        raise ValueError("DFlash T11 device mismatch")
    if (input_batch.positions.dtype != torch.int64
            or input_batch.idx_mapping.dtype != torch.int32
            or input_batch.query_start_loc.dtype != torch.int32
            or num_sampled.dtype != torch.int32
            or num_rejected.dtype != torch.int32
            or block_table.dtype != torch.int32
            or input_buffers.input_ids.dtype != torch.int32
            or input_buffers.positions.dtype != torch.int64
            or input_buffers.query_start_loc.dtype != torch.int32
            or input_buffers.seq_lens.dtype != torch.int32
            or query_slot_mapping.dtype != torch.int32
            or context_slot_mapping.dtype != torch.int32
            or context_positions.dtype != torch.int64
            or sample_indices.dtype != torch.int64
            or sample_pos.dtype != torch.int64
            or sample_idx_mapping.dtype != torch.int32):
        raise ValueError("DFlash T11 dtype ABI mismatch")
    if (num_sampled.shape != (n,) or num_rejected.shape != (n,)
            or input_batch.positions.numel() < nctx
            or input_batch.query_start_loc.numel() < n + 1
            or input_batch.idx_mapping.numel() < n
            or input_buffers.input_ids.numel() < max_num_tokens
            or input_buffers.positions.numel() < max_num_tokens
            or input_buffers.query_start_loc.numel() < max_num_reqs + 1
            or input_buffers.seq_lens.numel() < max_num_reqs
            or query_slot_mapping.numel() < max_num_tokens
            or context_positions.numel() < max_num_tokens
            or context_slot_mapping.numel() < max_num_tokens
            or sample_indices.numel() < max_num_reqs * num_speculative_steps
            or sample_pos.numel() < max_num_reqs * num_speculative_steps
            or sample_idx_mapping.numel() < max_num_reqs * num_speculative_steps
            or temperature.numel() < max_num_reqs or seeds.numel() < max_num_reqs
            or input_temperature.numel() < max_num_reqs
            or input_seeds.numel() < max_num_reqs
            or last_sampled.numel() < max_num_reqs
            or next_prefill_tokens.numel() < max_num_reqs):
        raise ValueError("DFlash T11 output/input buffer shape mismatch")

    q = num_query_per_req
    req_index = torch.tensor(req_ids.astype(np.int64), device=device)
    ends = torch.tensor(offsets[1:].astype(np.int64), device=device)
    lengths = torch.tensor(np.diff(offsets).astype(np.int64), device=device)
    # No .item() or host sync on device-resident sample/rejection counts.
    # The actual torch-npu _assert_async contract is still unverified.
    torch._assert_async(torch.all((num_rejected >= 0) &
                                  (num_rejected.to(torch.int64) < lengths)),
                        "DFlash T11 invalid rejected token count")
    torch._assert_async(torch.all((num_sampled >= 0) &
                                  (num_sampled <= q)),
                        "DFlash T11 invalid sampled token count")
    last_indices = ends - num_rejected.to(torch.int64) - 1
    last_positions = input_batch.positions.index_select(0, last_indices)
    query_offsets = torch.arange(q, device=device, dtype=torch.int64)
    query_pos = last_positions[:, None] + 1 + query_offsets[None, :]
    torch._assert_async(torch.all((query_pos >= 0) &
                                  (query_pos < max_model_len)),
                        "DFlash T11 query position exceeds KV capacity")
    context_pos = input_batch.positions[:nctx]
    torch._assert_async(torch.all((context_pos >= 0) &
                                  (context_pos < max_model_len)),
                        "DFlash T11 context position exceeds KV capacity")
    # CPU offsets determine shapes only; dynamic GPU counts never move to host.
    ctx_req = torch.tensor(np.repeat(np.arange(n), np.diff(offsets)),
                           dtype=torch.int64, device=device)

    def slot_ids(positions, rows):
        block_indices = positions.to(torch.int64) // block_size
        selected = block_table.index_select(0, rows)
        ids = selected.gather(1, block_indices[:, None]).squeeze(1)
        torch._assert_async(torch.all(ids >= 0),
                            "DFlash T11 unallocated KV block")
        return (ids.to(torch.int64) * block_size +
                positions.to(torch.int64) % block_size).to(torch.int32)

    context_slots = slot_ids(context_pos, ctx_req)
    query_rows = torch.arange(n, device=device).repeat_interleave(q)
    query_slots = slot_ids(query_pos.reshape(-1), query_rows)
    bonus = torch.where(
        num_sampled > 0,
        last_sampled.reshape(-1).index_select(0, req_index),
        next_prefill_tokens.reshape(-1).index_select(0, req_index),
    ).to(torch.int32)
    query_ids = torch.where(
        torch.arange(q, device=device)[None, :] == 0,
        bonus[:, None],
        torch.full((n, q), parallel_drafting_token_id,
                   dtype=torch.int32, device=device),
    )

    # Construct full buffers then copy once; no partial persistent output
    # write. NPU D2D semantics and stream ordering remain NO-GO evidence.
    def full_copy(dest, prefix, fill):
        length = prefix.numel()
        full = torch.cat((
            prefix.to(dest.dtype).reshape(-1),
            torch.full((dest.numel() - length,), fill,
                       dtype=dest.dtype, device=device),
        ))
        dest.copy_(full)

    full_copy(input_buffers.input_ids, query_ids.reshape(-1), 0)
    full_copy(input_buffers.positions,
              query_pos.clamp(max=max_model_len - 1).reshape(-1), 0)
    full_copy(input_buffers.query_start_loc,
              torch.arange(max_num_reqs + 1, dtype=torch.int32, device=device)
              .clamp(max=n) * q, 0)
    full_copy(input_buffers.seq_lens,
              torch.cat([last_positions.to(torch.int32) + 1 + q,
                         torch.zeros(max_num_reqs - n, dtype=torch.int32,
                                     device=device)]), 0)
    full_copy(query_slot_mapping, query_slots, -1)
    full_copy(context_positions, context_pos, 0)
    full_copy(context_slot_mapping, context_slots, -1)

    sample_idx = (torch.arange(n, device=device, dtype=torch.int64)[:, None] * q
                  + torch.arange(1, q, device=device, dtype=torch.int64)[None, :])
    sample_states = req_index[:, None].expand(n, num_speculative_steps)
    full_copy(sample_indices, sample_idx.reshape(-1), 0)
    full_copy(sample_pos, query_pos[:, 1:].reshape(-1), 0)
    full_copy(sample_idx_mapping, sample_states.reshape(-1), -1)

    new_temperature = temperature.clone()
    new_seeds = seeds.clone()
    new_temperature.index_copy_(0, req_index,
                                input_temperature.index_select(0, req_index))
    new_seeds.index_copy_(0, req_index,
                          input_seeds.index_select(0, req_index))
    temperature.copy_(new_temperature)
    seeds.copy_(new_seeds)

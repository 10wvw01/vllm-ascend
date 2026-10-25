"""Pure-Python CPU golden for 310P3 mixed-KV DFlash V2 input semantics.

This module is a *reference contract*, not an NPU implementation or runtime
patch. No torch import: the project-control CI intentionally has no torch-npu.
It models the frozen vLLM 0.27.1 sample_from_anchor=False K7 layout and
per-group physical slot addresses without aliasing old V1 runner internals.
"""
from dataclasses import dataclass


PAD_SLOT_ID = -1


@dataclass(frozen=True)
class DFlashGolden:
    input_ids: list[int]
    query_positions: list[int]
    query_start_loc: list[int]
    seq_lens: list[int]
    context_positions: list[int]
    context_slots: list[list[int]]
    query_slots: list[list[int]]
    sample_indices: list[int]
    sample_positions: list[int]
    sample_idx_mapping: list[int]


def physical_slot(position: int, block_ids: list[int], block_size: int) -> int:
    """Follow vLLM V2's physical per-group kernel-block mapping exactly."""
    if block_size <= 0 or position < 0:
        raise ValueError("block_size must be positive and position nonnegative")
    i, offset = divmod(position, block_size)
    if i >= len(block_ids) or block_ids[i] < 0:
        raise ValueError("out-of-range or unallocated physical block")
    return int(block_ids[i]) * block_size + offset


def build_k7_inputs(
    *,
    target_positions: list[int],
    target_query_start_loc: list[int],
    idx_mapping: list[int],
    last_sampled: list[int],
    next_prefill_tokens: list[int],
    num_sampled: list[int],
    num_rejected: list[int],
    group_block_tables: list[list[list[int]]],
    group_kernel_block_sizes: list[int],
    parallel_drafting_token_id: int,
    speculative_steps: int = 7,
    max_model_len: int = 8192,
) -> DFlashGolden:
    """Build reference context/query mappings separately for every KV group.

    Args:
      group_block_tables: [group][batch_req][logical_block_index] -> physical ID.
      idx_mapping: batch_req -> persistent request-state index.
      target_positions: concatenated, per-request token positions from Target.
      num_rejected: rejected Target tokens at tail of each request this step.
    """
    num_reqs = len(idx_mapping)
    n = speculative_steps
    if n <= 0 or max_model_len <= 0:
        raise ValueError("bad DFlash configuration")
    if len(target_query_start_loc) != num_reqs + 1:
        raise ValueError("invalid context offsets")
    if target_query_start_loc[0] != 0 or target_query_start_loc[-1] != len(target_positions):
        raise ValueError("context offset span mismatch")
    if len(num_sampled) != num_reqs or len(num_rejected) != num_reqs:
        raise ValueError("invalid sampling counts")
    if len(group_block_tables) != len(group_kernel_block_sizes) or not group_block_tables:
        raise ValueError("missing KV groups")
    for group in group_block_tables:
        if len(group) != num_reqs:
            raise ValueError("KV group request count mismatch")
    q_inputs = []
    q_positions = []
    ctx_positions = list(target_positions)
    ctx_slots = [[] for _ in group_kernel_block_sizes]
    q_slots = [[] for _ in group_kernel_block_sizes]
    seq_lens = []
    sample_indices = []
    sample_positions = []
    sample_idx_mapping = []
    query_width = 1 + n
    for req in range(num_reqs):
        req_state = idx_mapping[req]
        if req_state < 0 or req_state >= len(last_sampled) or req_state >= len(next_prefill_tokens):
            raise ValueError("request state index out of range")
        a, b = target_query_start_loc[req : req + 2]
        if not (0 <= a < b <= len(ctx_positions)):
            raise ValueError("empty or invalid context span")
        rejected = num_rejected[req]
        sampled = num_sampled[req]
        if rejected < 0 or rejected >= b - a or sampled < 0:
            raise ValueError("invalid sample/rejection count")
        # Upstream reads the last *accepted* Target position. Rejected tail
        # context positions still appear in the precompute input.
        base = ctx_positions[b - rejected - 1] + 1
        if base < 0:
            raise ValueError("invalid absolute sequence position")
        bonus = last_sampled[req_state] if sampled > 0 else next_prefill_tokens[req_state]
        q_inputs.extend([bonus] + [parallel_drafting_token_id] * n)
        query_p = [base + offset for offset in range(query_width)]
        q_positions.extend(min(position, max_model_len - 1) for position in query_p)
        seq_lens.append(base + query_width)
        for step in range(1, query_width):
            sample_indices.append(req * query_width + step)  # index into concatenated query batch
            sample_positions.append(query_p[step])
            sample_idx_mapping.append(req_state)
        for g, block_size in enumerate(group_kernel_block_sizes):
            table = group_block_tables[g][req]
            ctx_slots[g].extend(physical_slot(pos, table, block_size) for pos in ctx_positions[a:b])
            q_slots[g].extend(physical_slot(pos, table, block_size) for pos in query_p)
    return DFlashGolden(
        input_ids=q_inputs,
        query_positions=q_positions,
        query_start_loc=[req * query_width for req in range(num_reqs + 1)],
        seq_lens=seq_lens,
        context_positions=ctx_positions,
        context_slots=ctx_slots,
        query_slots=q_slots,
        sample_indices=sample_indices,
        sample_positions=sample_positions,
        sample_idx_mapping=sample_idx_mapping,
    )

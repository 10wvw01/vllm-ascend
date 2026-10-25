"""No-Torch CPU reference for frozen vLLM 0.27.1 T02/T04 input ABI."""
from __future__ import annotations


def _validate_cumulative(cumulative: list[int], n: int, capacity: int) -> None:
    if (len(cumulative) != n + 1 or cumulative[0] != 0
            or any(a > b for a, b in zip(cumulative, cumulative[1:]))
            or cumulative[-1] > capacity):
        raise ValueError("invalid cumulative lengths")


def positions_and_lengths(
    idx_mapping: list[int],
    query_start_loc: list[int],
    num_computed_tokens: list[int],
    padded_tokens: int,
    padded_reqs: int,
) -> tuple[list[int], list[int]]:
    if padded_tokens < 0 or padded_reqs < len(idx_mapping):
        raise ValueError("invalid padding")
    _validate_cumulative(query_start_loc, len(idx_mapping), padded_tokens)
    if any(i < 0 or i >= len(num_computed_tokens) for i in idx_mapping):
        raise ValueError("invalid request index")
    positions = [0] * padded_tokens
    seq_lens = [0] * padded_reqs
    for r, idx in enumerate(idx_mapping):
        start, end = query_start_loc[r:r + 2]
        computed = num_computed_tokens[idx]
        if computed < 0:
            raise ValueError("negative computed tokens")
        seq_lens[r] = computed + end - start
        positions[start:end] = range(computed, computed + end - start)
    return positions, seq_lens


def expanded_logits(
    idx_mapping: list[int],
    cu_num_logits: list[int],
    total_num_logits: int,
    max_expand_len: int,
) -> tuple[list[int], list[int]]:
    if total_num_logits < 0 or max_expand_len < 0:
        raise ValueError("invalid logits capacity")
    _validate_cumulative(cu_num_logits, len(idx_mapping), total_num_logits)
    if cu_num_logits[-1] != total_num_logits:
        raise ValueError("logit count mismatch")
    rows, positions = [], []
    for i, req_idx in enumerate(idx_mapping):
        count = cu_num_logits[i + 1] - cu_num_logits[i]
        if count > max_expand_len:
            raise ValueError("request logit expansion exceeds limit")
        rows.extend([req_idx] * count)
        positions.extend(range(count))
    return rows, positions

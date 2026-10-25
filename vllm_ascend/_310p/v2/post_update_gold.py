"""310P3 V2 T09 post_update CPU oracle, pinned to vLLM 0.27.1.

NOT a runtime/torch-npu implementation. The upstream GPU V2 post_update
is Triton and remains BLOCKED until a 310P-specific state-update kernel is
implemented, bound and reviewed on the actual model runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy


@dataclass
class PostUpdateState:
    computed: list[int]
    last_sampled: list[int]
    total_len: list[int]
    token_rows: list[list[int]]
    output_counts: list[list[int]] | None = None


def post_update_reference(
    *,
    state: PostUpdateState,
    idx_mapping: list[int],
    sampled: list[list[int]],
    num_sampled: list[int],
    num_rejected: list[int],
    query_start_loc: list[int] | None,
) -> PostUpdateState:
    """Pure reference for the vLLM 0.27.1 _post_update_kernel.

    Negative request indices are masked; positive state slots must be unique.
    Prevalidate the complete batch before mutating anything. Returns a new
    state; does not mutate any of the input containers.
    """
    n = len(idx_mapping)
    count = len(state.computed)
    if (len(state.last_sampled) != count or len(state.total_len) != count
            or len(state.token_rows) != count
            or (state.output_counts is not None
                and len(state.output_counts) != count)):
        raise ValueError("inconsistent persistent request-state sizes")
    if len(sampled) != n or len(num_sampled) != n or len(num_rejected) != n:
        raise ValueError("inconsistent sampled batch sizes")
    if query_start_loc is not None:
        if (len(query_start_loc) != n + 1
                or query_start_loc[0] != 0
                or any(a < 0 or b < a for a, b in zip(
                    query_start_loc, query_start_loc[1:]))):
            raise ValueError("invalid query prefix offsets")
    valid = [idx for idx in idx_mapping if idx >= 0]
    if len(valid) != len(set(valid)):
        raise ValueError("duplicate live state indices")
    if any(idx < -1 or idx >= count for idx in idx_mapping):
        raise ValueError("state index out of bounds")
    for batch, idx in enumerate(idx_mapping):
        if idx < 0:
            continue
        n_sampled = num_sampled[batch]
        n_rejected = num_rejected[batch]
        query_len = (0 if query_start_loc is None
                     else query_start_loc[batch + 1] - query_start_loc[batch])
        if (n_sampled < 0 or n_sampled > len(sampled[batch])
                or n_rejected < 0 or n_rejected > query_len):
            raise ValueError("invalid sample/rejection lengths")
        old_len = state.total_len[idx]
        new_computed = state.computed[idx] + query_len - n_rejected
        if (old_len < 0 or old_len + n_sampled > len(state.token_rows[idx])
                or new_computed < 0):
            raise ValueError("token row or computed-token capacity exceeded")
        if state.output_counts is not None:
            if any(token < 0 or token >= len(state.output_counts[idx])
                   for token in sampled[batch][:n_sampled]):
                raise ValueError("sampled token outside output vocabulary")

    result = deepcopy(state)
    for batch, idx in enumerate(idx_mapping):
        if idx < 0:
            continue
        n_sampled = num_sampled[batch]
        if n_sampled:
            start = result.total_len[idx]
            prefix = sampled[batch][:n_sampled]
            result.token_rows[idx][start:start + n_sampled] = prefix
            result.last_sampled[idx] = prefix[-1]
            result.total_len[idx] = start + n_sampled
            if result.output_counts is not None:
                for token in prefix:
                    result.output_counts[idx][token] += 1
        query_len = (0 if query_start_loc is None
                     else query_start_loc[batch + 1] - query_start_loc[batch])
        result.computed[idx] += query_len - num_rejected[batch]
    return result

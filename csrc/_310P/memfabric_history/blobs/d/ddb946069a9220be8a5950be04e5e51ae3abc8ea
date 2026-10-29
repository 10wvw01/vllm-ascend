"""T01 prefill/lookahead CPU oracle for vLLM 0.27.1 (no Torch/NPU)."""
from __future__ import annotations


def prepare_prefill_inputs(
    input_ids: list[int],
    next_prefill_tokens: list[list[int]],
    idx_mapping: list[int],
    query_start_loc: list[int],
    all_token_ids: list[list[int]],
    prefill_len: list[int],
    num_computed_tokens: list[int],
) -> tuple[list[int], list[list[int]]]:
    """Return copied buffers; non-prefill rows and inactive slots stay intact."""
    n = len(idx_mapping)
    capacity = len(all_token_ids)
    if (len(query_start_loc) != n + 1 or query_start_loc[0] != 0
            or len(prefill_len) != capacity or len(num_computed_tokens) != capacity
            or any(len(row) != capacity for row in next_prefill_tokens)
            or any(a > b for a, b in zip(query_start_loc, query_start_loc[1:]))
            or query_start_loc[-1] > len(input_ids)
            or len(set(idx_mapping)) != n
            or any(i < 0 or i >= capacity for i in idx_mapping)):
        raise ValueError("invalid T01 batch metadata")
    width = len(all_token_ids[0]) if capacity else 0
    if any(len(row) != width for row in all_token_ids):
        raise ValueError("T01 all_token_ids rows must have equal capacity")
    for state, prefill, computed in zip(
        range(capacity), prefill_len, num_computed_tokens
    ):
        if not (0 <= computed <= width and 0 <= prefill <= width):
            raise ValueError(f"invalid T01 state {state} bounds")
    result = input_ids.copy()
    next_result = [row.copy() for row in next_prefill_tokens]
    for row, state in enumerate(idx_mapping):
        computed = num_computed_tokens[state]
        prefill = prefill_len[state]
        if computed >= prefill:
            continue
        start, end = query_start_loc[row:row + 2]
        if computed + end - start > width:
            raise ValueError("T01 query exceeds token row")
        result[start:end] = all_token_ids[state][computed:computed + end - start]
        for look in range(len(next_result)):
            position = computed + end - start + look
            next_result[look][state] = (
                all_token_ids[state][position] if position < prefill else 0
            )
    return result, next_result

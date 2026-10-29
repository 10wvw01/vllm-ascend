"""No-Torch vLLM 0.27.1 T03 sampled/draft input assembly gold.

Exact reference: vllm/v1/worker/gpu/input_batch.py:
_combine_sampled_and_draft_tokens_kernel @ 6e448d0ea9bf.
"""
from __future__ import annotations


def combine_sampled_and_draft_tokens(
    input_ids: list[int], idx_mapping: list[int],
    last_sampled_tokens: list[list[int]], query_start_loc: list[int],
    seq_lens: list[int], prefill_len: list[int], draft_tokens: list[list[int]],
    cu_num_logits: list[int], num_logits: int,
    num_new_sampled_tokens: int = 1,
) -> tuple[list[int], list[int]]:
    n = len(idx_mapping)
    if num_new_sampled_tokens not in (0, 1) or num_logits < 0:
        raise ValueError("invalid bonus token count or logits length")
    if (len(seq_lens) != n or len(query_start_loc) != n + 1
            or len(cu_num_logits) != n + 1 or query_start_loc[0] != 0
            or cu_num_logits[0] != 0 or query_start_loc[-1] > len(input_ids)
            or cu_num_logits[-1] != num_logits
            or any(a > b for a, b in zip(query_start_loc, query_start_loc[1:]))
            or any(a > b for a, b in zip(cu_num_logits, cu_num_logits[1:]))):
        raise ValueError("invalid cumulative input/logits")
    output = input_ids.copy()
    logits_indices = [0] * num_logits
    for r, req in enumerate(idx_mapping):
        if (req < 0 or req >= len(prefill_len) or req >= len(draft_tokens)
                or req >= len(last_sampled_tokens) or not last_sampled_tokens[req]
                or seq_lens[r] < 0 or prefill_len[req] < 0):
            raise ValueError("invalid request state")
        start, end = query_start_loc[r:r + 2]
        logit_start, logit_end = cu_num_logits[r:r + 2]
        count = logit_end - logit_start
        draft_count = count - num_new_sampled_tokens
        if (count > end - start or draft_count < 0
                or draft_count > len(draft_tokens[req])):
            raise ValueError("invalid query/logits/draft capacity")
        logits_indices[logit_start:logit_end] = range(end - count, end)
        if seq_lens[r] <= prefill_len[req]:
            continue
        first_logit_seq_pos = seq_lens[r] - count
        if num_new_sampled_tokens and first_logit_seq_pos >= prefill_len[req]:
            output[end - count] = last_sampled_tokens[req][0]
        if draft_count:
            output[end - draft_count:end] = draft_tokens[req][:draft_count]
    return output, logits_indices

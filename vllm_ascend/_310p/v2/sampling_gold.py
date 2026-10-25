"""No-Torch vLLM 0.27.1 T07/T08 sampling reference for 310P3 R2-A.

Frozen source: vllm/v1/worker/gpu/input_batch.py:get_num_sampled_and_rejected
and spec_decode/rejection_sampler.py:_flatten_sampled_kernel @ 6e448d0ea9bf.
No GPU/NPU correctness is implied by this reference implementation.
"""
from __future__ import annotations


def _logit_counts(cu_num_logits: list[int], num_reqs: int) -> list[int]:
    if (len(cu_num_logits) != num_reqs + 1 or not cu_num_logits
            or cu_num_logits[0] != 0
            or any(a < 0 or b < a for a, b in zip(cu_num_logits, cu_num_logits[1:]))):
        raise ValueError("invalid cumulative logits")
    return [b - a for a, b in zip(cu_num_logits, cu_num_logits[1:])]


def sampled_and_rejected(
    num_sampled: list[int], seq_lens: list[int], cu_num_logits: list[int],
    idx_mapping: list[int], prefill_len: list[int],
) -> tuple[list[int], list[int]]:
    """T07: chunked prefill must zero *both* sampled and rejected counts."""
    n = len(idx_mapping)
    if len(num_sampled) != n or len(seq_lens) != n:
        raise ValueError("sampling batch shape mismatch")
    counts = _logit_counts(cu_num_logits, n)
    out_sampled, out_rejected = [], []
    for r, req_idx in enumerate(idx_mapping):
        if req_idx < 0 or req_idx >= len(prefill_len):
            raise ValueError("invalid request index")
        sampled = num_sampled[r]
        if sampled < 0 or sampled > counts[r] or seq_lens[r] < 0 or prefill_len[req_idx] < 0:
            raise ValueError("invalid sampled count or sequence length")
        if seq_lens[r] < prefill_len[req_idx]:
            out_sampled.append(0)
            out_rejected.append(0)
        else:
            out_sampled.append(sampled)
            out_rejected.append(counts[r] - sampled)
    return out_sampled, out_rejected


def flattened_sampled(
    sampled: list[list[int]], num_sampled: list[int],
    cu_num_logits: list[int], num_logits: int,
) -> list[int]:
    """T08: zero-fill unaccepted logits, copy only each accepted prefix."""
    n = len(sampled)
    if len(num_sampled) != n or num_logits < 0:
        raise ValueError("invalid sampled tensor shape or total logits")
    counts = _logit_counts(cu_num_logits, n)
    if cu_num_logits[-1] != num_logits:
        raise ValueError("total logits mismatch")
    out = [0] * num_logits
    for r, (tokens, count, logits_count) in enumerate(zip(sampled, num_sampled, counts)):
        if count < 0 or count > logits_count or count > len(tokens):
            raise ValueError("invalid sampled prefix")
        start = cu_num_logits[r]
        out[start:start + count] = tokens[:count]
    return out

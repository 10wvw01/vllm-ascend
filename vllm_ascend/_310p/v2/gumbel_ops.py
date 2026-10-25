"""310P3 V2 T17/T18 greedy/temperature Eager candidates (NOT runtime wired).

Stochastic Gumbel requires frozen Philox(seed,pos) parity and rejection
distribution verification. Nonzero temperature fails closed rather than
silently replacing the random sampler with a biased approximation.
"""
from __future__ import annotations


def _temperature_per_token(logits, expanded_idx_mapping, temperature):
    import torch

    if logits.ndim != 2 or expanded_idx_mapping.shape != (logits.shape[0],):
        raise ValueError("T17/T18 invalid logits/index shape")
    if (expanded_idx_mapping.device != logits.device or
            temperature.device != logits.device or
            expanded_idx_mapping.dtype != torch.int32 or
            temperature.ndim != 1):
        raise ValueError("T17/T18 device/dtype mismatch")
    if logits.numel() == 0:
        raise ValueError("T17/T18 empty logits")
    torch._assert_async(
        torch.all((expanded_idx_mapping >= 0) &
                  (expanded_idx_mapping < temperature.numel())),
        "T17/T18 invalid persistent request index",
    )
    return temperature.index_select(0, expanded_idx_mapping.to(torch.int64))


def apply_temperature(logits, expanded_idx_mapping, temperature) -> None:
    """Frozen Ascend V2 temperature semantics with a full-buffer D2D write."""
    import torch

    per_token = _temperature_per_token(logits, expanded_idx_mapping, temperature)
    torch._assert_async(torch.all(torch.isfinite(per_token) & (per_token >= 0)),
                        "T18 invalid temperature")
    per_token = per_token.to(torch.float32)
    scale = torch.where((per_token == 0) | (per_token == 1),
                        torch.ones_like(per_token), per_token)
    updated = (logits.to(torch.float32) / scale[:, None]).to(logits.dtype)
    logits.copy_(updated)


def gumbel_sample(
    logits, expanded_idx_mapping, temperature, seed, pos,
    apply_temperature: bool, output_processed_logits=None,
    output_processed_logits_col=None, use_fp64: bool = False,
):
    """Frozen Gumbel ABI, correct greedy subset only; stochastic is NO-GO."""
    import torch

    per_token = _temperature_per_token(logits, expanded_idx_mapping, temperature)
    if (seed.device != logits.device or pos.device != logits.device
            or seed.ndim != 1 or pos.shape != (logits.shape[0],)):
        raise ValueError("T17 invalid seed/position ABI")
    if use_fp64:
        raise NotImplementedError("310P3 T17 FP64 Gumbel not implemented")
    if output_processed_logits is not None or output_processed_logits_col is not None:
        raise NotImplementedError("310P3 T17 processed-logits output unreviewed")
    torch._assert_async(torch.all(per_token == 0),
                        "310P3 T17 stochastic Gumbel/Philox is not implemented")
    return torch.argmax(logits, dim=-1).to(torch.int64)

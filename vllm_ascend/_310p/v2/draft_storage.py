"""310P3 K7 draft storage update: write complete 64-byte int64 rows.

Upstream GPUModelRunner writes a 7-wide view. 310P3 must not issue a short
update that could corrupt adjacent request rows or the padding element.
Torch-CPU contract only; 310P3 actual device index_copy_ stream/writes need
static operator review before Worker is unblocked.
"""
from __future__ import annotations


def store_padded_draft_tokens(storage, logical_view, idx_mapping, idx_mapping_np, draft_tokens):
    import numpy as np
    import torch

    cpu = np.asarray(idx_mapping_np)
    if (storage.ndim != 2 or storage.shape[1] != 8
            or logical_view.ndim != 2 or logical_view.shape != (storage.shape[0], 7)
            or draft_tokens.ndim != 2 or draft_tokens.shape != (len(cpu), 7)
            or idx_mapping.ndim != 1 or idx_mapping.numel() != len(cpu)
            or not np.issubdtype(cpu.dtype, np.integer)):
        raise ValueError("310P3 K7 padded draft shape mismatch")
    if (len(set(cpu.tolist())) != len(cpu)
            or (cpu < 0).any() or (cpu >= storage.shape[0]).any()):
        raise ValueError("duplicate or out-of-range draft request indices")
    if (storage.dtype != torch.int64 or logical_view.dtype != torch.int64
            or idx_mapping.dtype != torch.int32 or draft_tokens.dtype != torch.int64
            or any(x.device != storage.device for x in
                   (logical_view, idx_mapping, draft_tokens))):
        raise ValueError("310P3 K7 draft storage dtype/device mismatch")
    # Preserve the view identity expected by input_batch and draft-token handler.
    if logical_view.data_ptr() != storage.data_ptr():
        raise ValueError("draft view must alias the padded storage")
    if not len(cpu):
        return
    # The host's persistent request-state indices are the scheduling
    # authority; asynchronously assert device idx agrees before all writes.
    target = torch.tensor(cpu, dtype=torch.int32, device=storage.device)
    torch._assert_async(torch.all(idx_mapping == target),
                        "310P3 draft device/CPU request-state mapping diverged")
    # Full contiguous 64B allocation rows. Clear stale padding explicitly.
    padded = torch.cat(
        (draft_tokens, torch.zeros((len(cpu), 1),
                    dtype=torch.int64, device=storage.device)),
        dim=1,
    )
    storage.index_copy_(0, idx_mapping.long(), padded)

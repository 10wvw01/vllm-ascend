"""310P3 DFlash K7 V2 proposer: scope T11 to frozen upstream propose.

The decorated frozen vLLM 0.27.1 DFlashSpeculator.propose must be
inspect.unwrap()-ed before cloning; otherwise its globals belong to
torch.inference_mode rather than the vLLM DFlash module. The original
Ascend build_attn_metadata_wrapper is retained for per-group metadata.
This candidate is NOT registered in NPUWorker310 (STATIC NO-GO).
"""
from __future__ import annotations

import inspect

import torch
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm_ascend.worker.v2.attn_utils import build_attn_metadata_wrapper
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator

from .dflash_input_ops import prepare_dflash_inputs
from .model_runner import _clone_upstream_function


class NPUDFlashSpeculator310V2(AscendDFlashSpeculator):
    @torch.inference_mode()
    def propose(self, *args, **kwargs):
        # Frozen vLLM 0.27.1 loops over draft_kv_cache_group_ids and calls
        # prepare_dflash_inputs once per group. Only this instance receives
        # the Eager function; other Ascend/GPU DFlash instances are unchanged.
        upstream_propose = inspect.unwrap(DFlashSpeculator.propose)
        scoped = _clone_upstream_function(
            upstream_propose,
            {"prepare_dflash_inputs": prepare_dflash_inputs},
        )
        with build_attn_metadata_wrapper():
            return scoped(self, *args, **kwargs)

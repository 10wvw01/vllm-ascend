"""310P3 scoped V2 sampler bindings (greedy-only, STATIC NO-GO).

Frozen vLLM 0.27.1 methods are cloned with local globals rather than
process-global monkeypatches. Remaining GPU/Ascend V2 penalties, top-k,
bad-words, grammar and logprobs Triton paths are not certified. This module
is consumed only by the unregistered NPUModelRunner310V2Candidate.
"""
from __future__ import annotations

from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from .gumbel_ops import gumbel_sample
from .rejection_ops import rejection_sample
from .sampling_ops import get_num_sampled_and_rejected
from .model_runner import _clone_upstream_function


class NPUSampler310V2(Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # FlashInfer is a CUDA implementation, never silently route 310P here.
        self.use_flashinfer = False

    def __call__(self, *args, **kwargs):
        call = _clone_upstream_function(
            Sampler.__call__,
            {"get_num_sampled_and_rejected": get_num_sampled_and_rejected},
        )
        return call(self, *args, **kwargs)

    def sample(self, *args, **kwargs):
        sample = _clone_upstream_function(
            Sampler.sample,
            {"gumbel_sample": gumbel_sample},
        )
        return sample(self, *args, **kwargs)


class NPURejectionSampler310V2(RejectionSampler):
    def __call__(self, *args, **kwargs):
        call = _clone_upstream_function(
            RejectionSampler.__call__,
            {"get_num_sampled_and_rejected": get_num_sampled_and_rejected},
        )
        return call(self, *args, **kwargs)

    def _verify(self, *args, **kwargs):
        verify = _clone_upstream_function(
            RejectionSampler._verify,
            {"rejection_sample": rejection_sample},
        )
        return verify(self, *args, **kwargs)

    def _get_logprobs_tensors(self, *args, **kwargs):
        # The frozen method otherwise launches _flatten_sampled_kernel
        # (T08) and patched Ascend compute_topk_scores (T22) Triton.
        # Reject until BOTH have device-reviewed Eager implementations.
        if kwargs:
            max_num_logprobs = kwargs.get("max_num_logprobs")
        else:
            max_num_logprobs = args[5] if len(args) >= 6 else None
        if max_num_logprobs == NO_LOGPROBS:
            return None
        raise NotImplementedError(
            "310P3 V2 rejection logprobs T08/T22 are not implemented"
        )

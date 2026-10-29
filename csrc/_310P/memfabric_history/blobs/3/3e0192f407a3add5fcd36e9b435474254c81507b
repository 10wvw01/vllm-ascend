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
    def __init__(
        self, max_num_reqs, vocab_size, device, req_states,
        logprobs_mode="raw_logprobs", num_speculative_tokens=1,
        use_fp64_gumbel=False,
    ):
        # Frozen Sampler.__init__ nests FIVE constructors that allocate UVA.
        # Bind every one locally BEFORE construction, not replace afterwards.
        from .sampler_state import make_npu_sampler_state_factories

        overrides = make_npu_sampler_state_factories(device)
        overrides["flashinfer_sampler_supported"] = lambda: False
        init = _clone_upstream_function(Sampler.__init__, overrides)
        init(
            self, max_num_reqs=max_num_reqs, vocab_size=vocab_size,
            device=device, req_states=req_states, logprobs_mode=logprobs_mode,
            num_speculative_tokens=num_speculative_tokens,
            use_fp64_gumbel=use_fp64_gumbel,
        )
        # FlashInfer is a CUDA implementation, never silently route 310P here.
        self.use_flashinfer = False

    def apply_sampling_params(
        self, logits, expanded_idx_mapping, idx_mapping_np, pos,
        input_ids, expanded_local_pos, skip_top_k_top_p=False,
    ):
        """Fail closed before patched Ascend Triton penalties/top-p kernels.

        Frozen Sampler.apply_sampling_params invokes modules whose Eager
        implementations are not audited for 310P3. The greedy fast path
        returns logits without any such kernels, exactly as upstream does.
        Do NOT silently approximate penalties, temperatures or bad words.
        """
        if self._requires_logits_processing(idx_mapping_np):
            raise NotImplementedError(
                "310P3 V2 non-default sampling parameter kernels T18-T24 "
                "are not yet implemented; refusing the Triton route"
            )
        return logits

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

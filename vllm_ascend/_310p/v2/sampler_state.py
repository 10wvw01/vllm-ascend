"""Scoped, non-UVA sampler state construction for 310P3 V2.

Frozen vLLM 0.27.1 Sampler.__init__ constructs FIVE nested state objects,
whose constructors use UvaBackedTensor and/or GPU StagedWriteTensor. Merely
replacing RequestState does not remove those allocations. Bind each frozen
constructor to NPU-owned device buffers in a *private* globals dictionary.
No upstream or process-global symbols are changed.

Candidate only: NPU device op/stream semantics and ModelState remain NO-GO.
"""
from __future__ import annotations


def _scoped_construct(cls, device, *args, **kwargs):
    """Call a frozen state constructor once, without its GPU/UVA buffers."""
    from .model_runner import _clone_upstream_function
    from .staged_buffers import NPUBackedTensor, NPUStagedWriteTensor

    globs = cls.__init__.__globals__
    if "UvaBackedTensor" not in globs:
        raise RuntimeError(f"{cls.__name__} frozen UVA constructor ABI changed")

    def backed(size, dtype):
        return NPUBackedTensor(size, dtype, device)

    overrides = {"UvaBackedTensor": backed}
    if "StagedWriteTensor" in globs:
        overrides["StagedWriteTensor"] = NPUStagedWriteTensor
    init = _clone_upstream_function(cls.__init__, overrides)
    state = cls.__new__(cls)
    init(state, *args, **kwargs)
    return state


def make_npu_sampler_state_factories(device):
    """Return all FIVE names referenced by frozen Sampler.__init__."""
    from vllm.v1.worker.gpu.sample.states import SamplingStates
    from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
    from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
    from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
    from vllm.v1.worker.gpu.sample.logprob import LogprobTokenIdsState

    classes = {
        "SamplingStates": SamplingStates,
        "PenaltiesState": PenaltiesState,
        "LogitBiasState": LogitBiasState,
        "BadWordsState": BadWordsState,
        "LogprobTokenIdsState": LogprobTokenIdsState,
    }

    def bind(cls):
        return lambda *args, **kwargs: _scoped_construct(
            cls, device, *args, **kwargs
        )

    return {name: bind(cls) for name, cls in classes.items()}


class NPUStructuredOutputsWorker310V2:
    """Never create upstream StructuredOutputsWorker's CUDA Stream.

    Greedy-only 310P3 requests do not use grammar. Keep the runtime
    attribute available for frozen load_model ABI; fail closed if a grammar
    call reaches the candidate through a later code path.
    """

    def __init__(self, max_num_logits, vocab_size, device):
        if max_num_logits <= 0 or vocab_size <= 0:
            raise ValueError("invalid 310P3 structured-output capacity")
        self.device = device

    def apply_grammar_bitmask(self, *args, **kwargs):
        raise NotImplementedError(
            "310P3 V2 grammar T24 requires an audited non-Triton backend"
        )

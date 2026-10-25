"""310P3 V2 DraftTokensHandler without CUDA Stream/Event construction.

Frozen vLLM 0.27.1 GPUModelRunner.__init__ creates DraftTokensHandler,
whose own module-global torch.cuda.Stream/Event cannot be overridden by a
constructor-local torch proxy. The greedy-only scope has no structured
outputs, so no D2H draft validation is needed. Keep the same public methods.
"""
from __future__ import annotations


class NPUDraftTokensHandler310V2:
    def __init__(self, device=None):
        self.device = device
        self.req_ids: list[str] = []
        self.num_draft_tokens = 0
        self.draft_tokens_np = None

    def set_draft_tokens(self, input_batch, draft_tokens):
        # Reject before mutating request state: grammar requires the upstream
        # async D2H path and is not supported by this NPU candidate.
        if input_batch.has_structured_output_reqs:
            raise NotImplementedError(
                "310P3 V2 structured draft validation requires a non-CUDA backend"
            )
        if (draft_tokens.ndim != 2
                or draft_tokens.shape[0] != len(input_batch.req_ids)):
            raise ValueError("310P3 V2 draft token shape/request mismatch")
        self.req_ids = list(input_batch.req_ids)
        self.num_draft_tokens = draft_tokens.shape[1]
        self.draft_tokens_np = None

    def get_draft_tokens(self):
        from vllm.v1.outputs import DraftTokenIds

        # Exact frozen non-grammar fallback: the scheduler has no draft-token
        # grammar validation to perform. No stream, event or host D2H needed.
        return DraftTokenIds(
            self.req_ids,
            [[-1] * self.num_draft_tokens for _ in self.req_ids],
        )

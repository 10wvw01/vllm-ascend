"""310P3 V2 correctness-first sampled-output snapshot without CUDA.

Frozen vLLM 0.27.1 AsyncOutput constructs torch.cuda.Event and calls
torch.cuda.set_stream even when the model runner's stream is an NPU stream.
This is a synchronous Eager D2H fallback: it captures immutable CPU results
BEFORE post_update and DFlash proposal reuse device state. No overlap is
claimed; true NPU async copy/event support requires a separate device gate.
"""
from __future__ import annotations

from vllm.v1.outputs import AsyncModelRunnerOutput


class NPUAsyncOutput310V2(AsyncModelRunnerOutput):
    def __init__(
        self, model_runner_output, sampler_output, num_sampled_tokens,
        main_stream, copy_stream, check_ep_fault=False,
    ):
        if check_ep_fault:
            raise NotImplementedError(
                "310P3 V2 async EP fault snapshot has no NPU backend"
            )
        if (sampler_output.logprobs_tensors is not None
                or sampler_output.num_nans is not None
                or model_runner_output.prompt_logprobs_dict):
            raise NotImplementedError(
                "310P3 V2 logprobs/NaN async output requires NPU backend"
            )
        # No copy-stream launch. Blocking D2H on the active NPU stream
        # snapshots values before postprocess mutates persistent buffers.
        # The caller must ensure all sampled tensors are produced on this
        # same stream. main_stream/copy_stream are accepted for frozen ABI.
        self.model_runner_output = model_runner_output
        self.sampled_token_ids = (
            sampler_output.sampled_token_ids.detach()
            .to("cpu", non_blocking=False).numpy().copy()
        )
        self.num_sampled_tokens_np = (
            num_sampled_tokens.detach()
            .to("cpu", non_blocking=False).numpy().copy()
        )
        if (self.sampled_token_ids.ndim != 2
                or self.num_sampled_tokens_np.ndim != 1
                or self.sampled_token_ids.shape[0]
                != self.num_sampled_tokens_np.shape[0]
                or self.sampled_token_ids.shape[0]
                != len(model_runner_output.req_ids)):
            raise ValueError("310P3 V2 sampled output shape mismatch")
        self._consumed = False

    def get_output(self):
        if self._consumed:
            raise RuntimeError("310P3 V2 async output can only be consumed once")
        rows = self.sampled_token_ids.tolist()
        counts = self.num_sampled_tokens_np.tolist()
        for row, count in zip(rows, counts, strict=True):
            if not 0 <= count <= len(row):
                raise ValueError("310P3 V2 invalid sampled token count")
            del row[count:]
        self.model_runner_output.sampled_token_ids = rows
        self._consumed = True
        return self.model_runner_output

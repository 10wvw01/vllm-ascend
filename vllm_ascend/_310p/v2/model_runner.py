"""310P3 dedicated V2 model-runner construction (staged; device NO-GO).

The frozen vLLM 0.27.1 GPUModelRunner initializes RequestState via a module
global. It is unsafe to let that run and replace the result afterwards: the
first constructor allocates UVA-backed state. Instead, call the same frozen
function code with a *private globals dictionary* containing the 310P state
factory. This does not patch process-wide GPU/Ascend imports.

Runtime is intentionally not exposed through NPUWorker310 until all required
Eager functions, samplers, mixed-attention and KV ownership are connected.
"""
from __future__ import annotations

import types
from functools import cached_property

import torch
from vllm.v1.core.sched.output import GrammarOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu import pcp_manager as pcp
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.eplb_utils import step_eplb_after


def _clone_upstream_function(method, globals_overrides: dict):
    """Bind a frozen upstream method to locally overridden dependencies.

    Preserves its code/defaults/closure without mutating the module that owns
    it. Avoids changing other model runners in the same Python process.
    """
    if not isinstance(method, types.FunctionType):
        raise TypeError("upstream constructor must be an ordinary Python method")
    globs = method.__globals__.copy()
    missing = set(globals_overrides).difference(globs)
    if missing:
        raise RuntimeError(f"upstream GPUModelRunner ABI drift: {sorted(missing)}")
    globs.update(globals_overrides)
    bound = types.FunctionType(
        method.__code__, globs, name=method.__name__,
        argdefs=method.__defaults__, closure=method.__closure__,
    )
    bound.__kwdefaults__ = method.__kwdefaults__
    bound.__annotations__ = method.__annotations__
    return bound


class _TorchNPUProxy:
    """310P constructor-only CUDA Stream translation, no global mutation."""

    def __init__(self, torch_module):
        self._torch = torch_module

    @property
    def cuda(self):
        # Fail closed if the NPU runtime is missing; never create a real CUDA
        # stream in a supposedly Triton-free 310P3 V2 process.
        return self._torch.npu

    def __getattr__(self, name):
        return getattr(self._torch, name)


class NPUModelRunner310V2:
    """Mixin to be combined with Ascend V2 runner once runtime is ready.

    This class deliberately has no Worker registration. The present
    bootstrapping hook is consumed only by the dedicated subclass below.
    """


def _validate_310p_v2_constructor_contract(vllm_config, device):
    """Abort before allocations outside frozen 310P3 TP2 Eager scope."""
    import vllm
    from vllm.config.compilation import CUDAGraphMode

    if not vllm.__version__.startswith("0.27.1"):
        raise RuntimeError("310P V2 requires frozen upstream vLLM 0.27.1")
    if (vllm_config.parallel_config.tensor_parallel_size != 2
            or vllm_config.parallel_config.pipeline_parallel_size != 1
            or str(device).split(":")[0] != "npu"
            or vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE):
        raise RuntimeError("310P V2 currently permits only NPU TP2 / PP1 / Eager")
    if not vllm_config.use_v2_model_runner:
        raise RuntimeError("explicit V2 model-runner opt-in is required")


# Imported here rather than in NPUModelRunner310V2.__init__: the class is
# never registered as a worker, and loading this module alone does not create
# a CUDA stream, import torch_npu or construct a GPU RequestState.
from vllm_ascend.worker.v2.model_runner import NPUModelRunner as _AscendNPUModelRunner


class NPUModelRunner310V2Candidate(_AscendNPUModelRunner):
    """Staged non-UVA 310P3 V2 constructor, not yet an executable Runner."""

    @cached_property
    def main_stream(self):
        """Bind inherited GPUModelRunner.main_stream to the actual NPU stream.

        The upstream 0.27.1 cached property uses torch.cuda.current_stream.
        A constructor-local torch proxy does not replace that descriptor:
        this dedicated override prevents a later CUDA stream lookup from
        accidentally escaping the private globals construction boundary.
        """
        return torch.npu.current_stream(self.device)

    def __init__(self, vllm_config, device):
        _validate_310p_v2_constructor_contract(vllm_config, device)
        super().__init__(vllm_config, device)

    def _310p_v2_parent_constructor(self, vllm_config, device):
        import torch
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner

        from .request_state import NPURequestState

        # Upstream GPU init is invoked *exactly once*. The temporary GPU
        # DFlashSpeculator is skipped because Ascend V2 creates its own draft
        # speculator after this parent constructor returns.
        ctor = _clone_upstream_function(
            GPUModelRunner.__init__,
            {
                "RequestState": NPURequestState,
                "init_speculator": lambda _cfg, _device: None,
                "torch": _TorchNPUProxy(torch),
            },
        )
        ctor(self, vllm_config, device)

    @staticmethod
    def _310p_v2_state_factory(**kwargs):
        from .request_state import NPURequestState

        return NPURequestState(**kwargs)

    @staticmethod
    def _310p_v2_speculator_factory(vllm_config, device):
        """Reject unreviewed draft methods before allocating their state."""
        spec_config = vllm_config.speculative_config
        if spec_config is None or not spec_config.use_dflash():
            raise NotImplementedError(
                "310P3 V2 currently supports only DFlash K7 draft"
            )
        if spec_config.num_speculative_tokens != 7:
            raise NotImplementedError("310P3 V2 requires DFlash K7")
        from .dflash_speculator import NPUDFlashSpeculator310V2

        return NPUDFlashSpeculator310V2(vllm_config, device)

    def add_requests(self, scheduler_output):
        """Reject unsupported generation paths before any request/KV mutation."""
        from .sampling_scope import validate_greedy_310p_sampling_params

        if scheduler_output.has_structured_output_requests:
            raise NotImplementedError(
                "310P V2 structured-output/grammar is not Triton-free"
            )
        for new_req in scheduler_output.scheduled_new_reqs:
            validate_greedy_310p_sampling_params(new_req.sampling_params)
        return super().add_requests(scheduler_output)

    def prepare_inputs(self, scheduler_output, batch_desc):
        """Bind frozen Ascend V2 input to 310P3 non-Triton T01-T04 ops.

        All four imports in the inherited Ascend method are module-level
        upstream GPU Triton functions. Defining the replacements under
        _310p/v2 does not change those bindings. This scoped clone applies
        the four substitutes only during this candidate runner call, without
        modifying vllm_ascend.worker.v2.model_runner module globals.
        """
        from vllm_ascend.worker.v2.model_runner import NPUModelRunner
        from .input_ops import (
            expand_idx_mapping,
            prepare_pos_seq_lens,
            prepare_prefill_inputs,
        )
        from .token_ops import combine_sampled_and_draft_tokens

        prepare = _clone_upstream_function(
            NPUModelRunner.prepare_inputs,
            {
                "prepare_prefill_inputs": prepare_prefill_inputs,
                "prepare_pos_seq_lens": prepare_pos_seq_lens,
                "expand_idx_mapping": expand_idx_mapping,
                "combine_sampled_and_draft_tokens":
                    combine_sampled_and_draft_tokens,
            },
        )
        return prepare(self, scheduler_output, batch_desc)

    def postprocess_sampled(
        self, idx_mapping, sampled_tokens, num_sampled, num_rejected,
        query_start_loc=None,
    ):
        """310P-specific T09 call chain: never call upstream Triton post_update."""
        from .post_update_ops import post_update

        if self.input_batch is None:
            raise RuntimeError("310P V2 post_update requires prepared input_batch")
        if self.is_last_pp_rank:
            assert self.sampler is not None
            output_bin_counts = self.sampler.penalties_state.output_bin_counts
        else:
            output_bin_counts = None

        post_update(
            idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            query_start_loc,
            self.req_states.all_token_ids.gpu,
            self.req_states.total_len.gpu,
            idx_mapping_np=self.input_batch.idx_mapping_np,
            last_sampled_storage=self.req_states._last_sampled_storage,
        )
        self.model_state.postprocess_state(
            idx_mapping, num_sampled, self.req_states.num_computed_tokens.gpu
        )
        self._copy_num_computed_tokens_to_cpu()

    def load_model(self, *args, **kwargs):
        """Scope the frozen GPU V2 sampler constructors to 310P candidates.

        Does not register this runner or permit non-greedy/spec logprobs.
        """
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from .sampler import NPUSampler310V2, NPURejectionSampler310V2
        from .sampler_state import NPUStructuredOutputsWorker310V2

        load = _clone_upstream_function(
            GPUModelRunner.load_model,
            {
                "Sampler": NPUSampler310V2,
                "RejectionSampler": NPURejectionSampler310V2,
                "StructuredOutputsWorker": NPUStructuredOutputsWorker310V2,
            },
        )
        return load(self, *args, **kwargs)

    def _310p_v2_store_draft_tokens(self, input_batch, draft_tokens):
        """Write all eight padded int64 elements per K7 row, not seven."""
        from .draft_storage import store_padded_draft_tokens

        store_padded_draft_tokens(
            self.req_states._draft_storage,
            self.req_states.draft_tokens,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            draft_tokens,
        )

    # Copied from pinned vLLM 0.27.1 GPUModelRunner.sample_tokens with
    # exactly one mutation: 7-token assignment -> full 8-token row write.
    # Keep async output copy, sampler state and KV connector ordering intact.
    @torch.inference_mode()
    @step_eplb_after()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> AsyncOutput | ModelRunnerOutput | None:
        if self.execute_model_state is None:
            # The prior execute_model call must have failed.
            return None

        input_batch = self.execute_model_state.input_batch
        attn_metadata = self.execute_model_state.attn_metadata
        slot_mappings_by_layer = self.execute_model_state.slot_mappings_by_layer
        hidden_states = self.execute_model_state.hidden_states
        aux_hidden_states = self.execute_model_state.aux_hidden_states
        finished_req_ids = self.execute_model_state.finished_req_ids
        self.execute_model_state = None

        if not self.is_last_pp_rank:
            # Non-last PP rank: hidden_states is None because this rank produced
            # IntermediateTensors instead of final hidden states. Receive the
            # sampled tokens broadcast from the last rank and update local state.
            assert self.pp_handler is not None
            all_decode_next = self.pp_handler.receive(input_batch)
            # Optimistically update num_computed_tokens for entire batch here.
            # Will be adjusted for rejections if necessary in update_requests.
            self.postprocess_num_computed_tokens(input_batch)
            if not all_decode_next:
                # Might contain non-final prefill chunks, which will be scheduled
                # in the immediate next step (rather than in pp_size steps).
                self.model_state.postprocess_state(input_batch.idx_mapping, 0)

            # Post-step KV connector related operations.
            kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
            return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

        # Last rank: sample tokens
        hidden_states, input_batch = pcp.maybe_restore_pcp_for_sampling(
            self.pcp_manager, hidden_states, input_batch
        )

        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )

        if self.pp_handler is not None:
            # Broadcast to non-last PP ranks (handles spec decode multi-token).
            self.pp_handler.broadcast(
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                input_batch,
            )

        assert self.prompt_logprobs_worker is not None
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            hidden_states,
            input_batch,
            self.req_states.all_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len.np,
        )

        # Prepare the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            # NOTE(woosuk): req_id_to_index is unused in this model runner.
            # Only for compatibility with the existing model runner and scheduler.
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,  # type: ignore
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
        )
        # Start async output copy here so that it can overlap with speculator proposal.
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            main_stream=self.main_stream,
            copy_stream=self.output_copy_stream,
            check_ep_fault=self.check_ep_fault,
        )

        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None
        if self.speculator is not None and self.speculator.supports_mm_inputs:
            # Get cached multimodal embeddings for draft forward.
            # NOTE: This is done here because postprocess updates
            # num_computed_prefill_tokens.
            # The EAGLE/MTP drafter reads one position ahead of the target.
            # TODO(TheEpicDolphin): Gather MM embeddings for all speculative
            # steps during multi-module MTP.
            mm_inputs = self.model_state.gather_mm_embeddings(
                input_batch, draft_lookahead=1
            )

        # Postprocess results and update request states.
        # NOTE: This is intentionally done after creating the AsyncOutput,
        # ensuring that `copy_event` is recorded before calling postprocess.
        # This sequencing may slightly reduce latency as async D2H copy does not
        # need to wait for the postprocess to finish.
        self.postprocess_sampled(
            input_batch.idx_mapping,
            sampler_output.sampled_token_ids,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
        )

        if self.speculator is not None:
            assert self.sampler is not None
            # Let the target override the hidden state fed to the drafter
            # (e.g. DeepSeek V4 MTP needs the pre-hc_head residual). The
            # target returns a persistent buffer sized at max_num_batched_tokens;
            # slice to the active token count that propose() expects.
            spec_hidden_states = hidden_states
            if hasattr(self.model, "get_mtp_target_hidden_states"):
                pre_hc_hidden_states = self.model.get_mtp_target_hidden_states()
                spec_hidden_states = pre_hc_hidden_states[: hidden_states.shape[0]]  # type: ignore[union-attr]
            draft_tokens = self.speculator.propose(
                input_batch,
                attn_metadata,
                slot_mappings_by_layer,
                spec_hidden_states,
                aux_hidden_states,
                num_sampled,
                num_rejected,
                self.req_states.last_sampled_tokens,
                self.req_states.next_prefill_tokens,
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
                mm_inputs=mm_inputs,
            )
            self._310p_v2_store_draft_tokens(input_batch, draft_tokens)

        if self.num_speculative_steps > 0:
            # Spec-decode and diffusion LLMs both use draft tokens but the latter does
            # not have a speculator (i.e. self.speculator is None)
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                self.req_states.draft_tokens[input_batch.idx_mapping],
            )

        # Post-step KV connector related operations.
        kv_connector_output = self.kv_connector.post_forward(finished_req_ids)
        model_runner_output.kv_connector_output = kv_connector_output

        return async_output

    def initialize_kv_cache(self, kv_cache_config):
        """Use the frozen GPU V2 KV orchestration with 310P-owned BlockTables.

        The original GPUModelRunner method computes real KV group capacities,
        kernel block sizes, attention group/ModelState wiring and DFlash
        set_attn. A private-globals copy substitutes the 310P BlockTables
        and Ascend ACL graph manager *before* that method constructs them.
        This avoids both process-global GPU monkeypatching and fallback to
        upstream Triton KV staged writes.

        It remains an unregistered candidate; backend, KV cache allocation,
        mixed-group attention, graph-none manager and torch-npu op contracts
        are not proven by CPU tests.
        """
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm_ascend.worker.v2.aclgraph_utils import ModelAclGraphManager

        from .block_tables import NPUBlockTables

        def make_acl_graph_manager(
            vllm_config, device, cudagraph_mode, decode_query_len,
            lora_capture_cases=None,
        ):
            from vllm.config.compilation import CUDAGraphMode

            if cudagraph_mode != CUDAGraphMode.NONE:
                raise NotImplementedError(
                    "310P3 V2 Graph capture is a separate R3 gate"
                )
            return ModelAclGraphManager(
                vllm_config, device, cudagraph_mode, decode_query_len,
                self, lora_capture_cases=lora_capture_cases,
            )

        kv_init = _clone_upstream_function(
            GPUModelRunner.initialize_kv_cache,
            {
                "BlockTables": NPUBlockTables,
                "ModelCudaGraphManager": make_acl_graph_manager,
            },
        )
        try:
            kv_init(self, kv_cache_config)
        except Exception:
            # KV state may be allocated/cached before an attention-backend
            # failure. Never retry in-place after partial initialization.
            self._310p_v2_kv_poisoned = True
            raise


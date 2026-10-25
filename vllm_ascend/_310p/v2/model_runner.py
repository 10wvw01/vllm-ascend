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


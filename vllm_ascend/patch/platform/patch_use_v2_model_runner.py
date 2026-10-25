"""310P3-only opt-in to replace upstream GPU V2's Triton validation gate.

This gate override intentionally does *not* enable the V2 NPU runtime.
See _310p/worker_310p.py: a visible fail-closed runtime gate prevents an
accidental V1 runner masquerading as V2 after config validation succeeds.
"""
import logging
import os
from importlib.metadata import version as _distribution_version

import vllm.envs as envs
from vllm.config.compilation import CUDAGraphMode
from vllm.config.vllm import VllmConfig
from vllm.triton_utils import HAS_TRITON
from vllm_ascend.utils import is_310p


_FLAG = "VLLM_ASCEND_310P_EXPERIMENTAL_V2_NO_TRITON"
_UPSTREAM_VALIDATE_V2 = VllmConfig._validate_v2_model_runner
_LOGGER = logging.getLogger(__name__)


def _patched_use_v2_model_runner(self) -> bool:
    """Keep existing Ascend policy: explicit VLLM_USE_V2_MODEL_RUNNER only."""
    use_v2 = envs.VLLM_USE_V2_MODEL_RUNNER
    if use_v2 is not None:
        return use_v2
    return False


def _patched_validate_v2_model_runner(self) -> None:
    """Bypass ONLY the Triton presence check, only for frozen 310P3 Eager/TP2.

    Preserve other original V2 restrictions. Non-310P, no opt-in, or a
    real Triton backend always use upstream validation unchanged.
    """
    if (
        os.environ.get(_FLAG) != "1"
        or not is_310p()
        or HAS_TRITON
    ):
        return _UPSTREAM_VALIDATE_V2(self)

    # A misplaced bypass flag must never silently select a different runner.
    if envs.VLLM_USE_V2_MODEL_RUNNER is not True:
        raise RuntimeError(
            f"{_FLAG}=1 requires VLLM_USE_V2_MODEL_RUNNER=1"
        )
    # The wrapper mirrors the non-Triton part of _validate_v2_model_runner
    # from exactly vLLM 0.27.1. For a new upstream release, review afresh.
    if _distribution_version("vllm").split("+", 1)[0] != "0.27.1":
        raise RuntimeError(
            f"{_FLAG}=1 requires audited upstream vLLM 0.27.1; "
            "review before using a newer/older V2 validation API"
        )
    if self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
        raise ValueError(
            "310P3 Triton-free V2 experimental gate permits Eager "
            "(cudagraph_mode=NONE) only; Graph remains unvalidated"
        )
    if self.parallel_config.tensor_parallel_size != 2:
        raise ValueError("310P3 Triton-free V2 experimental gate requires TP=2")
    spec = self.speculative_config
    if spec is not None and spec.method != "dflash":
        raise ValueError(
            "310P3 Triton-free V2 experimental gate covers only "
            "non-speculative or DFlash configurations"
        )

    # Preserve ALL remaining v0.27.1 upstream feature rejection logic.
    unsupported = self._get_v2_model_runner_unsupported_features()
    if unsupported:
        raise ValueError(
            "Model Runner V2 does not yet support: " + ", ".join(unsupported)
        )
    if self.reasoning_config is not None:
        _LOGGER.warning(
            "Model Runner V2 does not yet support the thinking_token_budget "
            "request parameter. Set VLLM_USE_V2_MODEL_RUNNER=0 if required."
        )

    _LOGGER.warning(
        "310P3 experimental V2 Triton *validation* waiver active. "
        "HAS_TRITON remains false; 310P V2 execution is NOT enabled "
        "until its worker, input, sampling and DFlash kernels are ported."
    )


VllmConfig.use_v2_model_runner = property(_patched_use_v2_model_runner)
VllmConfig._validate_v2_model_runner = _patched_validate_v2_model_runner

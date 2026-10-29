import os
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.v1.spec_decode.dflash import DFlashProposer
import vllm_ascend.platform
vllm_ascend.platform.NPUPlatform.pre_register_and_update(None)
import vllm_ascend._310p.model_runner_310p
import vllm_ascend.patch.worker.patch_idex_310
import vllm.config.vllm as cfgv
print("HAS_TRITON in config/vllm.py:", cfgv.HAS_TRITON)
from vllm.config.vllm import VllmConfig
try:
    v = VllmConfig()
    print("VllmConfig() OK; use_v2_model_runner =", v.use_v2_model_runner)
except Exception as e:
    print("VllmConfig() FAILED:", type(e).__name__, str(e)[:400])

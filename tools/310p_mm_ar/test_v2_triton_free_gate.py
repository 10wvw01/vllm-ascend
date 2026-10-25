"""CPU-only contract for the narrowly-scoped 310P3 upstream Triton gate waiver."""
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


GATE = Path(__file__).resolve().parents[2] / (
    "vllm_ascend/patch/platform/patch_use_v2_model_runner.py"
)
WORKER = Path(__file__).resolve().parents[2] / "vllm_ascend/_310p/worker_310p.py"
OPT_IN = "VLLM_ASCEND_310P_EXPERIMENTAL_V2_NO_TRITON"


class _Config:
    def __init__(self):
        self.compilation_config = types.SimpleNamespace(cudagraph_mode="NONE")
        self.parallel_config = types.SimpleNamespace(tensor_parallel_size=2)
        self.speculative_config = types.SimpleNamespace(method="dflash")
        self.reasoning_config = None
        self.unsupported = []

    def _get_v2_model_runner_unsupported_features(self):
        return self.unsupported

    def _validate_v2_model_runner(self):
        raise ValueError("Model Runner V2 requires Triton.")


def _load_gate(*, is_310p=True, has_triton=False):
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    envs = types.ModuleType("vllm.envs")
    envs.VLLM_USE_V2_MODEL_RUNNER = True
    compilation = types.ModuleType("vllm.config.compilation")
    compilation.CUDAGraphMode = types.SimpleNamespace(NONE="NONE")
    config_pkg = types.ModuleType("vllm.config")
    config_pkg.__path__ = []
    config = types.ModuleType("vllm.config.vllm")
    config.VllmConfig = type("VllmConfig", (_Config,), {})
    triton = types.ModuleType("vllm.triton_utils")
    triton.HAS_TRITON = has_triton
    ascend = types.ModuleType("vllm_ascend")
    ascend.__path__ = []
    ascend_utils = types.ModuleType("vllm_ascend.utils")
    ascend_utils.is_310p = Mock(return_value=is_310p)
    vllm.envs = envs
    vllm.config = config_pkg
    config_pkg.compilation = compilation
    config_pkg.vllm = config
    modules = {
        "vllm": vllm, "vllm.envs": envs, "vllm.config": config_pkg,
        "vllm.config.compilation": compilation,
        "vllm.config.vllm": config, "vllm.triton_utils": triton,
        "vllm_ascend": ascend, "vllm_ascend.utils": ascend_utils,
    }
    namespace = {"__name__": "test_310p_v2_gate"}
    with patch.dict(sys.modules, modules):
        exec(compile(GATE.read_text(), str(GATE), "exec"), namespace)
    namespace["_distribution_version"] = lambda name: "0.27.1+empty"
    return namespace, config.VllmConfig, envs


class Gate310PTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {OPT_IN: "1"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_true_opt_in_waives_only_triton(self):
        _, cfg, _ = _load_gate()
        config = cfg()
        self.assertIsNone(config._validate_v2_model_runner())
        self.assertTrue(config.use_v2_model_runner)

    def test_without_exact_opt_in_upstream_rejects(self):
        _, cfg, _ = _load_gate()
        for value in (None, "", "true", "0"):
            with self.subTest(value=value):
                if value is None:
                    os.environ.pop(OPT_IN, None)
                else:
                    os.environ[OPT_IN] = value
                with self.assertRaisesRegex(ValueError, "requires Triton"):
                    cfg()._validate_v2_model_runner()

    def test_other_device_and_actual_triton_defer_to_upstream(self):
        for args in ({"is_310p": False}, {"has_triton": True}):
            with self.subTest(args=args):
                _, cfg, _ = _load_gate(**args)
                with self.assertRaisesRegex(ValueError, "requires Triton"):
                    cfg()._validate_v2_model_runner()

    def test_requires_explicit_model_runner_selection(self):
        _, cfg, envs = _load_gate()
        for forced in (None, False):
            envs.VLLM_USE_V2_MODEL_RUNNER = forced
            with self.assertRaisesRegex(RuntimeError, "requires VLLM_USE_V2"):
                cfg()._validate_v2_model_runner()

    def test_version_guard(self):
        namespace, cfg, _ = _load_gate()
        namespace["_distribution_version"] = lambda name: "0.28.0"
        with self.assertRaisesRegex(RuntimeError, "vLLM 0.27.1"):
            cfg()._validate_v2_model_runner()

    def test_graph_tp_and_other_speculative_rejected(self):
        _, cfg, _ = _load_gate()
        obj = cfg()
        obj.compilation_config.cudagraph_mode = "FULL_DECODE_ONLY"
        with self.assertRaisesRegex(ValueError, "Eager"):
            obj._validate_v2_model_runner()
        obj.compilation_config.cudagraph_mode = "NONE"
        obj.parallel_config.tensor_parallel_size = 4
        with self.assertRaisesRegex(ValueError, "TP=2"):
            obj._validate_v2_model_runner()
        obj.parallel_config.tensor_parallel_size = 2
        obj.speculative_config.method = "eagle"
        with self.assertRaisesRegex(ValueError, "DFlash"):
            obj._validate_v2_model_runner()

    def test_upstream_feature_rejections_remain(self):
        _, cfg, _ = _load_gate()
        obj = cfg()
        obj.unsupported = ["unsupported KV transfer"]
        with self.assertRaisesRegex(ValueError, "unsupported KV transfer"):
            obj._validate_v2_model_runner()
        obj.unsupported = []
        obj.speculative_config = None
        self.assertIsNone(obj._validate_v2_model_runner())

    def test_worker_must_not_silently_instantiate_v1(self):
        source = WORKER.read_text()
        self.assertIn("if self.vllm_config.use_v2_model_runner:", source)
        self.assertIn("Triton-free V2 NPUWorker/ModelRunner is NOT_IMPLEMENTED", source)
        self.assertLess(
            source.index("if self.vllm_config.use_v2_model_runner:"),
            source.index("self.device = self._init_device()"),
        )


if __name__ == "__main__":
    unittest.main()

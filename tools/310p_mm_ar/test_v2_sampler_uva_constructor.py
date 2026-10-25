"""T13/T15 CPU contract: nested sampler constructors cannot allocate UVA."""
import ast
import importlib.util
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "vllm_ascend/_310p/v2"


class TestSamplerUvaContract(unittest.TestCase):
    def test_scoped_bindings_cover_all_five_frozen_states_and_cuda_stream(self):
        s = (SRC / "sampler_state.py").read_text()
        runner = (SRC / "model_runner.py").read_text()
        sampler = (SRC / "sampler.py").read_text()
        for cls in (
            "SamplingStates", "PenaltiesState", "LogitBiasState",
            "BadWordsState", "LogprobTokenIdsState",
        ):
            self.assertIn('"' + cls + '": ' + cls, s)
        self.assertIn('"StructuredOutputsWorker": NPUStructuredOutputsWorker310V2', runner)
        self.assertIn("make_npu_sampler_state_factories(device)", sampler)
        self.assertIn("_clone_upstream_function(Sampler.__init__", sampler)
        self.assertNotIn("super().__init__(*args, **kwargs)", sampler)
        self.assertNotIn("torch.cuda.Stream(", s)
        self.assertNotIn("Sampler.__init__ =", sampler)
        self.assertNotIn("UvaBackedTensor =", s)

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch absent; source ABI contract still runs")
        cls.torch = torch

        # Import only the candidate buffer/constructor modules. CI has no
        # vLLM/torch-npu installation and must not instantiate GPU classes.
        name = "_r2_sampler_uva_isolated"
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(SRC)]
        sys.modules[name] = pkg
        for part in ("staged_buffers", "model_runner", "sampler_state"):
            full = name + "." + part
            if part == "model_runner":
                module = types.ModuleType(full)
                def clone(method, overrides):
                    globs = method.__globals__.copy()
                    missing = set(overrides).difference(globs)
                    if missing:
                        raise RuntimeError(str(sorted(missing)))
                    globs.update(overrides)
                    return types.FunctionType(
                        method.__code__, globs, method.__name__,
                        method.__defaults__, method.__closure__,
                    )
                module._clone_upstream_function = clone
            else:
                spec = importlib.util.spec_from_file_location(
                    full, SRC / (part + ".py")
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[full] = module
                spec.loader.exec_module(module)
            sys.modules[full] = module
        cls.module = sys.modules[name + ".sampler_state"]

    def test_scoped_construct_uses_device_storage_without_global_mutation(self):
        t = self.torch
        def forbidden(*args, **kwargs):
            raise AssertionError("GPU/UVA constructor must not be called")

        # These names deliberately exist in FakeState.__init__.__globals__
        # so the real private-globals cloning contract is exercised.
        global UvaBackedTensor, StagedWriteTensor
        UvaBackedTensor = forbidden
        StagedWriteTensor = forbidden

        class FakeState:
            def __init__(self, n, device):
                self.scalar = UvaBackedTensor(n, dtype=t.float32)
                self.rows = StagedWriteTensor(
                    (n, 5), dtype=t.int32, device=device
                )
                self.scalar.np[1] = 2.5
                self.scalar.copy_to_uva()
                self.rows.stage_write(1, 3, [4, 5])
                self.rows.apply_write()

        state = self.module._scoped_construct(FakeState, "cpu", 3, "cpu")
        self.assertEqual(state.scalar.gpu.tolist(), [0.0, 2.5, 0.0])
        self.assertEqual(state.rows.gpu[1].tolist(), [0, 0, 0, 4, 5])
        self.assertEqual(state.rows.gpu.stride(0), 8)
        with self.assertRaisesRegex(AssertionError, "GPU/UVA"):
            FakeState(3, "cpu")
        self.assertIs(UvaBackedTensor, forbidden)
        self.assertIs(StagedWriteTensor, forbidden)

    def test_grammar_constructor_has_no_cuda_stream(self):
        cls = self.module.NPUStructuredOutputsWorker310V2
        worker = cls(16, 1024, "cpu")
        with self.assertRaisesRegex(NotImplementedError, "T24"):
            worker.apply_grammar_bitmask(None, None, [], None)
        with self.assertRaises(ValueError):
            cls(0, 1024, "cpu")


if __name__ == "__main__":
    unittest.main()

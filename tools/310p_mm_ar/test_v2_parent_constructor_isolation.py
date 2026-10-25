"""No-Torch AST / Python rebind contract for 310P V2 non-UVA parent."""
import ast
import pathlib
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE = ROOT / "vllm_ascend/worker/v2/model_runner.py"
CHILD = ROOT / "vllm_ascend/_310p/v2/model_runner.py"


def _load_bridge_source():
    # Pure stdlib extraction: this module also defines a subclass of the
    # production Ascend Runner and cannot be imported in lightweight CI.
    tree = ast.parse(CHILD.read_text(encoding="utf-8"))
    funcs = [x for x in tree.body if isinstance(x, ast.FunctionDef)
             and x.name == "_clone_upstream_function"]
    namespace = {"types": types}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(CHILD), "exec"), namespace)
    return namespace["_clone_upstream_function"]


class ConstructorIsolationTests(unittest.TestCase):
    def test_parent_constructor_uses_local_globals_not_global_patch(self):
        source = BASE.read_text(encoding="utf-8")
        self.assertIn('_310p_v2_parent_constructor', source)
        self.assertIn('_310p_v2_state_factory', source)
        self.assertIn("with torch_cuda_wrapper():", source)

    def test_true_310p_candidate_remains_unregistered_and_kv_rejected(self):
        text = CHILD.read_text(encoding="utf-8")
        self.assertIn('raise NotImplementedError(', text)
        self.assertNotIn('HAS_TRITON = True', text)
        self.assertNotIn('GPUModelRunner.__init__ =', text)
        worker = (ROOT / "vllm_ascend/_310p/worker_310p.py").read_text()
        self.assertIn("Triton-free V2 NPUWorker/ModelRunner is NOT_IMPLEMENTED", worker)

    def test_rebinding_preserves_original_globals_and_kwdefaults(self):
        def parent(self, value, *, bonus=4):
            return (FAKE_CTOR_DEP(value), bonus)
        existing = {**parent.__globals__, "FAKE_CTOR_DEP": lambda n: n}
        parent = types.FunctionType(parent.__code__, existing,
                                    argdefs=parent.__defaults__,
                                    closure=parent.__closure__)
        parent.__kwdefaults__ = {"bonus": 4}
        original_globals = parent.__globals__
        original_code = parent.__code__
        substitute = _load_bridge_source()(parent, {"FAKE_CTOR_DEP": lambda n: n + 1})
        self.assertEqual(substitute(None, 9), (10, 4))
        self.assertIs(parent.__globals__, original_globals)
        self.assertIs(parent.__code__, original_code)
        self.assertEqual(original_globals["FAKE_CTOR_DEP"](9), 9)

    def test_missing_upstream_global_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "ABI drift"):
            _load_bridge_source()(lambda: None, {"RequestState": object()})

    def test_parent_is_declared_as_true_subclass(self):
        tree = ast.parse(CHILD.read_text(encoding="utf-8"))
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef)
                   and x.name == "NPUModelRunner310V2Candidate")
        self.assertEqual(cls.bases[0].id, "_AscendNPUModelRunner")
        methods = [x.name for x in cls.body if isinstance(x, ast.FunctionDef)]
        self.assertIn("_310p_v2_parent_constructor", methods)
        self.assertIn("initialize_kv_cache", methods)


if __name__ == "__main__":
    unittest.main()

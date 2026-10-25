"""Frozen 0.27.1 copied sample_tokens import/name closure (no vllm import)."""
import ast
import builtins
import pathlib
import symtable
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATH = ROOT / "vllm_ascend/_310p/v2/model_runner.py"


class TestCopiedSampleTokensGlobals(unittest.TestCase):
    def test_decorators_and_annotations_have_module_imports(self):
        src = PATH.read_text()
        tree = ast.parse(src)
        imports = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.update(a.asname or a.name.split(".")[0]
                               for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.update(a.asname or a.name for a in node.names)
        for name in ("torch", "step_eplb_after", "GrammarOutput",
                     "AsyncOutput", "ModelRunnerOutput", "pcp"):
            self.assertIn(name, imports)

    def test_all_free_names_in_copied_method_resolve(self):
        src = PATH.read_text()
        symbols = symtable.symtable(src, str(PATH), "exec")
        module_names = set(dir(builtins))
        tree = ast.parse(src)
        for node in tree.body:
            if isinstance(node, ast.Import):
                module_names.update(a.asname or a.name.split(".")[0]
                                    for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                module_names.update(a.asname or a.name for a in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                module_names.add(node.name)
        cls = next(c for c in symbols.get_children()
                   if c.get_name() == "NPUModelRunner310V2Candidate")
        method = next(m for m in cls.get_children()
                      if m.get_name() == "sample_tokens")
        missing = sorted(sym.get_name() for sym in method.get_symbols()
                         if sym.is_global() and sym.is_referenced()
                         and sym.get_name() not in module_names)
        self.assertEqual(missing, [])

    def test_padded_writer_replaces_original_short_write(self):
        src = PATH.read_text()
        self.assertIn("self._310p_v2_store_draft_tokens(input_batch, draft_tokens)", src)
        self.assertNotIn(
            "self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens",
            src,
        )


if __name__ == "__main__":
    unittest.main()

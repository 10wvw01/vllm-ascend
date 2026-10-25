# SPDX-License-Identifier: Apache-2.0
"""CPU-only test of the Stock+DFlash environment classifier; no vllm import."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import dflash_stock_preflight as m  # noqa: E402


def good():
    versions = dict(m.PINNED)
    versions.update({key: "installed" for key in m.OBSERVE})
    return versions


class TestDflashStockPreflight(unittest.TestCase):
    def test_nominal_exact_contract(self):
        got = m.verdict(good(), {"vllm/dflash": True},
                        {flag: "0" for flag in m.STOCK_DISABLED_FLAGS},
                        {"target": True, "draft": True})
        self.assertTrue(got["static_go"])

    def test_version_or_api_missing_fails(self):
        v = good()
        v["vllm"] = "0.23.0"
        got = m.verdict(v, {"vllm/v1/spec_decode/dflash.py": False},
                        {}, {"target": True, "draft": True})
        self.assertFalse(got["static_go"])
        self.assertTrue(any("PIN_MISMATCH:vllm" in x for x in got["blockers"]))
        self.assertTrue(any("MISSING_SOURCE_API" in x for x in got["blockers"]))

    def test_stock_must_disable_fused(self):
        got = m.verdict(good(), {}, {m.STOCK_DISABLED_FLAGS[0]: "YES"},
                        {"target": True, "draft": True})
        self.assertFalse(got["static_go"])
        self.assertTrue(any("STOCK_CONTAMINATED" in x for x in got["blockers"]))

    def test_missing_model_config_is_not_missing_python_dependency(self):
        got = m.verdict(good(), {}, {}, {"target": True, "draft": False})
        self.assertEqual(got["blockers"], ["MODEL_CONFIG_MISSING:draft"])

    def test_optional_upstream_declaration_not_hard_failure(self):
        v = good()
        v["triton-ascend"] = None
        got = m.verdict(v, {}, {}, {"target": True, "draft": True})
        self.assertTrue(got["static_go"])
        self.assertTrue(any("triton-ascend" in x for x in got["observations"]))

    def test_triton_only_pip_check_exception_is_narrow(self):
        line = "vllm-ascend 0.24.0 requires triton-ascend, which is not installed.\n"
        self.assertEqual(
            m.classify_pip_check(1, line),
            "UPSTREAM_310P_INTENTIONAL_TRITON_MISSING_ONLY")
        self.assertEqual(
            m.classify_pip_check(1, line + "vllm 0.24.0 requires torch, which is not installed."),
            "UNEXPECTED_DEPENDENCY_BREAKAGE")
        self.assertEqual(
            m.classify_pip_check(1, "vllm requires triton>=3.2, but you have triton 3.1."),
            "UNEXPECTED_DEPENDENCY_BREAKAGE")
        self.assertEqual(m.classify_pip_check(0, "No broken requirements found."), "PASS")

    def test_source_api_scan_does_not_import_external_package(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "v1/worker").mkdir(parents=True)
            (root / "v1/worker/cp_utils.py").write_text(
                "def get_total_cp_world_size(): return 1\n", encoding="utf-8")
            got = m.check_api({"vllm": root, "vllm_ascend": None})
            self.assertTrue(got["vllm/cp_utils.get_total_cp_world_size"])
            self.assertFalse(got["vllm/model_executor/models/qwen3_dflash.py"])


if __name__ == "__main__":
    unittest.main()

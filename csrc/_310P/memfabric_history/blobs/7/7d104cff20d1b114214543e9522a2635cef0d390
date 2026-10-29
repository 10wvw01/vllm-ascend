#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A1环境漂移只读取证的纯CPU拒绝性测试，禁止导入NPU运行时。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[1] / "perf" / "a1_reentry_readonly.py"
spec = importlib.util.spec_from_file_location("a1_reentry_readonly", MODULE)
assert spec is not None and spec.loader is not None
reentry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reentry)


class ReentryReadonlyUT(unittest.TestCase):
    def test_env_presence_means_hazard_even_value_zero(self):
        env = {"ASCEND_RT_VISIBLE_DEVICES": "4,5", "MF_KFC_BYPASS": "0"}
        obs = reentry.safe_env_view(env)
        self.assertEqual(obs["ASCEND_RT_VISIBLE_DEVICES"]["value"], "4,5")
        self.assertTrue(obs["MF_KFC_BYPASS"]["present"])
        self.assertFalse(obs["MF_SDMA_ORCH_DEPLOY_HELPER"]["present"])
        self.assertNotIn("SECRET_TOKEN", obs)

    def test_file_facts_missing_abs_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "libmf_hybm_core.so"
            path.write_bytes(b"SAFE_STATIC_BYTES")
            data = reentry.file_facts(str(path))
            self.assertEqual(data["status"], "FILE")
            self.assertEqual(data["size"], len(b"SAFE_STATIC_BYTES"))
            self.assertEqual(len(data["sha256"]), 64)
            self.assertEqual(reentry.file_facts(str(root / "none"))["status"], "MISSING")
            self.assertEqual(reentry.file_facts("relative.so")["status"], "NOT_ABSOLUTE")
            self.assertEqual(reentry.file_facts(None)["status"], "NOT_SUPPLIED")
            link = root / "link.so"
            link.symlink_to(path)
            self.assertTrue(reentry.file_facts(str(link))["is_symlink"])

    def test_mf_sources_require_all_three_setaccess_sites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            h = root / reentry.SOURCE_PATHS[0]
            a = root / reentry.SOURCE_PATHS[1]
            b = root / reentry.SOURCE_PATHS[2]
            for p in (h, a, b):
                p.parent.mkdir(parents=True, exist_ok=True)
            h.write_text("HybmPhysicalDeviceIdOf")
            a.write_text("HybmPhysicalDeviceIdOf AclrtMemSetAccess(")
            b.write_text("HybmPhysicalDeviceIdOf AclrtMemSetAccess( AclrtMemSetAccess(")
            self.assertTrue(reentry.source_facts(root)["three_setaccess_mapper_static_hint"])
            b.write_text("AclrtMemSetAccess( AclrtMemSetAccess(")
            self.assertFalse(reentry.source_facts(root)["three_setaccess_mapper_static_hint"])

    def test_paths_parser_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "paths.json"
            p.write_text(json.dumps({"host_system": "/tmp/a", "host_python": None}))
            self.assertEqual(reentry.parse_paths(p)["host_system"], "/tmp/a")
            p.write_text(json.dumps({"BAD_KEY": "/tmp/a"}))
            with self.assertRaises(ValueError):
                reentry.parse_paths(p)
            p.write_text(json.dumps({"device_so": 123}))
            with self.assertRaises(ValueError):
                reentry.parse_paths(p)
            p.write_text("[]")
            with self.assertRaises(ValueError):
                reentry.parse_paths(p)

    def test_explicit_g0_never_approved(self):
        self.assertIn('"G0_APPROVED": False', MODULE.read_text())
        self.assertIn('"DEPLOY_APPROVED": False', MODULE.read_text())
        self.assertIn('"SMOKE_APPROVED": False', MODULE.read_text())
        for prohibited in ("torchrun", "kfc", "rm -rf", "git reset --hard"):
            self.assertNotIn("subprocess.run([" + repr(prohibited), MODULE.read_text())


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""R1 Smoke安全自动续跑门禁的无设备CPU合成用例。"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

PERF = Path(__file__).resolve().parents[1] / "perf"
sys.path.insert(0, str(PERF))
import u1_spec

spec = importlib.util.spec_from_file_location("a1_r1_smoke_gate", PERF / "a1_r1_smoke_gate.py")
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class SmokeGateUT(unittest.TestCase):
    def create_fake(self, root: Path):
        smoke = root / "smoke"
        proof = root / "proof"
        smoke.mkdir()
        proof.mkdir()
        sha = gate.SHA
        sessions = smoke / "sessions"
        sessions.mkdir()
        (smoke / "manifest.json").write_text(json.dumps({
            "experiment": "U1_FULLCHAIN", "scope": "large", "phase": "smoke",
            "physical_devices": "2,3", "session_ids": [u1_spec.session_id("large", i) for i in (1, 2)],
            **sha,
        }))
        for stage in (1, 2):
            arm = "stock_fp16" if stage == 1 else "memfabric_d4"
            sid = u1_spec.session_id("large", stage)
            sd = sessions / sid
            sd.mkdir()
            (sd / "session.log").write_text("Smoke completed with no errors")
            for rank in (0, 1):
                samples = [{"i": i, "variant": (i + 3) % 2,
                            "event_us": 200. + i, "wall_us": 210. + i}
                           for i in range(6)]
                anchors = [{"i": i, "variant": i % 2,
                            "event_us": 200. + i, "wall_us": 210. + i}
                           for i in range(8)]
                data = {
                    "experiment": "U1_SMOKE_ONLY", "scope": "large", "phase": "smoke",
                    "session": sid, "stage": stage, "arm": arm, "rank": rank,
                    "TP": 2, "K": 2048, "N": 2048, "dtype": "fp16",
                    "M_order": [2048], "profile": False,
                    "reference": "CPU_GLOO_FP32_SUM_TO_FP16",
                    "stock_collective": "FP16_HCCL_no_FP32_conversion",
                    "comm_isolation": "separate process; never HCCL with MemFabric pool",
                    **sha,
                    "results": {
                        "2048": {"M": 2048, "arm": arm, "warmup": 3,
                                  "iters": 6, "status": None if stage == 1 else 0,
                                  "bit_exact": True, "samples": samples},
                        "_anchor_after": {"M": 2048, "samples": anchors},
                    },
                }
                (sd / f"{sid}_rank{rank}.json").write_text(json.dumps(data))
        evidence = []
        for n in range(6):
            name = f"rank{n%2}_load_or_die{n}.log"
            (proof / name).write_text("verified raw evidence " + str(n))
            evidence.append(name)
        audit = {
            "scope": "A1_R1_M2048_SMOKE_RUNTIME_ATTESTED",
            "physical_devices": "2,3",
            "evidence_files": evidence,
            **{k: True for k in gate.REQUIRED_CHECKS},
        }
        att = proof / "att.json"
        att.write_text(json.dumps(audit))
        return smoke, proof, att

    def test_all_positive_fields_still_need_original_proof_files(self):
        with tempfile.TemporaryDirectory() as d:
            smoke, proof, att = self.create_fake(Path(d))
            self.assertEqual(gate.evaluate(smoke, proof, att)["decision"], "ALLOW_SCREEN")
            (proof / "rank0_load_or_die0.log").unlink()
            with self.assertRaises(ValueError):
                gate.evaluate(smoke, proof, att)

    def test_bit_mismatch_stops(self):
        with tempfile.TemporaryDirectory() as d:
            smoke, proof, att = self.create_fake(Path(d))
            p = smoke / "sessions" / "02_large_memfabric_d4" / "02_large_memfabric_d4_rank1.json"
            o = json.loads(p.read_text())
            o["results"]["2048"]["bit_exact"] = False
            p.write_text(json.dumps(o))
            with self.assertRaisesRegex(ValueError, "bit_exact"):
                gate.evaluate(smoke, proof, att)

    def test_runtime_identity_attestation_missing_stops(self):
        with tempfile.TemporaryDirectory() as d:
            smoke, proof, att = self.create_fake(Path(d))
            data = json.loads(att.read_text())
            data["rank0_runtime_d4_kernel"] = False
            att.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "rank0_runtime_d4_kernel"):
                gate.evaluate(smoke, proof, att)

    def test_wrong_device_mapping_or_both_rank_logs_missing_stops(self):
        with tempfile.TemporaryDirectory() as d:
            smoke, proof, att = self.create_fake(Path(d))
            manifest = smoke / "manifest.json"
            dat = json.loads(manifest.read_text())
            dat["physical_devices"] = "6,7"
            manifest.write_text(json.dumps(dat))
            with self.assertRaisesRegex(ValueError, "die pair"):
                gate.evaluate(smoke, proof, att)
            dat["physical_devices"] = "2,3"
            manifest.write_text(json.dumps(dat))
            p = smoke / "sessions" / "01_large_stock_fp16" / "01_large_stock_fp16_rank1.json"
            p.unlink()
            with self.assertRaises(FileNotFoundError):
                gate.evaluate(smoke, proof, att)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
"""五层实验CPU合同：检查源码约束、数组平衡及分析器负例。无NPU依赖。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
PERF=ROOT/"test"/"perf"
KERNEL=ROOT/"src"/"op_kernel"
SPEC=importlib.util.spec_from_file_location("five_arm_assess",PERF/"five_arm_assess.py")
ASSESS=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSESS)


def source(p):
    return p.read_text(encoding="utf-8")


class FiveArmContract(unittest.TestCase):
    def test_balanced_schedule(self):
        self.assertEqual(len(ASSESS.SCHEDULE),10)
        self.assertEqual(ASSESS.SCHEDULE,tuple(reversed(ASSESS.SCHEDULE)))
        self.assertEqual({arm:ASSESS.SCHEDULE.count(arm) for arm in set(ASSESS.SCHEDULE)},
                         dict.fromkeys(("stock","b1","b2","d4","e1"),2))

    def test_full_mm_then_ar_gate(self):
        code=source(KERNEL/"mf310p_v2_b2_full_mm_ar.asc")
        self.assertEqual(code.count("mm.IterateAll(cg);"),1)
        # MM循环结束后才有唯一生产端WaitReady，且仅在下一循环里发送。
        mm=code.index("mm.IterateAll(cg);")
        ready=code.index("WaitReady(ibGm, ibUb, core);")
        signal=code.index("smem_shm_sdma_signal_at(")
        self.assertTrue(mm < ready < signal)
        self.assertEqual(code.count("WaitReady(ibGm, ibUb, core);"),1)
        self.assertEqual(code.count("WaitAddDone(ibGm, ibUb, core);"),1)
        self.assertTrue(code.index("WaitAddDone(ibGm, ibUb, core);") <
                        code.index("smem_shm_sdma_quiet_at("))
        self.assertIn("每个完整分包的Add必须全部完成",code)
        self.assertIn("kProductionAddTile",code)
        self.assertIn("smem_shm_sdma_quiet_at(",code)
        self.assertIn("kTagDone",code)

    def test_e1_dcci_and_protocol_preservation(self):
        src=source(KERNEL/"mf310p_v2_dcci_entire_candidate.asc")
        formal=source(KERNEL/"mf310p_v2_kernel.asc")
        self.assertEqual(src.count("FlushCoreDCache(ownC + start + coreOff);"),1)
        self.assertEqual(src.count("FlushCoreDCache(ownPeerC + base);"),2)
        self.assertEqual(src.count("DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(fence);"),1)
        self.assertIn("ReadyTree(ibGm, ibUb, core);",src)
        self.assertIn("IBWait<true>(ibGm, ibUb, core + 1, kIBReady);",src)
        self.assertEqual(sum(src.count(f"IBWait<true>(ibGm, ibUb, {i}, kIBReady);") for i in (1,2,4,6)),4)
        self.assertEqual(src.count("WaitReady(ibGm, ibUb, core);"),0)
        for phrase in ("ReleaseAdd(ibGm, ibUb, core);",
                       "WaitAddDone(ibGm, ibUb, core);",
                       "smem_shm_sdma_signal_at(", "smem_shm_sdma_wait_at(",
                       "smem_shm_sdma_quiet_at(", "kTagDone"):
            self.assertIn(phrase,src)
            self.assertIn(phrase,formal)
        self.assertNotIn('extern "C" int mf310p_v2_trial_launch_async',src)
        self.assertNotIn('mf310p_v2_ib_probe_kernel',src)
        self.assertIn('mf310p_v2_dcci_entire_launch_async',src)

    def test_campaign_does_not_replace_binaries(self):
        sh=source(PERF/"five_arm_campaign.sh")
        build=source(PERF/"five_arm_build_e.sh")
        self.assertIn("binaries_before.sha256",sh)
        self.assertIn("binaries_after.sha256",sh)
        self.assertIn("A1_FIVE_ARM_G0_PASSED",sh)
        self.assertIn("MF_KFC_BYPASS",sh)
        self.assertIn("FIVE_ARM_E_DCCI_ENTIRE_BUILD_V1",build)
        self.assertIn('-x asc "$SRC" -x asc "$SRC_B2" -x none',build)
        self.assertNotIn("setup.py build_ext --inplace",build)
        self.assertNotIn("git reset --hard",sh)

    def _generate(self,folder,drift=0.0,wrong_rank=False):
        mset=(2048,4096)
        for stage,arm in enumerate(ASSESS.SCHEDULE,1):
            for rank in (0,1):
                bias=(1+drift) if stage==10 and arm=="stock" else 1.
                results={}
                for m in mset:
                    val=(m/4+100+rank*0.1)*bias
                    vals=[val]*100
                    results[str(m)]={
                        "M":m,"rank":rank,"arm":arm,
                        "bit_exact":True,"protocol_status":None if arm=="stock" else 0,
                        "warmup":20,"iters":100,"npu_event_us":vals,
                        "host_wall_us":[val+10]*100
                    }
                obj={
                    "schema":"A1_FIVE_ARM_TP2_V1","stage":stage,"arm":arm,
                    "rank":1-rank if wrong_rank and stage==1 else rank,
                    "TP":2,"K":2048,"N":2048,"weight_format":"NZ29",
                    "jit_compile":False,"allow_internal_format":True,
                    "profile":False,"reference":"CPU_GLOO_FP32_SUM_TO_FP16",
                    "results":results,"candidate":{"work_head":"test-head"}
                        if arm in ("b2","e1") else None
                }
                (folder/f"{stage:02d}_{arm}_rank{rank}.json").write_text(
                    json.dumps(obj),encoding="utf-8")

    def test_analyzer_valid_complete(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self._generate(p)
            out=ASSESS.analyze(p)
            self.assertEqual(out["status"],"FIVE_ARM_TIMING_VALID")
            self.assertEqual(len(out["comparisons"]),2)
            self.assertEqual(len(out["raw_sessions"]),10)

    def test_analyzer_drift_nogo(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self._generate(p,drift=0.08)
            self.assertEqual(ASSESS.analyze(p)["status"],"INCONCLUSIVE_DRIFT")

    def test_analyzer_bad_rank_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self._generate(p,wrong_rank=True)
            with self.assertRaises(ValueError):
                ASSESS.analyze(p)

    def test_analyzer_missing_file_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self._generate(p)
            (p/"03_b2_rank1.json").unlink()
            with self.assertRaises(FileNotFoundError):
                ASSESS.analyze(p)


if __name__=="__main__":
    unittest.main()

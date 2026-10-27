# SPDX-License-Identifier: Apache-2.0
"""E2性能攻坚CPU合同：源代码阶段依赖、IB事件唯一生产消费、证据负例。"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
PERF=ROOT/"test"/"perf"
KERNEL=ROOT/"src"/"op_kernel"
SPEC=importlib.util.spec_from_file_location("e2_assess",PERF/"e2_assess.py")
ASSESS=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSESS)


def source(p):
    return p.read_text(encoding="utf-8")


class E2PipelineContract(unittest.TestCase):
    def test_schedule_is_balanced_and_palindrome(self):
        self.assertEqual(ASSESS.SCHEDULE, tuple(reversed(ASSESS.SCHEDULE)))
        self.assertEqual(len(ASSESS.SCHEDULE),10)
        self.assertEqual({a:ASSESS.SCHEDULE.count(a) for a in set(ASSESS.SCHEDULE)},
                         dict.fromkeys(("stock","e1","e2r","e2p","e2p512"),2))

    def test_kernel_exports_only_candidate_symbols(self):
        k=source(KERNEL/"mf310p_v2_e2_pipeline.asc")
        self.assertIn('extern "C" int mf310p_v2_e2_pipeline_launch_async',k)
        self.assertNotIn('extern "C" int mf310p_v2_trial_launch_async',k)
        self.assertNotIn("mf310p_v2_ib_probe_kernel",k)
        self.assertIn("E2PipelineKernel<tiling##ROWS, ROWS, MODE>",k)
        self.assertIn("V2_CASE(512)",k)
        self.assertIn("V2_CASE(1024)",k)
        self.assertNotIn("V2_CASE(256)",k)

    def test_release_tree_has_seven_unique_consumers(self):
        k=source(KERNEL/"mf310p_v2_e2_pipeline.asc")
        beg=k.index("__aicore__ inline void ReleaseTree(")
        end=k.index("constexpr uint64_t kE2TagReady",beg)
        tree=k[beg:end]
        for i in (1,2,4,6):
            self.assertEqual(tree.count(f"IBSet<true>(ibGm, ibUb, 0, kIBReleaseBase + {i});"),1)
        self.assertIn("IBSet<true>(ibGm, ibUb, core, kIBReleaseBase + core + 1);",tree)
        self.assertIn("IBWait<true>(ibGm, ibUb, core - 1, kIBReleaseBase + core);",tree)
        self.assertIn("IBWait<true>(ibGm, ibUb, 0, kIBReleaseBase + 1);",tree)
        self.assertEqual(k.count("ReleaseTree(ibGm, ibUb, core);"),2)
        self.assertEqual(k.count("ReadyTree(ibGm, ibUb, core);"),2)
        self.assertEqual(k.count("WaitAddDone(ibGm, ibUb, core);"),1)

    def test_preinvalidate_occurs_before_first_mm_and_data_signal(self):
        k=source(KERNEL/"mf310p_v2_e2_pipeline.asc")
        a=k.index("if constexpr (PRE_INVALIDATE) {")
        b=k.index("FlushCoreDCache(ownPeerC + static_cast<uint64_t>(core) * perCoreBytes);")
        c=k.index("ReadyTree(ibGm, ibUb, core);",a)
        d=k.index("kE2TagReady), status, 7);")
        e=k.index("smem_shm_sdma_wait_at(reserved);",d)
        f=k.index("ReleaseAdd(ibGm, ibUb, core);",e)
        g=k.index("mm.IterateAll(cg);",f)
        h=k.index("smem_shm_sdma_signal_at(",g)
        self.assertTrue(a<b<c<d<e<f<g<h)
        self.assertIn("ready.imm == kE2TagReady",k)
        self.assertIn("ready.dst == expectedCtrlDst",k)
        self.assertIn("if constexpr (!PRE_INVALIDATE)",k)

    def test_ready_control_has_unique_tag_and_done_is_last(self):
        k=source(KERNEL/"mf310p_v2_e2_pipeline.asc")
        self.assertIn("kE2TagReady = 0x4D46324532524459ULL",k)
        self.assertNotIn("kE2TagReady = kTagDone",k)
        self.assertIn("smem_shm_sdma_quiet_at(reserved)",k)
        self.assertIn("kTagDone",k)
        self.assertLess(k.index("smem_shm_sdma_quiet_at(reserved)"),
                        k.index("kTagDone), status, 5)"))
        self.assertEqual(k.count("FlushCoreDCache(ownPeerC + base);"),2)

    def test_private_host_mode_geometry_and_first_write(self):
        h=source(PERF/"e2_private_host.cpp")
        for k in ("mf310p_v2_e2_pipeline_launch_async","mf310p_e2_run_private",
                  "kind==5?512u:1024u","kind==3?1u:kind>=4?2u:1u",
                  "aclrtSynchronizeDevice","aclrtMemsetAsync"):
            self.assertIn(k,h)
        self.assertNotIn("MF_KFC_BYPASS",h)

    def test_build_two_generations_and_no_production_link(self):
        sh=source(PERF/"e2_build.sh")
        self.assertIn('-x asc "$SRC" -x asc "$SRC_B2" -x asc "$SRC_E2" -x none',sh)
        self.assertIn("E2_PIPELINE_BUILD_V1",sh)
        self.assertIn("mf310p_v2_e2_pipeline_launch_async",sh)
        self.assertIn("mf310p_e2_run_private",sh)
        self.assertNotIn("git reset --hard",sh)
        self.assertNotIn("setup.py build_ext --inplace",sh)

    def _data(self,p,drift=0.,wrong_rank=False):
        for stage,arm in enumerate(ASSESS.SCHEDULE,1):
            for rank in (0,1):
                results={}
                for m in ASSESS.MS:
                    value=(m/3+round(100*stage,2)+rank/10)
                    if drift and arm=="e2p" and stage==7:
                        value*=1+drift
                    results[str(m)]={
                        "M":m,"rank":rank,"arm":arm,
                        "bit_exact":True,
                        "protocol_status":None if arm=="stock" else 0,
                        "warmup":20,"iters":100,
                        "npu_event_us":[value]*100,
                        "host_wall_us":[value+15]*100,
                    }
                obj={
                    "schema":"E2_PIPELINE_TP2_V1","stage":stage,"arm":arm,
                    "rank":(1-rank if wrong_rank and stage==1 else rank),
                    "TP":2,"K":2048,"N":2048,
                    "weight_format":"NZ29","jit_compile":False,
                    "allow_internal_format":True,"profile":False,
                    "reference":"CPU_GLOO_FP32_SUM_TO_FP16",
                    "results":results,
                    "candidate":{"work_head":"test-E2"} if arm!="stock" else None,
                }
                (p/f"{stage:02d}_{arm}_rank{rank}.json").write_text(
                    json.dumps(obj),encoding="utf-8")

    def test_assessor_accepts_complete_e2_campaign(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self._data(p)
            result=ASSESS.analyze(p)
            self.assertEqual(result["status"],"E2_FULL_CHAIN_TIMING_VALID")
            self.assertEqual(len(result["comparisons"]),2)
            self.assertIn("E2_P_512_vs_E1_pct",result["comparisons"][0])
            self.assertEqual(len(result["raw_sessions"]),10)

    def test_assessor_fails_closed_on_drift(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self._data(p,drift=.5)
            self.assertEqual(ASSESS.analyze(p)["status"],"INCONCLUSIVE_DRIFT")

    def test_assessor_rejects_corrupt_rank(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self._data(p,wrong_rank=True)
            with self.assertRaises(ValueError):
                ASSESS.analyze(p)

    def test_assessor_rejects_missing_e2_arm(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self._data(p)
            (p/"05_e2p512_rank1.json").unlink()
            with self.assertRaises(FileNotFoundError):
                ASSESS.analyze(p)


if __name__=="__main__":
    unittest.main()

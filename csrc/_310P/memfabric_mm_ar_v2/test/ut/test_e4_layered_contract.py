# SPDX-License-Identifier: Apache-2.0
"""E4纯CPU协议审查：真实2048组、4MiB邮包、AddTile与全M规划。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
PERF=ROOT/"test/perf"
sys.path.insert(0,str(PERF))
import e4_shape_plan as P
import e4_assess as A

def source(path: str) -> str:
    return (ROOT/path).read_text(encoding="utf-8")

class E4Contracts(unittest.TestCase):
    def test_full_integer_domain_exact_and_no_overrun(self):
        r=P.audit_all_shapes()
        self.assertEqual(r["shapes"],2049)
        self.assertEqual(r["unknown_m_routes_to_stock"],2049)
        self.assertEqual(r["max_padding_rows"],255)
        for m in range(2048,4097):
            p=P.plan(m)
            self.assertEqual(p.requested_m,m)
            self.assertEqual(sum(g.rows for g in p.groups),p.padded_m)
            self.assertEqual(sum(g.valid_rows for g in p.groups),m)
            self.assertLessEqual(p.padded_m,4096)
            self.assertEqual(p.groups[0].start,0)
            for i,g in enumerate(p.groups):
                self.assertIn(g.rows,P.LEGAL_GROUPS)
                self.assertTrue(0<g.valid_rows<=g.rows)
                self.assertEqual(sum(g.packet_rows),g.rows)
                self.assertTrue(all(x*4096<=P.MAX_SDMA_PACKET_BYTES
                                    for x in g.packet_rows))
                if i: self.assertEqual(g.start,p.groups[i-1].start+p.groups[i-1].rows)
                self.assertEqual(P.choose_default_route(m),"stock")

    def test_boundaries_and_mixed_group(self):
        for m,expected,pad in (
            (2048,(2048,),0),
            (2049,(2048,256),255),
            (2560,(2048,512),0),
            (3072,(2048,1024),0),
            (3073,(2048,1024,256),255),
            (3584,(2048,1024,512),0),
            (4095,(2048,2048),1),
            (4096,(2048,2048),0),
        ):
            p=P.plan(m)
            self.assertEqual(tuple(g.rows for g in p.groups),expected)
            self.assertEqual(p.padding_rows,pad)
        for bad in (0,1024,2047,4097,99999,True,0.1,"4096",None):
            with self.assertRaises(ValueError):P.plan(bad)

    def test_fail_closed_route_requires_same_m_positive_l3(self):
        m=4096
        e={"m":m,"tp":2,"kn":(2048,2048),"stock_nz":True,
           "rank_bit_exact":True,"l1_stable":True,
           "l3_confirmed":True,"no_graph_mismatch":True,
           "gain_vs_stock_pct":0.5}
        self.assertEqual(P.choose_default_route(m,{m:e}),"e4")
        for key in tuple(e):
            altered=dict(e);altered.pop(key)
            self.assertEqual(P.choose_default_route(m,{m:altered}),"stock")
        self.assertEqual(P.choose_default_route(4095,{m:e}),"stock")
        self.assertEqual(P.choose_default_route(m,{m:dict(e,gain_vs_stock_pct=-.001)}),"stock")

    def test_kernel_variants_share_mm_transport_and_packet_guard(self):
        s=source("src/op_kernel/mf310p_v2_e4_layered_candidates.asc")
        self.assertIn("template <uint32_t ADD_TILE, bool PACKET_ADD>",s)
        for tile in (2048,4096,8192):
            self.assertIn(f"case {0 if tile==2048 else 1 if tile==4096 else 2}: E4_CASE({tile}, false)",s)
        self.assertIn("case 3: E4_CASE(2048, true)",s)
        self.assertIn("Matmul<AT, BT, CT, BiasT, tiling1024> mm;",s)
        self.assertIn("kComputeCalls = 2",s)
        self.assertIn("kE3PacketBytes == 4ULL * 1024 * 1024",s)
        self.assertIn("E3WaitGroup(reserved, expectedRecvDst, prev, status, 2)",s)
        self.assertIn("E4WaitPacketAndRelease(reserved, expectedRecvDst, prev,",s)
        self.assertIn("received == 3",s)
        self.assertIn("mail.imm == (kTagBase | tag)",s)
        self.assertIn("mail.dst == expectedRecvDst +",s)
        self.assertIn("mail.len == kE3PacketBytes",s)
        self.assertEqual(s.count("smem_shm_sdma_signal_at("),2) # data + DONE
        self.assertIn("WaitAddDone(ibGm, ibUb, core)",s)
        self.assertIn("smem_shm_sdma_quiet_at(reserved)",s)

    def test_packet_add_release_has_single_mail_consumer(self):
        s=source("src/op_kernel/mf310p_v2_e4_layered_candidates.asc")
        a=s.index("__aicore__ inline void E4WaitPacketAndRelease(")
        b=s.index("template <uint32_t ADD_TILE, bool PACKET_ADD>",a)
        f=s[a:b]
        self.assertIn("if (core != 0) {",f)
        self.assertIn("IBWait<true>(ibGm, ibUb, 0, kIBReleaseBase + core);",f)
        self.assertIn("const uint32_t first = (part == 0) ? 1 : 4;",f)
        self.assertIn("const uint32_t last = (part == 0) ? 3 : 7;",f)
        self.assertIn("for (uint32_t b = first; b <= last; ++b)",f)
        self.assertIn("!(received & bit)",f)
        self.assertIn("CheckOrTrap(valid, status, code)",f)
        # Core0 single reader. Nonzero cores must never consume peer mail.
        self.assertLess(f.index("if (core != 0)"),f.index("smem_shm_sdma_wait_at(reserved)"))
        self.assertNotIn("ReleaseAdd(ibGm, ibUb, core);\n            E4Wait",s)
        self.assertIn("WaitAddDone(ibGm, ibUb, core)",s)

    def test_host_abi_build_identity_and_runner(self):
        h=source("test/perf/e4_private_host.cpp")
        b=source("test/perf/e4_build.sh")
        r=source("test/perf/e4_arm.py")
        self.assertIn("s->chunk_rows!=1024",h)
        self.assertIn("kind<2 || kind>7",h)
        self.assertIn("kind-4u",h)
        self.assertIn("mf310p_v2_e4_group2048_launch_async",h)
        self.assertIn("-x asc \"$SRC\" -x asc \"$SRC_E3\" -x asc \"$SRC_E4\" -x none",b)
        self.assertIn("E4_LAYERED_ADD_BUILD_V1",b)
        self.assertIn("e4_source_sha256",b)
        self.assertIn("mf310p_e4_run_private",r)
        self.assertIn("torch.equal(got,ref)",r)
        self.assertIn("allow_internal_format=True",r)
        self.assertIn("jit_compile=False",r)
        self.assertIn("MF_KFC_BYPASS",r)

    def _dataset(self,p,slow_e4=None):
        values={"stock":2000.0,"e1":2200.0,"e3g":2100.0,
                "e4t2":2100.0,"e4t4":2050.0,"e4t8":2080.0,"e4p":2020.0}
        for stage,arm in enumerate(A.SCHEDULE,1):
            for rank in range(2):
                results={}
                for m in A.MS:
                    v=values[arm]+rank*.1+(m-2048)*.4
                    if slow_e4 and (arm,stage)==slow_e4:v*=1.25
                    results[str(m)]={
                        "M":m,"rank":rank,"arm":arm,"bit_exact":True,
                        "protocol_status":None if arm=="stock" else 0,
                        "warmup":20,"iters":100,
                        "npu_event_us":[v]*100,"host_wall_us":[v+10]*100,
                    }
                item={"schema":"E4_LAYERED_TP2_V1","stage":stage,"arm":arm,
                      "rank":rank,"TP":2,"K":2048,"N":2048,
                      "weight_format":"NZ29","jit_compile":False,
                      "allow_internal_format":True,"profile":False,
                      "reference":"CPU_GLOO_FP32_SUM_TO_FP16",
                      "results":results,
                      "candidate":{"work_head":"e4-tested"} if arm!="stock" else None}
                (p/f"{stage:02d}_{arm}_rank{rank}.json").write_text(json.dumps(item))

    def test_assessor_good_drift_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self._dataset(p)
            good=A.analyze(p)
            self.assertEqual(good["status"],"E4_LAYERED_TIMING_VALID")
            self.assertEqual(len(good["raw_sessions"]),14)
            self.assertEqual(good["comparisons"][0]["best_E4_arm"],"e4p")
            (p/"07_e4p_rank0.json").unlink()
            with self.assertRaises(FileNotFoundError):A.analyze(p)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self._dataset(p,slow_e4=("e4p",8))
            self.assertEqual(A.analyze(p)["status"],"INCONCLUSIVE_DRIFT")


if __name__=="__main__":
    unittest.main()

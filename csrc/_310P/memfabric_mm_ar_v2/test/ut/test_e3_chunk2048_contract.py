# SPDX-License-Identifier: Apache-2.0
"""E3 2048完整MM组、4MiB SDMA分包、双Rank同构对照的纯CPU合同。"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
PERF=ROOT/"test"/"perf"
KERNEL=ROOT/"src"/"op_kernel"
SPEC=importlib.util.spec_from_file_location("e3_assess",PERF/"e3_assess.py")
ASSESS=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSESS)


def src(path):
    return path.read_text(encoding="utf-8")


class E3Chunk2048(unittest.TestCase):
    def test_reversible_eight_session_schedule(self):
        self.assertEqual(ASSESS.SCHEDULE,tuple(reversed(ASSESS.SCHEDULE)))
        self.assertEqual(len(ASSESS.SCHEDULE),8)
        self.assertEqual({x:ASSESS.SCHEDULE.count(x) for x in set(ASSESS.SCHEDULE)},
                         dict.fromkeys(("stock","e1","e3g","e3t"),2))

    def test_true_group_geometry_and_transport_limit(self):
        k=src(KERNEL/"mf310p_v2_e3_group2048.asc")
        self.assertIn("kE3GroupRows = 2048;",k)
        self.assertIn("kE3PacketRows = 1024;",k)
        self.assertIn("kE3PacketsPerGroup = kE3GroupRows / kE3PacketRows;",k)
        self.assertIn("kE3PacketBytes == 4ULL * 1024 * 1024",k)
        self.assertNotIn("chunkBytes, kTagBase | k",k)
        self.assertIn("kE3PacketBytes, kTagBase | tag",k)
        self.assertIn("kE3GroupRows * kRowBytes == 8ULL * 1024 * 1024",k)
        self.assertEqual(k.count("smem_shm_sdma_signal_at("),2)  # Packet + DONE
        self.assertIn("kTagDone",k)

    def test_two_distinct_packet_mail_all_received_before_add(self):
        k=src(KERNEL/"mf310p_v2_e3_group2048.asc")
        self.assertIn("uint32_t received = 0;",k)
        self.assertIn("received |= bit;",k)
        self.assertIn("CheckOrTrap(received == 3, status, code);",k)
        self.assertIn("(received & bit) == 0",k)
        self.assertIn("mail.imm == (kTagBase | tag)",k)
        self.assertIn("mail.len == kE3PacketBytes",k)
        self.assertIn("mail.dst == expectedRecvDst +",k)
        self.assertLess(k.index("E3WaitGroup(reserved, expectedRecvDst, prev, status, 2);"),
                        k.index("ReleaseAdd(ibGm, ibUb, core);"))
        self.assertLess(k.index("E3WaitGroup(reserved, expectedRecvDst, last, status, 3);"),
                        k.rindex("ReleaseAdd(ibGm, ibUb, core);"))
        self.assertIn("smem_shm_sdma_quiet_at(reserved)",k)

    def test_g_vs_t_only_mm_tile_changes(self):
        k=src(KERNEL/"mf310p_v2_e3_group2048.asc")
        self.assertIn("E3Group2048Kernel<tiling1024, 1024>",k)
        self.assertIn("E3Group2048Kernel<e3Tiling2048, 2048>",k)
        self.assertIn("kComputeCalls = kE3GroupRows / MM_ROWS;",k)
        self.assertIn("mm.SetOrgShape(kComputeRowsPerCore, kN, kK, kK, kN);",k)
        self.assertIn("mm.IterateAll(cg);",k)
        self.assertIn("ReadyTree(ibGm, ibUb, core);",k)
        self.assertIn("FlushCoreDCache(ownC + start + coreOff);",k)
        self.assertNotIn("PRE_INVALIDATE",k)
        self.assertNotIn("kE2TagReady",k)

    def test_private_host_preserves_1024_pool_abi(self):
        h=src(PERF/"e3_private_host.cpp")
        self.assertIn("s->chunk_rows!=1024",h)
        self.assertIn("const uint32_t logicalChunk=(kind==2)?1024u:2048u;",h)
        self.assertIn("kind==3?1024u:2048u",h)
        self.assertIn("s->layout.arena_bytes",h)
        self.assertIn("aclrtSynchronizeDevice",h)
        self.assertIn("aclrtMemsetAsync",h)
        self.assertIn("mf310p_e3_run_private",h)

    def test_build_and_route_gates(self):
        b=src(PERF/"e3_build.sh")
        s=src(PERF/"e3_campaign.sh")
        a=src(PERF/"e3_arm.py")
        self.assertIn('-x asc "$SRC" -x asc "$SRC_E3" -x none',b)
        self.assertIn("E3_CHUNK2048_BUILD_V1",b)
        self.assertNotIn("setup.py build_ext --inplace",b)
        self.assertIn('SCHEDULE=(stock e1 e3g e3t e3t e3g e1 stock)',s)
        self.assertLess(s.index('five_arm_nz_route.py'),s.index("SMOKE_START $arm"))
        self.assertLess(s.index("SMOKE_START $arm"),s.index("SCHEDULE=(stock"))
        self.assertIn("binaries_before.sha256",s)
        self.assertIn("binaries_after.sha256",s)
        self.assertIn('rank,a.store_url.encode(),1024,ctypes.byref(handle)',a)
        self.assertIn("torch.equal(got,ref)",a)

    def _dataset(self,p,drift=0.,bad=False):
        val={"stock":1100,"e1":1250,"e3g":1170,"e3t":1110}
        for stage,arm in enumerate(ASSESS.SCHEDULE,1):
            for rank in (0,1):
                x=val[arm]+rank*.2
                if drift and arm=="e3g" and stage==6:x*=1+drift
                data={}
                for m in ASSESS.MS:
                    v=x+(m-2048)*.4
                    data[str(m)]={
                        "M":m,"rank":rank,"arm":arm,"bit_exact":True,
                        "protocol_status":None if arm=="stock" else 0,
                        "warmup":20,"iters":100,
                        "npu_event_us":[v]*100,"host_wall_us":[v+10]*100,
                    }
                item={
                    "schema":"E3_CHUNK2048_TP2_V1","stage":stage,"arm":arm,
                    "rank":(1-rank if bad and stage==1 else rank),
                    "TP":2,"K":2048,"N":2048,"weight_format":"NZ29",
                    "jit_compile":False,"allow_internal_format":True,
                    "profile":False,"reference":"CPU_GLOO_FP32_SUM_TO_FP16",
                    "results":data,
                    "candidate":{"work_head":"same-E3"} if arm!="stock" else None,
                }
                (p/f"{stage:02d}_{arm}_rank{rank}.json").write_text(
                    json.dumps(item),encoding="utf-8")

    def test_valid_e3_assessor(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self._dataset(p)
            out=ASSESS.analyze(p)
            self.assertEqual(out["status"],"E3_CHUNK2048_TIMING_VALID")
            self.assertEqual(len(out["raw_sessions"]),8)
            self.assertIn("E3_T_vs_E3_G_pct",out["comparisons"][0])

    def test_detect_e3_drift(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self._dataset(p,drift=.2)
            self.assertEqual(ASSESS.analyze(p)["status"],"INCONCLUSIVE_DRIFT")

    def test_bad_rank_or_missing_data_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self._dataset(p,bad=True)
            with self.assertRaises(ValueError):
                ASSESS.analyze(p)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self._dataset(p)
            (p/"03_e3g_rank0.json").unlink()
            with self.assertRaises(FileNotFoundError):
                ASSESS.analyze(p)


if __name__=="__main__":
    unittest.main()

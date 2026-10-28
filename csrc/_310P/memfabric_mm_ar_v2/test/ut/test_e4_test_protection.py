#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""纯CPU测试：E4实机主动STOP、真实npu-smi表、脱管进程及构建身份。"""
from __future__ import annotations
import json,os,signal,subprocess as sp,sys,tempfile,time
from pathlib import Path
from unittest import TestCase,mock
PERF=Path(__file__).resolve().parents[1]/"perf"
sys.path.insert(0,str(PERF))
import e4_stage_guard as G
import e4_detached as D

def info(target=(),others=(),health="OK",card=16384):
    rows=[f"| {card}   310P3        | {health:<15} | NA |",
          f"| {card}   310P3        | OK              | NA |",
          "| NPU  Chip     | Process id      | Process name | Process memory(MB) |"]
    rows += [f"| {card}  0     | {pid} | | 2759 |" for pid in target]
    rows += [f"| 5664   0     | {pid} | | 2759 |" for pid in others]
    return "\n".join(rows)+"\n"

class TestProtection(TestCase):
    def test_foreign_cross_card_stop(self):
        x=G.parse_info(info(target=(506263,515597,515598),
                            others=(506263,3589835)),16384)
        with self.assertRaisesRegex(G.LeaseLost,"cross-card PID"):
            G.assess(x,stage_running=True)

    def test_expected_two_rank_and_idle(self):
        self.assertEqual(G.assess(G.parse_info(info(),16384),stage_running=False)
                         ["target_pids"],[])
        x=G.assess(G.parse_info(info(target=(515597,515598)),16384),
                   stage_running=True)
        self.assertEqual(x["target_pids"],[515597,515598])
        self.assertIn("unverified",x["target_pid_origin"])

    def test_busy_preflight_and_three_pids_fail(self):
        with self.assertRaisesRegex(G.LeaseLost,"must be idle"):
            G.assess(G.parse_info(info(target=(99,)),16384),stage_running=False)
        with self.assertRaisesRegex(G.LeaseLost,"unexpected target process count"):
            G.assess(G.parse_info(info(target=(1,2,3)),16384),stage_running=True)

    def test_bad_health_header_and_npu_failure(self):
        for raw in ("garbage",info(health="Alarm"),info().replace("Process id","Process PID")):
            with self.assertRaises(G.LeaseLost):G.parse_info(raw,16384)
        with mock.patch.object(G.subprocess,"run",side_effect=OSError("missing")):
            with self.assertRaises(G.LeaseLost):G.sample(16384)

    def test_running_intrusion_stops_only_owned_torchrun(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/"guard.jsonl"
            child=mock.Mock(pid=4822)
            child.pid=4822
            child.poll.return_value=None
            raws=[info(),info(target=(11,12)),
                  info(target=(11,12,506263),others=(506263,))]
            snap=[G.parse_info(x,16384) for x in raws]
            with mock.patch.object(G,"sample",side_effect=list(zip(raws,snap))),\
                 mock.patch.object(G.subprocess,"Popen",return_value=child),\
                 mock.patch.object(G.time,"sleep"),\
                 mock.patch.object(G,"stop_owned") as stop:
                self.assertEqual(G.run_stage(16384,log,2,["torchrun"]),G.STOP_EXIT)
                stop.assert_called_once_with(child)
            self.assertEqual(json.loads(log.read_text().splitlines()[-1])["event"],"STOP")
            self.assertIn("506263", (Path(tmp)/"lease_guard_stop.npu.txt").read_text())

    def test_preflight_stop_never_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=Path(tmp)/"guard.jsonl"
            s=G.parse_info(info(target=(506263,)),16384)
            with mock.patch.object(G,"sample",return_value=("",s)),\
                 mock.patch.object(G.subprocess,"Popen") as popen:
                with self.assertRaises(G.LeaseLost):
                    G.run_stage(16384,log,2,["torchrun"])
                popen.assert_not_called()
            self.assertEqual(json.loads(log.read_text().splitlines()[-1])["event"],"STOP")

    def test_process_failure_is_not_smoke_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            child=mock.Mock(pid=4822);child.pid=4822;child.poll.return_value=17
            with mock.patch.object(G,"sample",return_value=("",G.parse_info(info(),16384))),\
                 mock.patch.object(G.subprocess,"Popen",return_value=child):
                self.assertEqual(G.run_stage(16384,Path(tmp)/"watch.jsonl",2,["torchrun"]),17)

    def test_campaign_has_four_guards_and_no_confirm(self):
        s=(PERF/"e4_campaign.sh").read_text()
        self.assertEqual(s.count("e4_stage torchrun --standalone"),4)
        self.assertIn("lease_guard.jsonl",s)
        self.assertIn("assert d['work_head']==head",s)
        self.assertIn("E4_CONFIRM_APPROVED:-NO",s)

    def test_detached_stale_head_and_unapproved_confirm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);state=root/"state";out=root/"out";build=root/"build";repo=root/"repo"
            build.mkdir();repo.mkdir()
            (build/"manifest.json").write_text(json.dumps(
                {"schema":"E4_LAYERED_ADD_BUILD_V1","work_head":"old"}))
            with mock.patch.dict(os.environ,{"E4_PHYSICAL_CARD_ID":"16384"},clear=True),\
                 mock.patch.object(D.subprocess,"check_output",side_effect=["new",""]):
                with self.assertRaisesRegex(ValueError,"stale"):
                    D.guard_inputs(state,out,build,repo)
            with mock.patch.dict(os.environ,{"E4_PHYSICAL_CARD_ID":"16384",
                                             "E4_CONFIRM_APPROVED":"YES"},clear=True):
                with self.assertRaisesRegex(ValueError,"Confirm"):
                    D.guard_inputs(state,out,build,repo)

    def test_detached_start_is_own_process_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);state=base/"state";out=base/"out";build=base/"build"
            build.mkdir()
            with mock.patch.object(D,"guard_inputs",return_value="deadbeef"),\
                 mock.patch.dict(os.environ,{"E4_PHYSICAL_CARD_ID":"16384"},clear=True),\
                 mock.patch.object(D.subprocess,"Popen") as popen:
                popen.return_value.pid=43210
                self.assertEqual(D.start(state,out,build),0)
                args,kwargs=popen.call_args
                self.assertIn("worker",args[0])
                self.assertTrue(kwargs["start_new_session"])
                self.assertEqual(kwargs["stdin"],D.subprocess.DEVNULL)
                j=json.loads((state/"launch.json").read_text())
                self.assertEqual(j["worker_pid"],43210)
                self.assertEqual(j["repo_head"],"deadbeef")
                self.assertFalse((state/"result.json").exists())

    def test_worker_abnormal_exit_must_write_stop_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);state=base/"state";out=base/"out";build=base/"build"
            for d in (state,out,build):d.mkdir()
            child=mock.Mock();child.wait.return_value=143
            with mock.patch.object(D.subprocess,"Popen",return_value=child):
                self.assertEqual(D.worker(state,out,build),143)
            result=json.loads((state/"result.json").read_text())
            self.assertEqual(result["state"],"STOP")
            self.assertEqual(result["exit_code"],143)

    def test_worker_sigterm_no_deadlock_and_cleans_campaign(self):
        """回归：SIGTERM打断child.wait()(持Popen._waitpid_lock)时handler不得
        再wait同一child——真进程级验证worker退出143、result.json落盘、
        campaign子进程被清理，超时即死锁失败。"""
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);state=base/"state";out=base/"out";build=base/"build"
            for d in (state,out,build):d.mkdir()
            helper=(
                "import sys,json\n"
                "from pathlib import Path\n"
                f"sys.path.insert(0,r'{PERF}')\n"
                "import subprocess as sp\n"
                "_REAL_POPEN=sp.Popen\n"
                "import e4_detached as D\n"
                "def fake_popen(cmd,**kw):\n"
                "    p=_REAL_POPEN(['bash','-c','exec sleep 30'],start_new_session=True,"
                "stdin=sp.DEVNULL,stdout=sp.DEVNULL,stderr=sp.STDOUT,close_fds=True)\n"
                f"    Path(r'{state}/camp.pid').write_text(str(p.pid))\n"
                "    return p\n"
                "D.subprocess.Popen=fake_popen\n"
                f"rc=D.worker(Path(r'{state}'),Path(r'{out}'),Path(r'{build}'))\n"
                "print(json.dumps({'worker_rc':rc}))\n")
            proc=sp.Popen([sys.executable,"-c",helper],
                          stdout=sp.PIPE,stderr=sp.STDOUT,text=True)
            try:
                for _ in range(100):
                    if (state/"camp.pid").exists() and proc.poll() is None:
                        break
                    time.sleep(0.05)
                time.sleep(0.3)  # 确保主线程已阻塞在child.wait()内部
                os.kill(proc.pid,signal.SIGTERM)
                outp,_=proc.communicate(timeout=20)
            except sp.TimeoutExpired:
                proc.kill();proc.wait()
                self.fail("worker deadlocked on SIGTERM (regression of handler-wait deadlock)")
            self.assertIn("worker_rc",outp)
            camp_pid=int((state/"camp.pid").read_text())
            for _ in range(60):
                try:os.kill(camp_pid,0)
                except ProcessLookupError:break
                time.sleep(0.1)
            else:
                self.fail("campaign child survived worker SIGTERM")
            result=json.loads((state/"result.json").read_text())
            self.assertEqual(result["state"],"STOP")
            self.assertEqual(result["exit_code"],143)

    def test_detached_exit_is_explicit_not_optimistic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(D.status(root),2)
            (root/"result.json").write_text(json.dumps({"state":"STOP","exit_code":97}))
            self.assertEqual(D.status(root),1)
            (root/"result.json").write_text(json.dumps({"state":"SCREEN_COMPLETE","exit_code":0}))
            self.assertEqual(D.status(root),0)

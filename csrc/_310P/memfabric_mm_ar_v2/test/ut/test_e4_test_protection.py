#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""纯CPU测试：E4实机主动STOP、真实npu-smi表、脱管进程及构建身份。"""
from __future__ import annotations
import json,os,sys,tempfile
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
            snap=[G.parse_info(info(),16384),
                  G.parse_info(info(target=(11,12)),16384),
                  G.parse_info(info(target=(11,12,506263),others=(506263,)),16384)]
            with mock.patch.object(G,"sample",side_effect=[("",x) for x in snap]),\
                 mock.patch.object(G.subprocess,"Popen",return_value=child),\
                 mock.patch.object(G.time,"sleep"),\
                 mock.patch.object(G,"stop_owned") as stop:
                self.assertEqual(G.run_stage(16384,log,2,["torchrun"]),G.STOP_EXIT)
                stop.assert_called_once_with(child)
            self.assertEqual(json.loads(log.read_text().splitlines()[-1])["event"],"STOP")

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

    def test_detached_exit_is_explicit_not_optimistic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual(D.status(root),2)
            (root/"result.json").write_text(json.dumps({"state":"STOP","exit_code":97}))
            self.assertEqual(D.status(root),1)
            (root/"result.json").write_text(json.dumps({"state":"SCREEN_COMPLETE","exit_code":0}))
            self.assertEqual(D.status(root),0)

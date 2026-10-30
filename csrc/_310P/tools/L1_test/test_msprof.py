# SPDX-License-Identifier: Apache-2.0
"""Capability, argv, TP2 orchestration and export failures without NPU access."""
import copy
import json
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from L1_test.profiler_tool import probe,validate_tool,command
from L1_test.profile_collect import collect_external,expected_windows,load_reports
from L1_test.profile_windows import marked_range
from L1_test import profile_windows
from L1_test.core import make_plan,sha
from L1_test.testing import reports


def tool(tmp_path):
    path=tmp_path/'fake msprof'
    path.write_text('#!/bin/sh\ncase "$1" in\n--help) echo "--application --output --msproftx --aic-metrics --task-time --runtime-api";;\n--version) echo "CPU_FIXTURE_ONLY";;\n*) exit 93;;\nesac\n')
    path.chmod(0o755)
    return path


def export(directory,report,plan,job,rank,broken=False):
    labels=expected_windows(plan,job,rank);report['profile_windows']=labels
    directory.mkdir(parents=True,exist_ok=True)
    (directory/f'rank{rank}.json').write_text(json.dumps(report))
    root=directory/f'msprof-rank{rank}';root.mkdir()
    events=[dict(ph='M',name='process_name',pid=1,args={'name':'CANN'}),
            dict(ph='M',name='process_name',pid=7,args={'name':'Ascend Hardware'})]
    for i,name in enumerate(labels[:-1] if broken else labels):
        offset=i*100
        events.extend([
            dict(ph='X',name=name,pid=1,tid=1,ts=offset+10,dur=80,cat='user_annotation'),
            dict(ph='X',name='aclnnMatmulWeightNz' if job['version']=='STOCK' else 'A3Kernel',pid=7,tid=2,ts=offset+20,dur=30,cat='kernel'),
            dict(ph='X',name='hcom_allreduce',pid=7,tid=3,ts=offset+55,dur=20,cat='kernel'),
            dict(ph='X',name='outside_check_ADD',pid=7,tid=2,ts=offset+93,dur=5,cat='kernel')])
    (root/'msprof_123.json').write_text(json.dumps(events))


def test_probe_freezes_actual_capabilities_and_preserves_quoted_application(tmp_path):
    spec=probe(tool(tmp_path),['--aic-metrics=PipeUtilization'])
    assert spec['options']==['--aic-metrics=PipeUtilization','--task-time=on','--runtime-api=on','--msproftx=on']
    assert spec['device_validation']=='NOT_RUN'
    application=['/python path/python','-m','module','--value','$(false); space']
    argv=command(spec,tmp_path/'new output',application)
    assert shlex.split(next(x.split('=',1)[1] for x in argv if x.startswith('--application=')))==application
    assert argv[0]==spec['binary']
    Path(spec['binary']).write_text('changed')
    with pytest.raises(ValueError,match='identity changed'):validate_tool(spec)


@pytest.mark.parametrize('option',['--application=other','--devices=0,1','--delay=1','--duration=1',
                                   '--system-wide=on','--unsupported=on','--aic-metrics=x;whoami'])
def test_unsupported_options_and_capture_scope_overrides_rejected(tmp_path,option):
    with pytest.raises(ValueError):probe(tool(tmp_path),[option])


def test_markers_own_execution_only_and_mstx_failure_is_terminal(monkeypatch):
    calls=[]
    class FakeLib:
        def __init__(self,fail=False):self.fail=fail
        def mstxRangeStartA(self,message,stream):
            if self.fail:return None
            calls.append('start');return 0x10
        def mstxRangeEnd(self,token):
            assert token==0x10;calls.append('end')
    monkeypatch.setattr(profile_windows,'_MSTX_LIB',None)
    monkeypatch.setattr(profile_windows,'_mstx_lib',lambda:FakeLib())
    with marked_range('label','msprof',None):calls.append('operator_and_join')
    calls.append('status_and_numeric')
    assert calls==['start','operator_and_join','end','status_and_numeric']
    monkeypatch.setattr(profile_windows,'_mstx_lib',lambda:FakeLib(fail=True))
    with pytest.raises(RuntimeError,match='MSTX'):
        with marked_range('label','msprof',None):pytest.fail('must not execute')


def test_export_preserves_raw_report_filters_checks_and_checks_evidence_identity(tmp_path):
    p=make_plan(['V3-5'],samples=3,warmup=2,shapes=[2048],scenarios=['steady'],collection='msprof')
    job=next(j for j in p['jobs'] if j['stage']=='msprof')
    directory=tmp_path/job['id']
    for rank,report in enumerate(reports(p,job)):
        export(directory,report,p,job,rank)
        original=sha(directory/f'rank{rank}.json')
        result=collect_external(directory,p,job,rank,{})
        assert result['application_report_sha256']==original==sha(directory/f'rank{rank}.json')
        assert result['timeline']['window_count']==2
        assert not any(k['name']=='outside_check_ADD' for k in result['profile']['kernel_statistics'])
    assert len(load_reports(directory,p,job))==2
    (directory/'msprof-rank0/msprof_123.json').write_text('[]')
    with pytest.raises(ValueError,match='export changed'):load_reports(directory,p,job)


@pytest.mark.parametrize('broken',[False,True])
def test_common_campaign_runs_off_msprof_off_and_stops_on_incomplete_export(tmp_path,monkeypatch,broken):
    from L1_test import cli
    p=make_plan(['V3-5'],samples=2,warmup=2,shapes=[2048],scenarios=['steady'],collection='msprof',rounds=1)
    manifest=dict(card=16384,devices='2,3',msprof=probe(tool(tmp_path)),
                  runtimes={'STOCK':{'version':'STOCK','environment':{}},
                            'V3-5':{'version':'V3-5','environment':{},'route_pattern':'A3Kernel'}})
    (tmp_path/'run.json').write_text(json.dumps(dict(plan=p,manifest=manifest,job_timeout=60)))
    monkeypatch.setattr(cli,'check_manifest',lambda *args:None)
    seen=[];before=os.environ.copy()
    def guarded(card,log,poll,argv,timeout,**kw):
        job=next(j for j in p['jobs'] if j['id']==argv[argv.index('--job')+1]);seen.append(job['stage'])
        module=argv[argv.index('--module')+1]
        assert module==('L1_test.rank_launcher' if job['stage']=='msprof' else 'L1_test.worker')
        for rank,report in enumerate(reports(p,job,identity=manifest['runtimes'][job['version']])):
            if job['stage']=='msprof':export(log.parent,report,p,job,rank,broken)
            else:(log.parent/f'rank{rank}.json').write_text(json.dumps(report))
        return 0
    monkeypatch.setitem(sys.modules,'stage_guard',SimpleNamespace(run_stage=guarded))
    assert cli.execute(tmp_path)==(1 if broken else 0)
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['status']==('STOP' if broken else 'PASS_L1_ONLY')
    assert seen==(['off-before']*4+['msprof'] if broken else ['off-before']*4+['msprof']*4+['off-after']*4)
    assert os.environ==before
    if not broken:
        analysis=json.loads((tmp_path/'analysis.json').read_text())
        assert len(analysis['timeline_statistics'])==8
        assert all(r['benefit_verdict']=='DIAGNOSTIC_ONLY' for r in analysis['results'] if r['stage']=='msprof')


def test_rank_launcher_executes_only_the_diagnostic_rank_in_existing_group(tmp_path,monkeypatch):
    from L1_test import rank_launcher
    p=make_plan(['V3-5'],samples=2,warmup=2,rounds=1,collection='msprof')
    job=next(j for j in p['jobs'] if j['stage']=='msprof')
    spec=probe(tool(tmp_path));run=tmp_path/'run.json'
    run.write_text(json.dumps(dict(plan=p,manifest={'msprof':spec})))
    (tmp_path/job['id']).mkdir()
    monkeypatch.setenv('WORLD_SIZE','2');monkeypatch.setenv('RANK','1')
    monkeypatch.setattr(sys,'argv',['rank_launcher','--run',str(run),'--job',job['id']])
    class ExecCalled(Exception): pass
    captured=[]
    def execv(binary,argv):captured.append((binary,argv));raise ExecCalled()
    monkeypatch.setattr(os,'execv',execv)
    with pytest.raises(ExecCalled):rank_launcher.main()
    payload=json.loads((tmp_path/job['id']/'msprof-launch-rank1.json').read_text())
    assert captured==[(spec['binary'],payload['argv'])]
    assert payload['application'][1:3]==['-m','L1_test.worker']
    monkeypatch.setenv('RANK','2')
    with pytest.raises(ValueError,match='TP2'):rank_launcher.main()


def test_persistent_aicpu_task_is_not_counted_as_a_full_operator_invocation(tmp_path):
    p=make_plan(['V3-5'],samples=2,warmup=2,shapes=[2048],scenarios=['steady'],collection='msprof')
    job=next(j for j in p['jobs'] if j['stage']=='msprof');directory=tmp_path/job['id']
    report=reports(p,job)[0];export(directory,report,p,job,0)
    trace=directory/'msprof-rank0/msprof_123.json'
    data=json.loads(trace.read_text())
    data.append(dict(ph='X',name='persistent_orchestrator',pid=7,tid=999,ts=0,dur=10000,args={'Task Type':'AI_CPU'}))
    trace.write_text(json.dumps(data))
    result=collect_external(directory,p,job,0,{})
    assert len(result['profile']['spanning_tasks'])==1
    assert not any(r['name']=='persistent_orchestrator' for r in result['profile']['kernel_statistics'])
    assert result['timeline']['documents'][0]['windows'][0]['kind_union_us']['aicpu']==80


def test_help_advertisement_does_not_authorize_system_or_attach_switches(tmp_path):
    from L1_test.profiler_tool import validate_options
    help_text='--application --output --msproftx --host-sys-pid --sys-devices'
    with pytest.raises(ValueError):validate_options(['--msproftx=on','--host-sys-pid=1'],help_text)
    with pytest.raises(ValueError):validate_options(['--msproftx=on','--sys-devices=all'],help_text)

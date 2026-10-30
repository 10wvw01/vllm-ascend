# SPDX-License-Identifier: Apache-2.0
"""Complete matrix, frozen case isolation, STOP and summary without NPU."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from L1_test import campaign as c
from L1_test.core import CONTRACT, ROOT, analyze, pair_job
from L1_test.testing import reports


def test_complete_fixed_m_round_and_optional_history_have_distinct_bindings():
    full=c.make_campaign()
    assert full['case_count']==43 and full['tp2_sessions']==170
    assert full['paired_samples_per_arm_per_case_per_stage']==200
    assert {x['version'] for x in full['cases'] if x['version'].startswith('V3-')}=={f'V3-{n}' for n in range(8)}
    assert all(x['M']==[2048,4096] for x in full['cases'] if x['version'].startswith('V3-'))
    assert not any(x['version'].startswith('V1-') for x in full['cases'])
    for case in full['cases']:
        plan=c.case_plan(full,case)
        assert plan['W']==20 and plan['N']==100 and plan['R']==1
        assert plan['scenarios']==['ready','steady']
        assert plan['consumer_validation']=='untimed_immediate_same_stream_before_after'
        if case['version']!='STOCK':assert [j['version'] for j in plan['jobs']]==['STOCK',case['version'],case['version'],'STOCK']
    active=c.make_campaign('active')
    assert active['case_count']==30 and active['tp2_sessions']==118
    assert c.make_campaign(collection='host')['tp2_sessions']==510
    all_v2=c.make_campaign(v2_history=True)
    assert all_v2['case_count']==64 and all_v2['tp2_sessions']==254
    template=c.runtime_template(all_v2)
    assert template['runtimes']['V2-7']['api']=='v2_selected_v1'
    assert 'private_library' in template['runtimes']['V2-7.legacy']
    assert all_v2['cases'][0]['version']=='STOCK'


@pytest.mark.parametrize('change', [lambda p:p['cases'].pop(),lambda p:p['cases'][4]['config'].update(comm_rows=13),
                                    lambda p:p['M'].append(3072),lambda p:p.update(tp2_sessions=1)])
def test_campaign_cannot_silently_drop_or_relabel_experiments(change):
    plan=c.make_campaign();change(plan)
    with pytest.raises(ValueError):c.validate_campaign(plan)


@pytest.fixture
def frozen_round(tmp_path,monkeypatch):
    plan=c.make_campaign(samples=2,warmup=2)
    template=c.runtime_template(plan,repo=str(ROOT))
    files_dir=tmp_path/'fixtures';files_dir.mkdir()
    local={}
    def existing(path):
        if path.startswith(str(ROOT/'csrc/_310P/common')) or path.endswith('/src/torch_adapter/tensor_entry.py'):
            return path
        if path not in local:
            f=files_dir/str(len(local));f.write_bytes(('CPU_ONLY_FIXTURE:'+path).encode());local[path]=str(f)
        return local[path]
    for key,spec in template['runtimes'].items():
        spec['source_head']='a'*40
        spec['files']=[existing(p) for p in spec['files']]
        for field in ('library','private_library','tensor_adapter','selected_device_library'):
            if field in spec:spec[field]=existing(spec[field])
        if 'python_modules' in spec:spec['python_modules']={k:existing(p) for k,p in spec['python_modules'].items()}
        if 'historical_identity_verified' in spec:spec['historical_identity_verified']=True
        admission=template['admission_by_runtime'][key]
        for name in ('g0_readonly','cpu_contract','private_compile','dual_rank_smoke','binary_identity'):admission[name]='PASS'
        admission['evidence_files']=[existing(p) for p in admission['evidence_files']]
    # CPU fixtures are not real admission evidence; the machine checker is the external boundary.
    monkeypatch.setattr(c,'check_manifest',lambda manifest,plan:None)
    out=tmp_path/'round'
    return plan,template,out


def test_freeze_and_bundle_keep_exact_per_configuration_identity(frozen_round):
    plan,template,out=frozen_round
    result=c.freeze_campaign(plan,template,out)
    assert result['device_test']=='NOT_RUN' and not (out/'runs').exists()
    c.check_bundle(out)
    with pytest.raises(ValueError,match='unknown active case'):c.check_bundle(out,'nonexistent')
    for case in plan['cases']:
        data=json.loads((out/'prepared'/case['id']/'manifest.json').read_text())
        if case['version']!='STOCK':
            assert data['runtimes'][case['version']]['config']==case['config']
        assert data['runtimes']['STOCK']['environment']=={}
    path=out/'prepared'/'v3-1-g512-serial'/'manifest.json'
    path.write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):c.check_bundle(out)


def test_missing_historical_module_blocks_entire_round_before_output_or_device(frozen_round):
    plan,template,out=frozen_round
    template['runtimes']['V3-0']['python_modules']={}
    with pytest.raises(ValueError,match='BLOCKED_BEFORE_DEVICE'):c.freeze_campaign(plan,template,out)
    assert not out.exists()


def test_first_failure_stops_without_retry_or_next_experiment(frozen_round,monkeypatch):
    plan,template,out=frozen_round;c.freeze_campaign(plan,template,out)
    calls=[]
    def fail(directory):
        calls.append(directory)
        c.save(directory/'result.json',dict(status='STOP',device_test='NOT_RUN',error='CPU injected failure'))
        return 1
    monkeypatch.setattr(c,'execute',fail)
    assert c.execute_campaign(out,'tcp://CPU',60)==1
    assert len(calls)==1 and calls[0].name=='stock'
    result=json.loads((out/'campaign-result.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    assert result['completed']==[] and len(result['pending'])==43
    assert summary['case_status']['stock']=='STOP' and summary['case_status']['v2-e1-m2048']=='NOT_RUN'
    assert not (out/'runs'/'v2-e1-m2048').exists()


def test_summary_error_is_reported_as_stop_instead_of_losing_final_status(frozen_round,monkeypatch):
    plan,template,out=frozen_round;c.freeze_campaign(plan,template,out)
    monkeypatch.setattr(c,'execute',lambda directory:1)
    def broken_summary(directory):raise ValueError('CPU injected summary failure')
    monkeypatch.setattr(c,'write_summary',broken_summary)
    assert c.execute_campaign(out,'tcp://CPU',60)==1
    result=json.loads((out/'campaign-result.json').read_text())
    assert result['status']=='STOP' and 'summary failure' in result['summary_error']


def simulated_case(out,case,plan,scale):
    directory=out/'runs'/case['id'];directory.mkdir(parents=True)
    manifest=json.loads((out/'prepared'/case['id']/'manifest.json').read_text())
    c.save(directory/'run.json',dict(manifest=manifest))
    jobs={}
    for job in plan['jobs']:
        value=scale*(1 if job['version']=='STOCK' else 2)
        raw=reports(plan,job,values=lambda rank,i:value+rank,identity=manifest['runtimes'][job['version']])
        jobs[job['id']]=pair_job(plan,job,*raw)
    c.save(directory/'analysis.json',analyze(plan,jobs))
    c.save(directory/'result.json',dict(status='PASS_L1_ONLY',device_test='CPU_SIMULATION_ONLY'))


def test_summary_does_not_pool_stock_or_confuse_configurations(frozen_round):
    plan,template,out=frozen_round;c.freeze_campaign(plan,template,out)
    selected=[x for x in plan['cases'] if x['id'] in ('stock','v3-1-g512-serial','v3-1-g512-pipeline')]
    for case,scale in zip(selected,(100,200,300)):
        simulated_case(out,case,c.case_plan(plan,case),scale)
    result=c.write_summary(out)
    assert len(result['completed'])==3 and len(result['pending'])==40 and result['release_go'] is False
    for case,expected in [('v3-1-g512-serial',201),('v3-1-g512-pipeline',301)]:
        rows=[r for r in result['rows'] if r['experiment']==case]
        assert rows and all(r['stock_mean_us']==expected and r['calls']==4 and r['n']==(4 if r['scenario']=='ready' else 2) for r in rows)
        assert {r['metric'] for r in rows}=={'wall_us','event_us'}
        assert all(r['downstream_validation']=='PASS_UNTIMED' for r in rows)
        assert all(r['config']['pipeline']==case.endswith('pipeline') for r in rows)
    assert (out/'summary.csv').is_file()


def test_recursive_framework_identity_covers_each_plugin():
    from L1_test.cli import framework_identity
    found=framework_identity()
    for name in ('stock','v1','v2','v3'):
        assert 'csrc/_310P/tools/L1_test/adapter_plugins/'+name+'.py' in found

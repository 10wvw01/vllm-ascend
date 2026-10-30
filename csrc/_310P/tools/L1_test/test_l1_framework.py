# SPDX-License-Identifier: Apache-2.0
"""Regression tests for measurement errors that can reverse a performance verdict."""
import copy
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from L1_test.core import CONTRACT, ROOT, analyze, catalog, make_plan, pair_job, sha, validate_plan, validate_runtime
from L1_test.worker import timed, validate_downstream
from L1_test.cli import clean_environment
from L1_test.collect import import_legacy, read_profile


def reports(plan,job):
    out=[]
    for rank in (0,1):
        cases=[]
        for m in plan['M']:
            for scene in plan['scenarios']:
                # Opposite outliers: max(rank means)=55, mean(paired maxima)=100.
                cases.append(dict(M=m,scenario=scene,warmup_count=plan['W'],samples=[
                    dict(i=i,variant=i%2,correct=True,
                         **{k:([10,100][(i+rank)%2]) for k in
                            ('wall_us','event_us')})
                    for i in range(plan['N'])]))
        out.append(dict(contract=CONTRACT,plan=plan,job=job,rank=rank,status='PASS',cases=cases,
                        downstream_validation=dict(status='PASS',timed=False,same_stream=True,cases=[
                            dict(phase=phase,M=m,variant=v,correct=True)
                            for phase in ('before','after') for m in plan['M'] for v in (0,1)]),
                        runtime_identity={'frozen':'one'},route={'verified':True}))
    return out


def paired(plan):
    return {job['id']:pair_job(plan,job,*reports(plan,job)) for job in plan['jobs']}


def test_stock_is_independent_and_equal_counts():
    p=make_plan(samples=2,warmup=2,rounds=2,shapes=[2048],scenarios=['ready'])
    assert all(j['version']=='STOCK' for j in p['jobs'])
    result=analyze(p,paired(p))['results'][0]
    assert result['stats']['STOCK']['n']==8
    assert result['stats']['STOCK']['mean_us']==100
    assert 'speedup' not in result


def test_per_index_rank_max_before_mean_no_outlier_removal():
    p=make_plan(['V3-5'],samples=2,warmup=2,shapes=[2048],scenarios=['ready'])
    result=analyze(p,paired(p))['results'][0]
    assert result['stats']['STOCK']['mean_us']==100
    assert result['stats']['V3-5']['n']==8
    assert result['speedup']==1


@pytest.mark.parametrize('mutation',[
    lambda r:r[1]['cases'][0]['samples'].pop(),
    lambda r:r[1]['cases'][0]['samples'][0].update(i=7),
    lambda r:r[1]['downstream_validation']['cases'][0].update(correct=False),
    lambda r:r[1]['cases'][0]['samples'][0].update(wall_us=float('nan')),
    lambda r:r[1]['cases'].append(copy.deepcopy(r[1]['cases'][0])),
    lambda r:r[1].update(runtime_identity={'frozen':'other'}),
    lambda r:r[1].update(route={'verified':False})])
def test_reject_mixed_missing_wrong_or_invalid_evidence(mutation):
    p=make_plan(['V2-1'],samples=2,warmup=2)
    job=p['jobs'][0]
    r=reports(p,job)
    mutation(r)
    with pytest.raises(ValueError): pair_job(p,job,*r)


def test_missing_abba_job_and_modified_plan_are_rejected():
    p=make_plan(['V3-6'],samples=2,warmup=2)
    jobs=paired(p)
    jobs.pop(next(iter(jobs)))
    with pytest.raises(ValueError): analyze(p,jobs)
    p['jobs'].pop()
    with pytest.raises(ValueError): validate_plan(p)


def test_stable_brackets_do_not_hide_between_round_or_candidate_drift():
    p=make_plan(['V3-7'],samples=2,warmup=2,shapes=[2048],scenarios=['ready'])
    jobs=paired(p)
    for item in jobs.values():
        if item['job']['block']==1:
            for c in item['cases']:
                for k in c['paired_values']:
                    c['paired_values'][k]=[v*2 for v in c['paired_values'][k]]
                    c['metrics'][k]['mean_us']*=2
    assert all(r['comparison_status']=='INCONCLUSIVE_DRIFT' for r in analyze(p,jobs)['results'])
    p['mode']='shared'
    # Paired jobs keep the same schedule; shared drift still has a distinct result label.
    assert all(r['comparison_status']=='OBSERVATIONAL_SHARED' for r in analyze(p,jobs)['results'])


def test_collectors_have_off_controls_and_no_diagnostic_speedup_verdict():
    p=make_plan(['V3-5'],samples=2,warmup=2,collection='host',shapes=[2048],scenarios=['ready'])
    rows=analyze(p,paired(p))['results']
    assert set(r['stage'] for r in rows)=={'off-before','host','off-after'}
    assert all(r['comparison_status']=='DIAGNOSTIC_ONLY' for r in rows if r['stage']=='host')
    assert all(r['instrumentation_delta']['V3-5']==0 for r in rows if r['stage']=='host')


def test_no_historical_alias_substitution_or_unfrozen_library(tmp_path):
    f=tmp_path/'lib.so'; f.write_bytes(b'frozen')
    spec=dict(version='V2-0',source_head='a'*40,files={str(f):sha(f)},library=str(f),supported_M=[2048])
    with pytest.raises(ValueError,match='historical'): validate_runtime(spec,'V2-0')
    spec['historical_identity_verified']=True
    validate_runtime(spec,'V2-0')
    f.write_bytes(b'changed')
    with pytest.raises(ValueError,match='mismatch'): validate_runtime(spec,'V2-0')


def test_environment_leaks_and_trace_cannot_change_comparison(tmp_path):
    with patch.dict(os.environ,{'MF_SDMA_ORCH_JSON':'old-v11','VLLM_ASCEND_310P_MEMFABRIC_MM_AR_TRACE':'1','LD_PRELOAD':'fusion.so'},clear=True):
        env=clean_environment({'environment':{}})
        assert not any(k in env for k in ('MF_SDMA_ORCH_JSON','VLLM_ASCEND_310P_MEMFABRIC_MM_AR_TRACE','LD_PRELOAD'))
    with patch.dict(os.environ,{'MF310P_V2_NOWARMUP':'1'},clear=True):
        with pytest.raises(ValueError): clean_environment({})
    f=tmp_path/'f';f.write_bytes(b'a')
    s=dict(version='STOCK',source_head='a'*40,files={str(f):sha(f)},environment={'MF_SDMA_ORCH_JSON':'path'})
    with pytest.raises(ValueError): validate_runtime(s,'STOCK')


def test_operator_timing_excludes_downstream_and_gate_preserves_immediate_dependency(monkeypatch):
    calls=[]
    class Event:
        def record(self,stream): calls.append('event')
        def elapsed_time(self,end): return .1
    class Stream:
        def synchronize(self): calls.append('join')
    def op(): calls.append('op');return 5
    def consumer(y): calls.append('consumer');return y+1
    clock=iter([1000,201000])
    y,row=timed(op,Stream(),[Event(),Event()],lambda:next(clock))
    assert y==5 and 'consumer' not in calls
    assert calls.count('join')==1 and row['wall_us']==200
    assert set(row)=={'wall_us','event_us'} and calls.count('event')==2
    calls.clear()
    def no_clock():raise AssertionError('functional downstream gate must not read a clock')
    monkeypatch.setattr('L1_test.worker.time.perf_counter_ns',no_clock)
    y,z=validate_downstream(op,consumer,Stream())
    assert (y,z)==(5,6) and calls==['op','consumer','join']
    assert row==dict(wall_us=200,event_us=100)


def test_chain_and_old_contract_are_rejected_instead_of_relabelled():
    with pytest.raises(ValueError,match='untimed'):make_plan(scenarios=['chain'])
    plan=make_plan(samples=2,warmup=2)
    plan['contract']='L1_TENSOR_IO_V2'
    with pytest.raises(ValueError):validate_plan(plan)


@pytest.mark.parametrize('change',[
    lambda r:r[0].pop('downstream_validation'),
    lambda r:r[0]['downstream_validation'].update(timed=True),
    lambda r:r[0]['downstream_validation'].update(same_stream=False),
    lambda r:r[0]['downstream_validation']['cases'].pop(),
    lambda r:r[0]['downstream_validation']['cases'].append(copy.deepcopy(r[0]['downstream_validation']['cases'][0])),
    lambda r:r[0]['cases'][0]['samples'][0].update(consumer_event_us=1),
])
def test_untimed_consumer_evidence_required_and_legacy_metrics_rejected(change):
    plan=make_plan(samples=2,warmup=2,shapes=[2048]);job=plan['jobs'][0]
    raw=reports(plan,job);change(raw)
    with pytest.raises(ValueError):pair_job(plan,job,*raw)


def test_csv_keeps_pipe_counters_without_inventing_stage_shares(tmp_path):
    (tmp_path/'kernel_details.csv').write_text('Name,Duration(us),aic_scalar_time(us),aic_mte2_time(us)\nA3Kernel,100,80,80\n')
    data=read_profile(tmp_path,'A3Kernel')
    assert data['verified'] and data['rows'][0]['raw']['aic_mte2_time(us)']=='80'
    assert 'phases' not in data and 'overlap' in data['warning']
    assert not read_profile(tmp_path,'absent')['verified']


def test_all_versions_have_distinct_docs_resolved_links():
    known=catalog()
    assert len(known)>=35 and len({v['document'] for v in known.values()})==len(known)
    for vid,v in known.items():
        doc=ROOT/v['document']
        text=doc.read_text()
        assert vid in text and '```mermaid' in text and 'UNKNOWN' in text and '输入' in text
        import re
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if not link.startswith(('http','#')):
                target=(doc.parent/link).resolve()
                if not target.exists():
                    import subprocess
                    subprocess.run(['git','cat-file','-e','HEAD:'+str(target.relative_to(ROOT))],cwd=ROOT,check=True)
    with pytest.raises(ValueError): make_plan(['V1-4'])
    with pytest.raises(ValueError): make_plan(['V2-19'])


def test_historical_import_keeps_p50_and_coverage_limits():
    root=ROOT/'csrc/_310P/memfabric_mm_ar_v3/experiments/a3-unified-shared-card-20261029-r1'
    data=import_legacy(root)
    assert len(data['profiles'])==6
    assert all(p['exclusive_phases']=='UNKNOWN' for p in data['profiles'])
    assert data['reports'][0]['cases'][0]['wall_us'].get('p50')
    assert 'mean_us' not in data['reports'][0]['cases'][0]['wall_us']


@pytest.mark.parametrize('fail',[False,True])
def test_campaign_launches_tp2_isolates_arms_and_stops_on_first_failure(tmp_path,monkeypatch,fail):
    from types import SimpleNamespace
    from L1_test import cli
    p=make_plan(['V3-5'],samples=2,warmup=2,rounds=1,shapes=[2048],scenarios=['ready'])
    runtimes={'STOCK':{'version':'STOCK','environment':{}},
              'V3-5':{'version':'V3-5','environment':{'MF_SDMA_ORCH_JSON':'private-v28'}}}
    run=dict(plan=p,manifest=dict(card=16384,devices='2,3',runtimes=runtimes),
             store_url='tcp://127.0.0.1:8594',job_timeout=60)
    (tmp_path/'run.json').write_text(json.dumps(run))
    monkeypatch.setattr(cli,'check_manifest',lambda *args:None)
    original=os.environ.copy()
    seen=[]
    def guarded(card,log,poll,command,timeout,**options):
        assert '--nproc-per-node=2' in command and '--module' in command
        job=next(j for j in p['jobs'] if j['id']==command[command.index('--job')+1])
        seen.append(job['version'])
        if job['version']=='STOCK': assert 'MF_SDMA_ORCH_JSON' not in os.environ
        else: assert os.environ['MF_SDMA_ORCH_JSON']=='private-v28'
        if fail:return 97
        for rank,report in enumerate(reports(p,job)):
            report['runtime_identity']=runtimes[job['version']]
            (log.parent/f'rank{rank}.json').write_text(json.dumps(report))
        return 0
    monkeypatch.setitem(sys.modules,'stage_guard',SimpleNamespace(run_stage=guarded))
    assert cli.execute(tmp_path)==(1 if fail else 0)
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['status']==('STOP' if fail else 'PASS_L1_ONLY')
    assert seen==(['STOCK'] if fail else ['STOCK','V3-5','V3-5','STOCK'])
    assert os.environ==original
    assert not (tmp_path/'analysis.json').exists() if fail else (tmp_path/'analysis.json').exists()


def test_only_observed_host_phases_are_attributed_no_device_invention():
    from L1_test.collect import attribution
    p=make_plan(['V3-5'],samples=2,warmup=2,collection='host',shapes=[2048],scenarios=['ready'])
    job=next(j for j in p['jobs'] if j['stage']=='host')
    raw=reports(p,job)
    for rank in raw:
        for s in rank['cases'][0]['samples']:
            s['collection']={'host_us':dict(validate_reap=30,reserve=5,wrap=10,enqueue=20)}
    row=attribution(raw)[0]
    assert row['host_phase_us']==dict(host_prepare=30,output_acquire_wrap=15,enqueue=20)
    assert 'host_prepare' not in row['missing_phases'] and 'mm' in row['missing_phases']
    assert row['largest_measured_host_stage']=='validate_reap' and row['additive'] is False


def test_relocated_entry_and_worker_resolve_from_another_directory(tmp_path):
    import subprocess
    entry=Path(__file__).with_name('l1_bench.py').resolve()
    plan_path=tmp_path/'plan.json'
    subprocess.run([sys.executable,str(entry),'plan','--samples','2','--warmup','2',
                    '--rounds','1','--out',str(plan_path)],cwd=tmp_path,check=True,capture_output=True,text=True)
    plan=json.loads(plan_path.read_text())
    validate_plan(plan)
    assert plan['versions']==[] and len(plan['jobs'])==2
    env=clean_environment({'environment':{}})
    result=subprocess.run([sys.executable,'-m','L1_test.worker','--help'],cwd=tmp_path,
                          env=env,check=True,capture_output=True,text=True)
    assert '--run' in result.stdout and '--job' in result.stdout


def test_relocated_freeze_records_new_framework_paths(tmp_path):
    import subprocess
    frozen=tmp_path/'runtime.so';frozen.write_bytes(b'CPU-only identity fixture')
    evidence=tmp_path/'evidence.json';evidence.write_text('{}')
    template=tmp_path/'template.json'
    template.write_text(json.dumps(dict(runtimes={'STOCK':dict(version='STOCK',source_head='a'*40,
                        files=[str(frozen)],environment={})},admission=dict(evidence_files=[str(evidence)]))))
    output=tmp_path/'frozen.json'
    subprocess.run([sys.executable,str(Path(__file__).with_name('l1_bench.py')),'freeze',
                    '--template',str(template),'--out',str(output)],cwd=tmp_path,check=True,capture_output=True,text=True)
    result=json.loads(output.read_text())
    paths=result['framework_files']
    prefix='csrc/_310P/tools/L1_test/'
    assert all(prefix+name in paths for name in ('cli.py','core.py','worker.py','versions.json','l1_bench.py'))
    assert result['runtimes']['STOCK']['files'][str(frozen)]==sha(frozen)

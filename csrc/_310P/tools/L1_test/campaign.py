# SPDX-License-Identifier: Apache-2.0
"""One fixed-shape Stock/V2/V3 matrix, reusing every common L1 execution path."""
import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from .core import CONTRACT, ROOT, catalog, make_plan, sha
from .cli import check_manifest, clean_environment, execute, freeze_data
from .worker import save

SCHEMA = 'MF310P_L1_FIXED_M_CAMPAIGN_V2'


def make_campaign(profile='full', samples=100, warmup=20, rounds=1, mode='exclusive', collection='off', v2_history=False):
    if profile not in ('full', 'active'):
        raise ValueError('profile must be full or active')
    if type(v2_history) is not bool:
        raise ValueError('v2_history must be boolean')
    cases = []
    def add(case_id, version, config=None, shapes=(2048,4096), api=None):
        entry = catalog().get(version, {})
        binding=version+'.legacy' if case_id.startswith('v2-history-') and version in ('V2-7','V2-15') else version
        cases.append(dict(id=case_id, version=version, runtime_key=binding,
                          config={**entry.get('config',{}),**(config or {})}, M=list(shapes), api=api,
                          historical=bool(entry.get('historical'))))
    add('stock', 'STOCK')
    add('v2-e1-m2048', 'V2-7', shapes=(2048,), api='v2_selected_v1')
    add('v2-t8-m4096', 'V2-15', shapes=(4096,), api='v2_selected_v1')
    if v2_history:
        # Small-M T1 and non-full diagnostic paths are intentionally separate.
        for n in (0,1,2,5,6,7,8,9,10,11,12,13,14,15,16):
            add(f'v2-history-{n}',f'V2-{n}',api='v2_trial_v1' if n in (0,1,2) else None)
        for chunk in (256,512,1024):
            for overlap in (0,1):
                add(f'v2-f2-c{chunk}-o{overlap}','V2-3',dict(chunk_rows=chunk,overlap=overlap),api='v2_trial_v1')
    for n in range(8):
        if profile == 'active' and n in (0,2,4):
            continue
        version = f'V3-{n}'
        if n in (0,1):
            for group in ((1024,2048) if n == 0 else (512,1024,2048)):
                for pipeline in (False,True):
                    add(f'v3-{n}-g{group}-'+('pipeline' if pipeline else 'serial'),version,
                        dict(comm_rows=group,pipeline=pipeline),api='python_session_v1')
        elif n in (2,3):
            for group in (512,1024,2048):
                add(f'v3-{n}-g{group}',version,dict(comm_rows=group),api='python_session_v1')
        else:
            for compute in (512,1024):
                for group in (512,1024,2048):
                    add(f'v3-{n}-c{compute}-g{group}',version,dict(compute_rows=compute,comm_rows=group),api='python_session_v1')
    control = make_plan(samples=samples,warmup=warmup,rounds=rounds,mode=mode,collection=collection)
    jobs = sum(len(case_plan(dict(W=warmup,N=samples,R=rounds,mode=mode,collection=collection),c)['jobs']) for c in cases)
    return dict(schema=SCHEMA,profile=profile,v2_history=v2_history,W=control['W'],N=control['N'],R=control['R'],
                mode=mode,collection=collection,scenarios=['ready'],M=[2048,4096],cases=cases,
                consumer_validation=control['consumer_validation'],
                case_count=len(cases),tp2_sessions=jobs,paired_samples_per_arm_per_case_per_stage=2*rounds*samples,
                V1='adapter available; historical source/binary lookup and measurement deferred',
                policy='each candidate/config has its own common ABBA and Stock controls; no pooled cross-campaign Stock')


def validate_campaign(campaign):
    expected=make_campaign(campaign['profile'],campaign['N'],campaign['W'],campaign['R'],campaign['mode'],
                           campaign['collection'],campaign['v2_history'])
    if campaign != expected:
        raise ValueError('campaign differs from its complete canonical matrix')


def case_plan(campaign, case):
    return make_plan([] if case['version']=='STOCK' else [case['version']],campaign['N'],campaign['W'],campaign['R'],
                     campaign['mode'],('ready',),case['M'],campaign['collection'])


def runtime_template(campaign, repo='/ABS/REPO'):
    result=dict(card=16384,devices='2,3',cann='/ABS/CANN',runtimes={},admission_by_runtime={})
    for case in campaign['cases']:
        version=case['version']
        binding=case['runtime_key']
        if binding in result['runtimes']:
            continue
        base=dict(version=version,source_head='REPLACE_WITH_MATCHING_40_HEX_SOURCE_SHA',environment={})
        if version=='STOCK':
            base['files']=['/ABS/TORCH_NPU_LIBRARY.so']
        elif version.startswith('V2-'):
            directory='/ABS/PRIVATE/'+binding
            base.update(library=directory+'/vllm_ascend_C.so',supported_M=[2048,4096],
                        files=[directory+'/vllm_ascend_C.so',directory+'/libmf310p_v2_trial_device.so'],
                        route_pattern='V2TrialKernel',environment={'MF_SDMA_ORCH_JSON':'/ABS/SDK_V11/libmf_sdma_orch_v11.json',
                            'LD_LIBRARY_PATH':directory+':/ABS/SDK_V11:/ABS/CANN/aarch64-linux/lib64'})
            if case['historical']:
                base['historical_identity_verified']=False
            if case['api']=='v2_selected_v1':
                base.update(api=case['api'],supported_M=case['M'],selected_device_library=directory+'/libmf310p_v2_selected_device.so')
                base['files'].append(base['selected_device_library'])
                if version=='V2-15':base['route_pattern']='E4Group2048Kernel'
            elif case['api']=='v2_trial_v1':
                base['api']=case['api']
            elif catalog()[version]['runtime']=='v2_private':
                base['private_library']=directory+'/MATCHING_PRIVATE_HOST.so'
                base['files'].append(base['private_library'])
                base['files'].append(directory+'/MATCHING_PRIVATE_DEVICE.so')
                base['route_pattern']='REPLACE_WITH_EXACT_PRIVATE_KERNEL_REGEX'
            base['files']+=['/ABS/SDK_V11/libmf_smem.so','/ABS/SDK_V11/libmf_hybm_core.so','/ABS/SDK_V11/libmf_sdma_orch_v11.json']
        else:
            entry=catalog()[version]
            historical=entry.get('historical',False)
            directory='/ABS/PRIVATE/'+(version if historical else 'V3_CURRENT')
            pyroot=('/ABS/FROZEN_SOURCE/'+version+'/vllm_ascend/_310p/ops') if historical else repo+'/vllm_ascend/_310p/ops'
            family=entry['adapter_family']
            roles=['memfabric_mm_ar_v3']+(['memfabric_mm_ar_v32'] if family=='v32' else ['memfabric_mm_ar_a3'] if family=='a3' else [])
            modules={role:pyroot+'/'+role+'.py' for role in roles}
            base.update(api='python_session_v1',library=directory+'/libmf310p_v3_host.so',supported_M=[2048,4096],
                        python_modules=modules,route_pattern={'v3':'V3Kernel','v32':'V32Kernel','a3':
                            'A3SeatKernel' if version in ('V3-6','V3-7') else 'A3Kernel'}[family],
                        files=[directory+'/libmf310p_v3_host.so',directory+'/libmf310p_v3_device.so',*modules.values(),
                               '/ABS/SDK_V28/libmf_smem.so','/ABS/SDK_V28/libmf_hybm_core.so','/ABS/SDK_V28/libmf_sdma_orch_v28.json'],
                        environment={'MF_SDMA_ORCH_JSON':'/ABS/SDK_V28/libmf_sdma_orch_v28.json',
                                     'MEMFABRIC_HYBRID_HOME_PATH':'/ABS/SDK_V28',
                                     'LD_LIBRARY_PATH':directory+':/ABS/SDK_V28:/ABS/CANN/aarch64-linux/lib64'})
            if family in ('v32','a3'):
                base['tensor_adapter']=directory+'/mf310p_v32_tensor.so'
                base['files'].append(base['tensor_adapter'])
            if historical:
                base['historical_identity_verified']=False
                base['source_head']=entry['historical_source_hint']
        result['runtimes'][binding]=base
        result['admission_by_runtime'][binding]=dict(g0_readonly='NOT_RUN',cpu_contract='NOT_RUN',private_compile='NOT_RUN',
            dual_rank_smoke='NOT_RUN',binary_identity='NOT_RUN',evidence_files=['/ABS/ADMISSION/'+binding+'.json'])
    return result


def freeze_campaign(campaign, template, out):
    validate_campaign(campaign)
    prepared=[]
    errors=[]
    # Resolve every source/binary before creating output or running any device.
    for case in campaign['cases']:
        try:
            version=case['version']
            binding=case['runtime_key']
            candidate=copy.deepcopy(template['runtimes'][binding])
            if candidate.get('api')!=case['api']:
                raise ValueError('case API differs from explicit runtime binding')
            candidate['config']={**candidate.get('config',{}),**case['config']}
            manifest={k:copy.deepcopy(template[k]) for k in ('card','devices','cann')}
            manifest['admission']=copy.deepcopy(template['admission_by_runtime'][binding])
            manifest['runtimes']={'STOCK':copy.deepcopy(template['runtimes']['STOCK'])}
            if version!='STOCK':manifest['runtimes'][version]=candidate
            frozen=freeze_data(manifest)
            plan=case_plan(campaign,case)
            # Admission is checked here and again before each real process.
            check_manifest(frozen,plan)
            for spec in frozen['runtimes'].values():clean_environment(spec)
            prepared.append((case,plan,frozen))
        except (KeyError,ValueError,OSError) as exc:
            errors.append(dict(case=case['id'],error=str(exc)))
    if errors:
        raise ValueError(json.dumps(dict(status='BLOCKED_BEFORE_DEVICE',errors=errors),ensure_ascii=False))
    out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    save(out/'campaign.json',campaign)
    files={'campaign.json':sha(out/'campaign.json')}
    for case,plan,manifest in prepared:
        directory=out/'prepared'/case['id'];directory.mkdir(parents=True)
        for name,data in [('plan.json',plan),('manifest.json',manifest)]:
            path=directory/name;save(path,data);files[str(path.relative_to(out))]=sha(path)
    save(out/'bundle.json',dict(schema=SCHEMA,campaign_sha256=sha(out/'campaign.json'),files=files,
                               source_head=prepared[0][2]['framework_head'],device_test='NOT_RUN'))
    return dict(status='FROZEN_READY',cases=len(prepared),tp2_sessions=campaign['tp2_sessions'],device_test='NOT_RUN')


def check_bundle(out, active_case=None):
    out=Path(out).resolve()
    bundle=json.loads((out/'bundle.json').read_text())
    campaign=json.loads((out/'campaign.json').read_text())
    validate_campaign(campaign)
    if active_case is not None and active_case not in {c['id'] for c in campaign['cases']}:
        raise ValueError('unknown active case')
    expected={'campaign.json'}|{f'prepared/{c["id"]}/{name}.json' for c in campaign['cases'] for name in ('plan','manifest')}
    if bundle.get('schema')!=SCHEMA or set(bundle['files'])!=expected or bundle['campaign_sha256']!=sha(out/'campaign.json'):
        raise ValueError('bundle shape/identity differs from complete campaign')
    for relative,digest in bundle['files'].items():
        path=(out/relative).resolve(strict=True)
        if not path.is_relative_to(out) or sha(path)!=digest:
            raise ValueError('prepared campaign artifact changed')
    for case in campaign['cases']:
        if active_case is not None and case['id']!=active_case:
            continue
        d=out/'prepared'/case['id']
        plan=json.loads((d/'plan.json').read_text());manifest=json.loads((d/'manifest.json').read_text())
        if plan!=case_plan(campaign,case):raise ValueError('prepared case plan changed')
        spec=manifest['runtimes'][case['version']]
        if spec.get('api')!=case['api'] or spec.get('config',{})!=case['config']:
            raise ValueError('prepared case API/configuration differs from matrix')
        if manifest['framework_head']!=bundle['source_head']:
            raise ValueError('bundle/framework HEAD mismatch')
        check_manifest(manifest,plan)
    return campaign


def collect_summary(out):
    out=Path(out)
    campaign=json.loads((out/'campaign.json').read_text())
    validate_campaign(campaign)
    rows=[];completed=[];pending=[];case_status={}
    for case in campaign['cases']:
        path=out/'runs'/case['id']
        status=json.loads((path/'result.json').read_text())['status'] if (path/'result.json').exists() else 'NOT_RUN'
        case_status[case['id']]=status
        if status!='PASS_L1_ONLY':
            pending.append(case['id']);continue
        completed.append(case['id'])
        runtime=json.loads((path/'run.json').read_text())['manifest']['runtimes'][case['version']]
        analysis=json.loads((path/'analysis.json').read_text())
        if analysis.get('contract')!=CONTRACT or analysis['plan']!=case_plan(campaign,case):
            raise ValueError('completed analysis belongs to a different plan')
        if analysis.get('downstream_validation')!='PASS_UNTIMED':
            raise ValueError('completed case lacks untimed downstream validation')
        for row in analysis['results']:
            version=case['version'];stat=row['stats'][version]
            rows.append(dict(experiment=case['id'],version=version,config=case['config'],M=row['M'],scenario=row['scenario'],
                             stage=row['stage'],metric=row['metric'],status=row['comparison_status'],n=stat['n'],
                             source_head=runtime['source_head'],mean_us=stat['mean_us'],std_us=stat['std_us'],
                             downstream_validation=analysis['downstream_validation'],
                             p50_us=stat['p50_us'],p90_us=stat['p90_us'],p99_us=stat['p99_us'],
                             arm_drift=row['arm_drift'],stock_bracket_drift=row['stock_bracket_drift'],
                             stock_mean_us=row['stats']['STOCK']['mean_us'],speedup=row.get('speedup'),
                             latency_reduction=row.get('latency_reduction')))
    return dict(schema=SCHEMA,contract=CONTRACT,completed=completed,pending=pending,case_status=case_status,rows=rows,release_go=False,
                note='Each config uses its own paired Stock controls; no pooling across case identities.')


def write_summary(out):
    out=Path(out);data=collect_summary(out)
    save(out/'summary.json',data)
    if data['rows']:
        with (out/'summary.csv').open('x',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data['rows'][0]));writer.writeheader()
            writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v
                              for k,v in row.items()} for row in data['rows'])
    return data


def execute_campaign(out, store_url, timeout):
    out=Path(out).resolve()
    result=dict(status='STOP',device_test='NOT_RUN',completed=[],pending=[])
    try:
        campaign=check_bundle(out)
        (out/'runs').mkdir(exist_ok=False)
        for case in campaign['cases']:
            check_bundle(out,case['id'])
            directory=out/'prepared'/case['id']
            plan=json.loads((directory/'plan.json').read_text());manifest=json.loads((directory/'manifest.json').read_text())
            run=out/'runs'/case['id'];run.mkdir()
            save(run/'run.json',dict(plan=plan,manifest=manifest,manifest_sha256=sha(directory/'manifest.json'),
                                    store_url=store_url,job_timeout=timeout))
            result['device_test']='STARTED'
            save(out/f'started-{case["id"]}.json',dict(id=case['id'],started_unix=time.time()))
            if execute(run)!=0:
                raise RuntimeError(case['id']+' STOP; no retry or next experiment')
            result['completed'].append(case['id'])
            save(out/f'completed-{case["id"]}.json',dict(id=case['id'],finished_unix=time.time()))
        result.update(status='PASS_L1_CAMPAIGN_ONLY',device_test='PASS')
    except BaseException as exc:
        result['error']=repr(exc)
    finally:
        try:
            summary=write_summary(out)
            result['pending']=summary['pending']
        except BaseException as exc:
            result.update(status='STOP',summary_error=repr(exc))
        result['finished_unix']=time.time()
        save(out/'campaign-result.json',result)
    return 0 if result['status']=='PASS_L1_CAMPAIGN_ONLY' else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    plan=sub.add_parser('plan')
    plan.add_argument('--profile',choices=('full','active'),default='full')
    plan.add_argument('--samples',type=int,default=100);plan.add_argument('--warmup',type=int,default=20)
    plan.add_argument('--rounds',type=int,default=1);plan.add_argument('--mode',choices=('exclusive','shared'),default='exclusive')
    plan.add_argument('--collection',choices=('off','host','profiler'),default='off');plan.add_argument('--v2-history',action='store_true')
    plan.add_argument('--out',type=Path,required=True)
    template=sub.add_parser('template');template.add_argument('--campaign',type=Path,required=True);template.add_argument('--out',type=Path,required=True)
    freeze=sub.add_parser('freeze');freeze.add_argument('--campaign',type=Path,required=True);freeze.add_argument('--template',type=Path,required=True);freeze.add_argument('--out',type=Path,required=True)
    for action in ('start','status','_run'):
        child=sub.add_parser(action);child.add_argument('--out',type=Path,required=True)
        if action!='status':
            child.add_argument('--store-url',default='tcp://127.0.0.1:8594');child.add_argument('--job-timeout',type=int,default=1800)
    args=parser.parse_args(argv)
    if args.action=='plan':
        data=make_campaign(args.profile,args.samples,args.warmup,args.rounds,args.mode,args.collection,args.v2_history)
        save(args.out,data);print(json.dumps(dict(cases=data['case_count'],tp2_sessions=data['tp2_sessions'],device_test='NOT_RUN')))
    elif args.action=='template':
        data=json.loads(args.campaign.read_text());validate_campaign(data);save(args.out,runtime_template(data,str(ROOT)))
    elif args.action=='freeze':
        print(json.dumps(freeze_campaign(json.loads(args.campaign.read_text()),json.loads(args.template.read_text()),args.out)))
    elif args.action=='status':
        print((args.out/'campaign-result.json').read_text() if (args.out/'campaign-result.json').exists() else
              json.dumps(dict(status='RUNNING_OR_INTERRUPTED',completed=sorted(p.stem for p in args.out.glob('completed-*.json')))))
    else:
        if not 60<=args.job_timeout<=7200:raise ValueError('per-job timeout must be 60..7200 seconds')
        if args.action=='_run':return execute_campaign(args.out,args.store_url,args.job_timeout)
        check_bundle(args.out)
        if (args.out/'launch-campaign.json').exists() or (args.out/'runs').exists():
            raise ValueError('campaign already started; use a fresh frozen directory')
        command=[sys.executable,str(ROOT/'csrc/_310P/tools/L1_test/l1_bench.py'),'campaign','_run','--out',str(args.out.resolve()),
                 '--store-url',args.store_url,'--job-timeout',str(args.job_timeout)]
        with (args.out/'campaign-console.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        save(args.out/'launch-campaign.json',dict(pid=child.pid,argv=command,started_unix=time.time()))
        print(json.dumps(dict(status='STARTED_DETACHED',pid=child.pid,out=str(args.out))))
    return 0

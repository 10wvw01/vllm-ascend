# SPDX-License-Identifier: Apache-2.0
"""One Stock-only / version campaign entry, lease guard, collector and report."""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .collect import attribution, import_legacy, profile_statistics
from .core import ROOT, analyze, catalog, make_plan, pair_job, sha, validate_plan, validate_runtime
from .worker import save

GUARD=ROOT/'csrc/_310P/memfabric_mm_ar_v3/tools'
SERVER_CARDS={5664:'0,1',16384:'2,3',38144:'4,5',43520:'6,7'}
FORBIDDEN=('ASCEND_RT_VISIBLE_DEVICES','MF_KFC_BYPASS','MF_SDMA_ORCH_DISABLE','MF310P_V2_NOWARMUP')


def framework_identity():
    directory=ROOT/'csrc/_310P/tools/L1_test'
    paths=set(directory.glob('*.py'))
    paths.update((directory/'adapter_plugins').rglob('*.py'))
    paths.update((ROOT/'vllm_ascend/_310p/ops').glob('memfabric_mm_ar*.py'))
    paths.update([ROOT/'csrc/_310P/tools/L1_test/versions.json',ROOT/'csrc/_310P/tools/L1_test/l1_bench.py',
                  GUARD/'stage_guard.py',ROOT/'csrc/_310P/memfabric_mm_ar_v3/test/quick_worker.py'])
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def freeze_data(template):
    """Hash declared artifacts only; no discovery, build or device execution."""
    data=copy.deepcopy(template)
    for spec in data['runtimes'].values():
        spec['files']={str(Path(path).resolve(strict=True)):sha(path) for path in spec['files']}
        for key in ('library','private_library','factory_file','tensor_adapter','selected_device_library'):
            if spec.get(key):
                spec[key]=str(Path(spec[key]).resolve(strict=True))
        if spec.get('python_modules'):
            spec['python_modules']={k:str(Path(p).resolve(strict=True)) for k,p in spec['python_modules'].items()}
        validate_runtime(spec,spec['version'])
    data['admission']['evidence_files']={str(Path(path).resolve(strict=True)):sha(path)
                                        for path in data['admission']['evidence_files']}
    data['framework_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    data['framework_files']=framework_identity()
    return data


def check_manifest(manifest,plan):
    if SERVER_CARDS.get(manifest.get('card'))!=manifest.get('devices'):
        raise ValueError('card/physical-device pair differs from server map')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if manifest.get('framework_head')!=head:
        raise ValueError('framework HEAD differs from frozen manifest')
    expected_framework=framework_identity()
    if subprocess.check_output(['git','status','--porcelain','--',*expected_framework],cwd=ROOT,text=True).strip():
        raise ValueError('framework/dependency sources must be committed before a device run')
    if manifest.get('framework_files')!=expected_framework:
        raise ValueError('framework source hashes differ from frozen manifest')
    for version in set(plan['versions'])|{'STOCK'}:
        spec=manifest['runtimes'][version]
        validate_runtime(spec,version)
        if version!='STOCK' and not set(plan['M'])<=set(spec['supported_M']):
            raise ValueError(f'{version} does not admit all requested physical M values')
        if version!='STOCK' and not spec.get('route_pattern'):
            raise ValueError('candidate must freeze actual kernel route regex')
    admission=manifest.get('admission',{})
    required=('g0_readonly','cpu_contract','dual_rank_smoke','binary_identity')
    if plan['versions']:
        required += ('private_compile',)
    if not all(admission.get(k)=='PASS' for k in required):
        raise ValueError('complete G0/CPU/private-build/dual-rank-smoke/identity evidence required')
    for path,expected in admission.get('evidence_files',{}).items():
        if not Path(path).is_absolute() or sha(path)!=expected:
            raise ValueError('admission evidence identity changed')
    if not admission.get('evidence_files'):
        raise ValueError('admission results need frozen evidence files')


def clean_environment(spec):
    if any(k in os.environ for k in FORBIDDEN):
        raise ValueError('device remapping, KFC bypass or first-write bypass present')
    # No legacy tracing / LD_PRELOAD / fusion environment leaks into independent Stock.
    env={k:v for k,v in os.environ.items() if not (k.startswith(('MF_','MF310P_','MEMFABRIC_','VLLM_ASCEND_310P_')) or k in ('V2_HOST_LIB','LD_PRELOAD'))}
    env.update(spec.get('environment',{}))
    env['PYTHONPATH']=str(ROOT/'csrc/_310P/tools')
    return env


def analyze_directory(out):
    run=json.loads((out/'run.json').read_text())
    plan=run['plan']
    jobs={}
    ranks=[]
    for job in plan['jobs']:
        pair=[json.loads((out/job['id']/f'rank{rank}.json').read_text()) for rank in (0,1)]
        expected=run['manifest']['runtimes'][job['version']]
        if any(r['runtime_identity']!=expected for r in pair):
            raise ValueError('rank runtime differs from the frozen run manifest')
        jobs[job['id']]=pair_job(plan,job,*pair)
        ranks.extend(pair)
    result=analyze(plan,jobs)
    result['attribution']=attribution(ranks)
    result['kernel_statistics']=profile_statistics(ranks)
    return result


def execute(out):
    run=json.loads((out/'run.json').read_text())
    plan,manifest=run['plan'],run['manifest']
    result=dict(status='STOP',device_test='NOT_RUN')
    original_env=os.environ.copy()
    try:
        validate_plan(plan)
        check_manifest(manifest,plan)
        sys.path.insert(0,str(GUARD))
        from stage_guard import run_stage
        for job in plan['jobs']:
            check_manifest(manifest,plan)
            path=out/job['id']
            path.mkdir(exist_ok=False)
            spec=manifest['runtimes'][job['version']]
            env=clean_environment(spec)
            os.environ.clear()
            os.environ.update(env)
            command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc-per-node=2',
                     '--module','L1_test.worker','--run',str(out/'run.json'),'--job',job['id']]
            result['device_test']='STARTED'
            with (path/'console.log').open('x') as log:
                # Guard inherits this process's environment; stdout redirection is scoped to own child.
                old_out,old_err=os.dup(1),os.dup(2)
                try:
                    os.dup2(log.fileno(),1)
                    os.dup2(log.fileno(),2)
                    rc=run_stage(manifest['card'],path/'lease_guard.jsonl',2,command,run['job_timeout'],
                                 deploy_flag_dir=path,allow_external_target=plan['mode']=='shared')
                finally:
                    os.dup2(old_out,1)
                    os.dup2(old_err,2)
                    os.close(old_out)
                    os.close(old_err)
            os.environ.clear()
            os.environ.update(original_env)
            if rc:
                raise RuntimeError(f'{job["id"]} STOP rc={rc}; no retry or next arm')
            reports=[json.loads((path/f'rank{r}.json').read_text()) for r in (0,1)]
            save(path/'paired.json',pair_job(plan,job,*reports))
        report=analyze_directory(out)
        save(out/'analysis.json',report)
        result.update(status='PASS_L1_ONLY',device_test='PASS')
    except BaseException as exc:
        result['error']=repr(exc)
    finally:
        os.environ.clear()
        os.environ.update(original_env)
        result['finished_unix']=time.time()
        save(out/'result.json',result)
    return 0 if result['status']=='PASS_L1_ONLY' else 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('versions')
    campaign=sub.add_parser('campaign')
    campaign.add_argument('arguments',nargs=argparse.REMAINDER)
    plan=sub.add_parser('plan')
    plan.add_argument('--versions',nargs='*',default=[])
    plan.add_argument('--samples',type=int,default=100)
    plan.add_argument('--warmup',type=int,default=20)
    plan.add_argument('--rounds',type=int,default=2)
    plan.add_argument('--mode',choices=['exclusive','shared'],default='exclusive')
    plan.add_argument('--collection',choices=['off','host','profiler'],default='off')
    plan.add_argument('--M',nargs='+',type=int,default=[2048,4096])
    plan.add_argument('--scenarios',nargs='+',choices=['ready','chain'],default=['ready','chain'])
    plan.add_argument('--out',type=Path,required=True)
    freeze=sub.add_parser('freeze')
    freeze.add_argument('--template',type=Path,required=True)
    freeze.add_argument('--out',type=Path,required=True)
    start=sub.add_parser('start')
    start.add_argument('--plan',type=Path,required=True)
    start.add_argument('--manifest',type=Path,required=True)
    start.add_argument('--out',type=Path,required=True)
    start.add_argument('--job-timeout',type=int,default=1800)
    start.add_argument('--store-url',default='tcp://127.0.0.1:8594')
    for action in ('_run','status','analyze'):
        s=sub.add_parser(action)
        s.add_argument('--out',type=Path,required=True)
    imp=sub.add_parser('import-legacy')
    imp.add_argument('--root',type=Path,required=True)
    imp.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.action=='versions':
        print(json.dumps(catalog(),ensure_ascii=False,indent=2))
    elif a.action=='campaign':
        from .campaign import main as campaign_main
        return campaign_main(a.arguments)
    elif a.action=='plan':
        save(a.out,make_plan(a.versions,a.samples,a.warmup,a.rounds,a.mode,a.scenarios,a.M,a.collection))
    elif a.action=='freeze':
        save(a.out,freeze_data(json.loads(a.template.read_text())))
    elif a.action=='import-legacy':
        save(a.out,import_legacy(a.root))
    else:
        out=a.out.resolve()
        if a.action=='status':
            print((out/'result.json').read_text() if (out/'result.json').exists() else 'RUNNING_OR_INTERRUPTED; inspect console.log/launch.json')
        elif a.action=='analyze':
            save(out/'analysis-recomputed.json',analyze_directory(out))
        elif a.action=='_run':
            return execute(out)
        else:
            plan=json.loads(a.plan.read_text())
            manifest=json.loads(a.manifest.read_text())
            validate_plan(plan)
            check_manifest(manifest,plan)
            if not 60<=a.job_timeout<=7200:
                raise ValueError('per-job deadline must be 60..7200 seconds')
            # Validate inherited state now, before claiming a detached task was started.
            for spec in manifest['runtimes'].values():
                clean_environment(spec)
            out.mkdir(parents=True,exist_ok=False)
            save(out/'run.json',dict(plan=plan,manifest=manifest,manifest_sha256=sha(a.manifest),
                                    store_url=a.store_url,job_timeout=a.job_timeout))
            command=[sys.executable,str(ROOT/'csrc/_310P/tools/L1_test/l1_bench.py'),'_run','--out',str(out)]
            with (out/'console.log').open('x') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            save(out/'launch.json',dict(pid=child.pid,argv=command,started_unix=time.time()))
            print(json.dumps(dict(status='STARTED_DETACHED',pid=child.pid,out=str(out))))
    return 0

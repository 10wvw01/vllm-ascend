# SPDX-License-Identifier: Apache-2.0
"""TP2 fresh-process worker; tensor input through ND-ready or real downstream consumption."""
import argparse
import json
import os
import re
import sys
import time
from datetime import timedelta
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from . import adapters
from .collect import capabilities, read_profile
from .core import CONTRACT, ROOT, sha, validate_plan, validate_runtime


def save(path,data):
    with Path(path).open('x') as f:
        json.dump(data,f,ensure_ascii=False,indent=2)
        f.write('\n')


def timed(op,consumer,stream,events,clock,scenario,host=False):
    start,middle,end=events
    begin=clock()
    start.record(stream)
    call_begin=clock() if host else None
    output=op()
    call_end=clock() if host else None
    middle.record(stream)
    consumed=consumer(output) if scenario=='chain' else None
    end.record(stream)
    stream.synchronize()
    wall=(clock()-begin)/1000
    row=dict(wall_us=wall,event_us=start.elapsed_time(end)*1000,
             operator_event_us=start.elapsed_time(middle)*1000)
    if scenario=='ready':
        # Consumer verifies actual usability, but is outside the ready-output boundary.
        middle.record(stream)
        consumed=consumer(output)
        end.record(stream)
        stream.synchronize()
    row['consumer_event_us']=middle.elapsed_time(end)*1000
    if host:
        row['collection']={'host_us':{'adapter_call_including_blocking':(call_end-call_begin)/1000}}
    return output,consumed,row


def verify_maps(spec,version):
    maps=Path('/proc/self/maps').read_text()
    frozen={str(Path(p).resolve()) for p in spec['files']}
    for key in ('library','private_library','tensor_adapter','selected_device_library'):
        if key in spec and str(Path(spec[key]).resolve()) not in maps:
            raise RuntimeError(f'expected library not mapped: {key}')
    mapped={line.split()[-1] for line in maps.splitlines() if '/libmf_' in line}
    if version=='STOCK' and mapped:
        raise RuntimeError('Stock contains MemFabric libraries')
    if mapped-frozen:
        raise RuntimeError(f'unfrozen MemFabric dependencies: {sorted(mapped-frozen)}')
    return sorted(mapped)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--job',required=True)
    a=p.parse_args()
    run=json.loads(a.run.read_text())
    plan=run['plan']
    validate_plan(plan)
    job=next(j for j in plan['jobs'] if j['id']==a.job)
    spec=run['manifest']['runtimes'][job['version']]
    validate_runtime(spec,job['version'])
    rank=int(os.environ['RANK'])
    if os.environ.get('WORLD_SIZE')!='2' or rank not in (0,1) or 'ASCEND_RT_VISIBLE_DEVICES' in os.environ:
        raise RuntimeError('direct physical TP2 required')
    out=a.run.parent/job['id']
    report=dict(contract=CONTRACT,plan=plan,job=job,rank=rank,status='RUNNING',cases=[],
                runtime_identity=spec,route={})
    try:
        import torch
        import torch.distributed as dist
        import torch.nn.functional as F
        import torch_npu
        sys.path.insert(0,str(ROOT/'csrc/_310P/memfabric_mm_ar_v3/test'))
        from quick_worker import physical_device
        selected=int(run['manifest']['devices'].split(',')[rank])
        torch.npu.set_device(selected)
        if physical_device(run['manifest']['cann'],selected)!=selected:
            raise RuntimeError('logical/physical identity mismatch')
        device=f'npu:{selected}'
        flag=out/f'deploy_window_rank{rank}.flag'
        flag.write_text('startup+pool-create')
        torch.npu.config.allow_internal_format=True
        torch.npu.set_compile_mode(jit_compile=False)
        dist.init_process_group('hccl' if job['version']=='STOCK' else 'gloo',timeout=timedelta(seconds=120))
        cpu=dist.new_group(backend='gloo') if job['version']=='STOCK' else dist.group.WORLD
        stream=torch.npu.current_stream()
        ctx=SimpleNamespace(torch=torch,dist=dist,linear=F.linear,rank=rank,url=run['store_url'],
                            stream=stream,device=device,shapes=plan['M'])
        torch.manual_seed(plan['seed'])
        w=torch_npu.npu_format_cast(torch.randn((2048,2048),dtype=torch.float16,device=device),29)
        xs,refs,residuals={}, {}, {}
        for m in plan['M']:
            xs[m],refs[m],residuals[m]=[],[],[]
            for v in (0,1):
                torch.manual_seed(m+rank*17+v*1009)
                x=torch.randn((m,2048),dtype=torch.float16,device=device)
                ref=F.linear(x,w).float().cpu()
                dist.all_reduce(ref,group=cpu)
                xs[m].append(x)
                refs[m].append(ref.to(device=device,dtype=torch.float16))
                residuals[m].append(torch.full_like(x,.125))
        stream.synchronize()
        begin=time.perf_counter_ns()
        stage=job['stage']
        adapter=adapters.create(job['version'],spec,ctx,stage)
        def checked(y,z,m,v):
            collected=adapter.after_sample()
            if (tuple(y.shape)!=(m,2048) or y.dtype!=torch.float16 or torch_npu.get_npu_format(y)!=2
                    or not y.is_contiguous() or not torch.equal(y,refs[m][v])):
                raise RuntimeError('full FP16/ND/bit-exact gate')
            if not torch.equal(z,refs[m][v]+residuals[m][v]):
                raise RuntimeError('downstream read gate')
            return collected
        # First-touch, cold allocations, pool initialization and route capture never become hot samples.
        for m in plan['M']:
            for v in (0,1):
                y=adapter.run(xs[m][v],w,v)
                z=y+residuals[m][v]
                stream.synchronize()
                checked(y,z,m,v)
                del y,z
        pattern=spec.get('route_pattern',r'aclnnMatmulWeightNz')
        route_cases=[]
        for m in plan['M']:
            route=out/f'route-rank{rank}'/f'm{m}'
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(route))):
                y=adapter.run(xs[m][0],w,0)
                z=y+residuals[m][0]
                stream.synchronize()
                checked(y,z,m,0)
                del y,z
            evidence=read_profile(route,pattern)
            names='\n'.join(row['name'] for row in evidence['rows'])
            if not evidence['verified'] or (job['version']=='STOCK' and not re.search('hcom.*allreduce|hccl.*allreduce',names,re.I)):
                raise RuntimeError(f'actual NZ/fused/HCCL route unproven for M={m}')
            route_cases.append(dict(M=m,**evidence))
        report['route']=dict(verified=True,cases=route_cases)
        report['mapped_libraries']=verify_maps(spec,job['version'])
        report['runtime']=dict(torch=str(torch.__version__),torch_npu=str(torch_npu.__version__),physical_device=selected)
        report['cold_setup_us']=(time.perf_counter_ns()-begin)/1000
        flag.unlink()
        events=tuple(torch.npu.Event(enable_timing=True) for _ in range(3))
        for m in plan['M']:
            for scenario in plan['scenarios']:
                samples=[]
                for phase,count in [('warmup',plan['W']),('measure',plan['N'])]:
                    profile_dir=out/f'profile-rank{rank}'/f'm{m}-{scenario}'
                    capture=torch_npu.profiler.profile(
                        activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                        experimental_config=torch_npu.profiler._ExperimentalConfig(
                            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization),
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir))
                    ) if stage=='profiler' and phase=='measure' else nullcontext()
                    with capture:
                        for i in range(count):
                            dist.barrier(group=cpu)
                            v=i%2
                            y,z,row=timed(lambda:adapter.run(xs[m][v],w,v),lambda y:y+residuals[m][v],
                                          stream,events,time.perf_counter_ns,scenario,stage=='host')
                            detail=checked(y,z,m,v)
                            if stage=='host' and detail.get('host_us'):
                                row['collection']['host_us'].update(detail['host_us'])
                            del y,z
                            if phase=='measure':
                                samples.append(dict(i=i,variant=v,correct=True,consumer_correct=True,**row))
                case=dict(M=m,scenario=scenario,warmup_count=plan['W'],samples=samples)
                report['cases'].append(case)
                save(out/f'checkpoint-rank{rank}-m{m}-{scenario}.json',case)
        if stage=='profiler':
            report['profile']=read_profile(out/f'profile-rank{rank}',pattern)
        report['capabilities']=capabilities(job['version'],stage,any(
            'reserve' in s.get('collection',{}).get('host_us',{}) for c in report['cases'] for s in c['samples']))
        validate_runtime(spec,job['version'])
        adapter.close()
        report['status']='PASS'
        save(out/f'rank{rank}.json',report)
        dist.destroy_process_group()
    except BaseException as exc:
        report.update(status='STOP',error=repr(exc))
        save(out/f'stop_rank{rank}.json',report)
        raise  # Never destroy/retry an uncertain MemFabric pool.


if __name__=='__main__':
    main()

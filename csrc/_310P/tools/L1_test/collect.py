# SPDX-License-Identifier: Apache-2.0
"""Common CSV collector. Hardware pipe counters overlap and are not phase wall times."""
import csv
import json
import math
import re
from pathlib import Path

from .core import CONTRACT, PHASES, sha, summarize


def read_profile(root, route_pattern):
    root=Path(root)
    paths=sorted(root.rglob('kernel_details*.csv')) or sorted(root.rglob('op_summary*.csv'))
    rows=[]
    for path in paths:
        for row in csv.DictReader(path.open(newline='',encoding='utf-8-sig')):
            name=row.get('Name') or row.get('Op Name') or row.get('Kernel Name') or ''
            duration=row.get('Duration(us)') or row.get('Task Duration(us)')
            if not duration:
                continue
            value=float(duration)
            if not math.isfinite(value) or value<=0:
                raise ValueError('invalid profiler duration')
            rows.append(dict(name=name,duration_us=value,task_type=row.get('Task Type'),
                             source=str(path.relative_to(root)),raw=row))
    if not rows:
        raise ValueError('no usable device profiler CSV rows')
    verified=any(re.search(route_pattern,row['name']) for row in rows)
    groups={}
    for row in rows:
        groups.setdefault((row['source'],row['name']),[]).append(row['duration_us'])
    kernels=[dict(source=source,name=name,**summarize(values)) for (source,name),values in groups.items()]
    return dict(kernel_statistics=kernels,verified=verified,files={str(p.relative_to(root)):sha(p) for p in paths},rows=rows,
                warning='Kernel durations and per-pipe busy counters overlap; not exclusive MM/wait/SDMA phases.')


def capabilities(version, collection, host_phases=False):
    return dict(version=version,collection=collection,wall=True,event=True,downstream=True,downstream_timing=False,
                host_call_total=collection=="host",timeline=collection in ('profiler','msprof'),host_detail_phases=host_phases,device_phase_wall=False,sdma_link_occupancy=False,
                reason='310P3 dav-2002 existing AICore clock is unavailable; no synthetic phase durations')


def attribution(rank_reports):
    out=[]
    for report in rank_reports:
        for case in report['cases']:
            host={}
            for sample in case['samples']:
                for name,value in sample.get('collection',{}).get('host_us',{}).items():
                    if not math.isfinite(value) or value<0:
                        raise ValueError('invalid Host duration')
                    host.setdefault(name,[]).extend([value]*sample.get('calls',1))
            for block in case.get('host_blocks',[]):
                for call in block['host_calls']:
                    for name,value in call.items():
                        if isinstance(value,bool):
                            continue
                        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                            raise ValueError('invalid operator Host phase duration')
                        name={'enqueue_us':'entry_total','run_us':'enqueue'}.get(name,name.removesuffix('_us'))
                        host.setdefault(name,[]).append(value)
            means={k:sum(v)/len(v) for k,v in host.items()}
            phases={}
            if 'validate_reap' in means:
                phases['host_prepare']=means['validate_reap']
            acquire='reserve' if 'reserve' in means else 'acquire'
            if acquire in means and 'wrap' in means:
                phases['output_acquire_wrap']=means[acquire]+means['wrap']
            if 'enqueue' in means:
                phases['enqueue']=means['enqueue']
            detailed={k:v for k,v in means.items() if k not in ('adapter_call_including_blocking','entry_total')}
            out.append(dict(version=report['job']['version'],stage=report['job']['stage'],rank=report['rank'],
                            M=case['M'],scenario=case['scenario'],host_mean_us=means,host_phase_us=phases,
                            largest_measured_host_stage=max(detailed,key=detailed.get) if detailed else None,
                            missing_phases={k:'UNKNOWN: no calibrated phase clock/evidence' for k in PHASES if k not in phases},
                            additive=False))
    return out


def profile_statistics(rank_reports):
    return [dict(version=r['job']['version'],stage=r['job']['stage'],rank=r['rank'],
                 kernels=r['profile']['kernel_statistics'],spanning_tasks=r['profile'].get('spanning_tasks',[]),exclusive_phase_wall=False,
                 collection_scope='analysis restricted to operator execution ranges; external raw capture is process-wide')
            for r in rank_reports if 'profile' in r]



def timeline_statistics(rank_reports):
    rows=[]
    from .timeline import WINDOW
    for report in rank_reports:
        for doc in report.get('timeline',{}).get('documents',[]):
            for window in doc['windows']:
                match=WINDOW.fullmatch(window['label'])
                rows.append(dict(job=report['job'],rank=report['rank'],source=doc['source'],
                    M=int(match[3]),scenario=match[4],first=int(match[5]),calls=int(match[6]),
                    label=window['label'],window_us=window['window_us'],
                    host_device_correlation=window['host_device_correlation'],device_timeline=window['device_timeline'],
                    kind_union_us=window['kind_union_us'],
                    visible_device_task_union_us=window['visible_device_task_union_us'],
                    uncovered_visible_device_us=window['uncovered_visible_device_us'],
                    visible_compute_communication_task_overlap_us=window['visible_compute_communication_task_overlap_us'],
                    links=len(window['links']),host_to_device_flow_mean_us=window['host_to_device_flow_mean_us'],
                    host_queue_flow_mean_us=window['host_queue_flow_mean_us'],status='DIAGNOSTIC_ONLY',internal_fused_overlap='UNKNOWN'))
    return rows

def import_legacy(root):
    """Read existing unified summaries without changing their p50 into an arithmetic mean."""
    root=Path(root)
    reports=[]
    for path in sorted(root.glob('campaign-r4-paired/*.paired.json')):
        data=json.loads(path.read_text())
        reports.append(dict(source=str(path.relative_to(root)),sha256=sha(path),
                            configuration=data['configuration'],cases=data['cases'],
                            attribution=data.get('attribution',[]),
                            status=f'HISTORICAL_SHARED_CARD; p50 remains p50; not {CONTRACT}'))
    profiles=[]
    for path in sorted((root/'pipe-profiler').glob('kernel_details*.csv')):
        # Each archived filename is a final CSV, not all 18 entries in its source index.
        grouped={}
        for row in csv.DictReader(path.open(newline='',encoding='utf-8-sig')):
            name=row.get('Name') or row.get('Op Name') or ''
            if 'A3Kernel' not in name:
                continue
            grouped.setdefault(name,[]).append(float(row['Duration(us)']))
        profiles.append(dict(source=str(path.relative_to(root)),sha256=sha(path),
                             fused_kernels={k:summarize(v) for k,v in grouped.items()},
                             geometry='compute1024; source probe overwrites same CSV basename across comm configs; do not infer all configs',
                             exclusive_phases='UNKNOWN',sdma_link_occupancy='UNKNOWN'))
    return dict(schema='MF_L1_LEGACY_EVIDENCE_V1',reports=reports,profiles=profiles,
                conclusion='Host phases are measurable in instrumented runs. No exact internal per-phase wall decomposition.',
                release_go=False)

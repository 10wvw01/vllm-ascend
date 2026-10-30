# SPDX-License-Identifier: Apache-2.0
"""Collect external exports after application exit; preserve the original rank report."""
import json
from pathlib import Path

from .core import summarize
from .measurement import groups
from .profile_windows import label
from .profiler_tool import digest
from .timeline import DEVICE_KINDS, read_timeline, write_timeline


def expected_windows(plan,job,rank):
    return [label(job,rank,m,scenario,first,count) for m in plan['M'] for scenario in plan['scenarios']
            for first,count in groups(plan['N'],plan['batch_size'])]


def inventory(root):
    root=Path(root).resolve(strict=True);files={}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            if not path.resolve().is_relative_to(root):raise ValueError('capture file escapes output directory')
            files[str(path.relative_to(root))]=dict(bytes=path.stat().st_size,sha256=digest(path))
    if not files:raise ValueError('empty external capture')
    return files



def scoped_statistics(timeline):
    grouped={};spanning=[]
    for doc in timeline['documents']:
        for event in doc['events']:
            if event['kind'] in DEVICE_KINDS and event['duration_us']>0:
                contained=any(w['window_start_us']<=event['start_us'] and event['end_us']<=w['window_end_us'] for w in doc['windows'])
                if contained:
                    grouped.setdefault((doc['source'],event['name']),[]).append(event['duration_us'])
                else:
                    spanning.append(dict(source=doc['source'],event=event,scope='spans measurement boundary; not an operator invocation duration'))
    return dict(spanning_tasks=spanning,kernel_statistics=[dict(source=source,name=name,**summarize(values)) for (source,name),values in grouped.items()])

def collect_external(directory, plan, job, rank, spec):
    directory=Path(directory)
    original=directory/f'rank{rank}.json'
    report=json.loads(original.read_text())
    if report.get('status')!='PASS':raise ValueError('msprof application did not finish successfully')
    expected=expected_windows(plan,job,rank)
    if report.get('profile_windows')!=expected:raise ValueError('application profile-window schedule mismatch')
    root=directory/f'msprof-rank{rank}'
    pattern=spec.get('route_pattern',r'aclnnMatmulWeightNz')
    timeline=read_timeline(root,expected,pattern,job['version']=='STOCK')
    windows=[w for d in timeline['documents'] for w in d['windows']]
    if not all(w['route_verified'] for w in windows):raise ValueError('measured msprof window lacks actual NZ/fused/HCCL route')
    report.update(timeline=timeline,profile=dict(verified=True,
        **scoped_statistics(timeline),
        files=inventory(root),scope='measured L1 ranges only; raw capture includes setup/warmup/checks'),
        application_report_sha256=digest(original),route=dict(verified=True,source='external_msprof_measured_windows'),
        collector='msprof_cli')
    write_timeline(timeline,directory/f'timeline-rank{rank}')
    with (directory/f'rank{rank}.profiled.json').open('x') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    return report


def load_reports(directory,plan,job):
    directory=Path(directory);reports=[]
    for rank in (0,1):
        suffix='.profiled' if job['stage']=='msprof' else ''
        report=json.loads((directory/f'rank{rank}{suffix}.json').read_text())
        if job['stage'] in ('profiler','msprof'):
            expected=expected_windows(plan,job,rank)
            timeline=report.get('timeline',{})
            actual=[w['label'] for d in timeline.get('documents',[]) for w in d['windows']]
            if (report.get('profile_windows')!=expected or timeline.get('scope')!='L1_MEASURED_WINDOWS'
                    or len(actual)!=len(expected) or set(actual)!=set(expected)
                    or any(w.get('route_verified') is not True for d in timeline.get('documents',[]) for w in d['windows'])):
                raise ValueError('diagnostic timeline window evidence missing/incomplete')
            root=directory/f'{"msprof" if job["stage"]=="msprof" else "profile"}-rank{rank}'
            for doc in timeline['documents']:
                if digest(root/doc['source'])!=doc['sha256']:raise ValueError('timeline export changed')
            if job['stage']=='msprof':
                if digest(directory/f'rank{rank}.json')!=report['application_report_sha256']:
                    raise ValueError('original application report changed')
                if inventory(root)!=report['profile']['files']:
                    raise ValueError('raw msprof capture inventory changed')
            else:
                for name,expected_sha in report['profile']['files'].items():
                    if digest(root/name)!=expected_sha:raise ValueError('profiler CSV export changed')
        reports.append(report)
    return reports

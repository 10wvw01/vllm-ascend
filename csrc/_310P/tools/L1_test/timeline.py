# SPDX-License-Identifier: Apache-2.0
"""Official Chrome traces: scoped intervals and explicit flow links, never nearest-time guesses."""
import csv
import gzip
import json
import math
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .profiler_tool import digest

SCHEMA = 'MF310P_L1_TIMELINE_V1'
KINDS = ('host_submit','host_wait','host_other','compute','fused','communication','transport','aicpu','device_other','unknown')
DEVICE_KINDS = {'compute','fused','communication','transport','aicpu','device_other'}
WINDOW = re.compile(r'^L1/([^/]+)/r([01])/m([0-9]+)/(ready|steady)/i([0-9]+)/n([0-9]+)$')


def number(value):
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError('invalid trace timestamp/duration') from None
    if not result.is_finite():
        raise ValueError('nonfinite trace timestamp/duration')
    return result


def merged(intervals):
    result=[]
    for begin,end in sorted(intervals):
        if not math.isfinite(begin+end) or end<begin:
            raise ValueError('invalid timeline interval')
        if end==begin:
            continue
        if result and begin<=result[-1][1]:
            result[-1]=(result[-1][0],max(result[-1][1],end))
        else:
            result.append((begin,end))
    return result


def length(intervals):
    return math.fsum(b-a for a,b in merged(intervals))


def overlap(left,right):
    a,b=merged(left),merged(right);i=j=0;total=0.
    while i<len(a) and j<len(b):
        total+=max(0.,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<=b[j][1]:i+=1
        else:j+=1
    return total


def classify(event, process, thread, fused_pattern):
    args=event.get('args') or {}
    cat=str(event.get('cat','')).lower()
    task=str(args.get('Task Type',args.get('task_type',''))).lower().replace('_','').replace(' ','')
    name=str(event.get('name',''));lower=name.lower()
    proc=process.lower()
    host_layer=any(x in proc for x in ('cann','python','framework','host','msproftx','mstx'))
    device = not host_layer and (any(x in proc for x in ('ascend hardware','device','npu hardware','communication','aicpu','ai cpu','ai core')) or
              cat in ('kernel','npu','aicore','aicpu','communication') or
              task in ('aicore','aivectorcore','aicpu','mix','mixaic','mixaiv','sdma','memcpy','memcpyasync'))
    if device:
        if fused_pattern and re.search(fused_pattern,name):return 'fused'
        if task in ('sdma','memcpy','memcpyasync') or re.search(r'\b(sdma|memcpy|memcpy_async)\b',lower):return 'transport'
        if task=='communication' or 'communication' in proc or cat=='communication' or 'communication' in thread.lower() or re.search(r'hccl|hcom',lower):return 'communication'
        if task=='aicpu' or cat=='aicpu' or 'aicpu' in proc or 'ai cpu' in proc:return 'aicpu'
        if task in ('aicore','aivectorcore','mix','mixaic','mixaiv') or cat in ('kernel','aicore'):return 'compute'
        return 'device_other'
    if any(x in proc for x in ('cann','python','framework','host','msproftx','mstx')) or cat in ('cpu_op','user_annotation','runtime'):
        if re.search(r'(synchronize|streamwait|eventwait)',lower):return 'host_wait'
        if re.search(r'(launch|enqueue|dequeue|memcpyasync)',lower):return 'host_submit'
        return 'host_other'
    return 'unknown'


def decode_events(data, fused_pattern=None):
    raw=data.get('traceEvents') if isinstance(data,dict) else data
    if not isinstance(raw,list):raise ValueError('Chrome trace must be a list or contain traceEvents')
    # Chrome ts/dur are microseconds. displayTimeUnit changes UI formatting only.
    processes={};threads={}
    for event in raw:
        if event.get('ph')=='M':
            name=(event.get('args') or {}).get('name','')
            if event.get('name')=='process_name':processes[str(event.get('pid'))]=str(name)
            if event.get('name')=='thread_name':threads[str(event.get('pid')),str(event.get('tid'))]=str(name)
    timed=[number(e['ts']) for e in raw if e.get('ph') not in ('M',None) and 'ts' in e]
    if not timed:raise ValueError('empty timeline')
    origin=min(timed)
    if any(not math.isfinite(float(t-origin)) for t in timed):raise ValueError('trace time range exceeds numeric precision')
    events=[];flows=[];stacks={};async_stacks={};unmatched=0
    def append(event,ts,dur,index):
        if dur<0:raise ValueError('negative trace interval')
        pid,tid=str(event.get('pid')),str(event.get('tid'))
        name=str(event.get('name',''));args=event.get('args') or {}
        start=float(ts-origin);duration=float(dur)
        if not math.isfinite(start+duration):raise ValueError('trace duration exceeds numeric precision')
        row=dict(id=f'e{index}',name=name,start_us=start,end_us=start+duration,duration_us=duration,
                 pid=pid,tid=tid,process=processes.get(pid,''),thread=threads.get((pid,tid),''),cat=str(event.get('cat','')),
                 kind=classify(event,processes.get(pid,''),threads.get((pid,tid),''),fused_pattern),
                 stream_id=str(args.get('Physic Stream Id',args.get('Stream Id',args.get('stream_id',tid)))),
                 task_id=str(args.get('Task Id',args.get('task_id','UNKNOWN'))),raw_event_index=index)
        events.append(row)
    for index,event in enumerate(raw):
        ph=event.get('ph');pid,tid=str(event.get('pid')),str(event.get('tid'))
        if ph in ('X','B','E','b','e','s','t','f') and 'ts' not in event:raise ValueError('timed event missing ts')
        if ph=='X':
            if 'dur' not in event:raise ValueError('complete event missing duration')
            append(event,number(event['ts']),number(event['dur']),index)
        elif ph=='B':stacks.setdefault((pid,tid),[]).append((event,index))
        elif ph=='E':
            stack=stacks.get((pid,tid),[])
            if not stack:unmatched+=1;continue
            begin,original=stack.pop();append(begin,number(begin['ts']),number(event['ts'])-number(begin['ts']),original)
        elif ph in ('b','e'):
            key=(pid,str(event.get('cat','')),str(event.get('id',event.get('id2',''))))
            if ph=='b':
                if key in async_stacks:raise ValueError('ambiguous asynchronous range id')
                async_stacks[key]=(event,index)
            elif key in async_stacks:
                begin,original=async_stacks.pop(key);append(begin,number(begin['ts']),number(event['ts'])-number(begin['ts']),original)
            else:unmatched+=1
        elif ph in ('s','t','f'):
            identity=event.get('id2',event.get('id'))
            if identity is None:unmatched+=1;continue
            if isinstance(identity,dict):
                if len(identity)!=1 or not set(identity)<={'local','global'}:raise ValueError('unsupported flow id2')
                mode,value=next(iter(identity.items()));scope=pid if mode=='local' else 'global';identity=str(value)
            else:scope=str(event.get('scope','global'));identity=str(identity)
            flows.append(dict(key=(str(event.get('cat','')),scope,identity),ph=ph,pid=pid,tid=tid,
                              ts=float(number(event['ts'])-origin),raw_event_index=index))
    unmatched+=sum(len(s) for s in stacks.values())+len(async_stacks)
    return events,flows,dict(origin_us=str(origin),unmatched_ranges_or_flows=unmatched)


def flow_links(events,flows):
    grouped={};links=[];unresolved=[]
    for f in flows:grouped.setdefault(f['key'],[]).append(f)
    def enclosing(point,device):
        candidates=[e for e in events if e['pid']==point['pid'] and e['tid']==point['tid']
                    and not e['name'].startswith('L1/') and e['start_us']<=point['ts']<=e['end_us']
                    and ((e['kind'] in DEVICE_KINDS) if device else e['kind'].startswith('host_'))]
        if not candidates:return None
        size=min(e['duration_us'] for e in candidates)
        smallest=[e for e in candidates if e['duration_us']==size]
        return smallest[0] if len(smallest)==1 else None
    for key,points in grouped.items():
        starts=[p for p in points if p['ph']=='s'];ends=[p for p in points if p['ph']=='f']
        if len(starts)!=1 or len(ends)!=1:
            unresolved.append(dict(flow=key,reason='missing/duplicate endpoints; no temporal guessing'));continue
        start,end=starts[0],ends[0]
        device=enclosing(end,True);src=enclosing(start,False)
        dst=device or enclosing(end,False)
        if src is None or dst is None:
            unresolved.append(dict(flow=key,reason='endpoint has no unique enclosing interval'));continue
        delta=end['ts']-start['ts']
        links.append(dict(flow=key,source_event=src['id'],target_event=dst['id'],source_name=src['name'],target_name=dst['name'],
                          kind='host_to_device' if device else 'host_queue' if 'queue' in key[0].lower() else 'host_dependency',start_us=start['ts'],end_us=end['ts'],
                          flow_start_to_end_us=delta,api_return_to_device_start_us=dst['start_us']-src['end_us'] if device else None,
                          clock_order='NONNEGATIVE' if delta>=0 else 'NEGATIVE_CLOCK_ALIGNMENT_OR_TRACE_ERROR'))
    return links,unresolved


def summarize_window(window, events, links):
    begin,end=window['start_us'],window['end_us']
    selected=[e for e in events if e['duration_us']>0 and e['start_us']<end and e['end_us']>begin
              and not e['name'].startswith('L1/')]
    clipped=lambda e:(max(begin,e['start_us']),min(end,e['end_us']))
    intervals={kind:[clipped(e) for e in selected if e['kind']==kind] for kind in KINDS}
    device=[clipped(e) for e in selected if e['kind'] in DEVICE_KINDS]
    streams={}
    for e in selected:
        if e['kind'] in DEVICE_KINDS:streams.setdefault((e['pid'],e['tid'],e['stream_id']),[]).append(clipped(e))
    per_stream=[]
    for key,spans in sorted(streams.items()):
        spans=merged(spans);cursor=begin;gaps=[]
        for a,b in spans:
            if a>cursor:gaps.append(dict(start_us=cursor,end_us=a,duration_us=a-cursor))
            cursor=max(cursor,b)
        if cursor<end:gaps.append(dict(start_us=cursor,end_us=end,duration_us=end-cursor))
        per_stream.append(dict(pid=key[0],tid=key[1],stream_id=key[2],visible_task_union_us=length(spans),gaps=gaps,
                               gap_scope='absence of exported tasks on this stream; not proof of hardware idle'))
    visible=length(device)
    inside_links=[x for x in links if begin<=x['start_us']<=end and begin<=x['end_us']<=end]
    flow_means={}
    for kind in ('host_to_device','host_queue'):
        values=[x['flow_start_to_end_us'] for x in inside_links if x['kind']==kind]
        flow_means[kind+'_flow_mean_us']=math.fsum(values)/len(values) if values and all(v>=0 for v in values) else None
        flow_means[kind+'_flow_count']=len(values)
    return dict(**flow_means,label=window['name'],window_us=end-begin,window_start_us=begin,window_end_us=end,
        event_count=len(selected),host_device_correlation='OBSERVED' if any(x['kind']=='host_to_device' for x in inside_links) else 'UNKNOWN_NO_RESOLVED_FLOW',
        device_timeline='OBSERVED' if device else 'UNKNOWN_NO_VISIBLE_TASK',kind_union_us={k:length(v) for k,v in intervals.items()},
        visible_device_task_union_us=visible,uncovered_visible_device_us=max(0.,end-begin-visible),
        visible_compute_communication_task_overlap_us=None if intervals['fused'] else overlap(intervals['compute'],intervals['communication']+intervals['transport']),
        internal_fused_overlap='UNKNOWN: no exported internal MM/SDMA phase boundaries',
        streams=per_stream,links=inside_links,event_ids=[e['id'] for e in selected],
        clipped_event_count=sum(e['start_us']<begin or e['end_us']>end for e in selected))


def trace_paths(root):
    root=Path(root)
    if root.is_file():return [root]
    found=sorted(set(root.rglob('trace_view.json'))|set(root.rglob('trace_view.json.gz'))|
                 {p for p in root.rglob('msprof_*.json*') if re.fullmatch(r'msprof_[0-9].*\.json(?:\.gz)?',p.name)})
    if not found:raise ValueError('no official Chrome timeline export; CSV alone cannot establish task chronology')
    return found


def read_timeline(root, expected_labels=(), route_pattern=None, stock=False, whole_trace=False):
    root=Path(root).resolve(strict=True);paths=trace_paths(root);documents=[];seen=[]
    for path in paths:
        if not path.resolve().is_relative_to(root if root.is_dir() else root.parent):
            raise ValueError('timeline export escapes declared capture directory')
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt',encoding='utf-8-sig') as stream:
            data=json.load(stream,parse_float=Decimal)
        events,flows,meta=decode_events(data,None if stock else route_pattern)
        windows=[e for e in events if WINDOW.fullmatch(e['name'])]
        if whole_trace:
            if expected_labels:raise ValueError('whole-trace analysis cannot replace measured-window evidence')
            windows=[dict(name='WHOLE_TRACE',start_us=min(e['start_us'] for e in events),end_us=max(e['end_us'] for e in events))]
        if not whole_trace:
            ordered=sorted(windows,key=lambda w:w['start_us'])
            if any(a['end_us']>b['start_us'] for a,b in zip(ordered,ordered[1:])):
                raise ValueError('measured L1 windows overlap; refusing duplicate attribution')
        links,unresolved=flow_links(events,flows)
        results=[]
        for window in windows:
            if expected_labels and window['name'] not in expected_labels:raise ValueError('unexpected L1 range in capture')
            if window['end_us']<=window['start_us']:raise ValueError('empty measured range')
            seen.append(window['name'])
            summary=summarize_window(window,events,links)
            names=[e['name'] for e in events if e['id'] in summary['event_ids']]
            summary['route_verified']=(any(re.search(route_pattern,n) for n in names) if route_pattern else None)
            if route_pattern and not any(e['kind'] in DEVICE_KINDS for e in events if e['id'] in summary['event_ids']):
                summary['route_verified']=False
            if stock and route_pattern and not any(re.search('hcom.*allreduce|hccl.*allreduce',n,re.I) for n in names):
                summary['route_verified']=False
            results.append(summary)
        selected_ids={i for w in results for i in w['event_ids']}
        documents.append(dict(source=str(path.relative_to(root) if root.is_dir() else path.name),sha256=digest(path),
            clock_domain='this export only; ranks/exports are not merged',clock=meta,
            windows=results,events=[e for e in events if e['id'] in selected_ids],unresolved_flows=unresolved))
    if not seen:raise ValueError('missing measured L1 ranges; use explicit whole-trace analysis for unscoped historical data')
    if not whole_trace and len(seen)!=len(set(seen)):raise ValueError('duplicate L1 ranges/alternate exports; refusing double counting')
    if expected_labels and set(seen)!=set(expected_labels):raise ValueError('missing measured L1 ranges')
    return dict(schema=SCHEMA,status='DIAGNOSTIC_ONLY',scope='WHOLE_TRACE' if whole_trace else 'L1_MEASURED_WINDOWS',
        documents=documents,window_count=len(seen),host_device_links=sum(len(w['links']) for d in documents for w in d['windows']),
        warning='Intervals overlap; union is not additive attribution. Communication envelopes may include waits; task overlap is not proof of concurrent data movement. Visible-task gaps are not physical idle or waits. Flow latency includes export clock alignment uncertainty.',
        sdma_link_occupancy='UNKNOWN',exclusive_phase_wall='UNKNOWN',release_go=False)


def write_timeline(result, prefix):
    prefix=Path(prefix)
    if any(Path(str(prefix)+suffix).exists() for suffix in ('.json','.csv')):
        raise ValueError('timeline output must be fresh; refusing overwrite')
    rows=[]
    for doc in result['documents']:
        for window in doc['windows']:
            rows.append(dict(source=doc['source'],label=window['label'],window_us=window['window_us'],
                visible_device_task_union_us=window['visible_device_task_union_us'],uncovered_visible_device_us=window['uncovered_visible_device_us'],
                visible_compute_communication_task_overlap_us=window['visible_compute_communication_task_overlap_us'],
                links=len(window['links']),host_to_device_flow_mean_us=window['host_to_device_flow_mean_us'],
                host_queue_flow_mean_us=window['host_queue_flow_mean_us'],internal_fused_overlap='UNKNOWN',**window['kind_union_us']))
    if not rows:raise ValueError('no timeline windows to export')
    with Path(str(prefix)+'.json').open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
    with Path(str(prefix)+'.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

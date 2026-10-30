# SPDX-License-Identifier: Apache-2.0
"""Synthetic official-format trace fixtures; never an NPU measurement."""
import copy
import gzip
import json
import sys
from pathlib import Path

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from L1_test.timeline import decode_events, read_timeline, write_timeline, length, overlap

LABEL='L1/V3-5-msprof-r0-s0/r0/m2048/steady/i0/n2'


def trace(name='MatMul',label=LABEL):
    return [
        dict(ph='M',name='process_name',pid=1,args={'name':'CANN'}),
        dict(ph='M',name='process_name',pid=7,args={'name':'Ascend Hardware'}),
        dict(ph='M',name='thread_name',pid=7,tid=4,args={'name':'Stream 4'}),
        dict(ph='X',name=label,pid=1,tid=2,ts=10,dur=100,cat='user_annotation'),
        dict(ph='X',name='cold_MM',pid=7,tid=4,ts=0,dur=5,cat='kernel'),
        dict(ph='X',name='aclrtLaunchKernel',pid=1,tid=2,ts=20,dur=5),
        dict(ph='X',name=name,pid=7,tid=4,ts=30,dur=50,cat='kernel',args={'Task Id':9,'Task Type':'AI_CORE','Stream Id':4}),
        dict(ph='X',name='nested_device_interval',pid=7,tid=4,ts=40,dur=10,cat='kernel'),
        dict(ph='X',name='MEMCPY_ASYNC',pid=7,tid=8,ts=60,dur=30,args={'Task Type':'SDMA'}),
        dict(ph='X',name='downstream_ADD',pid=7,tid=4,ts=120,dur=15,cat='kernel'),
        dict(ph='X',name='hcom_allreduce',pid=7,tid=9,ts=65,dur=10,cat='kernel'),
        dict(ph='s',name='HostToDevice1',cat='HostToDevice',id='9007199254740993',pid=1,tid=2,ts=20),
        dict(ph='f',name='HostToDevice1',cat='HostToDevice',id='9007199254740993',pid=7,tid=4,ts=30)]


def write_trace(tmp_path,data,name='trace_view.json'):
    path=tmp_path/name
    if path.suffix=='.gz':
        with gzip.open(path,'wt') as stream:json.dump(data,stream)
    else:path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize('wrapped',[False,True])
@pytest.mark.parametrize('compressed',[False,True])
def test_scoped_union_overlap_flow_and_no_cold_or_downstream_contamination(tmp_path,wrapped,compressed):
    data=trace();data={'traceEvents':data,'displayTimeUnit':'ms'} if wrapped else data
    path=write_trace(tmp_path,data,'trace_view.json'+('.gz' if compressed else ''))
    result=read_timeline(path,[LABEL],'MatMul',stock=True)
    window=result['documents'][0]['windows'][0]
    assert result['scope']=='L1_MEASURED_WINDOWS' and window['route_verified']
    assert window['window_us']==100 and window['kind_union_us']['compute']==50
    assert window['visible_device_task_union_us']==60 and window['uncovered_visible_device_us']==40
    assert window['visible_compute_communication_task_overlap_us']==20
    link=window['links'][0]
    assert link['flow_start_to_end_us']==10 and link['api_return_to_device_start_us']==5
    assert link['source_name']=='aclrtLaunchKernel' and link['target_name']=='MatMul'
    assert 'cold_MM' not in [e['name'] for e in result['documents'][0]['events']]
    assert 'downstream_ADD' not in [e['name'] for e in result['documents'][0]['events']]
    stream=next(s for s in window['streams'] if s['tid']=='4')
    assert [g['duration_us'] for g in stream['gaps']]==[20,30]  # Nested interval must not create a false gap.
    write_timeline(result,tmp_path/'analysis')
    assert (tmp_path/'analysis.csv').is_file()


def test_fused_kernel_cannot_reveal_internal_mm_sdma_overlap(tmp_path):
    result=read_timeline(write_trace(tmp_path,trace('A3Kernel')),[LABEL],'A3Kernel')
    window=result['documents'][0]['windows'][0]
    assert window['visible_compute_communication_task_overlap_us'] is None
    assert window['kind_union_us']['fused']==50
    assert 'UNKNOWN' in window['internal_fused_overlap'] and result['sdma_link_occupancy']=='UNKNOWN'


@pytest.mark.parametrize('mutation',[
    lambda d:d.pop(3),
    lambda d:d.append(copy.deepcopy(d[3])),
    lambda d:d[3].update(name=LABEL.replace('/n2','/n1')),
    lambda d:d[3].update(dur=-1),
    lambda d:d[6].update(ts='NaN'),
    lambda d:d[6].pop('dur'),
])
def test_incomplete_ambiguous_or_malformed_evidence_fails(tmp_path,mutation):
    data=trace();mutation(data)
    with pytest.raises(ValueError):read_timeline(write_trace(tmp_path,data),[LABEL])


def test_duplicate_exports_not_counted_twice_and_unmarked_historical_data_explicit(tmp_path):
    path=write_trace(tmp_path,trace())
    write_trace(tmp_path,trace(),'msprof_123.json')
    with pytest.raises(ValueError,match='duplicate'):read_timeline(tmp_path,[LABEL])
    data=trace();data.pop(3);path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='missing measured'):read_timeline(path)
    assert read_timeline(path,whole_trace=True)['scope']=='WHOLE_TRACE'
    with pytest.raises(ValueError):read_timeline(path,[LABEL],whole_trace=True)


def test_endpoint_ids_required_no_nearest_time_fallback_and_negative_delay_preserved(tmp_path):
    data=trace();data[-2]['id']='different'
    result=read_timeline(write_trace(tmp_path,data),[LABEL])
    assert result['host_device_links']==0 and result['documents'][0]['unresolved_flows']
    data=trace();data[6]['ts']=23;data[-1]['ts']=23
    result=read_timeline(write_trace(tmp_path,data),[LABEL])
    assert result['documents'][0]['windows'][0]['links'][0]['api_return_to_device_start_us']==-2
    data.append(copy.deepcopy(data[-2]))
    result=read_timeline(write_trace(tmp_path,data),[LABEL])
    assert result['host_device_links']==0


def test_large_absolute_timestamps_keep_small_intervals_and_be_ranges_work(tmp_path):
    data=trace()
    for event in data:
        if 'ts' in event:event['ts']=str(10**20+event['ts'])+'.125'
    window=data.pop(3);end={**window,'ph':'E','ts':str(10**20+110)+'.125'}
    window['ph']='B';window.pop('dur');data.extend([window,end])
    result=read_timeline(write_trace(tmp_path,data),[LABEL])
    assert result['documents'][0]['windows'][0]['window_us']==100
    assert result['documents'][0]['windows'][0]['links'][0]['flow_start_to_end_us']==10


def test_metadata_prevents_cann_api_from_being_misclassified_as_device_and_unknown_not_guessed():
    data=trace();data[5]['args']={'Task Type':'AI_CORE'}
    data.append(dict(ph='X',name='MysteryKernel',pid=999,tid=0,ts=20,dur=5))
    events,_,_=decode_events(data)
    assert next(e for e in events if e['name']=='aclrtLaunchKernel')['kind']=='host_submit'
    assert next(e for e in events if e['name']=='MysteryKernel')['kind']=='unknown'
    assert length([(0,10),(2,3),(8,12)])==12
    assert overlap([(0,10)],[(5,7),(6,12)])==5


def test_real_cpu_torch_chrome_export_with_named_range_is_supported(tmp_path):
    import subprocess
    script=r'''
import sys,torch
from L1_test.profile_windows import marked_range
from L1_test.timeline import read_timeline
name=sys.argv[2]
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
    with marked_range(name,'profiler',torch):
        x=torch.ones(2,2);y=x.mm(x)
    with torch.autograd.profiler.record_function('outside_CHECK'):
        z=y+1
profile.export_chrome_trace(sys.argv[1])
result=read_timeline(sys.argv[1],[name])
assert result['window_count']==1
assert not any(e['name']=='outside_CHECK' for e in result['documents'][0]['events'])
assert result['documents'][0]['windows'][0]['visible_device_task_union_us']==0
'''
    import os
    env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parent.parent)}
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path/'trace_view.json'),LABEL],env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr

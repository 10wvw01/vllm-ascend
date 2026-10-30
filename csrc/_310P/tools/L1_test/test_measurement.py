# SPDX-License-Identifier: Apache-2.0
"""Timing mechanics and evidence claims that matter for performance decisions."""
import sys
from pathlib import Path

import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from L1_test.measurement import measure_group, groups, disjoint, speedup_interval, warmup_quality
from L1_test.core import make_plan, execution_contract, summarize_units, analyze, pair_job
from L1_test.testing import reports
from L1_test.campaign import make_campaign, validate_campaign


@pytest.mark.parametrize('scenario,joins', [('ready',2),('steady',1)])
def test_bounded_group_retains_all_outputs_and_only_joins_at_declared_boundary(scenario,joins):
    import weakref
    calls=[];refs=[]
    class Output: pass
    class Event:
        def record(self, stream): calls.append('event')
        def elapsed_time(self, end): return .2
    class Stream:
        def synchronize(self):
            calls.append('join')
            assert all(r() is not None for r in refs)
    def op(i):
        assert all(r() is not None for r in refs)
        output=Output();refs.append(weakref.ref(output));calls.append(i)
        return output
    clock=iter(range(0,10000000,100000))
    outputs,rows=measure_group(op,4,2,scenario,Stream(),[Event(),Event()],lambda:next(clock))
    assert len(outputs)==2 and calls.count('join')==joins
    assert sum(r['calls'] for r in rows)==2
    assert [r['i'] for r in rows]==([4,5] if scenario=='ready' else [4])
    assert set(calls)=={'event','join',4,5}
    del outputs
    assert all(r() is None for r in refs)


def test_partial_blocks_have_call_weighted_means_and_explicit_quantile_scope():
    assert groups(5,2)==[(0,2),(2,2),(4,1)]
    stats=summarize_units([10,20,100],[2,2,1])
    assert stats['mean_us']==32 and stats['n']==3 and stats['calls']==5
    assert stats['quantile_scope']=='timing_block_mean_per_call'
    assert disjoint([(100,200),(200,300)]) and not disjoint([(100,201),(200,300)])
    assert not disjoint([(0,200)])


def test_capacity_rejected_without_changing_requested_window():
    plan=make_plan(['V3-5'],batch_size=3)
    with pytest.raises(ValueError,match='capacity 2'):execution_contract('V3-5',{},plan)
    assert execution_contract('STOCK',{},plan)['max_retained_calls']==32
    p=make_plan(['V3-1'],batch_size=32)
    assert execution_contract('V3-1',{},p)['output_slots']==32
    with pytest.raises(ValueError):execution_contract('V1-0',{},p)


def test_rounds_and_selection_are_explicit_not_pseudoreplication():
    plan=make_plan(['V3-5'],samples=5,rounds=3)
    assert [j['version'] for j in plan['jobs'][:8]]==['STOCK','V3-5','V3-5','STOCK','V3-5','STOCK','STOCK','V3-5']
    assert speedup_interval([10],[5])['low'] is None
    ci=speedup_interval([100,101,99],[80,81,79])
    assert 1.2<ci['low']<=ci['high']<1.3
    campaign=make_campaign(case_ids=['v3-5-c512-g1024'])
    validate_campaign(campaign)
    assert campaign['case_count']==1 and campaign['tp2_sessions']==4
    with pytest.raises(ValueError):make_campaign(case_ids=['unregistered'])
    assert warmup_quality([100]*3)['status']=='UNVERIFIED_SHORT_WARMUP'
    assert warmup_quality([100]*8)['status']=='STABLE_TIMING_WINDOW'
    assert warmup_quality([100,200]*4)['status']=='UNSTABLE_TIMING_WINDOW'


def test_fast_results_require_rounds_warmup_overhead_and_exclusivity():
    def result(rounds=3, mode='exclusive', overhead=.01, warmup='STABLE_TIMING_WINDOW'):
        p=make_plan(['V3-5'],samples=5,rounds=rounds,mode=mode)
        jobs={}
        for job in p['jobs']:
            value=100 if job['version']=='STOCK' else 80
            raw=reports(p,job,values=lambda rank,i:value)
            for rank in raw:
                for case in rank['cases']:
                    case['timer_overhead_ratio']=overhead
                    case['warmup']['status']=warmup
            jobs[job['id']]=pair_job(p,job,*raw)
        return {r['benefit_verdict'] for r in analyze(p,jobs)['results']}
    assert result()=={'FASTER_MEASURED_ENTRY'}
    assert result(rounds=1)=={'SCREEN_ONLY'}
    assert result(mode='shared')=={'OBSERVATIONAL_ONLY'}
    assert result(overhead=.1)=={'INCONCLUSIVE_TIMER_OVERHEAD'}
    assert result(warmup='UNSTABLE_TIMING_WINDOW')=={'INCONCLUSIVE_WARMUP'}

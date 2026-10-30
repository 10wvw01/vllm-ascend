# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract, frozen identities and exact-count TP2 arithmetic means."""
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from .measurement import speedup_interval, units

ROOT = Path(__file__).resolve().parents[4]
CONTRACT = "L1_TENSOR_IO_V4"
METRICS = ("wall_us", "event_us")
PHASES = ("host_prepare", "output_acquire_wrap", "enqueue", "mm", "seed", "local_ready_wait",
          "peer_ready_wait", "transport_submit", "receive_reduce", "gather", "output_copy", "drain_quiet_done")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def catalog():
    entries = json.loads(Path(__file__).with_name("versions.json").read_text())["versions"]
    return {v["id"]: v for v in entries}


def make_plan(versions=(), samples=100, warmup=20, rounds=2, mode="exclusive", scenarios=("ready", "steady"),
              shapes=(2048, 4096), collection="off", drift_limit=.05, batch_size=2, working_set=4):
    known = catalog()
    if len(set(versions)) != len(versions) or any(v not in known for v in versions):
        raise ValueError("select distinct registered VN-M candidates; empty means standalone Stock")
    if any(known[v].get("category") in ("graph", "diagnostic") for v in versions):
        raise ValueError("Graph/diagnostic versions are not Eager full MM+AR candidates")
    if any(type(x) is not int for x in (warmup, samples, rounds)) or not (2 <= warmup <= 1000 and 1 <= samples <= 10000 and 1 <= rounds <= 20):
        raise ValueError("W>=2, N>=1, R>=1 with bounded integer counts required")
    if mode not in ("exclusive", "shared") or collection not in ("off", "host", "profiler"):
        raise ValueError("invalid mode or collector")
    if not scenarios or len(set(scenarios)) != len(scenarios) or set(scenarios)-{'ready','steady'}:
        raise ValueError("only ready/steady time MM+AR; downstream consumption is an untimed gate")
    if type(batch_size) is not int or not 1 <= batch_size <= 32 or type(working_set) is not int or not 2 <= working_set <= 16:
        raise ValueError('batch_size in [1,32] and working_set in [2,16] required')
    if not shapes or len(set(shapes)) != len(shapes) or any(type(m) is not int or m < 1 or m > 4096 for m in shapes):
        raise ValueError("distinct physical M in [1,4096] required")
    if not math.isfinite(drift_limit) or not 0 < drift_limit < 1:
        raise ValueError("invalid drift threshold")
    stages = ["off"] if collection == "off" else ["off-before", collection, "off-after"]
    jobs = []
    for version in versions or ["STOCK"]:
        for stage in stages:
            for block in range(rounds):
                arms = ["STOCK", version, version, "STOCK"] if versions else ["STOCK", "STOCK"]
                if versions and block % 2:
                    arms = [version, "STOCK", "STOCK", version]
                for slot, arm in enumerate(arms):
                    jobs.append(dict(id=f"{version}-{stage}-r{block}-s{slot}", candidate=version,
                                     block=block, slot=slot, version=arm, stage=stage))
    return dict(contract=CONTRACT, W=warmup, N=samples, R=rounds, mode=mode, scenarios=list(scenarios),
                M=list(shapes), collection=collection, stages=stages, drift_limit=drift_limit,
                versions=list(versions), jobs=jobs, consumer="residual_add_fp16_ND", seed=4242,
                consumer_validation="untimed_immediate_same_stream_before_after",
                batch_size=batch_size, working_set=working_set, gate_variants=max(batch_size,working_set),
                checking='status_at_group_boundary; numeric_pre_post_and_last_group',
                output_retention='all_group_outputs_until_completion',
                paired_samples_per_arm_per_case_per_stage=2*rounds*samples,
                paired_units_per_arm={s:2*rounds*len(units(samples,batch_size,s)) for s in scenarios},
                statistic="rank-max timing unit totals / total calls; N counts calls; quantiles label timing-unit scope")


def validate_plan(plan):
    expected = make_plan(plan["versions"], plan["N"], plan["W"], plan["R"], plan["mode"],
                         plan["scenarios"], plan["M"], plan["collection"], plan["drift_limit"],
                         plan['batch_size'],plan['working_set'])
    if plan != expected:
        raise ValueError("plan differs from the canonical complete schedule")


def execution_contract(version, spec, plan):
    kind = 'stock' if version == 'STOCK' else catalog()[version]['runtime']
    raw_limit = min(32,64//len(plan['M']))
    if kind == 'stock':
        entry, storage, limit, sync = 'F.linear+HCCL', 'framework_allocation_per_call', 32, 'BACKEND_MANAGED'
    elif kind == 'v1' or kind == 'external':
        entry, storage, limit, sync = kind, 'UNVERIFIED', 1, 'NOT_AUDITED'
    elif kind.startswith('v2'):
        entry, storage, limit = 'torch.ops.memfabric310p.mm_ar', 'operator_owned_persistent_pool_and_tensor_lease', raw_limit
        sync = 'YES' if kind == 'v2_b1' else 'NOT_AUDITED' if kind == 'v2_private' and spec.get('api')!='v2_selected_v1' else 'NO_AFTER_WARMUP'
    else:
        pool = kind in ('v32','a3','a3-2','a3-tail')
        entry, storage, limit = 'torch.ops.memfabric310p.mm_ar', 'operator_owned_tensor_lease', 2 if pool else raw_limit
        sync = 'YES' if kind == 'v32' else 'NOT_AUDITED' if version == 'V3-0' else 'NO_AFTER_WARMUP'
    if plan['batch_size'] > limit:
        raise ValueError(f'{version}: retained output group {plan["batch_size"]} exceeds audited capacity {limit}; no silent downgrade')
    return dict(caller_contract='Torch dispatch: Tensor X/W -> complete ND Tensor Y', implementation_entry=entry,
                output_storage=storage,max_retained_calls=limit,output_slots=max(2,plan['batch_size']),
                operator_internal_stream_sync=sync,public_dispatch_alignment='UNVERIFIED_EXTERNAL' if kind=='external' else 'STOCK_TENSOR_BOUNDARY',
                model_callsite_integration='NOT_RUN',
                comparison_scope='public Tensor operator boundary; not pure kernel or model latency')


def validate_runtime(spec, version):
    if spec.get("version") != version or not re.fullmatch(r"[a-f0-9]{40}", spec.get("source_head", "")) or not spec.get("files"):
        raise ValueError("runtime version/full source SHA/files identity missing")
    for path, expected in spec["files"].items():
        if not Path(path).is_absolute() or sha(path) != expected:
            raise ValueError(f"frozen runtime file mismatch: {path}")
    environment = spec.get("environment", {})
    allowed = {"LD_LIBRARY_PATH", "MF_SDMA_ORCH_JSON", "MEMFABRIC_HYBRID_HOME_PATH",
               "MEMFABRIC_HYBRID_EXTEND_LIB_PATH", "V2_HOST_LIB",
               "VLLM_ASCEND_310P_MEMFABRIC_STORE_URL", "VLLM_ASCEND_310P_MEMFABRIC_MM_AR_SMALL_M_PATH",
               "VLLM_ASCEND_310P_MEMFABRIC_MM_AR_SINGLE_WAVE_MAX_M"}
    for key, value in environment.items():
        if key not in allowed or not isinstance(value, str):
            raise ValueError(f"unapproved runtime environment key: {key}")
    if version == "STOCK":
        if environment or any(spec.get(k) for k in ("library", "private_library", "factory_file", "api", "selected_device_library", "python_modules", "tensor_adapter")):
            raise ValueError("standalone Stock must not load fusion libraries/environment")
        return spec
    entry = catalog()[version]
    api = spec.get("api")
    if api not in (None, "v2_selected_v1", "v2_trial_v1", "python_session_v1"):
        raise ValueError("unknown runtime API")
    selected = api == "v2_selected_v1"
    session = api == "python_session_v1"
    base_v2 = api == "v2_trial_v1"
    if base_v2 and version not in ("V2-0", "V2-1", "V2-2", "V2-3"):
        raise ValueError("base V2 ABI is restricted to matching T0/D4/C3/F2 builds")
    family = entry.get("adapter_family", entry["runtime"])
    if session:
        roles = {"v3": {"memfabric_mm_ar_v3"},
                 "v32": {"memfabric_mm_ar_v3", "memfabric_mm_ar_v32"},
                 "a3": {"memfabric_mm_ar_v3", "memfabric_mm_ar_a3"}}
        modules = spec.get("python_modules", {})
        if not version.startswith("V3-") or family not in roles or set(modules) != roles[family]:
            raise ValueError("exact frozen Python session module roles required for V3")
        for path in modules.values():
            if not Path(path).is_absolute() or path not in spec["files"]:
                raise ValueError("Python session module must be in frozen files")
    if selected:
        expected_m = {"V2-7": [2048], "V2-15": [4096]}
        if version not in expected_m or spec.get("supported_M") != expected_m[version] or spec.get("config"):
            raise ValueError("selected V2 requires its fixed version/M and no configuration override")
        if spec.get("private_library") or spec.get("factory_file"):
            raise ValueError("selected V2 cannot load a historical private adapter")
    if entry.get("historical") and spec.get("historical_identity_verified") is not True:
        raise ValueError("historical VN-M requires explicit source/binary audit, no alias substitution")
    if selected:
        required = ["library", "selected_device_library"]
    elif base_v2:
        required = ["library"]
    elif session:
        required = ["library"] + (["tensor_adapter"] if family in ("v32", "a3") else [])
    elif entry["runtime"] == "external":
        required = ["factory_file"]
    else:
        required = ["library"]
        if entry["runtime"] == "v2_private":
            required.append("private_library")
        if entry["runtime"] in ("v32", "a3", "a3-2", "a3-tail"):
            required.append("tensor_adapter")
    for key in required:
        if not spec.get(key) or spec[key] not in spec["files"]:
            raise ValueError(f"{key} must be in frozen files")
    if version.startswith(('V2-','V3-')) and entry['runtime']!='external':
        if not spec.get('tensor_adapter') or spec['tensor_adapter'] not in spec['files']:
            raise ValueError('public Tensor operator requires its frozen TorchNPU Storage bridge')
        public_sources=[ROOT/'csrc/_310P/common/torch_tensor_entry.py',
                        ROOT/'csrc/_310P'/('memfabric_mm_ar_v2' if version.startswith('V2-') else 'memfabric_mm_ar_v3')/'src/torch_adapter/tensor_entry.py']
        if any(str(path) not in spec['files'] for path in public_sources):
            raise ValueError('operator-owned public Tensor entry sources must be frozen')
    if not spec.get("supported_M") or any(type(m) is not int or m < 1 for m in spec["supported_M"]):
        raise ValueError("candidate must declare admitted physical M shapes")
    # Configuration variants are explicit separate runtime specs, never silent defaults.
    conf = {**entry.get("config", {}), **spec.get("config", {})}
    if set(spec.get("config", {})) - set(entry.get("config", {})):
        raise ValueError("unregistered configuration field")
    if family == "a3" or entry["runtime"].startswith("a3"):
        if conf["compute_rows"] not in (512, 1024) or conf["comm_rows"] not in (512, 1024, 2048):
            raise ValueError("unsupported A3 compute/communication geometry")
    if version.startswith("V3-"):
        if conf["comm_rows"] not in ((256, 512, 1024, 2048) if family == "v3" else (512, 1024, 2048)):
            raise ValueError("unsupported V3 communication geometry")
        if conf.get("packet_bytes") not in (1 << 20, 4 << 20):
            raise ValueError("unsupported frozen V3 packet size")
        if family == "v3" and type(conf.get("pipeline")) is not bool:
            raise ValueError("pipeline must be an explicit boolean")
        if family == "v32" and (type(conf.get("merged_wave")) is not bool or
                conf["merged_wave"] != entry["config"]["merged_wave"]):
            raise ValueError("classic/merged identity differs from registered VN-M")
    if entry["runtime"] == "v1" and conf.get("batch_basem_count", 1) not in (1, 2, 4):
        raise ValueError("V1 q must be 1, 2 or 4")
    return spec


def summarize(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("invalid/empty duration")
    ordered = sorted(values)
    def percentile(p):
        z=(len(ordered)-1)*p
        lo=int(z)
        return ordered[lo]+(ordered[min(lo+1,len(ordered)-1)]-ordered[lo])*(z-lo)
    return dict(n=len(values), mean_us=math.fsum(values)/len(values), p50_us=percentile(.5),
                p90_us=percentile(.9), p99_us=percentile(.99), std_us=statistics.pstdev(values),
                min_us=min(values), max_us=max(values))


def summarize_units(values, counts):
    if len(values) != len(counts) or any(type(n) is not int or n < 1 for n in counts):
        raise ValueError('invalid timing-unit call counts')
    result = summarize(values)
    calls = sum(counts)
    mean = math.fsum(v*n for v,n in zip(values,counts))/calls
    result.update(mean_us=mean, calls=calls,
                  std_us=math.sqrt(math.fsum(n*(v-mean)**2 for v,n in zip(values,counts))/calls),
                  quantile_scope='single_call_latency' if all(n==1 for n in counts) else 'timing_block_mean_per_call')
    return result


def pair_job(plan, job, a, b):
    validate_plan(plan)
    if job not in plan['jobs']:
        raise ValueError('unregistered job')
    for rank,report in enumerate((a,b)):
        if report.get('rank')!=rank or report.get('status')!='PASS' or report.get('contract')!=CONTRACT:
            raise ValueError('rank identity/status/contract mismatch')
        if report.get('job')!=job or report.get('plan')!=plan:
            raise ValueError('mixed plans/jobs')
        if not report.get('route',{}).get('verified') or not report.get('runtime_identity'):
            raise ValueError('missing route/runtime evidence')
        if report.get('execution_contract')!=execution_contract(job['version'],report['runtime_identity'],plan):
            raise ValueError('public Tensor entry execution contract mismatch')
        evidence=report.get('downstream_validation',{})
        if evidence.get('status')!='PASS' or evidence.get('timed') is not False or evidence.get('same_stream') is not True:
            raise ValueError('untimed same-stream downstream validation required')
        expected_checks={(phase,m,v) for phase in ('before','after') for m in plan['M'] for v in range(plan['gate_variants'])}
        checks=evidence.get('cases',[])
        actual=[(c.get('phase'),c.get('M'),c.get('variant')) for c in checks]
        if len(actual)!=len(expected_checks) or set(actual)!=expected_checks or any(c.get('correct') is not True for c in checks):
            raise ValueError('missing/duplicate/failed downstream validation cases')
        submission=report.get('submission_validation',{})
        cases=submission.get('cases',[])
        if (submission.get('status')!='PASS' or submission.get('timed') is not False or
                len(cases)!=len(plan['M']) or {c.get('M') for c in cases}!=set(plan['M']) or
                any(c.get('calls')!=plan['batch_size'] or c.get('correct') is not True or c.get('disjoint') is not True for c in cases)):
            raise ValueError('retained-output submission gate required')
    if a['runtime_identity']!=b['runtime_identity']:
        raise ValueError('different rank runtime identities')
    expected={(m,s) for m in plan['M'] for s in plan['scenarios']}
    for report in (a,b):
        actual=[(x['M'],x['scenario']) for x in report['cases']]
        if len(actual)!=len(expected) or set(actual)!=expected:
            raise ValueError('missing/duplicate cases')
    right={(x['M'],x['scenario']):x for x in b['cases']}
    cases=[]
    for left in a['cases']:
        other=right[left['M'],left['scenario']]
        schedule=units(plan['N'],plan['batch_size'],left['scenario'])
        for case in (left,other):
            if (len(case['samples'])!=len(schedule) or case['warmup_count']!=plan['W'] or
                    case.get('checking')!=plan['checking'] or
                    case.get('validation')!={'status':'PASS','scope':'pre_post_and_last_group'}):
                raise ValueError('sample count/checking/validation mismatch')
            ratio=case.get('timer_overhead_ratio')
            if not isinstance(ratio,(int,float)) or not math.isfinite(ratio) or ratio<0:
                raise ValueError('timer calibration missing/invalid')
            if case.get('calibration',{}).get('subtracted') is not False or not case.get('warmup',{}).get('status'):
                raise ValueError('calibration must be reported without subtraction; warmup evidence required')
        values={k:[] for k in METRICS}
        counts=[]
        for (first,count),s,t in zip(schedule,left['samples'],other['samples']):
            for sample in (s,t):
                if (sample.get('i')!=first or sample.get('calls')!=count or
                        sample.get('variants')!=[i%plan['working_set'] for i in range(first,first+count)]):
                    raise ValueError('wrong timing unit/order/input rotation')
                if set(sample)&{'consumer_event_us','operator_event_us','consumer_correct','correct'}:
                    raise ValueError('legacy timing/per-call validation fields are not part of clean sampling')
            counts.append(count)
            for metric in METRICS:
                if any(not math.isfinite(x[metric]) or x[metric]<=0 for x in (s,t)):
                    raise ValueError('invalid rank duration')
                values[metric].append(max(s[metric],t[metric])/count)
        if sum(counts)!=plan['N']:
            raise ValueError('operator call count differs from N')
        cases.append(dict(M=left['M'],scenario=left['scenario'],paired_values=values,unit_calls=counts,
                          metrics={k:summarize_units(v,counts) for k,v in values.items()},
                          timer_overhead_ratio=max(left['timer_overhead_ratio'],other['timer_overhead_ratio']),
                          warmup_status=[x['warmup']['status'] for x in (left,other)]))
    return dict(job=job,runtime_identity=a['runtime_identity'],execution_contract=a['execution_contract'],cases=cases)


def analyze(plan, jobs):
    validate_plan(plan)
    if set(jobs)!={j['id'] for j in plan['jobs']}:
        raise ValueError('incomplete ABBA/BAAB blocks')
    for job in plan['jobs']:
        if jobs[job['id']]['job']!=job:
            raise ValueError('misfiled job')
    results=[]
    for version in plan['versions'] or ['STOCK']:
        selected=[jobs[j['id']] for j in plan['jobs'] if j['candidate']==version]
        for arm in {'STOCK',version}:
            identities=[x['runtime_identity'] for x in selected if x['job']['version']==arm]
            if any(x!=identities[0] for x in identities):
                raise ValueError('runtime changed between blocks/stages')
        for m in plan['M']:
            for scenario in plan['scenarios']:
                for metric in METRICS:
                    stage_results=[]
                    for stage in plan['stages']:
                        samples={arm:[] for arm in {'STOCK',version}}
                        counts={arm:[] for arm in samples}
                        rounds={arm:[] for arm in samples}
                        drifts=[];qualities=[]
                        for block in range(plan['R']):
                            centers={arm:[] for arm in samples}
                            for x in selected:
                                if x['job']['block']!=block or x['job']['stage']!=stage:
                                    continue
                                case=next(c for c in x['cases'] if (c['M'],c['scenario'])==(m,scenario))
                                arm=x['job']['version']
                                samples[arm].extend(case['paired_values'][metric])
                                counts[arm].extend(case['unit_calls'])
                                centers[arm].append(case['metrics'][metric]['mean_us'])
                                qualities.append(case)
                            for arm in rounds:
                                rounds[arm].append(statistics.mean(centers[arm]))
                            drifts.append(max(centers['STOCK'])/min(centers['STOCK'])-1)
                        if any(sum(v)!=2*plan['R']*plan['N'] for v in counts.values()):
                            raise ValueError('unequal effective operator calls')
                        stats={k:summarize_units(v,counts[k]) for k,v in samples.items()}
                        stage_results.append(dict(stage=stage,stats=stats,stock_bracket_drift=drifts,
                            paired_round_speedup=speedup_interval(rounds['STOCK'],rounds[version],plan['seed']) if version!='STOCK' else None,
                            max_timer_overhead_ratio=max(c['timer_overhead_ratio'] for c in qualities),
                            warmup_stable=all(s=='STABLE_TIMING_WINDOW' for c in qualities for s in c['warmup_status'])))
                    arm_drift={}
                    for arm in {'STOCK',version}:
                        centers=[c['metrics'][metric]['mean_us'] for x in selected
                                 if x['job']['version']==arm and x['job']['stage'].startswith('off')
                                 for c in x['cases'] if (c['M'],c['scenario'])==(m,scenario)]
                        arm_drift[arm]=max(centers)/min(centers)-1
                    stable=max(arm_drift.values())<=plan['drift_limit']
                    for row in stage_results:
                        diagnostic=not row['stage'].startswith('off')
                        row.update(version=version,M=m,scenario=scenario,metric=metric,arm_drift=arm_drift,
                            comparison_status='DIAGNOSTIC_ONLY' if diagnostic else 'OBSERVATIONAL_SHARED' if plan['mode']=='shared'
                            else 'COMPARABLE_EXCLUSIVE' if stable else 'INCONCLUSIVE_DRIFT',
                            execution_contracts={x['job']['version']:x['execution_contract'] for x in selected},
                            scope='public Torch Tensor entry; not pure kernel or model TTFT')
                        verdict='BASELINE_ONLY'
                        if version!='STOCK':
                            stock,candidate=row['stats']['STOCK']['mean_us'],row['stats'][version]['mean_us']
                            row.update(speedup=stock/candidate,latency_reduction=1-candidate/stock)
                            ci=row['paired_round_speedup']
                            verdict=('DIAGNOSTIC_ONLY' if diagnostic else 'OBSERVATIONAL_ONLY' if plan['mode']=='shared'
                                     else 'INCONCLUSIVE_CALL_LAYER' if any(x['execution_contract']['public_dispatch_alignment']!='STOCK_TENSOR_BOUNDARY' for x in selected)
                                     else 'INCONCLUSIVE_DRIFT' if not stable else 'SCREEN_ONLY' if ci['low'] is None
                                     else 'INCONCLUSIVE_WARMUP' if not row['warmup_stable']
                                     else 'INCONCLUSIVE_TIMER_OVERHEAD' if row['max_timer_overhead_ratio']>.05
                                     else 'FASTER_MEASURED_ENTRY' if ci['low']>1
                                     else 'SLOWER_MEASURED_ENTRY' if ci['high']<1 else 'INCONCLUSIVE_UNCERTAINTY')
                        row['benefit_verdict']=verdict
                        if diagnostic:
                            row['instrumentation_delta']={arm:row['stats'][arm]['mean_us']/statistics.mean(
                                r['stats'][arm]['mean_us'] for r in stage_results if r['stage'].startswith('off'))-1
                                for arm in row['stats']}
                        results.append(row)
    return dict(contract=CONTRACT,plan=plan,results=results,release_go=False,
                downstream_validation='PASS_UNTIMED',
                note='N counts calls; steady quantiles describe block means. Independent rounds, not calls, are bootstrapped.')

# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract, frozen identities and exact-count TP2 arithmetic means."""
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = "L1_TENSOR_IO_V2"
METRICS = ("wall_us", "event_us", "operator_event_us", "consumer_event_us")
PHASES = ("host_prepare", "output_acquire_wrap", "enqueue", "mm", "seed", "local_ready_wait",
          "peer_ready_wait", "transport_submit", "receive_reduce", "gather", "output_copy", "drain_quiet_done")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def catalog():
    entries = json.loads(Path(__file__).with_name("versions.json").read_text())["versions"]
    return {v["id"]: v for v in entries}


def make_plan(versions=(), samples=100, warmup=20, rounds=2, mode="exclusive", scenarios=("ready", "chain"),
              shapes=(2048, 4096), collection="off", drift_limit=.05):
    known = catalog()
    if len(set(versions)) != len(versions) or any(v not in known for v in versions):
        raise ValueError("select distinct registered VN-M candidates; empty means standalone Stock")
    if any(known[v].get("category") in ("graph", "diagnostic") for v in versions):
        raise ValueError("Graph/diagnostic versions are not Eager full MM+AR candidates")
    if any(type(x) is not int for x in (warmup, samples, rounds)) or not (2 <= warmup <= 1000 and 1 <= samples <= 10000 and 1 <= rounds <= 20):
        raise ValueError("W>=2, N>=1, R>=1 with bounded integer counts required")
    if mode not in ("exclusive", "shared") or collection not in ("off", "host", "profiler"):
        raise ValueError("invalid mode or collector")
    if not scenarios or len(set(scenarios)) != len(scenarios) or set(scenarios) - {"ready", "chain"}:
        raise ValueError("invalid scenarios")
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
                for slot, arm in enumerate(arms):
                    jobs.append(dict(id=f"{version}-{stage}-r{block}-s{slot}", candidate=version,
                                     block=block, slot=slot, version=arm, stage=stage))
    return dict(contract=CONTRACT, W=warmup, N=samples, R=rounds, mode=mode, scenarios=list(scenarios),
                M=list(shapes), collection=collection, stages=stages, drift_limit=drift_limit,
                versions=list(versions), jobs=jobs, consumer="residual_add_fp16_ND", seed=4242,
                paired_samples_per_arm_per_case_per_stage=2*rounds*samples,
                statistic="arithmetic mean of same-index two-rank maxima; all N retained")


def validate_plan(plan):
    expected = make_plan(plan["versions"], plan["N"], plan["W"], plan["R"], plan["mode"],
                         plan["scenarios"], plan["M"], plan["collection"], plan["drift_limit"])
    if plan != expected:
        raise ValueError("plan differs from the canonical complete schedule")


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
        if environment or any(spec.get(k) for k in ("library", "private_library", "factory_file")):
            raise ValueError("standalone Stock must not load fusion libraries/environment")
        return spec
    entry = catalog()[version]
    if entry.get("historical") and not spec.get("historical_identity_verified"):
        raise ValueError("historical VN-M requires explicit source/binary audit, no alias substitution")
    if entry["runtime"] == "external":
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
    if not spec.get("supported_M") or any(type(m) is not int or m < 1 for m in spec["supported_M"]):
        raise ValueError("candidate must declare admitted physical M shapes")
    # Configuration variants are explicit separate runtime specs, never silent defaults.
    conf = {**entry.get("config", {}), **spec.get("config", {})}
    if set(spec.get("config", {})) - set(entry.get("config", {})):
        raise ValueError("unregistered configuration field")
    if entry["runtime"].startswith("a3"):
        if conf["compute_rows"] not in (512, 1024) or conf["comm_rows"] not in (512, 1024, 2048):
            raise ValueError("unsupported A3 compute/communication geometry")
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


def pair_job(plan, job, a, b):
    validate_plan(plan)
    if job not in plan["jobs"]:
        raise ValueError("unregistered job")
    for rank, report in enumerate((a,b)):
        if report.get("rank") != rank or report.get("status") != "PASS" or report.get("contract") != CONTRACT:
            raise ValueError("rank identity/status/contract mismatch")
        if report.get("job") != job or report.get("plan") != plan:
            raise ValueError("mixed plans/jobs")
        if not report.get("route", {}).get("verified") or not report.get("runtime_identity"):
            raise ValueError("missing route/runtime evidence")
    if a["runtime_identity"] != b["runtime_identity"]:
        raise ValueError("different rank runtime identities")
    expected={(m,s) for m in plan["M"] for s in plan["scenarios"]}
    for report in (a,b):
        actual=[(x["M"],x["scenario"]) for x in report["cases"]]
        if len(actual)!=len(expected) or set(actual)!=expected:
            raise ValueError("missing/duplicate cases")
    right={(x["M"],x["scenario"]):x for x in b["cases"]}
    cases=[]
    for left in a["cases"]:
        other=right[left["M"],left["scenario"]]
        if any(len(x["samples"])!=plan["N"] or x["warmup_count"]!=plan["W"] for x in (left,other)):
            raise ValueError("unequal warmup/sample count")
        values={k:[] for k in METRICS}
        for i,(s,t) in enumerate(zip(left["samples"],other["samples"])):
            if any(x["i"]!=i or x["variant"]!=i%2 or x["correct"] is not True or x["consumer_correct"] is not True for x in (s,t)):
                raise ValueError("wrong sample/order/correctness")
            for metric in METRICS:
                for x in (s,t):
                    if not math.isfinite(x[metric]) or x[metric]<=0:
                        raise ValueError("invalid rank duration")
                values[metric].append(max(s[metric],t[metric]))
        cases.append(dict(M=left["M"],scenario=left["scenario"],paired_values=values,
                          metrics={k:summarize(v) for k,v in values.items()}))
    return dict(job=job,runtime_identity=a["runtime_identity"],cases=cases)


def analyze(plan, jobs):
    validate_plan(plan)
    if set(jobs)!={j["id"] for j in plan["jobs"]}:
        raise ValueError("incomplete ABBA blocks")
    for job in plan["jobs"]:
        if jobs[job["id"]]["job"] != job:
            raise ValueError("misfiled job")
    results=[]
    for version in plan["versions"] or ["STOCK"]:
        selected=[jobs[j["id"]] for j in plan["jobs"] if j["candidate"]==version]
        for arm in {"STOCK",version}:
            identities=[x["runtime_identity"] for x in selected if x["job"]["version"]==arm]
            if any(x!=identities[0] for x in identities):
                raise ValueError("runtime changed between blocks/stages")
        for m in plan["M"]:
            for scenario in plan["scenarios"]:
                for metric in METRICS:
                    stage_results=[]
                    for stage in plan["stages"]:
                        samples={arm:[] for arm in {"STOCK",version}}
                        drifts=[]
                        for block in range(plan["R"]):
                            centers=[]
                            for x in selected:
                                if x["job"]["block"]!=block or x["job"]["stage"]!=stage:
                                    continue
                                case=next(c for c in x["cases"] if (c["M"],c["scenario"])==(m,scenario))
                                samples[x["job"]["version"]].extend(case["paired_values"][metric])
                                if x["job"]["version"]=="STOCK":
                                    centers.append(case["metrics"][metric]["mean_us"])
                            drifts.append(max(centers)/min(centers)-1)
                        if any(len(v)!=2*plan["R"]*plan["N"] for v in samples.values()):
                            raise ValueError("unequal effective N")
                        stats={k:summarize(v) for k,v in samples.items()}
                        stage_results.append(dict(stage=stage,stats=stats,stock_bracket_drift=drifts))
                    # Between-round/stage drift catches changes hidden by locally stable brackets.
                    arm_drift={}
                    for arm in {"STOCK",version}:
                        centers=[c["metrics"][metric]["mean_us"] for x in selected
                                 if x["job"]["version"]==arm and x["job"]["stage"].startswith("off")
                                 for c in x["cases"] if (c["M"],c["scenario"])==(m,scenario)]
                        arm_drift[arm]=max(centers)/min(centers)-1
                    stable=max(arm_drift.values())<=plan["drift_limit"]
                    for row in stage_results:
                        diagnostic=not row["stage"].startswith("off")
                        row.update(version=version,M=m,scenario=scenario,metric=metric,arm_drift=arm_drift,
                                   comparison_status="DIAGNOSTIC_ONLY" if diagnostic else
                                   "OBSERVATIONAL_SHARED" if plan["mode"]=="shared" else
                                   "COMPARABLE_EXCLUSIVE" if stable else "INCONCLUSIVE_DRIFT")
                        if version!="STOCK":
                            stock,candidate=row["stats"]["STOCK"]["mean_us"],row["stats"][version]["mean_us"]
                            row.update(speedup=stock/candidate,latency_reduction=1-candidate/stock)
                        if diagnostic:
                            row["instrumentation_delta"]={arm:row["stats"][arm]["mean_us"]/
                                statistics.mean(r["stats"][arm]["mean_us"] for r in stage_results if r["stage"].startswith("off"))-1
                                for arm in row["stats"]}
                        results.append(row)
    return dict(contract=CONTRACT,plan=plan,results=results,release_go=False,
                note="Mean of per-index rank maxima. Shared drift retained. Parallel phase durations are not additive.")

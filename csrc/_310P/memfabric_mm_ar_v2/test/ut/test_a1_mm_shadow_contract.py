# SPDX-License-Identifier: Apache-2.0
"""纯CPU合成MM-shadow端到端合同：安全、真实路由、漂移、不可误认D4内部时间。"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PERF=Path(__file__).resolve().parents[1]/"perf"
MODULE=PERF.parents[1]
sys.path.insert(0,str(PERF))
from a1_mm_shadow_assess import BASE, MEMORY, MS, evaluate


def make_fixture(root: Path, slowdown: float = 1.25):
    (root/"sessions").mkdir()
    (root/"routes/native").mkdir(parents=True)
    (root/"routes/nd").mkdir(parents=True)
    sha="a"*64
    (root/"build_manifest.json").write_text(
        json.dumps({"schema":"A1_MM_SHADOW_BUILD_V1","so_sha256":sha}))
    for arm in ("native","nd"):
        for rank in (0,1):
            for m in MS:
                csv=root/"routes"/arm/f"rank{rank}_m{m}_kernel_details.csv"
                csv.write_text("actual synthetic kernel csv; not real device data\n")
                meta={"schema":"A1_MM_SHADOW_ROUTE_V1","arm":arm,"rank":rank,
                      "M":m,"kernel_hits":6,"shadow_sha256":sha,"bit_exact":True,
                      "device_kernel_durations_us":[800.0 if arm=="native" else 1100.0]*6,
                      "device_kernel_p50_us":800.0 if arm=="native" else 1100.0,
                      "csv_sha256":hashlib.sha256(csv.read_bytes()).hexdigest()}
                (csv.parent/f"rank{rank}_m{m}_route.json").write_text(json.dumps(meta))
    for i,arm in enumerate(BASE+MEMORY):
        d=root/"sessions"/f"{i+1:02d}_{arm}"
        d.mkdir()
        for rank in (0,1):
            records=[]
            for m in MS:
                baseline=780 if m==2048 else 1510
                factor={"native":1.0,"nd":slowdown,"sym":slowdown*1.04,
                        "sym_clean":slowdown*1.08}[arm]
                samples=[{"i":j,"variant":j%2,
                          "event_us":float(baseline*factor+rank+j%2),
                          "wall_us":float(baseline*factor+50+rank)}
                         for j in range(100)]
                records.append({"M":m,"rank":rank,"arm":arm,"warmup":20,
                                "iters":100,"bit_exact":True,"samples":samples})
            report={"schema":"A1_MM_SHADOW_RANK_V1","arm":arm,"rank":rank,
                    "world_size":2,"shadow_sha256":sha,
                    "d4_sha256":"2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19",
                    "weight_layout":"NZ29","profile":False,
                    "sym_pool_created":arm in ("sym","sym_clean"),"results":records}
            (d/f"{arm}_rank{rank}.json").write_text(json.dumps(report))
    return root


def test_synthetic_complete_12_stage_cpu_contract(tmp_path):
    result=evaluate(make_fixture(tmp_path))
    assert result["status"]=="MM_ONLY_COMPARISON_VALID"
    assert result["native_stock_route_verified"] is True
    assert len(result["native_and_shadow_route_proofs"])==8
    assert len(result["comparisons"])==2
    assert result["comparisons"][1]["shadow_slowdown_pct"]>20
    assert result["real_full_d4_internal_mm_time"]=="UNKNOWN"
    assert result["formal_D4_phase_causality_proven"] is False


def test_missing_original_profiler_is_not_accepted(tmp_path):
    make_fixture(tmp_path)
    (tmp_path/"routes/native/rank0_m4096_kernel_details.csv").unlink()
    with pytest.raises(ValueError,match="missing independent"):
        evaluate(tmp_path)


def test_original_profiler_sha_mismatch_rejected(tmp_path):
    make_fixture(tmp_path)
    p=tmp_path/"routes/nd/rank0_m2048_kernel_details.csv"
    p.write_text("tampered")
    with pytest.raises(ValueError,match="hash mismatch"):
        evaluate(tmp_path)


def test_missing_rank_or_wrong_binary_rejected(tmp_path):
    make_fixture(tmp_path)
    p=tmp_path/"sessions/02_nd/nd_rank1.json"
    j=json.loads(p.read_text())
    j["shadow_sha256"]="wrong"
    p.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="identity mismatch"):
        evaluate(tmp_path)


def test_corrupt_sample_and_incomplete_100_rejected(tmp_path):
    make_fixture(tmp_path)
    p=tmp_path/"sessions/01_native/native_rank0.json"
    j=json.loads(p.read_text())
    j["results"][0]["samples"].pop()
    p.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="raw samples"):
        evaluate(tmp_path)


def test_no_cherrypick_bad_drift_must_stay_inconclusive(tmp_path):
    make_fixture(tmp_path)
    p=tmp_path/"sessions/08_native/native_rank0.json"
    q=tmp_path/"sessions/08_native/native_rank1.json"
    for file in (p,q):
        j=json.loads(file.read_text())
        for rec in j["results"]:
            for sample in rec["samples"]:
                sample["event_us"]*=1.25
        file.write_text(json.dumps(j))
    result=evaluate(tmp_path)
    assert result["status"]=="INCONCLUSIVE_DRIFT"
    assert result["comparisons"][1]["session_drift"]["native"]>0.05


@pytest.mark.parametrize("fname",[
    "a1_mm_shadow_build.sh","a1_mm_shadow_campaign.sh"])
def test_shell_parses_and_has_explicit_isolation(fname):
    p=PERF/fname
    result=subprocess.run(["bash","-n",str(p)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    s=p.read_text()
    assert "U1_G0_APPROVED" in s or "bisheng" in s.lower()
    assert "sudo" not in s and "pkill -f" not in s


def test_shadow_source_is_not_modified_production():
    source=(MODULE/"src/op_kernel/mf310p_v2_mm_shadow.asc").read_text()
    production=(MODULE/"src/op_kernel/mf310p_v2_kernel.asc").read_text()
    assert "V2MmShadowKernel" in source
    assert "V2TrialKernel" in production
    assert "kShadowAddTile = 2048" in source
    assert "TOTAL_UB_SIZE - kShadowVectorReserve - kUBForIB" in source
    assert "GetMatmulApiTiling" in (MODULE/"src/op_kernel/mf310p_v2_kernel_common.h").read_text()
    assert "smem_shm_sdma_signal_at" not in source
    assert "IBWait" not in source


def test_cpu_python_sources_compile():
    import ast
    for file in ("a1_mm_shadow_arm.py","a1_mm_shadow_profile.py",
                 "a1_mm_shadow_assess.py"):
        ast.parse((PERF/file).read_text(encoding="utf-8"),filename=file)

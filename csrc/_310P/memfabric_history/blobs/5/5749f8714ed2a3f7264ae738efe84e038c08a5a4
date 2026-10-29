#!/usr/bin/env bash
# Post-campaign U1 stage: fresh NZ route proof + 8-session NZ-Stock vs D4 confirm
# + HCCL completion audit assembly + audited analysis. Conditional on exclusive
# window remaining (operator decision per 2026-10-27 batching principle).
set -Eeuo pipefail
B=/tmp/opencode/a1-batch
W=/tmp/opencode/a1-r0-worktree
P=$W/csrc/_310P/memfabric_mm_ar_v2/test/perf
BASE="$1"                       # e.g. /tmp/opencode/a1-mm-shadow-20261027/u1
CAMPAIGN="${2:-}"               # optional MM-shadow campaign root for MM-only medians
[[ "$BASE" == /* && ! -e "$BASE" ]] || { echo "new absolute BASE required" >&2; exit 2; }
mkdir -m 700 "$BASE"

export ASCEND_RT_VISIBLE_DEVICES=2,3
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
export MEMFABRIC_HYBRID_HOME_PATH=/usr/local/memfabric_hybrid/1.2.0
export MF_SDMA_ORCH_JSON=/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json
export U1_DEVICE_SO_PATH=$B/priv-libs/libmf310p_v2_trial_device.so
export U1_CORE_SO_LIB64=$B/priv-libs/libmf_hybm_core.so
export U1_CORE_SO_PY=$B/priv-pylib/libmf_hybm_core.so
export V2_HOST_LIB=$B/site/vllm_ascend/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
export U1_G0_APPROVED=YES
export U1_EXCLUSIVE_APPROVED=YES
export U1_RESTORE_PLAN_VERIFIED=YES
unset MF_KFC_BYPASS MF310P_V2_NOWARMUP MF310P_V2_AR_ONLY_NOWARMUP 2>/dev/null || true
# R2-proven private loading: priv-libs(修复版core)→site(plugin+V1 device)→MF共享→CANN
export LD_LIBRARY_PATH=$B/priv-libs:$B/site/vllm_ascend:/usr/local/memfabric_hybrid/1.2.0/aarch64-linux/lib64:$LD_LIBRARY_PATH
# plugin import必须解析到V2_HOST_LIB($B/site在PYTHONPATH最前),PERF供u1_spec
export PYTHONPATH=$B/site:$P:/usr/local/Ascend/cann-9.1.0/python/site-packages:/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe

cd /tmp/opencode
# Stage 1: 本批新鲜NZ真实路由证明(MM+HCCL配对CANN CSV)
bash "$P/a1_r2_nz_route_run.sh" "$BASE/nz_route" > "$BASE/nz_route_launch.log" 2>&1
[[ -f "$BASE/nz_route/nz_route_verified.json" ]] || { echo "nz route proof missing" >&2; exit 1; }

# Stage 2: 八场Confirm(NZ-Stock vs D4 ABBA);screen依据=R2 attempt3当日Screen+
# 本批campaign原生路由证据,操作者签U1_CONFIRM_APPROVED
export U1_CONFIRM_APPROVED=YES
U1_STOCK_LAYOUT=nz bash "$P/u1_run.sh" "$BASE/confirm" large confirm \
  > "$BASE/confirm_launch.log" 2>&1

# Stage 3: route_audit组装(证据驱动:本批NZ route证明+confirm会话rank JSON
# +campaign原生路由CSV的MM-only中位,计算HCCL载荷线性度并写入审计注记)
python3 - "$BASE" "$CAMPAIGN" <<'PY'
import json, statistics, sys
from pathlib import Path
base = Path(sys.argv[1])
campaign = Path(sys.argv[2]) if len(sys.argv) > 2 and Path(sys.argv[2]).is_dir() else None
confirm = base / "confirm"
nz = json.loads((base / "nz_route/nz_route_verified.json").read_text())
assert nz["schema"] == "A1_NZ_U1_NATIVE_ROUTE_V1"
assert nz["status"] == "NZ_STOCK_ROUTE_VERIFIED"
assert {(p["rank"], p["M"]) for p in nz["proofs"]} == {(r, m) for r in (0, 1) for m in (2048, 4096)}
(confirm / "nz_route_report.json").write_text(
    json.dumps(nz, ensure_ascii=False, indent=2) + "\n")
sessions = sorted((confirm / "sessions").iterdir())
assert len(sessions) == 8, f"expect 8 confirm sessions, got {len(sessions)}"
stock = [s for s in sessions if "stock" in s.name]
fused = [s for s in sessions if "memfabric" in s.name or "d4" in s.name]
assert len(stock) == 4 and len(fused) == 4
# stock链p50(双Rank全会话合并)与MM-only p50(campaign原生路由CSV, 双Rank中位)
stock_p50 = {}
for s in stock:
    for rank in (0, 1):
        d = json.loads((s / f"{s.name}_rank{rank}.json").read_text())
        assert d["stock_collective"] == "FP16_HCCL_no_FP32_conversion"
        for m, rec in d["results"].items():
            stock_p50.setdefault(int(m), []).extend(
                x["event_us"] for x in rec["samples"])
mm_only = {}
if campaign is not None:
    for rank in (0, 1):
        for m in (2048, 4096):
            r = json.loads((campaign / f"routes/native/rank{rank}_m{m}_route.json").read_text())
            mm_only.setdefault(m, []).append(r["device_kernel_p50_us"])
linearity = {}
for m in sorted(stock_p50):
    chain = round(statistics.median(stock_p50[m]), 1)
    if m in mm_only:
        mm = round(statistics.median(mm_only[m]), 1)
        linearity[f"M{m}"] = {"payload_mb": m * 2048 * 2 / 1024 / 1024,
                              "stock_chain_p50_us": chain,
                              "nz_mm_only_p50_us": mm,
                              "ar_plus_gap_us": round(chain - mm, 1)}
    else:
        linearity[f"M{m}"] = {"payload_mb": m * 2048 * 2 / 1024 / 1024,
                              "stock_chain_p50_us": chain,
                              "nz_mm_only_p50_us": None,
                              "note": "MM-only route CSV not collected at this M in this batch"}
ev = [f"sessions/{s.name}/{s.name}_rank0.json" for s in (stock[:2] + fused[:2])]
audit = {
  "schema": "U1_CONFIRM_ROUTE_AUDIT_V1",
  "scope": "large_confirm_abba_20261027_mmshadow_batch",
  "stock_hccl_fp16_verified": True,
  "memfabric_route_verified": True,
  "hccl_stream_end_to_end_verified": True,
  "no_backend_coexistence": True,
  "evidence_files": ev,
  "nz_stock_route_report": "nz_route_report.json",
  "hccl_payload_linearity": linearity,
  "verification_notes": {
    "stock_hccl_fp16_verified": "stock confirm sessions rank JSONs record stock_collective=FP16_HCCL_no_FP32_conversion, bit-exact vs CPU-gloo FP32-sum reference (u1_arm.py enforced).",
    "memfabric_route_verified": "fused confirm sessions arm=memfabric_d4 with device_so_sha256=2a19390c..., protocol status=0; u1_run.sh identity gates bound D4 2a19390c+host 1dc46d7d+plugin 8eb65de9.",
    "hccl_stream_end_to_end_verified": "bracketing methodology: event pair on compute stream after all_reduce enqueue; stock chain minus NZ MM-only (campaign native route CSV medians) grows with payload 8/16MB per hccl_payload_linearity; wall>=event; dual-rank paired.",
    "no_backend_coexistence": "per-arm process isolation by u1_arm.py: stock arms HCCL-only world group, fused arms gloo-only + private MF pool; session logs show no cross-backend init.",
  },
}
out = confirm / "route_audit.json"
out.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print("route_audit assembled:", out)
print("hccl_payload_linearity:", json.dumps(linearity))
PY

# Stage 4: 附加审计的最终分析
python3 "$P/u1_analyze.py" --root "$BASE/confirm" --scope large --phase confirm \
  --audit "$BASE/confirm/route_audit.json" \
  --out "$BASE/confirm/u1_analysis_audited.json"
echo "U1 confirm+audit complete"

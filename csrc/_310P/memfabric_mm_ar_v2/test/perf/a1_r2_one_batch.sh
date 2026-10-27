#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# A1-R2一批进场：NZ真实Stock基线→已有D4诊断→15/5/5/15s长尾对照。
# 开始前OpenCode须重新完成G0/独占/真实私有SO/回滚与原仓CPU审查。
set -Eeuo pipefail
[[ "$#" == 1 ]] || { echo "usage: $0 NEW_ABSOLUTE_PRIVATE_RUN_ROOT" >&2; exit 2; }
OUT="$1"
[[ "$OUT" == /* && ! -e "$OUT" && -d "$(dirname "$OUT")" ]] || {
  echo "output directory must not exist; parent must exist" >&2; exit 2;
}
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
PERF="$ROOT/csrc/_310P/memfabric_mm_ar_v2/test/perf"
[[ "${U1_G0_APPROVED:-}" == YES &&
   "${U1_EXCLUSIVE_APPROVED:-}" == YES &&
   "${U1_RESTORE_PLAN_VERIFIED:-}" == YES ]] || {
  echo "current-batch G0/dual-die/rollback not independently approved" >&2; exit 2; }
[[ -z "${MF_KFC_BYPASS+x}" && -z "${MF310P_V2_NOWARMUP+x}" &&
   -z "${MF310P_V2_AR_ONLY_NOWARMUP+x}" &&
   -z "${MF_SDMA_ORCH_EPOCH_BUDGET_S+x}" &&
   -z "${A1_R2_EPOCH_ARM+x}" ]] || {
  echo "unsafe pre-existing MemFabric epoch/bypass variable" >&2; exit 2;
}
[[ "${ASCEND_RT_VISIBLE_DEVICES:-}" =~ ^[0-9]+,[0-9]+$ ]] || exit 2
[[ "${ASCEND_RT_VISIBLE_DEVICES%,*}" != "${ASCEND_RT_VISIBLE_DEVICES#*,}" ]] || exit 2
: "${A1_R2_DIAG_STORE_PORT:?private diag port required}"
: "${A1_SERVE_STORE_PORT:?private Serve pool port required}"
: "${A1_SERVE_PORT:?private Serve HTTP port required}"
: "${A1_RUNTIME_IDENTITY_JSON:?fresh actual Host/DeviceSO/Plugin/KVER evidence required}"
for tool in python3 pytest torchrun timeout npu-smi; do
  command -v "$tool" >/dev/null || { echo "missing tool $tool" >&2; exit 2; }
done
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]] || {
  echo "tracked code dirty: refuse reproducibility risk" >&2; exit 2;
}
mkdir -m 700 "$OUT"
step="CPU_PRECHECK"
on_err() {
  rc="$?"
  printf '%s\n' "STOP at $step rc=$rc, preserve $OUT, do not retry/reboot/switch cards" >&2
  printf '{"schema":"A1_R2_CAMPAIGN_STOP_V1","step":"%s","exit":%s}\n' "$step" "$rc" \
    > "$OUT/STOP_STAGE.json"
}
trap on_err ERR
echo "R2 START $(date -u +%FT%TZ)" > "$OUT/runbook.txt"
git -C "$ROOT" rev-parse HEAD > "$OUT/source_sha.txt"
npu-smi info > "$OUT/g0_before.txt" 2>&1
python3 -m pytest -q "$ROOT/csrc/_310P/memfabric_mm_ar_v2/test/ut" \
  > "$OUT/cpu_pytest.log" 2>&1
python3 -m py_compile "$PERF"/a1_{nz_stock_profile_probe,nz_route_gate,r1_l2_budget,r1_tail_analyze,r2_epoch_assess,r2_diag_assess}.py \
  > "$OUT/compile.log" 2>&1
for f in a1_r2_nz_route_run.sh a1_r2_diag_run.sh a1_r2_epoch_run.sh \
         a1_prefill_serve_session.sh u1_run.sh; do
  bash -n "$PERF/$f"
done
step="NZ_REAL_ROUTE"
bash "$PERF/a1_r2_nz_route_run.sh" "$OUT/nz_route" \
  > "$OUT/nz_route_launch.log" 2>&1
python3 - "$OUT/nz_route/nz_route_verified.json" <<'PY'
import json,sys
doc=json.load(open(sys.argv[1]))
if (doc.get("status")!="NZ_STOCK_ROUTE_VERIFIED" or
        {(x["rank"],x["M"]) for x in doc.get("proofs",[])}
        != {(r,m) for r in (0,1) for m in (2048,4096)}):
    raise SystemExit("NZ real-MatMul proof insufficient, STOP")
PY
step="NZ_STOCK_SMOKE"
U1_STOCK_LAYOUT=nz bash "$PERF/u1_run.sh" "$OUT/nz_smoke" large smoke \
  > "$OUT/nz_smoke_launch.log" 2>&1
step="NZ_STOCK_SCREEN"
U1_STOCK_LAYOUT=nz bash "$PERF/u1_run.sh" "$OUT/nz_screen" large screen \
  > "$OUT/nz_screen_launch.log" 2>&1
# U1 screen has already checked two Rank status, bit-exact, anchors and all samples;
# fresh NZ route evidence provided by independent short Profiler above.
step="NZ_STOCK_CONFIRM"
U1_CONFIRM_APPROVED=YES U1_STOCK_LAYOUT=nz \
  bash "$PERF/u1_run.sh" "$OUT/nz_confirm" large confirm \
  > "$OUT/nz_confirm_launch.log" 2>&1
# Confirm is only a measured candidate until separate HCCL stream audit is attached;
# do not auto-promote the U1 aggregate to a model-grade GO.
step="D4_INNER_MECHANISM"
bash "$PERF/a1_r2_diag_run.sh" "$OUT/mechanism" \
  > "$OUT/mechanism_launch.log" 2>&1
step="EPOCH_ABBA"
bash "$PERF/a1_r2_epoch_run.sh" "$OUT/epoch" \
  > "$OUT/epoch_launch.log" 2>&1
step="R1_OFFLINE_CROSS_CHECK"
mkdir -m 700 "$OUT/offline"
R1="$ROOT/csrc/_310P/memfabric_mm_ar_v2/evidence/a1_r1_batch_20261027"
python3 "$PERF/a1_r1_l2_budget.py" --evidence-root "$R1" \
  --out "$OUT/offline/r1_l2_budget.json"
python3 "$PERF/a1_r1_tail_analyze.py" \
  --l3-json "$R1/s5_l3/l3_result.json" \
  --out "$OUT/offline/r1_periodic_tail.json"
step="DONE"
npu-smi info > "$OUT/npu_after.txt" 2>&1
printf '%s\n' "R2_EVIDENCE_COLLECTED; confirm/HCCL route and inner phase still need independent review" \
  > "$OUT/FINISH_STAGE.txt"
echo "R2 archived all stages; never infer INNER_PHASE GO or L3/L4 gains from this batch"

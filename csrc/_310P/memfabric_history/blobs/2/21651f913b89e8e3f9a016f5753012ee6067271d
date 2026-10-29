#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# R2四个独立V2 Serve: 15s → 5s → 5s → 15s，严格单变量Epoch预算。
set -Eeuo pipefail
[[ "$#" == 1 ]] || { echo "usage: $0 NEW_ABSOLUTE_DIR" >&2; exit 2; }
OUT="$1"; PERF="$(cd "$(dirname "$0")" && pwd)"
[[ "$OUT" == /* && ! -e "$OUT" && -d "$(dirname "$OUT")" ]] || exit 2
[[ "${U1_G0_APPROVED:-}" == YES && "${U1_EXCLUSIVE_APPROVED:-}" == YES &&
   "${U1_RESTORE_PLAN_VERIFIED:-}" == YES ]] || {
  echo "No verified G0/Die exclusive/rollback" >&2; exit 2; }
: "${A1_RUNTIME_IDENTITY_JSON:?current runtime SO/D4/model identity required}"
: "${A1_SERVE_PORT:?isolated empty API port required}"
: "${A1_SERVE_STORE_PORT:?isolated empty MemFabric port required}"
[[ -z "${A1_R2_EPOCH_ARM+x}" && -z "${MF_SDMA_ORCH_EPOCH_BUDGET_S+x}" &&
   -z "${MF_KFC_BYPASS+x}" && -z "${MF310P_V2_NOWARMUP+x}" &&
   -z "${MF310P_V2_AR_ONLY_NOWARMUP+x}" ]] || {
  echo "R2 caller cannot pre-set epoch/bypass variables" >&2; exit 2; }
mkdir -m 700 "$OUT"
mkdir -m 700 "$OUT/sessions"
python3 - "$OUT/campaign.json" "$A1_RUNTIME_IDENTITY_JSON" <<'PY'
import json,sys,os,subprocess
dest,src=sys.argv[1:]
idn=json.load(open(src,encoding="utf-8"))
m={"schema":"A1_R2_EPOCH_CONTROL_V1","sequence":[["E15A",15],["E5A",5],
   ["E5B",5],["E15B",15]],"server_identity":idn,
   "physical_devices":os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
   "instrumentation":"no-profiler NPU event; separate Serve per budget",
   "actual_epoch_boundary_log_available":False,
   "causality_permission":False}
with open(dest,"x") as f:json.dump(m,f,indent=2)
PY
for item in E15A:15 E5A:5 E5B:5 E15B:15; do
  sid="${item%%:*}"; budget="${item##*:}"
  [[ ! -e "$OUT/sessions/$sid" ]] || exit 2
  # 用户同一批次授权在安全前提下执行诊断5秒，不改变正式安装/配置。
  export A1_R2_EPOCH_ARM="$budget"
  export MF_SDMA_ORCH_EPOCH_BUDGET_S="$budget"
  echo "R2 isolated trial $sid budget=${budget}s; no profiler" >&2
  if ! timeout --signal=TERM --kill-after=45s 35m \
    bash "$PERF/a1_prefill_serve_session.sh" v2 l3_event "$sid" \
    "$OUT/sessions/$sid" > "$OUT/${sid}.log" 2>&1; then
    echo "STOP on error in $sid, preserve its own Serve logs; no retry/no switch" >&2
    exit 1
  fi
  # 本次脚本与模型动态加载调用都必须成功，不能只看server正常退出。
  [[ -s "$OUT/sessions/$sid/client_requests.jsonl" &&
     -s "$OUT/sessions/$sid/manifest.json" ]] || exit 1
  # 每一场后重新做设备只读健康快照；不要将诊断环境变量泄漏到下一任务。
  npu-smi info > "$OUT/${sid}_device_after.txt" 2>&1
done
unset A1_R2_EPOCH_ARM MF_SDMA_ORCH_EPOCH_BUDGET_S
python3 "$PERF/a1_r2_epoch_assess.py" --root "$OUT" \
  --out "$OUT/epoch_assessment.json"
echo "R2 four Serve evidence complete; epoch causal status still UNKNOWN without real boundary traces"

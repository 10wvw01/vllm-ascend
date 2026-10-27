#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# R2五类已有D4机制诊断：每Case双Rank独立进程、独立池，失败立刻STOP。
set -Eeuo pipefail
[[ "$#" == 1 ]] || { echo "usage: $0 NEW_ABSOLUTE_DIR" >&2; exit 2; }
OUT="$1"
[[ "$OUT" == /* && ! -e "$OUT" && -d "$(dirname "$OUT")" ]] || exit 2
PERF="$(cd "$(dirname "$0")" && pwd)"
[[ "${U1_G0_APPROVED:-}" == YES && "${U1_EXCLUSIVE_APPROVED:-}" == YES &&
   "${U1_RESTORE_PLAN_VERIFIED:-}" == YES ]] || {
   echo "revalidate current G0/dualdie/restoration" >&2; exit 2; }
[[ "${ASCEND_RT_VISIBLE_DEVICES:-}" =~ ^[0-9]+,[0-9]+$ ]] || exit 2
[[ "${ASCEND_RT_VISIBLE_DEVICES%,*}" != "${ASCEND_RT_VISIBLE_DEVICES#*,}" ]] || exit 2
[[ -z "${MF_KFC_BYPASS+x}" && -z "${MF310P_V2_NOWARMUP+x}" &&
   -z "${MF310P_V2_AR_ONLY_NOWARMUP+x}" &&
   -z "${MF_SDMA_ORCH_EPOCH_BUDGET_S+x}" ]] || {
  echo "no bypass or Epoch diagnostic during kernel mechanism measurements" >&2; exit 2;
}
for tool in torchrun python3 timeout npu-smi; do command -v "$tool" >/dev/null || exit 2; done
: "${A1_R2_DIAG_STORE_PORT:?explicit verified private store port required}"
[[ "$A1_R2_DIAG_STORE_PORT" =~ ^[0-9]+$ &&
   "$A1_R2_DIAG_STORE_PORT" -ge 1024 &&
   "$A1_R2_DIAG_STORE_PORT" -le 65535 ]] || exit 2
for var in U1_DEVICE_SHA256 U1_HOST_SHA256 U1_PLUGIN_SHA256 U1_KVER_SHA256; do
  [[ -n "${!var:-}" ]] || { echo "missing approved identity $var" >&2; exit 2; }
done
mkdir -m 700 "$OUT"
npu-smi info > "$OUT/npu_before.txt" 2>&1
for case in formal loop4 diag0 diag4 diag5; do
  mkdir -m 700 "$OUT/$case"
  # 每个case运行前确认本批专属端口未被其它进程使用。
  python3 - "$A1_R2_DIAG_STORE_PORT" <<'PY'
import socket,sys
with socket.socket() as s:
    if s.connect_ex(("127.0.0.1",int(sys.argv[1]))) == 0:
        raise SystemExit("MemFabric store port unexpectedly occupied; STOP")
PY
  npu-smi info > "$OUT/$case/device_before.txt" 2>&1
  if ! timeout --signal=TERM --kill-after=30s 15m torchrun --standalone --nproc-per-node=2 \
     "$PERF/u1_diag_large.py" --case "$case" \
     --out "$OUT/$case" --warmup 3 --iters 8 \
     --store-url "tcp://127.0.0.1:$A1_R2_DIAG_STORE_PORT" \
     > "$OUT/$case/torchrun.log" 2>&1; then
    echo "R2 diagnostic $case failed: freeze logs, no more device work" >&2
    exit 1
  fi
  npu-smi info > "$OUT/$case/device_after.txt" 2>&1
  for rank in 0 1; do
    [[ -s "$OUT/$case/${case}_rank${rank}_result.json" ]] || {
      echo "incomplete 2-rank $case" >&2; exit 1;
    }
  done
done
python3 "$PERF/a1_r2_diag_assess.py" --root "$OUT" --out "$OUT/diag_assessment.json"
# 同一批次的额外短Profiler仅用于看Whole Kernel及AI Core活动，
# 与No-Profiler数据分开，绝不拿单Chunk差额冒充内部MM/SDMA时间。
if [[ "${A1_R2_SHORT_PROFILER:-NO}" == YES ]]; then
  mkdir -m 700 "$OUT/short_profiler"
  for case in formal loop4 diag0; do
    mkdir -m 700 "$OUT/short_profiler/$case"
    npu-smi info > "$OUT/short_profiler/$case/device_before.txt" 2>&1
    if ! timeout --signal=TERM --kill-after=30s 15m torchrun --standalone --nproc-per-node=2 \
      "$PERF/u1_diag_large.py" --case "$case" --profile \
      --out "$OUT/short_profiler/$case" --warmup 3 --iters 4 \
      --store-url "tcp://127.0.0.1:$A1_R2_DIAG_STORE_PORT" \
      > "$OUT/short_profiler/$case/torchrun.log" 2>&1; then
      echo "short profiler $case failed; STOP" >&2
      exit 1
    fi
    for rank in 0 1; do
      [[ -s "$OUT/short_profiler/$case/${case}_rank${rank}_trace.json" ]] || {
        echo "missing profiled Rank $rank trace" >&2; exit 1;
      }
    done
    npu-smi info > "$OUT/short_profiler/$case/device_after.txt" 2>&1
  done
fi
npu-smi info > "$OUT/npu_after.txt" 2>&1
echo "D4_DIAG_COMPLETE; INNER_PHASE_UNKNOWN until independent supported profiler evidence"

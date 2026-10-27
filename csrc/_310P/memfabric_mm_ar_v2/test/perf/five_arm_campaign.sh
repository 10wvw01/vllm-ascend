#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 实机G0已批准后执行：五臂十会话单卡双Die，全程保留原始Rank和Host日志。
set -Eeuo pipefail
[[ $# == 2 ]] || { echo "usage: $0 FRESH_ABSOLUTE_RUN_ROOT VERIFIED_ISOLATED_BUILD_DIR" >&2; exit 2; }
OUT="$1"; BUILD="$2"
[[ "$OUT" == /* && ! -e "$OUT" && -d "$(dirname "$OUT")" ]] || {
  echo "NEW absolute OUT path needed" >&2; exit 2; }
[[ "$BUILD" == /* && -f "$BUILD/manifest.json" ]] || {
  echo "isolated verified B2/E1 build required" >&2; exit 2; }
[[ "${A1_FIVE_ARM_G0_PASSED:-NO}" == YES &&
   "${A1_FIVE_ARM_CARD_EXCLUSIVE:-NO}" == YES ]] || {
  echo "G0 health+binary identity+continuous exclusive lease not approved; STOP" >&2; exit 2; }
: "${ASCEND_RT_VISIBLE_DEVICES:?G0 verified actual same-card dual-Die mapping missing}"
: "${V2_HOST_LIB:?loaded plugin shared library path required}"
: "${U1_DEVICE_SO_PATH:?frozen formal device binary path required}"
: "${MF_SDMA_ORCH_JSON:?KVER11 config required}"
: "${U1_DEVICE_SHA256:?}"
: "${U1_PLUGIN_SHA256:?}"
: "${U1_HOST_SHA256:?}"
: "${U1_KVER_SHA256:?}"
: "${FIVE_ARM_STORE_PORT:?private local MemFabric store port needed}"
[[ "$FIVE_ARM_STORE_PORT" =~ ^[0-9]+$ ]] &&
  ((FIVE_ARM_STORE_PORT>=1024 && FIVE_ARM_STORE_PORT<=65535)) || exit 2
for name in MF_KFC_BYPASS MF310P_V2_NOWARMUP MF310P_V2_AR_ONLY_NOWARMUP             MF310P_V2_AR_ONLY_PARTIAL_GUARD MF_SDMA_ORCH_EPOCH_BUDGET_S; do
  [[ -z "${!name+x}" ]] || { echo "forbidden bypass: $name" >&2; exit 2; }
done
PERF="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$PERF/../../../../.." && pwd)"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]] || {
  echo "dirty tracked working copy: STOP" >&2; exit 2; }
mkdir -m 700 "$OUT"
mkdir -m 700 "$OUT/sessions"
git -C "$ROOT" rev-parse HEAD > "$OUT/commit.txt"
git -C "$ROOT" status --porcelain > "$OUT/git_status.txt"
sha256sum "$U1_DEVICE_SO_PATH" "$V2_HOST_LIB" "$BUILD/"*.so > "$OUT/binaries_before.sha256"
npu-smi info > "$OUT/npu_before.txt" 2>&1 || { echo "npu-smi failed" >&2; exit 1; }
# 不占计算资源的只读30s设备巡检；异常退出同样保留日志并停止采样。
(
  while true; do
    date -Is
    npu-smi info
    sleep 30
  done
) > "$OUT/card_watch_30s.log" 2>&1 &
WATCH_PID=$!
cleanup_watch() {
  kill "$WATCH_PID" 2>/dev/null || true
  wait "$WATCH_PID" 2>/dev/null || true
}
trap cleanup_watch EXIT
python3 - "$BUILD/manifest.json" <<'PY'
import json,sys,pathlib,hashlib
p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text())
for path,expected in ((d['candidate_so'],d['candidate_sha256']),
                      (d['private_host_so'],d['private_host_sha256'])):
    assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==expected
assert d['isolated_not_installed'] and d['device_arch']=='dav-2002'
PY
# 不允许让真正NZ Stock悄悄退化为通用MatMul+TransData；先独立采CANN证据。
mkdir -m 700 "$OUT/routes"
if ! torchrun --standalone --nproc-per-node=2 "$PERF/five_arm_nz_route.py" \
     --out "$OUT/routes" > "$OUT/routes/torchrun.log" 2>&1; then
  echo "REAL_NZ_ROUTE_STOP; preserve CANN CSV evidence" | tee -a "$OUT/progress.txt"
  exit 1
fi
for rank in 0 1; do
  for m in 2048 4096; do
    [[ -s "$OUT/routes/rank${rank}_m${m}_route.json" ]] || exit 1
  done
done
echo "REAL_NZ_ROUTE_PROVEN" | tee -a "$OUT/progress.txt"
# E1可能触发全DCache flush副作用，先用独立极短双Rank数值Smoke。
# B2确保首个完整MM/AR两阶段代码真实可运行；任错不启动正式十场。
for spec in "3:b2" "5:e1"; do
  stage="${spec%%:*}"; arm="${spec#*:}"
  echo "SMOKE_START $arm" | tee -a "$OUT/progress.txt"
  if ! torchrun --standalone --nproc-per-node=2 "$PERF/five_arm_arm.py" \
       --phase smoke --warmup 3 --iters 6 \
       --stage "$stage" --out "$OUT/sessions" \
       --build-manifest "$BUILD/manifest.json" \
       --store-url "tcp://127.0.0.1:$FIVE_ARM_STORE_PORT" \
       > "$OUT/sessions/smoke_$arm.log" 2>&1; then
    echo "SMOKE_STOP $arm; preserve evidence" | tee -a "$OUT/progress.txt"
    exit 1
  fi
  for rank in 0 1; do
    [[ -s "$OUT/sessions/$(printf 'smoke_%02d_%s_rank%d.json' "$stage" "$arm" "$rank")" ]] || exit 1
  done
  echo "SMOKE_PASS $arm" | tee -a "$OUT/progress.txt"
done
# 保留各场B1设备首写防护、结构化R0/R1 JSON、日志；任错立刻停止。
SCHEDULE=(stock b1 b2 d4 e1 e1 d4 b2 b1 stock)
for i in "${!SCHEDULE[@]}"; do
  stage=$((i+1))
  arm="${SCHEDULE[$i]}"
  label="$(printf '%02d_%s' "$stage" "$arm")"
  echo "BEGIN $label" | tee -a "$OUT/progress.txt"
  python3 - "$FIVE_ARM_STORE_PORT" <<'PY'
import socket,sys
port=int(sys.argv[1]);s=socket.socket();s.settimeout(0.25)
if s.connect_ex(("127.0.0.1",port))==0:
    raise SystemExit("private MF store port still in use; STOP")
s.close()
PY
  if ! torchrun --standalone --nproc-per-node=2 "$PERF/five_arm_arm.py" \
       --stage "$stage" --out "$OUT/sessions" \
       --build-manifest "$BUILD/manifest.json" \
       --store-url "tcp://127.0.0.1:$FIVE_ARM_STORE_PORT" \
       > "$OUT/sessions/$label.log" 2>&1; then
    echo "STOP $label (preserve all evidence; do not relaunch blindly)" | tee -a "$OUT/progress.txt"
    exit 1
  fi
  for rank in 0 1; do
    [[ -s "$OUT/sessions/${label}_rank$rank.json" ]] || {
      echo "missing rank$rank in $label; STOP" >&2; exit 1; }
  done
  npu-smi info > "$OUT/sessions/${label}.npu_after.txt" 2>&1 || {
    echo "device health command failure; STOP" >&2; exit 1; }
  echo "COMPLETE $label" | tee -a "$OUT/progress.txt"
done
python3 "$PERF/five_arm_assess.py" --sessions "$OUT/sessions" \
  --route-dir "$OUT/routes" --out "$OUT/five_arm_assessment.json"
sha256sum "$U1_DEVICE_SO_PATH" "$V2_HOST_LIB" "$BUILD/"*.so > "$OUT/binaries_after.sha256"
diff -u "$OUT/binaries_before.sha256" "$OUT/binaries_after.sha256" || {
  echo "binary identity modified during performance runs; STOP" >&2; exit 1; }
npu-smi info > "$OUT/npu_after.txt" 2>&1
echo "DONE: check paired results + independent CANN native NZ route + exclusive-card monitoring before signing L1"

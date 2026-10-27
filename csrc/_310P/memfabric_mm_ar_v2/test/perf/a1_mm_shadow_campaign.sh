#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# MM01-03影子MM同场比较；正式D4保持只读；禁止无独占/强制替换SO。
set -Eeuo pipefail
[[ "$#" == 1 ]] || { echo "usage: $0 NEW_ABSOLUTE_PRIVATE_DIR" >&2; exit 2; }
OUT="$1"; PERF="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$PERF/../../../../.." && pwd)"
[[ "$OUT" == /* && ! -e "$OUT" && -d "$(dirname "$OUT")" ]] || {
  echo "new absolute result directory required" >&2; exit 2; }
for key in U1_G0_APPROVED U1_EXCLUSIVE_APPROVED U1_RESTORE_PLAN_VERIFIED \
           A1_MM_SHADOW_EXCLUSIVE_APPROVED; do
  [[ "${!key:-}" == YES ]] || { echo "missing current $key" >&2; exit 2; }
done
[[ "${ASCEND_RT_VISIBLE_DEVICES:-}" =~ ^[0-9]+,[0-9]+$ &&
   "${ASCEND_RT_VISIBLE_DEVICES%,*}" != "${ASCEND_RT_VISIBLE_DEVICES#*,}" ]] || exit 2
[[ -z "${MF_KFC_BYPASS+x}" && -z "${MF310P_V2_NOWARMUP+x}" &&
   -z "${MF_SDMA_ORCH_EPOCH_BUDGET_S+x}" ]] || exit 2
for tool in python3 torchrun timeout npu-smi sha256sum git; do
  command -v "$tool" >/dev/null || exit 2
done
for key in U1_DEVICE_SO_PATH U1_CORE_SO_LIB64 MF_SDMA_ORCH_JSON; do
  f="${!key:-}"
  [[ "$f" == /* && -f "$f" ]] || { echo "missing private binary $key" >&2; exit 2; }
done
: "${A1_MM_SHADOW_STORE_PORT:?single unused localhost MemFabric port}"
[[ "$A1_MM_SHADOW_STORE_PORT" =~ ^[0-9]+$ &&
   "$A1_MM_SHADOW_STORE_PORT" -ge 1024 &&
   "$A1_MM_SHADOW_STORE_PORT" -le 65535 ]] || exit 2
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]] || exit 2
sha() { sha256sum "$1" | awk '{print $1}'; }
export U1_DEVICE_SHA256="$(sha "$U1_DEVICE_SO_PATH")"
export U1_HOST_SHA256="$(sha "$U1_CORE_SO_LIB64")"
export U1_KVER_SHA256="$(sha "$MF_SDMA_ORCH_JSON")"
[[ "$U1_DEVICE_SHA256" == 2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19 &&
   "$U1_HOST_SHA256" == 1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30 &&
   "$U1_KVER_SHA256" == 25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78 ]] || {
  echo "frozen SO / patched MemFabric / KVER identity mismatch; no device run" >&2; exit 2; }
mkdir -m 700 "$OUT"
step=PREPARE
trap 'rc=$?; printf "{\"step\":\"%s\",\"rc\":%s}\n" "$step" "$rc" > "$OUT/STOP_STAGE.json"; echo "STOP $step rc=$rc; do not change card/retry" >&2' ERR
git -C "$ROOT" rev-parse HEAD > "$OUT/source_sha.txt"
npu-smi info > "$OUT/g0_before.txt" 2>&1
step=CPU
python3 -m pytest -q "$ROOT/csrc/_310P/memfabric_mm_ar_v2/test/ut" > "$OUT/cpu_ut.log" 2>&1
python3 -m py_compile "$PERF"/a1_mm_shadow_{arm,profile,assess}.py
bash -n "$PERF"/a1_mm_shadow_build.sh
step=BUILD
bash "$PERF/a1_mm_shadow_build.sh" "$OUT/build" > "$OUT/build_driver.log" 2>&1
cp -n "$OUT/build/manifest.json" "$OUT/build_manifest.json"
LIB="$OUT/build/libmf310p_v2_mm_shadow.so"
for arm in native nd; do
  step="ROUTE_$arm"
  mkdir -p "$OUT/routes/$arm"
  timeout --signal=TERM --kill-after=30s 20m \
    torchrun --standalone --nproc-per-node=2 "$PERF/a1_mm_shadow_profile.py" \
    --arm "$arm" --out "$OUT/routes/$arm" \
    --shadow-so "$LIB" --build-manifest "$OUT/build_manifest.json" \
    > "$OUT/routes/$arm/torchrun.log" 2>&1
  npu-smi info > "$OUT/routes/$arm/device_after.txt" 2>&1
done
step=ABBA
mkdir "$OUT/sessions"
arms=(native nd nd native native nd nd native sym sym_clean sym_clean sym)
for idx in "${!arms[@]}"; do
  arm="${arms[$idx]}"
  printf -v number "%02d" "$((idx+1))"
  dir="$OUT/sessions/${number}_${arm}"
  step="MM_${number}_${arm}"
  mkdir -m 700 "$dir"
  python3 - "$A1_MM_SHADOW_STORE_PORT" <<'PY'
import socket,sys
with socket.socket() as s:
    if s.connect_ex(("127.0.0.1",int(sys.argv[1])))==0:
        raise SystemExit("MemFabric Store port occupied; STOP before launch")
PY
  npu-smi info > "$dir/device_before.txt" 2>&1
  timeout --signal=TERM --kill-after=30s 20m \
    torchrun --standalone --nproc-per-node=2 "$PERF/a1_mm_shadow_arm.py" \
    --arm "$arm" --out "$dir" --shadow-so "$LIB" \
    --build-manifest "$OUT/build_manifest.json" \
    --store-url "tcp://127.0.0.1:$A1_MM_SHADOW_STORE_PORT" \
    --warmup 20 --iters 100 > "$dir/torchrun.log" 2>&1
  [[ -s "$dir/${arm}_rank0.json" && -s "$dir/${arm}_rank1.json" ]] || exit 1
  npu-smi info > "$dir/device_after.txt" 2>&1
done
step=ANALYZE
python3 "$PERF/a1_mm_shadow_assess.py" --root "$OUT" \
  --out "$OUT/mm_shadow_assessment.json"
npu-smi info > "$OUT/npu_after.txt" 2>&1
echo "A1 MM shadow evidence collected; official D4 inside-MM wall time remains UNKNOWN" \
  > "$OUT/FINISH_STAGE.txt"

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# MM01-03影子MM同场比较；正式D4保持只读；禁止无独占/强制替换SO。
# 分批续跑语义(2026-10-27用户原则):外部占用中断时,操作者复核G0与监视器后
# 可对同一OUT按已完成阶段/会话粒度续跑;已完成会话必须通过与最终分析同源的
# read_session校验才可跳过;不完整目录先归档再整场重跑;绝不挑样本、绝不改
# 数值结果;数值/正确性/门禁失败仍是硬STOP,不由本脚本自动续跑。
set -Eeuo pipefail
[[ "$#" == 1 ]] || { echo "usage: $0 NEW_OR_RESUMABLE_ABSOLUTE_PRIVATE_DIR" >&2; exit 2; }
OUT="$1"; PERF="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$PERF/../../../../.." && pwd)"
[[ "$OUT" == /* && -d "$(dirname "$OUT")" ]] || {
  echo "absolute result directory required" >&2; exit 2; }
resume=0
if [[ -e "$OUT" ]]; then
  [[ -f "$OUT/source_sha.txt" ]] || {
    echo "existing dir is not a campaign root; refuse" >&2; exit 2; }
  resume=1
fi
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
for key in A1_MM_SHADOW_V1_DEVICE_LIB MEMFABRIC_HYBRID_HOME_PATH; do
  f="${!key:-}"
  [[ "$f" == /* && ( -f "$f" || -d "$f" ) ]] || { echo "missing $key" >&2; exit 2; }
done
[[ -f "$A1_MM_SHADOW_V1_DEVICE_LIB" ]] || { echo "A1_MM_SHADOW_V1_DEVICE_LIB must be a file" >&2; exit 2; }
# R2验证过的私有装载顺序：priv-libs(修复版hybm core)最前，site(含V1 device SO)
# 次之，MemFabric共享lib64再次；CANN/driver基础路径继承自外层环境。
MF_LIB64="$MEMFABRIC_HYBRID_HOME_PATH/$(uname -m)-linux/lib64"
[[ -f "$MF_LIB64/libmf_smem.so" ]] || MF_LIB64="$MEMFABRIC_HYBRID_HOME_PATH/lib64"
[[ -f "$MF_LIB64/libmf_smem.so" ]] || { echo "libmf_smem.so not found under MEMFABRIC_HYBRID_HOME_PATH" >&2; exit 2; }
PRIV_DIR="$(cd "$(dirname "$U1_CORE_SO_LIB64")" && pwd)"
V1_DIR="$(cd "$(dirname "$A1_MM_SHADOW_V1_DEVICE_LIB")" && pwd)"
export LD_LIBRARY_PATH="$PRIV_DIR:$V1_DIR:$MF_LIB64:${LD_LIBRARY_PATH:-}"
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
mkdir -m 700 "$OUT" 2>/dev/null || true
[[ -d "$OUT" ]] || { echo "cannot create result dir" >&2; exit 2; }
step=PREPARE
trap 'rc=$?; printf "{\"step\":\"%s\",\"rc\":%s,\"resume\":%s}\n" "$step" "$rc" "$resume" > "$OUT/STOP_STAGE.json"; echo "STOP $step rc=$rc; evidence archived. External-occupation interruptions may be resumed in batches by the operator after G0+monitor re-verification (user directive 2026-10-27); numeric/correctness/gate failures remain hard-STOP." >&2' ERR
if [[ "$resume" == 1 ]]; then
  printf 'RESUME %s head=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(git -C "$ROOT" rev-parse HEAD)" >> "$OUT/RESUME_LOG.txt"
fi
git -C "$ROOT" rev-parse HEAD > "$OUT/source_sha.txt.new"
if [[ -f "$OUT/source_sha.txt" ]] &&
   ! cmp -s "$OUT/source_sha.txt" "$OUT/source_sha.txt.new"; then
  echo "source SHA changed across resume; refuse mixing batches" >&2; exit 2
fi
mv -f "$OUT/source_sha.txt.new" "$OUT/source_sha.txt"
npu-smi info > "$OUT/g0_before_$(date -u +%Y%m%dT%H%M%SZ).txt" 2>&1
step=CPU
python3 -m pytest -q "$ROOT/csrc/_310P/memfabric_mm_ar_v2/test/ut" > "$OUT/cpu_ut.log" 2>&1
python3 -m py_compile "$PERF"/a1_mm_shadow_{arm,profile,assess}.py
bash -n "$PERF"/a1_mm_shadow_build.sh
# BUILD可续跑条件:已有manifest身份+产物SHA全部复核一致且HEAD未变,否则拒绝混合批次。
build_verified=0
if [[ "$resume" == 1 && -f "$OUT/build_manifest.json" &&
      -f "$OUT/build/libmf310p_v2_mm_shadow.so" &&
      -f "$OUT/build/libmf310p_v1_adapter_export.so" ]]; then
  if python3 - "$OUT/build_manifest.json" "$OUT/build" \
       "$A1_MM_SHADOW_V1_DEVICE_LIB" "$(git -C "$ROOT" rev-parse HEAD)" <<'PY'
import hashlib,json,sys,pathlib
man=pathlib.Path(sys.argv[1]); build=pathlib.Path(sys.argv[2])
v1dev=pathlib.Path(sys.argv[3]); head=sys.argv[4]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
try:
    m=json.loads(man.read_text(encoding="utf-8"))
    ok=(m.get("schema")=="A1_MM_SHADOW_BUILD_V1"
        and m.get("branch_head")==head
        and m.get("so_sha256")==digest(build/"libmf310p_v2_mm_shadow.so")
        and m.get("adapter_so_sha256")==digest(build/"libmf310p_v1_adapter_export.so")
        and m.get("v1_device_link_sha256")==digest(v1dev))
    raise SystemExit(0 if ok else 1)
except Exception:
    raise SystemExit(1)
PY
  then build_verified=1; fi
fi
step=BUILD
if [[ "$build_verified" == 1 ]]; then
  echo "BUILD stage already verified; resume without rebuild" | tee -a "$OUT/RESUME_LOG.txt"
else
  if [[ -d "$OUT/build" || -f "$OUT/build_manifest.json" ]]; then
    echo "existing build artifacts failed identity re-verification; refuse mixed batches" >&2; exit 2
  fi
  bash "$PERF/a1_mm_shadow_build.sh" "$OUT/build" > "$OUT/build_driver.log" 2>&1
  cp -n "$OUT/build/manifest.json" "$OUT/build_manifest.json"
fi
LIB="$OUT/build/libmf310p_v2_mm_shadow.so"
for arm in native nd; do
  step="ROUTE_$arm"
  # 路由证明可续跑:4份route.json+CSV全部存在且SHA/身份复核通过才跳过;
  # 部分完成则整体归档后重跑该arm的完整路由阶段(不与旧CSV混用)。
  route_ok=0
  if [[ "$resume" == 1 && -d "$OUT/routes/$arm" ]]; then
    if python3 - "$OUT/routes/$arm" "$OUT/build_manifest.json" "$arm" <<'PY'
import hashlib,json,sys,pathlib
folder=pathlib.Path(sys.argv[1]); man=json.loads(pathlib.Path(sys.argv[2]).read_text())
arm=sys.argv[3]; sha=man.get("so_sha256")
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ok=True
for rank in (0,1):
    for m in (2048,4096):
        meta=folder/f"rank{rank}_m{m}_route.json"
        csv=folder/f"rank{rank}_m{m}_kernel_details.csv"
        if not meta.is_file() or not csv.is_file():
            ok=False; continue
        try:
            r=json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            ok=False; continue
        if (r.get("schema")!="A1_MM_SHADOW_ROUTE_V1" or r.get("arm")!=arm
                or r.get("rank")!=rank or r.get("M")!=m
                or r.get("shadow_sha256")!=sha
                or r.get("bit_exact") is not True or r.get("kernel_hits",0)<3
                or r.get("csv_sha256")!=digest(csv)):
            ok=False
if not ok:
    raise SystemExit(1)
PY
    then route_ok=1; fi
  fi
  if [[ "$route_ok" == 1 ]]; then
    echo "ROUTE_$arm already verified; resume" | tee -a "$OUT/RESUME_LOG.txt"
    continue
  fi
  if [[ -d "$OUT/routes/$arm" ]]; then
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    mkdir -p "$OUT/aborted"
    mv "$OUT/routes/$arm" "$OUT/aborted/routes_${arm}_$ts"
    echo "ROUTE_$arm partial archived to aborted/routes_${arm}_$ts" >> "$OUT/RESUME_LOG.txt"
  fi
  mkdir -p "$OUT/routes/$arm"
  timeout --signal=TERM --kill-after=30s 20m \
    torchrun --standalone --nproc-per-node=2 "$PERF/a1_mm_shadow_profile.py" \
    --arm "$arm" --out "$OUT/routes/$arm" \
    --shadow-so "$LIB" --build-manifest "$OUT/build_manifest.json" \
    > "$OUT/routes/$arm/torchrun.log" 2>&1
  npu-smi info > "$OUT/routes/$arm/device_after.txt" 2>&1
done
step=ABBA
mkdir -m 700 "$OUT/sessions" 2>/dev/null || true
[[ -d "$OUT/sessions" ]] || { echo "cannot create sessions dir" >&2; exit 2; }
arms=(native nd nd native native nd nd native sym sym_clean sym_clean sym)
for idx in "${!arms[@]}"; do
  arm="${arms[$idx]}"
  printf -v number "%02d" "$((idx+1))"
  dir="$OUT/sessions/${number}_${arm}"
  step="MM_${number}_${arm}"
  # 会话可续跑:该场双Rank JSON必须通过与最终分析同源的read_session全量校验
  # (身份/100样本/bit_exact/统计),否则归档部分目录后整场重跑。
  if [[ -d "$dir" ]]; then
    if python3 - "$dir" "$arm" "$OUT/build_manifest.json" "$PERF" <<'PY'
import json,sys,pathlib
sys.path.insert(0,sys.argv[4])
from a1_mm_shadow_assess import read_session
man=json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
try:
    read_session(pathlib.Path(sys.argv[1]),sys.argv[2],man["so_sha256"])
except Exception:
    raise SystemExit(1)
PY
    then
      echo "MM_${number}_${arm} already verified; resume" | tee -a "$OUT/RESUME_LOG.txt"
      continue
    fi
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    mkdir -p "$OUT/aborted"
    mv "$dir" "$OUT/aborted/session_${number}_${arm}_$ts"
    echo "MM_${number}_${arm} partial archived to aborted/session_${number}_${arm}_$ts" >> "$OUT/RESUME_LOG.txt"
  fi
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
rm -f "$OUT/mm_shadow_assessment.json"
python3 "$PERF/a1_mm_shadow_assess.py" --root "$OUT" \
  --out "$OUT/mm_shadow_assessment.json"
npu-smi info > "$OUT/npu_after.txt" 2>&1
echo "A1 MM shadow evidence collected; official D4 inside-MM wall time remains UNKNOWN" \
  > "$OUT/FINISH_STAGE.txt"

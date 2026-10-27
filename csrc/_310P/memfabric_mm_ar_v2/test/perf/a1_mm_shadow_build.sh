#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 独立编译影子MM，绝不修改/install现有D4 Device SO。
set -Eeuo pipefail
[[ "$#" == 1 ]] || { echo "usage: $0 NEW_ABSOLUTE_OUTPUT_DIR" >&2; exit 2; }
OUT="$1"
[[ "$OUT" == /* && ! -e "$OUT" && -d "$(dirname "$OUT")" ]] || {
  echo "require unused absolute directory" >&2; exit 2; }
PERF="$(cd "$(dirname "$0")" && pwd)"
P="$(cd "$PERF/../.." && pwd)"
SRC="$P/src/op_kernel/mf310p_v2_mm_shadow.asc"
HDR="$P/src/op_kernel/mf310p_v2_kernel_common.h"
[[ -f "$SRC" && -f "$HDR" ]] || exit 2
: "${ASCEND_HOME_PATH:?CANN 9.1 root required}"
: "${MEMFABRIC_HYBRID_HOME_PATH:?installed MF public device header required}"
[[ "$ASCEND_HOME_PATH" == /* && "$MEMFABRIC_HYBRID_HOME_PATH" == /* ]] || exit 2
[[ -f "$ASCEND_HOME_PATH/aarch64-linux/include/acl/acl_rt.h" ]] || {
  echo "wrong CANN headers" >&2; exit 2; }
MF_INC="$MEMFABRIC_HYBRID_HOME_PATH/include/smem/device"
if [[ ! -f "$MF_INC/smem_shm_aicore_base_sdma.h" ]]; then
  MF_INC="$MEMFABRIC_HYBRID_HOME_PATH/$(uname -m)-linux/include/smem/device"
fi
[[ -f "$MF_INC/smem_shm_aicore_base_sdma.h" ]] || {
  echo "patched MemFabric public SDMA device header absent" >&2; exit 2; }
BISHENG="$ASCEND_HOME_PATH/bin/bisheng"
if [[ ! -x "$BISHENG" ]]; then
  BISHENG="$ASCEND_HOME_PATH/compiler/aarch64-linux/bin/bisheng"
fi
[[ -x "$BISHENG" ]] || { echo "CANN9.1 bisheng absent" >&2; exit 2; }
command -v sha256sum >/dev/null
command -v nm >/dev/null
command -v git >/dev/null
ROOT="$(cd "$P/../../.." && pwd)"
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]] || {
  echo "dirty tracked tree: do not compile uncontrolled variant" >&2; exit 2; }
mkdir -m 700 "$OUT"
LIB="$OUT/libmf310p_v2_mm_shadow.so"
# 与cmake/memfabric_310p_v2.cmake生产D4使用同一bisheng架构和依赖。
"$BISHENG" --npu-arch=dav-2002 -O2 -std=c++17 -w \
  -shared -fPIC -x asc "$SRC" -x none -o "$LIB" \
  -I"$ASCEND_HOME_PATH/aarch64-linux/include" \
  -I"$ASCEND_HOME_PATH/aarch64-linux/ascendc/include" \
  -I"$MF_INC" \
  -L"$ASCEND_HOME_PATH/aarch64-linux/lib64" \
  -lascendcl -lruntime -ldl -pthread -lm \
  "-Wl,-rpath,$ASCEND_HOME_PATH/aarch64-linux/lib64" \
  > "$OUT/build.log" 2>&1 || {
     echo "shadow BiSheng FAIL: preserve build.log; D4 untouched" >&2; exit 1; }
nm -D "$LIB" > "$OUT/symbols.txt"
grep -F 'mf310p_v2_mm_shadow_launch_async' "$OUT/symbols.txt" >/dev/null || {
  echo "missing exported shadow ABI" >&2; exit 1; }
python3 - "$OUT/manifest.json" "$ROOT" "$SRC" "$HDR" "$LIB" "$BISHENG" <<'PY'
import hashlib,json,subprocess,sys,pathlib
dst,root,src,hdr,lib,bisheng=map(pathlib.Path,sys.argv[1:])
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
data={"schema":"A1_MM_SHADOW_BUILD_V1",
      "branch_head":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
      "soc":"dav-2002","build_isolated":True,
      "source_sha256":digest(src),"common_header_sha256":digest(hdr),
      "so_sha256":digest(lib),"so_path":str(lib),
      "compiler_path":str(bisheng),
      "production_d4_sha256_untouched":"must separately verify pre/post D4 SO SHA"}
dst.write_text(json.dumps(data,indent=2)+"\n")
PY
echo "SHADOW_BUILD_READY $LIB (not installed, not linked into Serving)"

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
# 第二个独立产物：树内V1 adapter（与正式插件内建完全同源）导出mf310p_*池API，
# 供sym/sym_clean臂创建真实MemFabric对称arena。绝不安装、不改任何共享/冻结SO。
# 实机取证(2026-10-27)：私有Host SO(1dc46d7d)不导出mf310p_*；该符号只存在于
# 插件内部。导出SO是取得真实对称arena设备指针且不触碰冻结身份的唯一独立途径。
: "${A1_MM_SHADOW_V1_DEVICE_LIB:?absolute vllm_ascend V1 device SO (libmf310p_device.so) required}"
[[ "$A1_MM_SHADOW_V1_DEVICE_LIB" == /* && -f "$A1_MM_SHADOW_V1_DEVICE_LIB" ]] || {
  echo "A1_MM_SHADOW_V1_DEVICE_LIB must be absolute existing file" >&2; exit 2; }
command -v g++ >/dev/null || { echo "host g++ required for adapter export" >&2; exit 2; }
V1_ROOT="$(cd "$P/../memfabric_mm_ar" && pwd)"
V1_SRC="$V1_ROOT/memfabric310p_adapter.cpp"
V1_HDR="$V1_ROOT/memfabric310p_adapter_api.h"
[[ -f "$V1_SRC" && -f "$V1_HDR" ]] || { echo "V1 adapter source absent" >&2; exit 2; }
MF_HOST_INC="$MEMFABRIC_HYBRID_HOME_PATH/include/smem/host"
if [[ ! -f "$MF_HOST_INC/smem_shm.h" ]]; then
  MF_HOST_INC="$MEMFABRIC_HYBRID_HOME_PATH/$(uname -m)-linux/include/smem/host"
fi
[[ -f "$MF_HOST_INC/smem_shm.h" ]] || {
  echo "MemFabric public host headers absent" >&2; exit 2; }
MF_LIB64="$MEMFABRIC_HYBRID_HOME_PATH/$(uname -m)-linux/lib64"
if [[ ! -f "$MF_LIB64/libmf_smem.so" ]]; then
  MF_LIB64="$MEMFABRIC_HYBRID_HOME_PATH/lib64"
fi
[[ -f "$MF_LIB64/libmf_smem.so" ]] || {
  echo "MemFabric public libmf_smem.so absent" >&2; exit 2; }
V1_LIB_DIR="$(cd "$(dirname "$A1_MM_SHADOW_V1_DEVICE_LIB")" && pwd)"
ADP="$OUT/libmf310p_v1_adapter_export.so"
g++ -O2 -std=c++17 -fPIC -shared -o "$ADP" "$V1_SRC" \
  -I"$V1_ROOT" \
  -I"$ASCEND_HOME_PATH/aarch64-linux/include" \
  -I"$MF_HOST_INC" \
  -L"$V1_LIB_DIR" -lmf310p_device \
  -L"$MF_LIB64" -lmf_smem \
  -L"$ASCEND_HOME_PATH/aarch64-linux/lib64" -lascendcl \
  "-Wl,-rpath,$V1_LIB_DIR" "-Wl,-rpath,$MF_LIB64" \
  > "$OUT/adapter_build.log" 2>&1 || {
     echo "V1 adapter export g++ FAIL: preserve adapter_build.log" >&2; exit 1; }
nm -D "$ADP" > "$OUT/adapter_symbols.txt"
for sym in mf310p_create mf310p_control_barrier mf310p_exchange_geometry \
           mf310p_get_layout mf310p_destroy; do
  grep -F " T $sym" "$OUT/adapter_symbols.txt" >/dev/null || {
    echo "adapter export missing $sym" >&2; exit 1; }
done
python3 - "$OUT/manifest.json" "$ROOT" "$SRC" "$HDR" "$LIB" "$BISHENG" \
  "$V1_SRC" "$ADP" "$A1_MM_SHADOW_V1_DEVICE_LIB" <<'PY'
import hashlib,json,subprocess,sys,pathlib
dst,root,src,hdr,lib,bisheng,v1src,adp,v1dev=map(pathlib.Path,sys.argv[1:])
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
data={"schema":"A1_MM_SHADOW_BUILD_V1",
      "branch_head":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
      "soc":"dav-2002","build_isolated":True,
      "source_sha256":digest(src),"common_header_sha256":digest(hdr),
      "so_sha256":digest(lib),"so_path":str(lib),
      "compiler_path":str(bisheng),
      "adapter_source_sha256":digest(v1src),
      "adapter_so_sha256":digest(adp),"adapter_so_path":str(adp),
      "v1_device_link_lib":str(v1dev),
      "v1_device_link_sha256":digest(v1dev),
      "production_d4_sha256_untouched":"must separately verify pre/post D4 SO SHA"}
dst.write_text(json.dumps(data,indent=2)+"\n")
PY
echo "SHADOW_BUILD_READY $LIB (not installed, not linked into Serving)"
echo "ADAPTER_EXPORT_READY $ADP (pool API export only; not installed)"

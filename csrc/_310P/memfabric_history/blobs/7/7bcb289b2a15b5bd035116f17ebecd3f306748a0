#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# E4：E1 + 冻结E3-G + E4私有Kernel编译；绝不覆盖正式SO。
set -Eeuo pipefail
[[ $# == 1 ]] || { echo "usage: $0 FRESH_ABSOLUTE_OUTPUT_DIRECTORY" >&2; exit 2; }
OUT="$1"
[[ "$OUT" == /* && ! -e "$OUT" && -d "$(dirname "$OUT")" ]] || {
  echo "must use a fresh absolute output directory" >&2; exit 2; }
PERF="$(cd "$(dirname "$0")" && pwd)"
P="$(cd "$PERF/../.." && pwd)"
ROOT="$(cd "$P/../../.." && pwd)"
SRC="$P/src/op_kernel/mf310p_v2_dcci_entire_candidate.asc"
SRC_E3="$P/src/op_kernel/mf310p_v2_e3_group2048.asc"
SRC_E4="$P/src/op_kernel/mf310p_v2_e4_layered_candidates.asc"
PRIVATE_HOST="$PERF/e4_private_host.cpp"
BASE="$P/src/op_kernel/mf310p_v2_kernel.asc"
HDR="$P/src/op_kernel/mf310p_v2_kernel_common.h"
[[ -f "$SRC" && -f "$SRC_E3" && -f "$SRC_E4" && -f "$PRIVATE_HOST" && -f "$BASE" && -f "$HDR" ]] || exit 2
: "${ASCEND_HOME_PATH:?CANN 9.1 path is required}"
: "${MEMFABRIC_HYBRID_HOME_PATH:?installed MemFabric path is required}"
[[ "$ASCEND_HOME_PATH" == /* && "$MEMFABRIC_HYBRID_HOME_PATH" == /* ]] || exit 2
# 不允许拿未提交或未审核的内核临时文件执行性能实验。
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]] || {
  echo "tracked worktree dirty; STOP" >&2; exit 2; }
MF_INC="$MEMFABRIC_HYBRID_HOME_PATH/include/smem/device"
[[ -f "$MF_INC/smem_shm_aicore_base_sdma.h" ]] ||
  MF_INC="$MEMFABRIC_HYBRID_HOME_PATH/$(uname -m)-linux/include/smem/device"
[[ -f "$MF_INC/smem_shm_aicore_base_sdma.h" ]] || {
  echo "verified public MemFabric SDMA header missing" >&2; exit 2; }
BISHENG="$ASCEND_HOME_PATH/bin/bisheng"
[[ -x "$BISHENG" ]] ||
  BISHENG="$ASCEND_HOME_PATH/compiler/aarch64-linux/bin/bisheng"
[[ -x "$BISHENG" ]] || { echo "BiSheng missing" >&2; exit 2; }
[[ -f "$ASCEND_HOME_PATH/aarch64-linux/include/acl/acl_rt.h" ]] || exit 2
command -v sha256sum >/dev/null
command -v nm >/dev/null
mkdir -m 700 "$OUT"
LIB="$OUT/libmf310p_e4_candidates.so"
"$BISHENG" --npu-arch=dav-2002 -O2 -std=c++17 -w   -shared -fPIC -x asc "$SRC" -x asc "$SRC_E3" -x asc "$SRC_E4" -x none -o "$LIB"   -I"$ASCEND_HOME_PATH/aarch64-linux/include"   -I"$ASCEND_HOME_PATH/aarch64-linux/ascendc/include"   -I"$MF_INC"   -L"$ASCEND_HOME_PATH/aarch64-linux/lib64"   -lascendcl -lruntime -ldl -pthread -lm   "-Wl,-rpath,$ASCEND_HOME_PATH/aarch64-linux/lib64"   > "$OUT/build.log" 2>&1 || {
    echo "E Device build failed; preserve build.log; D4 stays unchanged" >&2; exit 1; }
nm -D "$LIB" > "$OUT/symbols.txt"
for ABI in mf310p_v2_dcci_entire_launch_async mf310p_v2_e3_group2048_launch_async mf310p_v2_e4_group2048_launch_async; do
  grep -F "$ABI" "$OUT/symbols.txt" >/dev/null || { echo "missing private ABI $ABI" >&2; exit 1; }
done
if grep -F " T mf310p_v2_trial_launch_async" "$OUT/symbols.txt"; then
  echo "candidate must not export/override formal D4 ABI" >&2; exit 1
fi
# 私有Host桥接：从正式plugin创建的trial上下文读取布局，只发射E1/E3G/E4候选。
# 仅与本批二进制配对；不链接/覆盖生产Host或Serving扩展。
command -v g++ >/dev/null
HOST="$OUT/libmf310p_e4_private_host.so"
g++ -O2 -std=c++17 -shared -fPIC -o "$HOST" "$PRIVATE_HOST" \
  -I"$P/src/op_host" \
  -I"$ASCEND_HOME_PATH/aarch64-linux/include" \
  -L"$OUT" -lmf310p_e4_candidates \
  -L"$ASCEND_HOME_PATH/aarch64-linux/lib64" -lascendcl \
  "-Wl,-rpath,$OUT" "-Wl,-rpath,$ASCEND_HOME_PATH/aarch64-linux/lib64" \
  > "$OUT/host_build.log" 2>&1 || {
    echo "private Host build failed; D4 untouched" >&2; exit 1; }
nm -D "$HOST" > "$OUT/host_symbols.txt"
grep -F 'mf310p_e4_run_private' "$OUT/host_symbols.txt" >/dev/null || {
  echo "private Host ABI missing" >&2; exit 1; }
python3 - "$OUT/manifest.json" "$ROOT" "$SRC" "$SRC_E3" "$SRC_E4" "$PRIVATE_HOST" "$HOST" "$BASE" "$HDR" "$LIB" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out,root,src,src_e3,src_e4,host_src,host,baseline,hdr,lib=map(pathlib.Path,sys.argv[1:])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
meta={"schema":"E4_LAYERED_ADD_BUILD_V1",
      "mainline":"dev/310p-memfabric-mm-ar",
      "work_head":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
      "device_arch":"dav-2002", "isolated_not_installed":True,
      "source_sha256":sha(src),"e3_source_sha256":sha(src_e3),
      "e4_source_sha256":sha(src_e4),
      "private_host_source_sha256":sha(host_src),"private_host_so":str(host),
      "private_host_sha256":sha(host),
      "frozen_baseline_source_sha256":sha(baseline),
      "common_sha256":sha(hdr),"candidate_so":str(lib),"candidate_sha256":sha(lib),
      "npu_correctness":"NOT_RUN","perf":"NOT_RUN",
      "formal_d4_binary_intact":"MUST_VERIFY_PRE_AND_POST_AT_G0"}
out.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
PY
echo "E4_ISOLATED_BUILD_READY $LIB $HOST ; NO_DEVICE_TEST_OR_INSTALL_PERFORMED"

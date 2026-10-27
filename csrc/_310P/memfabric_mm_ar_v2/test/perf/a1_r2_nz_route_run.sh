#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 原生NZ Stock双Rank短Profiler→真实CANN CSV→机器路由核验的一体入口。
# 禁止共享部署，禁止在未经本轮G0/独占/恢复审批的情况下执行。
set -Eeuo pipefail
[[ "$#" == 1 ]] || { echo "usage: $0 NEW_ABSOLUTE_DIR" >&2; exit 2; }
OUT="$1"
[[ "$OUT" == /* && ! -e "$OUT" && -d "$(dirname "$OUT")" ]] || {
  echo "require NEW absolute private output directory" >&2; exit 2;
}
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
PERF="$ROOT/csrc/_310P/memfabric_mm_ar_v2/test/perf"
[[ "${U1_G0_APPROVED:-}" == YES && "${U1_EXCLUSIVE_APPROVED:-}" == YES &&
   "${U1_RESTORE_PLAN_VERIFIED:-}" == YES ]] || { echo "G0/EXCLUSIVE/RESTORE NO_GO" >&2; exit 2; }
[[ "${ASCEND_RT_VISIBLE_DEVICES:-}" =~ ^[0-9]+,[0-9]+$ ]] || exit 2
[[ "${ASCEND_RT_VISIBLE_DEVICES%,*}" != "${ASCEND_RT_VISIBLE_DEVICES#*,}" ]] || exit 2
[[ -z "${MF_KFC_BYPASS+x}" && -z "${MF310P_V2_NOWARMUP+x}" &&
   -z "${MF310P_V2_AR_ONLY_NOWARMUP+x}" ]] || exit 2
for var in V2_HOST_LIB U1_DEVICE_SO_PATH U1_CORE_SO_LIB64 U1_CORE_SO_PY MF_SDMA_ORCH_JSON; do
  f="${!var:-}"
  [[ "$f" == /* && -f "$f" ]] || { echo "missing $var" >&2; exit 2; }
done
for exe in python3 torchrun timeout sha256sum npu-smi; do command -v "$exe" >/dev/null || exit 2; done
[[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]] || {
  echo "tracked working-tree changes forbidden for reproducible R2" >&2; exit 2;
}
sha() { sha256sum "$1" | awk '{print $1}'; }
export U1_DEVICE_SHA256="$(sha "$U1_DEVICE_SO_PATH")"
export U1_HOST_SHA256="$(sha "$U1_CORE_SO_LIB64")"
export U1_KVER_SHA256="$(sha "$MF_SDMA_ORCH_JSON")"
[[ "$(sha "$U1_CORE_SO_PY")" == "$U1_HOST_SHA256" ]] || exit 2
[[ "$U1_DEVICE_SHA256" == 2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19 ]] || exit 2
[[ "$U1_HOST_SHA256" == 1dc46d7dcbe583a5dbb78b4c65a0b5e22f9f352976c53ed31b761fa744f91a30 ]] || exit 2
[[ "$U1_KVER_SHA256" == 25e16b0af2f16989231ff0443698261257bd8142834783a0c1e4ad57cd9ebc78 ]] || exit 2
plugin_path="$(python3 -c 'import vllm_ascend.vllm_ascend_C as x;print(x.__file__)' 2>&1 | tail -n 1)"
[[ -f "$plugin_path" && "$(readlink -f "$plugin_path")" == "$(readlink -f "$V2_HOST_LIB")" ]] || {
  echo "wrong Python plugin imported" >&2; exit 2;
}
export U1_PLUGIN_SHA256="$(sha "$plugin_path")"
[[ "$U1_PLUGIN_SHA256" == 8eb65de98c3d8664ff10b421c1d7ad31318dc720c3f5679cd4d34e187203bea1 ]] || exit 2
mkdir -m 700 "$OUT"
npu-smi info > "$OUT/device_before.txt" 2>&1
python3 - "$OUT/run_identity.json" "$ROOT" <<'PY'
import json,os,subprocess,sys
out,root=sys.argv[1:]
data={"schema":"A1_R2_NZ_ROUTE_CAPTURE_V1",
      "repo_sha":subprocess.check_output(["git","-C",root,"rev-parse","HEAD"],text=True).strip(),
      "device_pair":os.environ["ASCEND_RT_VISIBLE_DEVICES"],
      "host_so_sha256":os.environ["U1_HOST_SHA256"],
      "device_so_sha256":os.environ["U1_DEVICE_SHA256"],
      "plugin_sha256":os.environ["U1_PLUGIN_SHA256"],
      "kver_sha256":os.environ["U1_KVER_SHA256"],
      "profile":True,"timing_go":False}
with open(out,"x") as f: json.dump(data,f,indent=2)
PY
if ! timeout --signal=TERM --kill-after=30s 20m torchrun --standalone --nproc-per-node=2 \
  "$PERF/a1_nz_stock_profile_probe.py" --out "$OUT" --iters 6 \
  > "$OUT/probe.log" 2>&1; then
  echo "NZ probe failed; DO NOT pass as NZ route verified" >&2
  exit 1
fi
for r in 0 1; do
  [[ -s "$OUT/nz_probe_rank${r}.json" ]] || { echo "missing Rank result" >&2; exit 1; }
  for m in 2048 4096; do
    [[ -s "$OUT/rank${r}_m${m}_kernel_details.csv" ]] || {
      echo "missing actual CANN CSV rank$r m$m; STOP not fake" >&2; exit 1;
    }
  done
done
python3 "$PERF/a1_nz_route_gate.py" --trace-root "$OUT" --out "$OUT/nz_route_verified.json" \
   > "$OUT/route_gate.log" 2>&1 || { echo "NZ_ROUTE_NO_GO" >&2; exit 1; }
npu-smi info > "$OUT/device_after.txt" 2>&1
echo "NZ_STOCK_ROUTE_VERIFIED (only profiler route; no formal ABBA speedup yet)"

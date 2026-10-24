#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# F2 D4 Tile2048 大M调度网格；每场独立TP2进程、不改正式核/Stock默认。
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
DRIVER="$ROOT/csrc/_310P/memfabric_mm_ar_v2/test/perf/f2_chunk_overlap_perf.py"
ANALYZE="$ROOT/csrc/_310P/memfabric_mm_ar_v2/test/perf/f2_sweep_analyze.py"
OUT="${1:?usage: f2_run_sweep.sh ABS_NEW_OUTPUT_DIR}"
[[ "$OUT" == /* && ! -e "$OUT" ]] || { echo "expected new absolute output dir" >&2; exit 2; }
[[ "${F2_EXCLUSIVE_APPROVED:-}" == "YES" ]] || { echo "F2_EXCLUSIVE_APPROVED=YES only after manual NPU exclusivity verification" >&2; exit 2; }
[[ "${F2_DEVICE_SO_PATH:-}" == /* && -f "$F2_DEVICE_SO_PATH" ]] || { echo "pin currently deployed F2_DEVICE_SO_PATH" >&2; exit 2; }
[[ "${MF_SDMA_ORCH_JSON:-}" == /*/libmf_sdma_orch_v11.json && -f "$MF_SDMA_ORCH_JSON" ]] || { echo "KVER11 JSON absolute path required" >&2; exit 2; }
[[ ! -v MF310P_V2_NOWARMUP && ! -v MF_KFC_BYPASS ]] || { echo "no warmup/KFC bypass" >&2; exit 2; }
[[ "${ASCEND_RT_VISIBLE_DEVICES:-}" =~ ^[0-9]+,[0-9]+$ ]] || { echo "pin TP2 physical die pair" >&2; exit 2; }
sha="$(sha256sum "$F2_DEVICE_SO_PATH" | awk '{print $1}')"
# D4 已完成实机验证的 Tile2048 设备 SO 哈希；不允许误装 C3 或旧 Tile1024。
D4_TILE2048_SHA="2a19390c83c3fe6b0bdb8946d88990cca66b331f765ca445dbda2e787c2d2a19"
[[ "$sha" == "$D4_TILE2048_SHA" ]] || {
  echo "BLOCKED: wrong Device SO SHA; expected D4 Tile2048=$D4_TILE2048_SHA actual=$sha" >&2
  echo "Do not silently override; audit CANN build/deployed kernel identity first." >&2
  exit 2
}
command -v timeout >/dev/null || { echo "Missing GNU timeout for hung-device protection" >&2; exit 2; }
[[ "$(git -C "$ROOT" status --porcelain --untracked-files=no)" == "" ]] || {
  echo "Tracked source tree dirty: halt, record diff, don't benchmark uncommitted code." >&2
  exit 2
}

mkdir -p "$OUT/sessions"
cat > "$OUT/manifest.txt" <<EOF
purpose=F2_D4_Tile2048_geometry_sweep
repo_head=$(git -C "$ROOT" rev-parse HEAD)
device_so_path=$F2_DEVICE_SO_PATH
device_so_sha256=$sha
verified_D4_tile2048_sha256=$D4_TILE2048_SHA
session_timeout=15m_plus_30s_kill
kver11_json=$MF_SDMA_ORCH_JSON
kver11_sha256=$(sha256sum "$MF_SDMA_ORCH_JSON" | awk '{print $1}')
devices=$ASCEND_RT_VISIBLE_DEVICES
shape=TP2_K2048_N2048_FP16
caveat=stock_vs_f2_baseline_not_measured_here
EOF
# M各组C1024/O1为D4正式baseline，每两组候选重复一次，便于发现热态漂移。
i=0
for m in 2048 4096; do
  for spec in "1024 1" "256 0" "256 1" "1024 1" \
              "512 0" "512 1" "1024 1" "1024 0" "1024 1"; do
    read -r chunk overlap <<<"$spec"
    i=$((i+1))
    name=$(printf "%02d_m%d_c%d_o%d" "$i" "$m" "$chunk" "$overlap")
    dir="$OUT/sessions/$name"
    mkdir "$dir"
    echo "F2 $name" >&2
    # 不自动忽略任何挂死、错误或非零状态；失败中止保全。
    if ! timeout --signal=TERM --kill-after=30s 15m torchrun --standalone --nproc-per-node=2 "$DRIVER" \
        --label "$name" --m "$m" --chunk-rows "$chunk" --overlap "$overlap" \
        --warmup 3 --iters 30 --device-so-sha "$sha" --out "$dir" > "$dir/session.log" 2>&1; then
      echo "F2 failed $name; preserve evidence. No automatic rerun, no F1 until device/KFC clean." >&2
      exit 1
    fi
  done
done
python3 "$ANALYZE" --root "$OUT/sessions" --device-so-sha "$sha" --out "$OUT/f2_screening.json"
echo "F2 screen finished: review drift, failures and separate Stock comparisons."

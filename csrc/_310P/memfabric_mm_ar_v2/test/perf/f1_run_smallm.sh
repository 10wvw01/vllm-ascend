#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# F1小M现有T1低开销路线基准筛选，必须保留Stock HCCL参考。
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
DRIVER="$ROOT/csrc/_310P/memfabric_mm_ar_v2/test/t1_nz_direct_trial.py"
ANALYZER="$ROOT/csrc/_310P/memfabric_mm_ar_v2/test/perf/f1_smallm_analyze.py"
OUT="$1"
[[ "$OUT" == /* && ! -e "$OUT" ]] || { echo "expected unused absolute output directory" >&2; exit 2; }
[[ "$F1_EXCLUSIVE_APPROVED" == "YES" ]] || { echo "check exclusive TP2 devices and Host/Device SO SHA first" >&2; exit 2; }
[[ "$ASCEND_RT_VISIBLE_DEVICES" =~ ^[0-9]+,[0-9]+$ ]] || { echo "physical TP2 die pair required" >&2; exit 2; }
[[ "$MF_SDMA_ORCH_JSON" == /*/libmf_sdma_orch_v11.json && -f "$MF_SDMA_ORCH_JSON" ]] || { echo "pin MemFabric KVER11" >&2; exit 2; }
[[ ! -v MF310P_V2_NOWARMUP && ! -v MF_KFC_BYPASS ]] || { echo "bypass not permitted" >&2; exit 2; }
mkdir "$OUT"
cat > "$OUT/manifest.txt" <<EOF
experiment=F1_low_M_M10_cost_screening
repo_head=$(git -C "$ROOT" rev-parse HEAD)
devices=$ASCEND_RT_VISIBLE_DEVICES
orch_sha256=$(sha256sum "$MF_SDMA_ORCH_JSON" | awk '{print $1}')
num_m=6
mm_cores=4,8
add_cores=1,4,8
host_timing=Stock_MM_HCCL_prepool_vs_T1_MM_MemFabric_postpool
warning=do_not_make_final_GO_from_prepool_vs_postpool_host_sync
EOF
# 一次执行36种形状分核，含M10以及邻近M8/M16，所有shape均逐次位精确校验。
torchrun --standalone --nproc-per-node=2 "$DRIVER" \
  --mode full --rows 1 8 10 16 32 64 \
  --mm-cores 4 8 --add-cores 1 4 8 --repeat 30 \
  --chunk-rows 256 > "$OUT/f1_t1_stock.log" 2>&1 || {
    echo "F1 failed; preserve logs and stop (no silent repair)." >&2
    exit 1
  }
python3 "$ANALYZER" --log "$OUT/f1_t1_stock.log" --out "$OUT/f1_screening.json"
echo "F1 initial screen only: must still separate MM, protocol/IB, wait, add by valid experiments."

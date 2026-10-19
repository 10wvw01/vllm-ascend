#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# T0/T1/T2/V1 regression driver (post G3 fixes). Runs on dev0,1.
set -uo pipefail
OUT=/tmp/opencode/g3/regression
mkdir -p "$OUT"
# Resolve checked-out repo independent of the original G3 machine home path.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
cd "$REPO_ROOT"
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/memfabric_hybrid/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1
export MF_SDMA_ORCH_JSON=/usr/local/memfabric_hybrid/latest/aarch64-linux/hybm/aicpu_kernel/libmf_sdma_orch_v11.json
export VLLM_ASCEND_310P_MEMFABRIC_STORE_URL=tcp://127.0.0.1:8581
# NOTE: no MF_KFC_BYPASS - the helper path is now the healthy default.
# T0 stand-alone driver was NOT committed by the device agent. Opt in
# explicitly to an existing driver checkout; never prepend a stale .so.
DIAG_DIR="${MF310P_V2_DIAG_DRIVER_DIR:-}"
if [ -n "$DIAG_DIR" ] && [ ! -f "$DIAG_DIR/v2_trial.py" ]; then
    echo "invalid MF310P_V2_DIAG_DRIVER_DIR: missing v2_trial.py" >&2
    exit 2
fi

run() {
    name=$1; shift
    echo "=== $name start $(date +%H:%M:%S) ==="
    (cd "$DIAG_DIR" && timeout 1200 torchrun --standalone --nproc-per-node=2 "$@" ) > "$OUT/$name.log" 2>&1
    rc=$?
    tail -2 "$OUT/$name.log" | head -1
    echo "=== $name rc=$rc $(date +%H:%M:%S) ===" | tee -a "$OUT/status.txt"
    sleep 45
}

if [ -n "$DIAG_DIR" ]; then
# T0 G0: official IB probes
run t0_g0 v2_trial.py --mode g0 --chunks 1024 --reps 1000
# T0 G1: serial multi-chunk correctness (reduced reps)
run t0_g1 v2_trial.py --mode g1 --chunks 256 512 1024 --rows 256 512 1024 --reps 100
# T0 G2: serial/overlap at M=2048/4096/8192
run t0_g2 v2_trial.py --mode g2 --chunks 1024 --rows 2048 4096 8192 --reps 20
# T0 stock baseline
run t0_stock v2_trial.py --mode stock --chunks 1024 --rows 2048 4096 8192 --reps 20
else
  echo "T0 standalone driver not in repository; SKIP G0/G1/G2/Stock (set MF310P_V2_DIAG_DRIVER_DIR)" | tee -a "$OUT/status.txt"
fi
# T1 small-M NZ direct
(timeout 1200 torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/test/t1_nz_direct_trial.py --mode full --rows 1 2 4 8 16 32 64 128 255 256 --repeat 20 ) > "$OUT/t1.log" 2>&1
echo "t1 rc=$?" | tee -a "$OUT/status.txt"
sleep 45
# T2 AR-only stress (bit-exact + status; staged AICore path)
timeout 1200 torchrun --standalone --nproc-per-node=2 \
  csrc/_310P/memfabric_mm_ar_v2/test/t2_ar_stress.py --chunk-rows 1024 --repeat 100 \
  > "$OUT/t2.log" 2>&1
echo "t2 rc=$?" | tee -a "$OUT/status.txt"
sleep 45
# V1 layer bench regression (bit-exact all rows)
timeout 1800 torchrun --standalone --nproc-per-node=2 \
  benchmarks/scripts/bench_310p_memfabric_mm_ar_layer.py \
  --rows 255 256 257 512 1024 2048 4096 8192 --repeat 5 > "$OUT/v1_bench.log" 2>&1
echo "v1 rc=$?" | tee -a "$OUT/status.txt"
echo "REGRESSION DONE"

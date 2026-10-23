#!/bin/bash
# D4-4 通用单会话 E2E 运行器（任意卡对/任意 epoch 预算）：
# 用法: d4_e2e_pair_one.sh <name> <so_path> <devs> <budget_s|default>
# 例:  d4_e2e_pair_one.sh cand_r1 $CAND_SO 4,5 default
#      d2_e2e_pair_one.sh cand_b5_r1 $CAND_SO 6,7 5
set -u
source /tmp/opencode/d4/d4_lib.sh
BENCH=$REPO/tests/e2e/_310p/bench_memfabric_mm_ar_serve.sh
STATUS=$D4/e2e_pair_status.txt

name=$1; so=$2; DEVS=$3; BUDGET=${4:-default}
outdir=$D4/e2e_pair/$name
mkdir -p "$outdir"

D1=${DEVS%,*}; D2=${DEVS#*,}
echo "$(date +%T) === pair[$DEVS] E2E $name ($(basename $so)) budget=${BUDGET}s ===" >> "$STATUS"

pair_idle() {
    local out rows busy=0
    out=$(timeout 30 npu-smi info 2>/dev/null) || return 1
    rows=$(echo "$out" | tr -d '|' | tr -s ' ' | awk -v d1="$D1" -v d2="$D2" '
        $1 ~ /^[0-9]+$/ && $3 ~ /^0000:/ && ($2==d1||$2==d2) {print $2, $4, int($5)}')
    [ -n "$rows" ] || return 1
    while read -r dev aicore mem; do
        [ "${aicore:-99}" = "0" ] || busy=1
        [ "${mem:-99999}" -lt 4000 ] || busy=1
    done <<< "$rows"
    return $busy
}

for i in $(seq 1 10); do pair_idle || { echo "$(date +%T) busy, wait $name" >> "$STATUS"; sleep 60; }; done
pair_idle || { echo "$(date +%T) no window, abort $name" >> "$STATUS"; exit 1; }

port=$((8300 + RANDOM % 100)); store_port=$((8760 + RANDOM % 40))

( while :; do
    ts=$(date +%T)
    timeout 20 npu-smi info 2>/dev/null | tr -d '|' | tr -s ' ' | \
      awk -v ts="$ts" -v d1="$D1" -v d2="$D2" \
      '$1 ~ /^[0-9]+$/ && $3 ~ /^0000:/ && ($2==d1||$2==d2) {print ts" dev"$2" aicore="$4"% mem="int($5)}'
    sleep 5
  done >> "$outdir/monitor.log" 2>&1 ) &
MON=$!

d4_swap_so "$so" | grep -q INSTALLED || { echo "$(date +%T) swap failed $name" >> "$STATUS"; kill $MON; exit 1; }
[ "$(sha256sum "$so" | cut -d' ' -f1)" = "$(cat "$D4/installed_device_so.sha")" ] || {
    echo "$(date +%T) sha mismatch $name" >> "$STATUS"; kill $MON; exit 1; }

BUDGET_ENV=()
[ "$BUDGET" != "default" ] && BUDGET_ENV=(MF_SDMA_ORCH_EPOCH_BUDGET_S="$BUDGET")

if ! ( cd "$REPO" && \
    env "${BUDGET_ENV[@]}" \
    MEMFABRIC_MM_AR_E2E_DEVICES=$DEVS \
    MEMFABRIC_MM_AR_E2E_NUM_PROMPTS=20 MEMFABRIC_MM_AR_E2E_OUTPUT_LEN=512 \
    MEMFABRIC_MM_AR_E2E_INPUT_LEN=4096 MEMFABRIC_MM_AR_E2E_CONCURRENCY=10 \
    MEMFABRIC_MM_AR_E2E_REPEAT=1 MEMFABRIC_MM_AR_E2E_V2=1 \
    MEMFABRIC_MM_AR_E2E_PORT=$port MEMFABRIC_MM_AR_E2E_STORE_PORT=$store_port \
    MEMFABRIC_MM_AR_E2E_CYCLE_TAG=1 \
    bash "$BENCH" "$outdir" "fused eager" >> "$outdir/runner.log" 2>&1 ); then
    echo "$(date +%T) SESSION_FAILED $name (bench rc!=0)" >> "$STATUS"
    kill $MON; pkill -9 -f 'vllm serve.*Qwen3.6-35B-A3B-w8a8' 2>/dev/null; exit 1
fi

res=$(ls "$outdir"/*rep1_run.json 2>/dev/null | head -1)
[ -n "$res" ] || { echo "$(date +%T) SESSION_FAILED $name (no json)" >> "$STATUS"; kill $MON; exit 1; }
read -r c f n <<< "$(python3 -c "import json;d=json.load(open('$res'));print(d['completed'],d['failed'],d['num_prompts'])" 2>/dev/null)"
if [ "${c:-0}" != "${n:-1}" ] || [ "${f:-1}" != "0" ]; then
    echo "$(date +%T) SESSION_FAILED $name (completed=$c failed=$f of $n — HANG/ERROR CONFIRMED)" >> "$STATUS"
    kill $MON; pkill -9 -f 'vllm serve.*Qwen3.6-35B-A3B-w8a8' 2>/dev/null; exit 1
fi
kill $MON 2>/dev/null
echo "$(date +%T) SESSION_OK $name (completed=$c failed=0)" >> "$STATUS"
exit 0

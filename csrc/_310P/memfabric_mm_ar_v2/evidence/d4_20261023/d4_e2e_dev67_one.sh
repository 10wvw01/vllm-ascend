#!/bin/bash
# D4-4 dev6/7 单会话 E2E 运行器（复跑实验用）：带 AICore 侧车监视 + bench 结果校验。
# 用法: d4_e2e_dev67_one.sh <name> <so_path>
set -u
source /tmp/opencode/d4/d4_lib.sh
BENCH=$REPO/tests/e2e/_310p/bench_memfabric_mm_ar_serve.sh
E2E_OUT=$D4/e2e_dev67
STATUS=$D4/e2e_dev67_status.txt
DEVS=6,7

name=$1; so=$2
outdir=$E2E_OUT/$name
mkdir -p "$outdir"

echo "$(date +%T) === dev67 E2E $name ($(basename $so)) [one-shot with monitor] ===" >> "$STATUS"

# 会话级唯一端口（bench serve / store）
port=$((8300 + RANDOM % 100))
store_port=$((8750 + RANDOM % 50))

# 侧车：每 5s 采样 dev6/7 AICore/mem（会话期间）
( while :; do
    ts=$(date +%T)
    timeout 20 npu-smi info 2>/dev/null | tr -d '|' | tr -s ' ' | \
      awk -v ts="$ts" '$1 ~ /^[0-9]+$/ && $3 ~ /^0000:/ && ($2=="6"||$2=="7") \
        {print ts" dev"$2" aicore="$4"% mem="int($5)}'
    sleep 5
  done >> "$outdir/monitor.log" 2>&1 ) &
MON=$!

if ! d4_swap_so "$so" | grep -q INSTALLED; then
    echo "$(date +%T) swap failed $name" >> "$STATUS"; kill $MON 2>/dev/null; exit 1
fi
want=$(sha256sum "$so" | cut -d' ' -f1); got=$(cat "$D4/installed_device_so.sha")
if [ "$want" != "$got" ]; then
    echo "$(date +%T) sha mismatch $name" >> "$STATUS"; kill $MON 2>/dev/null; exit 1
fi

if ! ( cd "$REPO" && \
    MEMFABRIC_MM_AR_E2E_DEVICES=$DEVS \
    MEMFABRIC_MM_AR_E2E_NUM_PROMPTS=20 MEMFABRIC_MM_AR_E2E_OUTPUT_LEN=512 \
    MEMFABRIC_MM_AR_E2E_INPUT_LEN=4096 MEMFABRIC_MM_AR_E2E_CONCURRENCY=10 \
    MEMFABRIC_MM_AR_E2E_REPEAT=1 MEMFABRIC_MM_AR_E2E_V2=1 \
    MEMFABRIC_MM_AR_E2E_PORT=$port MEMFABRIC_MM_AR_E2E_STORE_PORT=$store_port \
    MEMFABRIC_MM_AR_E2E_CYCLE_TAG=1 \
    bash "$BENCH" "$outdir" "fused eager" >> "$outdir/runner.log" 2>&1 ); then
    echo "$(date +%T) SESSION_FAILED $name (bench rc!=0)" >> "$STATUS"
    kill $MON 2>/dev/null; pkill -9 -f 'vllm serve.*Qwen3.6-35B-A3B-w8a8' 2>/dev/null; exit 1
fi

# 结果校验：必须 completed==num_prompts 且 failed==0
res=$(ls "$outdir"/*rep1_run.json 2>/dev/null | head -1)
if [ -z "$res" ]; then
    echo "$(date +%T) SESSION_FAILED $name (no result json)" >> "$STATUS"
    kill $MON 2>/dev/null; exit 1
fi
completed=$(python3 -c "import json;d=json.load(open('$res'));print(d['completed'],d['failed'],d['num_prompts'])" 2>/dev/null)
read -r c f n <<< "$completed"
if [ "${c:-0}" != "${n:-1}" ] || [ "${f:-1}" != "0" ]; then
    echo "$(date +%T) SESSION_FAILED $name (completed=$c failed=$f of $n — HANG/ERROR CONFIRMED)" >> "$STATUS"
    kill $MON 2>/dev/null; pkill -9 -f 'vllm serve.*Qwen3.6-35B-A3B-w8a8' 2>/dev/null; exit 1
fi

kill $MON 2>/dev/null
echo "$(date +%T) SESSION_OK $name (completed=$c failed=0)" >> "$STATUS"
exit 0

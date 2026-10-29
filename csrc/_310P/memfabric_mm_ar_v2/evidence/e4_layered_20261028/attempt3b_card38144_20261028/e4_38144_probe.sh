#!/usr/bin/env bash
# 38144逻辑/物理映射探针：ASCEND_RT_VISIBLE_DEVICES=4,5下双Rank最小设备验证。
# 输出设备数/属性并做小matmul；外部并行采样npu-smi核对进程实际落卡位置。
source /tmp/opencode/e4-launch-env-attempt3-20261028.sh
export ASCEND_RT_VISIBLE_DEVICES=4,5
export E4_PHYSICAL_CARD_ID=38144
OUT=/tmp/opencode/e4-38144-map-probe-20261028
rm -rf $OUT; mkdir -p $OUT
PERF=/home/w00804146/310P/e4wt/csrc/_310P/memfabric_mm_ar_v2/test/perf
cat > $OUT/probe.py <<'PY'
import os, torch, torch_npu
rank = int(os.environ.get("RANK", -1))
local = int(os.environ.get("LOCAL_RANK", -1))
assert int(os.environ["WORLD_SIZE"]) == 2
torch.npu.set_device(local)
cnt = torch.npu.device_count()
props = torch.npu.get_device_properties(local)
x = torch.randn(512, 512, dtype=torch.float16, device=f"npu:{local}")
y = x @ x
torch.npu.synchronize()
s = float(y.sum().cpu())
print(f"PROBE_OK rank={rank} local={local} device_count={cnt} "
      f"props={props.name if hasattr(props,'name') else props} sum={s:.3f}", flush=True)
PY
# 并行采样：探针运行期间每2秒记录所有卡的进程表
( for i in $(seq 1 30); do
    date -u -Is >> $OUT/smi_watch.log
    npu-smi info 2>/dev/null | awk '/Process id/{f=1;next} f&&/^\| [0-9]/{print}' >> $OUT/smi_watch.log
    sleep 2
  done ) &
WPID=$!
cd /home/w00804146/310P/e4wt
python3 "$PERF/e4_stage_guard.py" --card 38144 --log "$OUT/lease_guard.jsonl" --poll-sec 2 -- \
  torchrun --standalone --nproc-per-node=2 "$OUT/probe.py" > "$OUT/torchrun.log" 2>&1
RC=$?
kill $WPID 2>/dev/null
echo "PROBE_GUARD_RC=$RC"
grep PROBE_OK "$OUT/torchrun.log"
echo ===进程落卡位置===
python3 - "$OUT/smi_watch.log" <<'PY'
import re, sys
cards = {}
for line in open(sys.argv[1]):
    m = re.match(r"\| (\d+)\s+\d+\s+\|\s+(\d+)", line.strip())
    if m:
        cards.setdefault(int(m.group(1)), set()).add(int(m.group(2)))
print("cards_with_processes:", {k: sorted(v) for k, v in cards.items()})
PY
python3 -c "
import json
for l in open('$OUT/lease_guard.jsonl'):
    e=json.loads(l); print(e['event'], e.get('target_pids',''), str(e.get('reason',''))[:70])"
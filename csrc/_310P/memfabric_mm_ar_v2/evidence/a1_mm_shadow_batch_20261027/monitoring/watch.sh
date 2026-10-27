#!/usr/bin/env bash
# Read-only exclusivity monitor for NPU card 16384 (dev2/3). 30s interval.
OUT=/tmp/opencode/a1-mm-shadow-20261027/monitor
while true; do
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  npu-smi info > "$OUT/snapshots/$ts.txt" 2>&1
  # extract per-die memory + process table lines for card 16384
  python3 - "$OUT" "$ts" <<'PY' >> "$OUT/card16384_timeline.log"
import sys
out, ts = sys.argv[1], sys.argv[2]
text = open(f"{out}/snapshots/{ts}.txt").read()
lines = text.splitlines()
# find die rows for NPU 16384 and process rows
marks = []
proc = []
npu16384 = False
for i, l in enumerate(lines):
    if "NPU" in l and "Process id" in l:
        npu16384 = False
    if l.startswith("| 16384"):
        npu16384 = True
        marks.append(l)
    elif npu16384 and l.startswith("|") and "310P3" not in l and "Process id" not in l:
        marks.append(l)
# process table section
in_proc = False
cur_npu = ""
for l in lines:
    if "Process id" in l and "Process name" in l:
        in_proc = True
        continue
    if in_proc and l.startswith("|"):
        parts = [p.strip() for p in l.split("|") if p.strip()]
        if parts and parts[0].isdigit() and len(parts[0]) == 5:
            cur_npu = parts[0]
        if parts and cur_npu == "16384":
            proc.append(l)
mem = [m for m in marks if "/" in m]
print(f"{ts} | dies={len([m for m in marks if m.startswith(chr(124)+' 16384')])} | procs_on_16384={len(proc)}")
for p in proc:
    print(f"  PROC {p}")
PY
  sleep 30
done

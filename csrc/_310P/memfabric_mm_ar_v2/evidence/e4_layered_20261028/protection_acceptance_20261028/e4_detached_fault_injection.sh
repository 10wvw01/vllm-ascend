#!/usr/bin/env bash
# e4_detached.py 真进程级故障注入：独立git仓 + 字节相同副本 + stub campaign。
set -uo pipefail
E4WT=/home/w00804146/310P/e4wt
PERF="$E4WT/csrc/_310P/memfabric_mm_ar_v2/test/perf"
STUB=/tmp/opencode/e4-detached-fi-20261028
rm -rf "$STUB"; mkdir -p "$STUB/csrc/_310P/memfabric_mm_ar_v2/test/perf"
cp "$PERF/e4_detached.py" "$STUB/csrc/_310P/memfabric_mm_ar_v2/test/perf/"
DET="$STUB/csrc/_310P/memfabric_mm_ar_v2/test/perf/e4_detached.py"
PASS=0; FAIL=0
check() { if [ "$2" == "$3" ]; then echo "PASS $1"; PASS=$((PASS+1));
          else echo "FAIL $1 got=$2 want=$3"; FAIL=$((FAIL+1)); fi }

# 独立git仓（manifest work_head绑定的repo_root = STUB）
git -C "$STUB" init -q && git -C "$STUB" add -A && git -C "$STUB" -c user.email=t@t -c user.name=t commit -qm stub
HEAD="$(git -C "$STUB" rev-parse HEAD)"

# stub campaign：$1=OUT $2=BUILD；按STUB_MODE行为
cat > "$STUB/csrc/_310P/memfabric_mm_ar_v2/test/perf/e4_campaign.sh" <<'EOF'
#!/usr/bin/env bash
OUT="$1"
mkdir -m 700 -p "$OUT"
case "${STUB_MODE:-}" in
  ok)   echo "E4_SCREEN_COMPLETE / CONFIRM_NOT_AUTHORIZED" >> "$OUT/progress.txt"; exit 0;;
  nomarker) exit 0;;
  fail17) exit 17;;
  *) exec sleep 600;;
esac
EOF
chmod +x "$STUB/csrc/_310P/memfabric_mm_ar_v2/test/perf/e4_campaign.sh"
git -C "$STUB" add -A && git -C "$STUB" -c user.email=t@t -c user.name=t commit -qm stub2
HEAD="$(git -C "$STUB" rev-parse HEAD)"

mkdir -p "$STUB/build" "$STUB/base"
printf '{"schema":"E4_LAYERED_ADD_BUILD_V1","work_head":"%s"}\n' "$HEAD" > "$STUB/build/manifest.json"
printf '{"schema":"E4_LAYERED_ADD_BUILD_V1","work_head":"0000000000000000000000000000000000000000"}\n' > "$STUB/build_stale/manifest.json" 2>/dev/null || { mkdir -p "$STUB/build_stale"; printf '{"schema":"E4_LAYERED_ADD_BUILD_V1","work_head":"0000"}\n' > "$STUB/build_stale/manifest.json"; }

export E4_PHYSICAL_CARD_ID=16384

# ---- 1) Confirm拒绝（真实CLI层） ----
E4_CONFIRM_APPROVED=YES python3 "$DET" start --state "$STUB/base/s1" --out "$STUB/base/o1" --build "$STUB/build" >/dev/null 2>&1
check "confirm-rejected-exit2" "$?" "2"
[ ! -e "$STUB/base/s1" ] && echo "PASS confirm-no-state-dir" && PASS=$((PASS+1)) || { echo "FAIL confirm-no-state-dir"; FAIL=$((FAIL+1)); }

# ---- 2) stale manifest拒绝 ----
python3 "$DET" start --state "$STUB/base/s2" --out "$STUB/base/o2" --build "$STUB/build_stale" >/dev/null 2>&1
check "stale-manifest-exit2" "$?" "2"

# ---- 3) 会话独立性：start返回后worker仍存活（stub campaign sleep 600） ----
python3 "$DET" start --state "$STUB/base/s3" --out "$STUB/base/o3" --build "$STUB/build" > "$STUB/base/s3_start.json" 2>&1
check "start-exit0" "$?" "0"
sleep 1.2
python3 "$DET" status --state "$STUB/base/s3" > "$STUB/base/s3_status.json" 2>&1
check "status-running-3" "$?" "3"
WPID="$(python3 -c "import json;print(json.load(open('$STUB/base/s3/launch.json'))['worker_pid'])")"
CPID="$(pgrep -P "$WPID" 2>/dev/null | head -1)"
kill -0 "$WPID" 2>/dev/null && echo "PASS worker-alive-after-start-returns" && PASS=$((PASS+1)) || { echo "FAIL worker-alive"; FAIL=$((FAIL+1)); }
[ -n "$CPID" ] && kill -0 "$CPID" 2>/dev/null && echo "PASS campaign-child-alive" && PASS=$((PASS+1)) || { echo "FAIL campaign-child-alive"; FAIL=$((FAIL+1)); }

# ---- 4) SIGTERM worker → campaign进程组被清理、result.json=STOP/143、日志保全 ----
kill -TERM "$WPID"
for i in $(seq 1 40); do [ -f "$STUB/base/s3/result.json" ] && break; sleep 0.2; done
python3 "$DET" status --state "$STUB/base/s3" > /dev/null 2>&1
check "sigterm-status-1" "$?" "1"
R="$(python3 -c "import json;d=json.load(open('$STUB/base/s3/result.json'));print(d['state'],d['exit_code'])")"
check "sigterm-result-stop-143" "$R" "STOP 143"
[ -f "$STUB/base/s3/campaign.log" ] && echo "PASS sigterm-campaign-log-preserved" && PASS=$((PASS+1)) || { echo "FAIL sigterm-campaign-log-preserved"; FAIL=$((FAIL+1)); }
sleep 0.5
if [ -n "$CPID" ] && kill -0 "$CPID" 2>/dev/null; then echo "FAIL sigterm-campaign-killed"; FAIL=$((FAIL+1)); else echo "PASS sigterm-campaign-killed"; PASS=$((PASS+1)); fi
if kill -0 "$WPID" 2>/dev/null; then echo "FAIL sigterm-worker-exited"; FAIL=$((FAIL+1)); else echo "PASS sigterm-worker-exited"; PASS=$((PASS+1)); fi

# ---- 5) stub ok（含marker）→ SCREEN_COMPLETE/0 ----
STUB_MODE=ok python3 "$DET" start --state "$STUB/base/s5" --out "$STUB/base/o5" --build "$STUB/build" >/dev/null 2>&1
check "ok-start-0" "$?" "0"
for i in $(seq 1 60); do [ -f "$STUB/base/s5/result.json" ] && break; sleep 0.2; done
python3 "$DET" status --state "$STUB/base/s5" >/dev/null 2>&1
check "ok-status-0" "$?" "0"
R="$(python3 -c "import json;d=json.load(open('$STUB/base/s5/result.json'));print(d['state'],d['exit_code'])")"
check "ok-result-screen-complete-0" "$R" "SCREEN_COMPLETE 0"

# ---- 6) stub exit0但无marker → STOP/96（fail-closed） ----
STUB_MODE=nomarker python3 "$DET" start --state "$STUB/base/s6" --out "$STUB/base/o6" --build "$STUB/build" >/dev/null 2>&1
for i in $(seq 1 60); do [ -f "$STUB/base/s6/result.json" ] && break; sleep 0.2; done
R="$(python3 -c "import json;d=json.load(open('$STUB/base/s6/result.json'));print(d['state'],d['exit_code'])")"
check "nomarker-result-stop-96" "$R" "STOP 96"

# ---- 7) stub exit17 → STOP/17 ----
STUB_MODE=fail17 python3 "$DET" start --state "$STUB/base/s7" --out "$STUB/base/o7" --build "$STUB/build" >/dev/null 2>&1
for i in $(seq 1 60); do [ -f "$STUB/base/s7/result.json" ] && break; sleep 0.2; done
R="$(python3 -c "import json;d=json.load(open('$STUB/base/s7/result.json'));print(d['state'],d['exit_code'])")"
check "fail17-result-stop-17" "$R" "STOP 17"

# ---- 8) 重复state拒绝（证据不覆盖） ----
python3 "$DET" start --state "$STUB/base/s5" --out "$STUB/base/o8" --build "$STUB/build" >/dev/null 2>&1
check "state-reuse-rejected-2" "$?" "2"

# ---- 9) 清理残余stub进程 ----
pkill -TERM -f "e4-detached-fi-20261028.*e4_campaign" 2>/dev/null || true

echo; echo "===== DETACHED FI SUMMARY: PASS=$PASS FAIL=$FAIL ====="
[ "$FAIL" == "0" ]

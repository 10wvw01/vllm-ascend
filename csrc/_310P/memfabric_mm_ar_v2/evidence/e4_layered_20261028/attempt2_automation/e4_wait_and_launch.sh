#!/usr/bin/env bash
# 16384空闲监视器：每60s采样；双Die无进程、hugepages<100页、无Alarm连续3次(3分钟)
# 判定稳定空闲，exec续跑脚本。全程日志 /tmp/opencode/e4_card_watch.log。
set -u
LOG=/tmp/opencode/e4_card_watch.log
free_count=0
echo "[$(date -Is)] watcher start (criteria: no proc + hp<100 both dies + no Alarm, 3 consecutive samples)" >> "$LOG"
while true; do
  ts=$(date -Is)
  out=$(npu-smi info 2>/dev/null)
  if [ -z "$out" ]; then
    echo "$ts npu-smi-empty (treated busy)" >> "$LOG"; free_count=0; sleep 60; continue
  fi
  proc=$(echo "$out" | grep -cE "^\| 16384    [01] +\|")
  hp2=$(echo "$out" | grep -m1 "^| 16384" | awk '{print $(NF-3)}')
  hp3=$(echo "$out" | grep -m2 "^| 16384" | tail -1 | awk '{print $(NF-3)}')
  alarm=$(echo "$out" | grep -m2 "^| 16384" | grep -c "Alarm")
  if [ "${proc:-1}" -eq 0 ] && [ "${hp2:-9999}" -le 100 ] && [ "${hp3:-9999}" -le 100 ] && [ "${alarm:-1}" -eq 0 ]; then
    free_count=$((free_count+1))
    echo "$ts FREE_CANDIDATE $free_count/3 (hp=$hp2/$hp3 proc=$proc)" >> "$LOG"
    if [ "$free_count" -ge 3 ]; then
      echo "$ts CARD_FREE_STABLE -> auto relaunch" >> "$LOG"
      exec bash /tmp/opencode/e4_relaunch.sh >> /tmp/opencode/e4_auto_launch.log 2>&1
    fi
  else
    [ "$free_count" -gt 0 ] && echo "$ts free-run broken (hp=$hp2/$hp3 proc=$proc alarm=$alarm)" >> "$LOG"
    free_count=0
    echo "$ts busy (hp=$hp2/$hp3 proc=$proc alarm=$alarm)" >> "$LOG"
  fi
  sleep 60
done

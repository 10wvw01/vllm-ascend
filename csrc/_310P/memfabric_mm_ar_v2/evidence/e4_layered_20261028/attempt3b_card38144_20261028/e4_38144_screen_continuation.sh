#!/usr/bin/env bash
# 手动续跑38144 campaign剩余5场Screen（与e4_campaign.sh的e4_stage调用完全一致）。
# 背景：campaign在screen_03 preflight被外部进程2884891正确STOP（stock/e1两场已无污染完成）；
# 本续跑以同一守护合同完成screen_03..07并按campaign原逻辑生成summary与收尾证据。
set -uo pipefail
source /tmp/opencode/e4-launch-env-38144-attempt3b.sh
OUT=/tmp/opencode/e4-campaign-38144-attempt3b
BUILD=/tmp/opencode/e4-build-attempt3b-20261028
PERF=/home/w00804146/310P/e4wt/csrc/_310P/memfabric_mm_ar_v2/test/perf
cd /home/w00804146/310P/e4wt

e4_stage() {
  python3 "$PERF/e4_stage_guard.py" \
    --card "$E4_PHYSICAL_CARD_ID" \
    --log "$OUT/lease_guard.jsonl" --poll-sec 2 -- "$@"
}

for spec in "3:e3g" "4:e4t2" "5:e4t4" "6:e4t8" "7:e4p"; do
  stage="${spec%%:*}"; arm="${spec#*:}"
  label="$(printf 'screen_%02d_%s' "$stage" "$arm")"
  echo "SCREEN_BEGIN $label (manual continuation)" | tee -a "$OUT/progress.txt"
  if ! e4_stage torchrun --standalone --nproc-per-node=2 "$PERF/e4_arm.py" \
       --phase screen --warmup 10 --iters 30 \
       --stage "$stage" --out "$OUT/sessions" \
       --build-manifest "$BUILD/manifest.json" \
       --store-url "tcp://127.0.0.1:$FIVE_ARM_STORE_PORT" \
       > "$OUT/sessions/$label.log" 2>&1; then
    echo "SCREEN_STOP $label (manual continuation; preserve evidence, no retry)" | tee -a "$OUT/progress.txt"
    exit 1
  fi
  ok=1
  for rank in 0 1; do
    [[ -s "$OUT/sessions/${label}_rank${rank}.json" ]] || ok=0
  done
  if [[ $ok != 1 ]]; then echo "missing rank JSON for $label; STOP" | tee -a "$OUT/progress.txt"; exit 1; fi
  echo "SCREEN_ARM_DONE $label" | tee -a "$OUT/progress.txt"
done

# 与campaign完全一致的summary计算
python3 - "$OUT/sessions" "$OUT/e4_screen_summary.json" <<'PY'
import json,sys,statistics,math,pathlib
p=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2])
arms=("stock","e1","e3g","e4t2","e4t4","e4t8","e4p")
res={}
for stage,arm in enumerate(arms,1):
    q=[json.loads((p/f"screen_{stage:02d}_{arm}_rank{r}.json").read_text()) for r in (0,1)]
    assert all(t["schema"]=="E4_LAYERED_TP2_V1" and t["arm"]==arm and
               t["stage"]==stage and t["phase"]=="screen" and
               t["rank"]==r for r,t in enumerate(q))
    res[arm]={}
    for m in ("2048","4096"):
        a,b=(t["results"][m] for t in q)
        assert all(t["bit_exact"] and t["protocol_status"]==(None if arm=="stock" else 0)
                   and t["warmup"]>=10 and t["iters"]==30 for t in (a,b))
        pair=[max(x,y) for x,y in zip(a["npu_event_us"],b["npu_event_us"])]
        assert len(pair)==30 and all(math.isfinite(v) and v>0 for v in pair)
        res[arm][m]=round(statistics.median(pair),3)
flag={arm:res[arm]["4096"]<res["e3g"]["4096"] for arm in arms[3:]}
out.write_text(json.dumps({"schema":"E4_SCREEN_ONLY_V1","rank_pair_p50_us":res,
    "improves_E3G_M4096_screen_only":flag,
    "status":"SCREEN_ONLY_NO_SIGNED_PERFORMANCE"},indent=2))
print("SUMMARY_OK")
PY
[[ $? == 0 ]] || exit 1
# campaign原收尾：二进制身份前后一致 + npu_after + 完成标记
sha256sum "$U1_DEVICE_SO_PATH" "$V2_HOST_LIB" "$BUILD/"*.so > "$OUT/binaries_after.sha256"
diff -u "$OUT/binaries_before.sha256" "$OUT/binaries_after.sha256" || { echo "binary identity changed; STOP"; exit 1; }
npu-smi info > "$OUT/npu_after.txt" 2>&1
echo "E4_SCREEN_COMPLETE / CONFIRM_NOT_AUTHORIZED (manual continuation on card 38144)" | tee -a "$OUT/progress.txt"
echo MANUAL_CONTINUATION_DONE

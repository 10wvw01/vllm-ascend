#!/usr/bin/env python3
"""E4保护器无NPU故障注入验收（真实进程级，非mock）。

用真实归档npu-smi快照+伪npu-smi驱动真实e4_stage_guard.py CLI，
验证：跨卡506263、第三进程、npu-smi失败、Alarm、preflight占用、
SIGTERM干预、正常完成、非零退出透传、只清理自己的进程组。
"""
import json, os, pathlib, signal, subprocess, sys, time

E4WT = pathlib.Path("/home/w00804146/310P/e4wt")
PERF = E4WT / "csrc/_310P/memfabric_mm_ar_v2/test/perf"
EV = E4WT / "csrc/_310P/memfabric_mm_ar_v2/evidence/e4_layered_20261028"
WORK = pathlib.Path("/tmp/opencode/e4-guard-fault-injection-20261028v2")
GUARD = str(PERF / "e4_stage_guard.py")
PASS = FAIL = 0
import shutil
shutil.rmtree(WORK / "runs", ignore_errors=True)
WORK.mkdir(parents=True, exist_ok=True)

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"PASS {name}")
    else:
        FAIL += 1; print(f"FAIL {name} {detail}")

HEAD = """+------------------------------------------------------------------+
| npu-smi 26.1.1                                   Version: 26.1.1 |
+-------------------------------+-----------------+---------------+
| NPU     Name                  | Health          | Power(W)      |
| Chip    Device                | Bus-Id          | AICore(%)     |
+===============================+=================+===============+
| 16384   310P3                 | {h0:<15} | NA            |
| 0       2                     | 0000:41:00.0    | 0             |
+-------------------------------+-----------------+---------------+
| 16384   310P3                 | {h1:<15} | NA            |
| 1       3                     | 0000:41:00.0    | 0             |
+===============================+=================+===============+
| NPU     Chip                  | Process id      | Process name  |
+===============================+=================+===============+
"""
def snap(procs=(), h0="OK", h1="OK"):
    rows = "".join(f"| {n:<7} {d:<19} | {p:<15} | {'':<14} | {m:<25}|\n"
                   for n, d, p, m in procs)
    return HEAD.format(h0=h0, h1=h1) + rows + \
        "+===============================+=================+===============+\n"

IDLE = snap()
TWORANK = snap([(16384, 0, 111, 419), (16384, 1, 112, 419)])
CROSS = snap([(16384, 0, 111, 419), (16384, 1, 112, 419),
              (5664, 0, 506263, 2959), (16384, 0, 506263, 2959)])
THIRD = snap([(16384, 0, 111, 419), (16384, 1, 112, 419), (16384, 1, 113, 419)])
ALARM = snap([(16384, 0, 111, 419), (16384, 1, 112, 419)], h0="Alarm")
BUSY = snap([(16384, 0, 777, 419)])

def fake_npu(dirname, seq):
    d = pathlib.Path(dirname) / "bin"
    d.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(seq):
        if s != "FAIL":
            (d / f"s{i}.txt").write_text(s)
    (d / "seq.conf").write_text(
        "".join(f"FAIL\n" if s == "FAIL" else f"{d}/s{i}.txt\n" for i, s in enumerate(seq)))
    (d / "npu-smi").write_text(
        "#!/usr/bin/env bash\n"
        'D="$(dirname "$0")"\n'
        'N="$(cat "$D/calls" 2>/dev/null || echo 0)"; N=$((N+1)); echo "$N" > "$D/calls"\n'
        'L="$(sed -n "${N}p" "$D/seq.conf")"\n'
        'if [[ -z "$L" ]]; then L="$(tail -n1 "$D/seq.conf")"; fi\n'  # 耗尽后重复最后快照
        'if [[ "$L" == FAIL ]]; then exit 1; fi\n'
        'cat "$L"\n')
    (d / "npu-smi").chmod(0o755)
    return str(d)

def run_guard(name, seq, cmd, poll=0.5, wait=15, env_extra=None):
    d = WORK / "runs" / name
    env = dict(os.environ)
    env["PATH"] = fake_npu(d, seq) + ":" + env["PATH"]
    if env_extra:
        env.update(env_extra)
    log = d / "lease_guard.jsonl"
    p = subprocess.run(["python3", GUARD, "--card", "16384", "--log", str(log),
                        "--poll-sec", str(poll), "--"] + cmd,
                       capture_output=True, text=True, timeout=wait, env=env)
    return d, p

def events(d):
    return [json.loads(l) for l in (d / "lease_guard.jsonl").read_text().splitlines()]

def alive(pid):
    try:
        os.kill(pid, 0); return True
    except (ProcessLookupError, TypeError):
        return False

sleeper = WORK / "sleeper.sh"
sleeper.write_text("#!/usr/bin/env bash\necho $$ > $0.pid\nexec sleep 600\n")
sleeper.chmod(0o755)

# ---- 1) 真实归档快照解析 ----
sys.path.insert(0, str(PERF))
import e4_stage_guard as G
raw_idle = (EV / "g0/npu_smi_g0.txt").read_text()
s = G.parse_info(raw_idle, 16384)
check("real-g0-idle-parse", s["target"] == set())
raw_stop = (EV / "stop_incident/npu_stop_0701.txt").read_text()
s2 = G.parse_info(raw_stop, 16384)
check("real-stop-snapshot-has-506263", 506263 in s2["target"])
try:
    G.assess(s2, stage_running=True)
    check("real-cross-card-detected", False)
except G.LeaseLost as e:
    check("real-cross-card-detected", "cross-card PID" in str(e) and "506263" in str(e))

# ---- 2) 运行中跨卡506263 → STOP 97 + 清理子进程组 + 保全快照 ----
d, p = run_guard("cross_stop", [IDLE, TWORANK, CROSS, CROSS], ["bash", str(sleeper)])
check("cross_stop-exit97", p.returncode == 97, f"rc={p.returncode}")
check("cross_stop-stop-event", events(d)[-1]["event"] == "STOP")
check("cross_stop-reason", "cross-card PID" in events(d)[-1]["reason"])
check("cross_stop-npu-snapshot-saved",
      "506263" in (d / "lease_guard_stop.npu.txt").read_text())
cpid = int((WORK / "sleeper.sh.pid").read_text())
time.sleep(0.4)
check("cross_stop-own-child-killed", not alive(cpid))

# ---- 3) 无关进程不被误杀（不同session的标记进程） ----
marker = subprocess.Popen(["bash", "-c", "exec sleep 600"], start_new_session=True,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
sleeper2 = WORK / "sleeper2.sh"
sleeper2.write_text("#!/usr/bin/env bash\necho $$ > $0.pid\nexec sleep 600\n")
sleeper2.chmod(0o755)
d, p = run_guard("kill_scope", [IDLE, TWORANK, CROSS, CROSS], ["bash", str(sleeper2)])
check("kill_scope-exit97", p.returncode == 97)
time.sleep(0.4)
check("kill_scope-marker-survives", alive(marker.popen_pid if hasattr(marker, 'popen_pid') else marker.pid))
marker.kill(); marker.wait()

# ---- 4) 第三个进程 → STOP ----
d, p = run_guard("third_pid", [IDLE, TWORANK, THIRD, THIRD], ["bash", str(sleeper)])
check("third_pid-exit97", p.returncode == 97)
check("third_pid-reason", "unexpected target process count" in events(d)[-1]["reason"])
cpid = int((WORK / "sleeper.sh.pid").read_text())
time.sleep(0.4)
check("third_pid-child-killed", not alive(cpid))

# ---- 5) npu-smi失败 → STOP ----
d, p = run_guard("npu_fail", [IDLE, TWORANK, "FAIL", "FAIL"], ["bash", str(sleeper)])
check("npu_fail-exit97", p.returncode == 97)
check("npu_fail-reason", "npu-smi unavailable" in events(d)[-1]["reason"])
cpid = int((WORK / "sleeper.sh.pid").read_text())
time.sleep(0.4)
check("npu_fail-child-killed", not alive(cpid))

# ---- 6) preflight占用 → STOP且不启动torchrun ----
d, p = run_guard("preflight_busy", [BUSY, BUSY], ["bash", str(sleeper)])
check("preflight_busy-exit97", p.returncode == 97)
check("preflight_busy-no-stage-start",
      not any(e["event"] == "STAGE_STARTED" for e in events(d)))
check("preflight_busy-reason", "must be idle" in events(d)[-1]["reason"])

# ---- 7) 目标卡Alarm → STOP ----
d, p = run_guard("alarm_card", [ALARM, ALARM], ["bash", str(sleeper)])
check("alarm_card-exit97", p.returncode == 97)
check("alarm_card-reason", "two OK dies" in events(d)[-1]["reason"])

# ---- 8) 正常完成：exit 0、事件链完整、无STOP ----
d, p = run_guard("normal", [IDLE, TWORANK, IDLE], ["bash", "-c", "echo done"])
check("normal-exit0", p.returncode == 0, f"rc={p.returncode}")
evs = [e["event"] for e in events(d)]
check("normal-event-chain",
      evs[:2] == ["PREFLIGHT_START", "PREFLIGHT_PASS"] and
      "STAGE_STARTED" in evs and evs[-1] == "STAGE_EXIT" and "STOP" not in evs,
      str(evs))
check("normal-post-stage-lease", "POST_STAGE_LEASE" in evs)

# ---- 9) 子命令非零退出原样透传 ----
d, p = run_guard("childfail", [IDLE, TWORANK, IDLE], ["bash", "-c", "exit 17"])
check("childfail-exit17", p.returncode == 17, f"rc={p.returncode}")

# ---- 10) SIGTERM干预守护器 → STOP 97 + 清理子进程组 ----
dd = WORK / "runs" / "sigterm"
env = dict(os.environ)
env["PATH"] = fake_npu(dd, [IDLE, TWORANK, TWORANK]) + ":" + env["PATH"]
log = dd / "lease_guard.jsonl"
proc = subprocess.Popen(["python3", GUARD, "--card", "16384", "--log", str(log),
                         "--poll-sec", "0.5", "--", "bash", str(sleeper)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
time.sleep(1.5)
cpid = int((WORK / "sleeper.sh.pid").read_text())
os.kill(proc.pid, signal.SIGTERM)
rc = proc.wait(timeout=15)
check("sigterm-exit97", rc == 97, f"rc={rc}")
evs = events(dd)
check("sigterm-reason", any("guard received signal" in e.get("reason", "") for e in evs))
time.sleep(0.4)
check("sigterm-child-killed", not alive(cpid))

print(f"\n===== FAULT INJECTION SUMMARY: PASS={PASS} FAIL={FAIL} =====")
sys.exit(0 if FAIL == 0 else 1)

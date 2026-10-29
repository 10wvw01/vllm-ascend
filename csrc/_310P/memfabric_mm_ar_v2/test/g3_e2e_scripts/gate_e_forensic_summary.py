#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""从单轮 Gate E 原始结果生成可复核摘要。缺失数据记 UNKNOWN，不作 PASS。"""
import glob
import json
import pathlib
import re
import sys

PATTERNS = {
    "aicore_timeout": r"aicore.*timeout|the aicore execution times out|retCode=0x25",
    "acl_errors": r"507014|507017|507018|507899|EH0012|ERR00100",
    "v2_route": r"V2 MM\+MemFabric",
    "v2_correctness": r"first fused call verified bit-exact",
    "epoch_error": r"epoch exited with error|epoch relaunch failed|supervisor exit",
    "memfabric_diag": r"\[MF_DIAG\]",
    "error_lines": r"\bERROR\b|\bError\b|Traceback",
}


def count_or_zero(text, pattern):
    return len(re.findall(pattern, text, flags=re.I))


def numeric_errors(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return len(value)
    return None


def main():
    if len(sys.argv) != 2:
        print("用法: gate_e_forensic_summary.py <单轮证据目录>", file=sys.stderr)
        return 2
    d = pathlib.Path(sys.argv[1]).resolve()
    if not d.is_dir():
        print(f"证据目录不存在: {d}", file=sys.stderr)
        return 2
    total = failed = 0
    unknown_errors = 0
    readable = 0
    corrupt = []
    files = sorted(glob.glob(str(d / "stability_rep*.json")))
    for name in files:
        try:
            obj = json.loads(pathlib.Path(name).read_text(encoding="utf-8"))
            completed = obj.get("completed")
            errors = numeric_errors(obj.get("errors"))
            if not isinstance(completed, int) or isinstance(completed, bool) or completed < 0:
                corrupt.append(f"{name}: completed 无法确认")
                continue
            readable += 1
            total += completed
            if errors is None:
                unknown_errors += 1
            else:
                failed += errors
        except (OSError, ValueError, TypeError) as exc:
            corrupt.append(f"{name}: {exc}")
    serve = (d / "serve.log").read_text(encoding="utf-8", errors="replace") if (d / "serve.log").exists() else ""
    bench = (d / "bench.log").read_text(encoding="utf-8", errors="replace") if (d / "bench.log").exists() else ""
    runner = (d / "runner.log").read_text(encoding="utf-8", errors="replace") if (d / "runner.log").exists() else ""
    code_file = d / "runner_exit_code.txt"
    rc = code_file.read_text().strip() if code_file.exists() else "UNKNOWN"
    print(f"证据目录: {d}")
    print(f"runner 退出码: {rc}")
    print(f"有效结果文件: {readable}/{len(files)}")
    print(f"已完成请求: {total}")
    print(f"明确错误请求: {failed}")
    print(f"错误数字段未知的结果文件: {unknown_errors}")
    print(f"无效结果文件: {len(corrupt)}")
    for line in corrupt[:20]:
        print(f"  INVALID: {line}")
    for name, regex in PATTERNS.items():
        print(f"{name}: serve={count_or_zero(serve, regex)} bench={count_or_zero(bench, regex)}")
    print(f"bench 致命字样: {count_or_zero(bench, r'failed|exception|timeout|error')}")
    print(f"runner 异常字样: {count_or_zero(runner, r'failed|timeout|error|exited early')}")
    bench_failed = count_or_zero(runner, r"bench rep [0-9]+ failed") > 0
    print(f"bench 子测试失败: {bench_failed}")
    print("注意：日志行数不是设备事件数，且不代表每一轮实际启用了 V2。")
    print("须人工核查 40 目标层启用、双 rank V2 实际执行、HTTP 负载及设备原始日志。")
    print("初筛:", end=" ")
    if readable == 0 or rc == "UNKNOWN" or corrupt or unknown_errors:
        print("UNKNOWN（结果不全或错误数不可判定）")
    elif rc != "0" or failed or bench_failed or count_or_zero(serve, PATTERNS["aicore_timeout"]):
        print("FAIL（需保留现场）")
    elif total == 0:
        print("UNKNOWN（没有完成请求）")
    else:
        print("NO_OBSERVED_FAILURE（仅本轮未见错误；非长稳根因闭环）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

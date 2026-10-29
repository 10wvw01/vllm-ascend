#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Gate E 离线判定器：只读分析，避免服务挂死但 runner 返回 0 的假通过。

退出码：0=本轮未观察到故障；1=明确失败；2=证据不足。
此判定不证明长稳性，G1-A/G1-B 不能合并计算发生率。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

FATAL_RE = re.compile(
    r"507014|507017|507018|the aicore execution times out|"
    r"retCode\s*=\s*0x25|rtEventSynchronize execution failed|"
    r"aclnn\w+ failed, error code is 507014",
    re.I,
)
RUNNER_FATAL_RE = re.compile(r"bench rep \d+ failed|serve exited early|health timeout", re.I)
REP_RE = re.compile(r"stability_rep([1-9][0-9]*)\.json\Z")


def error_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, list):
        return len(value)
    return None


def positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def text_file(root: pathlib.Path, name: str) -> str:
    p = root / name
    return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""


def judge(root: pathlib.Path, per_rep: int) -> dict:
    failures: list[str] = []
    unknowns: list[str] = []
    rounds: list[dict] = []
    completed_total = 0
    explicit_errors = 0
    candidates: dict[int, pathlib.Path] = {}

    for path in root.glob("stability_rep*.json"):
        match = REP_RE.fullmatch(path.name)
        if match is None:
            unknowns.append("不符合命名规则的结果文件: " + path.name)
            continue
        number = int(match.group(1))
        if number in candidates:
            unknowns.append("重复轮次: " + str(number))
        candidates[number] = path

    if not candidates:
        unknowns.append("没有有效的 stability_rep*.json")
    else:
        expected_nums = set(range(1, max(candidates) + 1))
        missing = sorted(expected_nums - set(candidates))
        if missing:
            unknowns.append("缺失轮次: " + ",".join(map(str, missing)))

    for number, path in sorted(candidates.items()):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("JSON 根节点不是对象")
            completed = record.get("completed")
            if not positive_int(completed):
                raise ValueError("completed 缺失或类型错误")
            errors = error_count(record.get("errors"))
            completed_total += completed
            if errors is not None:
                explicit_errors += errors

            rounds.append({
                "rep": number, "completed": completed,
                "expected": per_rep, "reported_errors": errors,
            })
            if completed < per_rep:
                failures.append(f"rep{number} 完成 {completed}/{per_rep}，未完成请求不能计为成功")
            elif completed > per_rep:
                unknowns.append(f"rep{number} 完成数 {completed} 超过预期 {per_rep}，参数需复核")
            if errors is None:
                unknowns.append(f"rep{number} errors 字段不可判读")
            elif errors > 0:
                failures.append(f"rep{number} 上报 {errors} 个请求错误")
        except (OSError, ValueError, TypeError) as exc:
            unknowns.append(f"rep{number} 结果损坏: {exc}")

    serve = text_file(root, "serve.log")
    runner = text_file(root, "runner.log")
    if not serve:
        unknowns.append("serve.log 不存在或为空")
    if not runner:
        unknowns.append("runner.log 不存在或为空")
    serve_hits = FATAL_RE.findall(serve)
    if serve_hits:
        failures.append(f"serve.log 出现 {len(serve_hits)} 处 AICore/ACL 致命错误特征")
    if RUNNER_FATAL_RE.search(runner):
        failures.append("runner.log 报告 benchmark/health 启动失败")

    rc_file = root / "runner_exit_code.txt"
    if not rc_file.is_file():
        unknowns.append("runner_exit_code.txt 缺失")
    else:
        rc = rc_file.read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9]+", rc):
            unknowns.append("runner_exit_code.txt 非法: " + repr(rc))
        elif int(rc) != 0:
            failures.append("runner 原始退出码非零: " + rc)

    if not failures and not unknowns:
        verdict = "NO_OBSERVED_FAILURE"
    elif failures:
        verdict = "FAIL"
    else:
        verdict = "UNKNOWN"

    return {
        "verdict": verdict, "expected_per_rep": per_rep,
        "rounds": rounds, "completed": completed_total,
        "reported_errors": explicit_errors,
        "failures": failures, "unknowns": unknowns,
        "note": "本轮无故障不代表根因解决；Fused 和 Stock 需分开报告。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--prompts-per-rep", type=int, default=25)
    parser.add_argument("--json-output", type=pathlib.Path, default=None)
    args = parser.parse_args()
    if args.prompts_per_rep < 1:
        parser.error("--prompts-per-rep 必须为正整数")
    if not args.run_dir.is_dir():
        parser.error("结果目录不存在")
    result = judge(args.run_dir, args.prompts_per_rep)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(output, end="")
    if args.json_output:
        args.json_output.write_text(output, encoding="utf-8")
    return {"NO_OBSERVED_FAILURE": 0, "FAIL": 1, "UNKNOWN": 2}[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())

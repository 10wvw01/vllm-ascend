#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""A1 Qwen input4096/output1 sequential Prefill requests for L2 profiler/L3 no-profiler.

Only talks to already started, isolated local vLLM HTTP port.
Does not launch/terminate other tenants' servers or auto-enable V2.
Outputs complete individual records; failure preserves previous samples.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

PROMPT = [64] * 4096


def api_post(port: int, endpoint: str, payload: dict | None = None) -> dict:
    url = f"http://127.0.0.1:{port}/{endpoint.lstrip('/')}"
    raw = json.dumps(payload or {}).encode("utf-8") if payload is not None else b""
    headers = {"Content-Type": "application/json"}
    req = Request(url, data=raw, headers=headers, method="POST")
    with urlopen(req, timeout=300) as response:
        data = response.read()
    return json.loads(data) if data else {}


def send(port: int, model: str, session: str, index: int, phase: str) -> dict:
    begin = time.perf_counter_ns()
    result = api_post(port, "v1/completions", {
        "model": model, "prompt": PROMPT, "max_tokens": 1,
        "ignore_eos": True, "temperature": 0, "stream": False,
    })
    end = time.perf_counter_ns()
    usage = result.get("usage") or {}
    if usage.get("prompt_tokens") != 4096 or usage.get("completion_tokens") != 1:
        raise RuntimeError(f"wrong model actual token counts: {usage}")
    if result.get("error") or not isinstance(result.get("choices"), list) or len(result["choices"]) != 1:
        raise RuntimeError("missing completion or server returned error")
    return {"session": session, "phase": phase, "index": index, "id": result.get("id"),
            "prompt_tokens": 4096, "completion_tokens": 1,
            "begin_ns": begin, "end_ns": end, "client_latency_ms": round((end - begin) / 1e6, 5),
            "warning": "Client latency is NOT L3 device Prefill step time"}


def run(port: int, model: str, session: str, out: Path, warmup: int,
        measured: int, profile: bool) -> None:
    if out.exists() or not out.is_absolute() or not out.parent.is_dir():
        raise ValueError("output must be a fresh absolute path in existing directory")
    if profile and (warmup < 6 or measured < 5):
        raise ValueError("profiler window must have >=6 warmup, >=5 measured")
    if not profile and (warmup < 6 or measured < 10):
        raise ValueError("unprofiled window must have >=6 warmup, >=10 measured")
    for n in range(warmup):
        record = send(port, model, session, n, "warmup")
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    enabled = False
    try:
        if profile:
            api_post(port, "start_profile")
            enabled = True
        for n in range(measured):
            record = send(port, model, session, n, "measured")
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
    finally:
        if enabled:
            api_post(port, "stop_profile")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--measured", type=int, default=30)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    if a.port < 1024 or a.port > 65535:
        p.error("port must be an unprivileged local TCP port")
    run(a.port, a.model, a.session, a.out, a.warmup, a.measured, a.profile)


if __name__ == "__main__":
    main()

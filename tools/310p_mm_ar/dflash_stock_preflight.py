#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stock+DFlash environment audit; no model load, package install or NPU work by default."""
from __future__ import annotations

import argparse
import ast
import importlib.metadata as metadata
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys

PINNED = {
    "vllm": "0.24.0",
    "torch": "2.10.0",
    "torch-npu": "2.10.0.post2",
    "transformers": "5.13.0",
}
# Observed, not all mandatory to initialize Stock. Upstream requirements.txt
# lists these but the 310P Dockerfile explicitly UNINSTALLS triton-ascend.
OBSERVE = (
    "vllm-ascend", "arctic-inference", "numba", "regex",
    "compressed-tensors", "memfabric-hybrid", "memcache-hybrid", "triton-ascend",
)
API_FILES = {
    "vllm": (
        "model_executor/models/qwen3_dflash.py",
        "v1/spec_decode/dflash.py",
        "v1/worker/cp_utils.py",
    ),
    "vllm_ascend": (
        "_310p/model_runner_310p.py",
        "_310p/spec_decode/dflash_proposer_310.py",
        "patch/worker/patch_idex_310.py",
    ),
}
STOCK_DISABLED_FLAGS = (
    "VLLM_ASCEND_310P_ENABLE_MEMFABRIC_MM_AR",
    "VLLM_ASCEND_310P_MEMFABRIC_MM_AR_V2_SERVE",
)


def active(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def verdict(versions: dict[str, str | None], files: dict[str, bool],
            flags: dict[str, str | None], model_ready: dict[str, bool]) -> dict:
    blockers, observations = [], []
    for name, wanted in PINNED.items():
        found = versions.get(name)
        if found is None:
            blockers.append("MISSING_DISTRIBUTION:" + name)
        elif found != wanted and not found.startswith(wanted + "+"):
            blockers.append("PIN_MISMATCH:" + name + ":" + found + ":expected=" + wanted)
    for name, present in files.items():
        if not present:
            blockers.append("MISSING_SOURCE_API:" + name)
    for flag in STOCK_DISABLED_FLAGS:
        if active(flags.get(flag)):
            blockers.append("STOCK_CONTAMINATED_BY_MEMFABRIC:" + flag)
    for name, ready in model_ready.items():
        if not ready:
            blockers.append("MODEL_CONFIG_MISSING:" + name)
    for name in OBSERVE:
        if versions.get(name) is None:
            observations.append("UPSTREAM_DECLARED_OR_OPTIONAL_NOT_INSTALLED:" + name)
    return {"static_go": not blockers, "blockers": blockers, "observations": observations}


def package_root(name: str) -> tuple[str | None, Path | None]:
    try:
        spec = importlib.util.find_spec(name)
        if spec is None:
            return None, None
        places = list(spec.submodule_search_locations or [])
        if places:
            return str(places[0]), Path(places[0])
        if spec.origin:
            return str(spec.origin), Path(spec.origin).parent
    except (ImportError, AttributeError, ValueError) as exc:
        return "FIND_SPEC_ERROR: " + repr(exc), None
    return None, None


def distribution_versions() -> dict[str, str | None]:
    versions = {}
    for name in (*PINNED, *OBSERVE):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def check_api(package_roots: dict[str, Path | None]) -> dict[str, bool]:
    result = {}
    for package, rels in API_FILES.items():
        root = package_roots.get(package)
        for rel in rels:
            result[package + "/" + rel] = bool(root and (root / rel).is_file())
    root = package_roots.get("vllm")
    cp = root / "v1/worker/cp_utils.py" if root else None
    if cp and cp.is_file():
        try:
            tree = ast.parse(cp.read_text(encoding="utf-8"))
            result["vllm/cp_utils.get_total_cp_world_size"] = any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "get_total_cp_world_size" for node in tree.body
            )
        except (OSError, SyntaxError, UnicodeError):
            result["vllm/cp_utils.get_total_cp_world_size"] = False
    else:
        result["vllm/cp_utils.get_total_cp_world_size"] = False
    return result


def classify_pip_check(exit_code: int | None, output: str) -> str:
    """Allow only the exact 310P Docker intentional missing Triton packages.

    Dockerfile.310p installs requirements, then uninstalls triton-ascend/triton.
    A different broken dependency or Triton version conflict is NOT waived.
    """
    if exit_code == 0:
        return "PASS"
    if exit_code is None:
        return "UNREADABLE"
    lines = [x.strip().lower() for x in output.splitlines() if x.strip()]
    if lines and all(
        "which is not installed" in line
        and re.search(r"\brequires triton(?:-ascend)?(?:[ \t,;>=<!~]|$)", line)
        for line in lines
    ):
        return "UPSTREAM_310P_INTENTIONAL_TRITON_MISSING_ONLY"
    return "UNEXPECTED_DEPENDENCY_BREAKAGE"


def optional_probes(enable_imports: bool) -> dict:
    cmd = [sys.executable, "-m", "pip", "check"]
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=35, check=False)
        output = {"exit_code": p.returncode, "output": (p.stdout + p.stderr)[-8000:]}
        output["classification"] = classify_pip_check(output["exit_code"], output["output"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        output = {"error": repr(exc), "classification": "UNREADABLE"}
    result = {"pip_check": output}
    if enable_imports:
        # Runs in a CHILD process only. This may load torch-npu native libraries;
        # do not enable until the server's read-only G0 environment check is safe.
        imports = "\n".join([
            "from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM",
            "from vllm.v1.spec_decode.dflash import DFlashProposer",
            "import vllm_ascend._310p.model_runner_310p",
            "import vllm_ascend.patch.worker.patch_idex_310",
        ])
        try:
            p = subprocess.run([sys.executable, "-c", imports], capture_output=True,
                               text=True, timeout=90, check=False)
            result["isolated_import_probe"] = {
                "exit_code": p.returncode, "traceback_tail": (p.stderr + p.stdout)[-12000:]
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["isolated_import_probe"] = {"error": repr(exc)}
    else:
        result["isolated_import_probe"] = {"status": "NOT_RUN"}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="/home/models/Qwen3.6-35B-A3B-w8a8")
    parser.add_argument("--draft", default="/home/models/Qwen3.6-35B-A3B-DFlash")
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--import-smoke", action="store_true",
                        help="opt-in child-process imports; only after G0")
    parser.add_argument("--out", type=Path, help="new output JSON file (never overwrite)")
    args = parser.parse_args()
    if args.num_speculative_tokens != 7:
        parser.error("current project Stock+DFlash baseline freezes num_speculative_tokens=7")
    modules = {}
    roots = {}
    for name in API_FILES:
        origin, root = package_root(name)
        modules[name] = origin
        roots[name] = root
    versions = distribution_versions()
    files = check_api(roots)
    flags = {name: os.environ.get(name) for name in STOCK_DISABLED_FLAGS}
    models = {"target": (Path(args.target) / "config.json").is_file(),
              "draft": (Path(args.draft) / "config.json").is_file()}
    decision = verdict(versions, files, flags, models)
    probes = optional_probes(args.import_smoke)
    result = {
        "mode": "STOCK_DFLASH_G0_READ_ONLY",
        "num_speculative_tokens": args.num_speculative_tokens,
        "logical_verify_width_example": args.num_speculative_tokens + 1,
        "note": "logical width is not measured physical/padded M",
        "python": sys.executable, "python_version": sys.version.split()[0],
        "cwd": os.getcwd(), "pythonpath": os.environ.get("PYTHONPATH"),
        "sys_path": sys.path, "package_roots": modules, "versions": versions,
        "api_files": files, "memfabric_flags": flags, "model_config_present": models,
        "verdict": decision, "probes": probes,
        "limitations": "No vllm serve, CANN/NPU kernel, Graph, or actual model inference executed",
    }
    # The upstream 310P Dockerfile explicitly removes Triton despite declaring
    # it in packaging; do not reject that single documented inconsistency.
    pip_state = probes.get("pip_check", {}).get("classification")
    if pip_state not in ("PASS", "UPSTREAM_310P_INTENTIONAL_TRITON_MISSING_ONLY"):
        decision["static_go"] = False
        decision["blockers"].append("PIP_CHECK_UNEXPECTED_FAILURE_OR_UNREADABLE")
    elif pip_state != "PASS":
        decision["observations"].append("UPSTREAM_DOCKER_INTENTIONALLY_UNINSTALLED_TRITON")
    smoke = probes.get("isolated_import_probe", {})
    if args.import_smoke and smoke.get("exit_code", 1) != 0:
        decision["static_go"] = False
        decision["blockers"].append("ISOLATED_IMPORT_FAILED_OR_TIMED_OUT")
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)
    return 0 if decision["static_go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

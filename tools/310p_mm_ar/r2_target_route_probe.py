#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only, sanitized route evidence for ONE frozen 310P3 Qwen3.6+DFlash scenario.

Does not import vLLM, torch, torch-npu, DFlash code, model code or shared objects.
Does not open weights or model implementation files. Does not run inference.
Output is a fixed set of enums/booleans; no raw model config, paths, revisions,
tokens, secret keys or traceback text are ever printed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONTRACT = {
    "target_id": "Eco-Tech/Qwen3.6-35B-A3B-w8a8",
    "device": "310P3_DUAL_DIE",
    "tp": 2,
    "input_tokens": 4096,
    "output_tokens": 2048,
    "batch": 10,
    "speculative_method": "dflash",
    "k": 7,
}
MAX_CONFIG_BYTES = 1024 * 1024
ALLOWED_LAYER_TYPES = frozenset(
    ("sliding_attention", "full_attention", "linear_attention")
)


def read_model_config(path: str | None) -> tuple[dict | None, str]:
    """Only read the specified config.json, and only as a bounded JSON object.

    Status strings are closed enums; neither the path nor the exception is
    included even for invalid JSON or denied access.
    """
    if not path:
        return None, "NOT_PROVIDED"
    candidate = Path(path)
    if candidate.name != "config.json":
        return None, "NOT_CONFIG_JSON"
    try:
        if not candidate.is_file():
            return None, "UNREADABLE"
        if candidate.stat().st_size > MAX_CONFIG_BYTES:
            return None, "TOO_LARGE"
        data = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None, "INVALID_FORMAT"
        return data, "READ"
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError):
        return None, "UNREADABLE_OR_INVALID"


def read_public_source(root: Path, path: str) -> str:
    """Inspect project/framework source text only, with no imports or eval."""
    try:
        file = root / path
        if not file.is_file() or file.stat().st_size > 1024 * 1024:
            return ""
        return file.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def draft_layout(config: dict | None) -> str:
    if config is None:
        return "UNVERIFIED"
    levels = [config, config.get("text_config", {})]
    layers = next((obj.get("layer_types") for obj in levels
                   if isinstance(obj, dict)
                   and isinstance(obj.get("layer_types"), list)), None)
    if not layers or len(layers) > 256 or not all(
        isinstance(v, str) for v in layers
    ):
        return "UNDETERMINED"
    types = set(layers)
    if not types.issubset(ALLOWED_LAYER_TYPES):
        return "OTHER_OR_PRIVATE_LAYOUT"
    if types == {"full_attention"}:
        return "ALL_FULL"
    if "sliding_attention" in types and "full_attention" in types:
        return "MIXED_SLIDING_FULL"
    return "OTHER_STANDARD_LAYOUT"


def target_class(config: dict | None) -> str:
    if config is None:
        return "UNVERIFIED"
    text_config = config.get("text_config", {})
    arch = config.get("architectures", [])
    if not isinstance(text_config, dict) or not isinstance(arch, list):
        return "UNDETERMINED"
    if (config.get("model_type") == "qwen3_5_moe"
            and text_config.get("model_type") == "qwen3_5_moe_text"
            and "Qwen3_5MoeForConditionalGeneration" in arch):
        return "PUBLIC_QWEN35_MOE_ARCH_COMPATIBLE"
    return "NOT_VERIFIED_AS_PUBLIC_QWEN35_MOE"


def inspect_routes(repo_root: Path, vllm_root: Path | None) -> dict[str, str]:
    patch = read_public_source(
        repo_root, "vllm_ascend/patch/worker/patch_idex_310.py")
    proposer = read_public_source(
        repo_root, "vllm_ascend/_310p/spec_decode/dflash_proposer_310.py")
    v2 = read_public_source(
        repo_root, "vllm_ascend/_310p/v2/dflash_speculator.py")
    worker = read_public_source(
        repo_root, "vllm_ascend/_310p/worker_310p.py")
    gpu = (read_public_source(
        vllm_root, "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py")
        if vllm_root else "")
    ms = (read_public_source(
        vllm_root, "vllm/v1/worker/gpu/model_states/__init__.py")
        if vllm_root else "")
    hybrid = (read_public_source(
        vllm_root, "vllm/v1/worker/gpu/model_states/mamba_hybrid.py")
        if vllm_root else "")
    return {
        "legacy_310p_dflash_input": (
            "ASCENDC_SOURCE_BINDING_PRESENT"
            if ("AscendDflashProposer.set_inputs_first_pass" in patch
                and "AscendDflashProposer310.set_inputs_first_pass" in patch
                and "npu_copy_and_expand_dflash_inputs" in proposer)
            else "UNVERIFIED"),
        "v2_candidate_dflash_input": (
            "SCOPED_NON_TRITON_CANDIDATE_PRESENT"
            if ("inspect.unwrap(DFlashSpeculator.propose)" in v2
                and '"prepare_dflash_inputs": prepare_dflash_inputs' in v2)
            else "UNVERIFIED"),
        "frozen_gpu_v2_dflash_input": (
            "DIRECT_TRITON_SOURCE_PATH_PRESENT"
            if ("def propose(" in gpu and "prepare_dflash_inputs(" in gpu
                and "_prepare_dflash_inputs_kernel[" in gpu)
            else "UNVERIFIED"),
        "v2_worker": (
            "NOT_IMPLEMENTED_GUARD_PRESENT"
            if "Triton-free V2 NPUWorker/ModelRunner is NOT_IMPLEMENTED"
            in worker else "UNVERIFIED"),
        "gpu_hybrid_model_state": (
            "CONDITIONAL_FACTORY_PATH_PRESENT"
            if ("is_hybrid" in ms and "MambaHybridModelState" in ms
                and "preprocess_mamba_align_fused_kernel[" in hybrid
                and "_scatter_num_accepted_kernel[" in hybrid)
            else "UNVERIFIED"),
        "effective_runtime_runner": "NOT_OBSERVED",
        "effective_model_state": "NOT_OBSERVED",
        "actual_dflash_triton_launch": "NOT_OBSERVED",
        "actual_device_kernel": "NOT_RUN",
    }


def build_report(target: dict | None, target_status: str,
                 draft: dict | None, draft_status: str,
                 repo_root: Path, vllm_root: Path | None,
                 cache_mode: str) -> dict:
    return {
        "schema_version": 1,
        "purpose": "STATIC_METADATA_AND_SOURCE_ONLY",
        "contract": CONTRACT,
        "inputs": {
            "target_config": target_status,
            "draft_config": draft_status,
            "frozen_vllm_source": (
                "REQUESTED" if vllm_root is not None else "NOT_PROVIDED"),
        },
        "target": {
            "architecture_compatibility": target_class(target),
            "local_artifact_identity": "UNVERIFIED",
        },
        "draft": {
            "attention_layout": draft_layout(draft),
            "private_architecture_and_weights": "NOT_COLLECTED",
        },
        "declared_flags": {
            "mamba_cache_mode": cache_mode.upper(),
            "actual_process_flags": "NOT_OBSERVED",
        },
        "routes": inspect_routes(repo_root, vllm_root),
        "disclosure": "ALLOWLIST_ENUMS_ONLY_NO_PRIVATE_CONTENT",
        "evidence_level": "STATIC_ONLY_NOT_RUNTIME_OR_ONDEVICE_GO",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-config",
                        help="Local target config.json (read-only, never echoed)")
    parser.add_argument("--draft-config",
                        help="Local private draft config.json (read-only, never echoed)")
    parser.add_argument("--vllm-root",
                        help="Frozen upstream vLLM checkout (source reads only)")
    parser.add_argument("--repo-root", help=argparse.SUPPRESS)
    parser.add_argument("--mamba-cache-mode", choices=("align", "none", "all"),
                        default="align",
                        help="Declared R2 plan setting; not observed process")
    args = parser.parse_args(argv)
    target, target_status = read_model_config(args.target_config)
    draft, draft_status = read_model_config(args.draft_config)
    repo_root = (Path(args.repo_root) if args.repo_root
                 else Path(__file__).resolve().parents[2])
    report = build_report(
        target, target_status, draft, draft_status,
        repo_root, Path(args.vllm_root) if args.vllm_root else None,
        args.mamba_cache_mode)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    # A read-only report is not an admission gate; always exit zero on
    # incomplete evidence to allow offline/partial reports without faking GO.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

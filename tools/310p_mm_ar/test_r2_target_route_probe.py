"""Pure Python/NDA-safe R2 Target+DFlash route probe regression tests."""
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATH = ROOT / "tools/310p_mm_ar/r2_target_route_probe.py"
spec = importlib.util.spec_from_file_location("r2_target_route_probe", PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class TestR2TargetRouteProbe(unittest.TestCase):
    def test_target_recognizes_public_qwen_architecture_only(self):
        valid = {"model_type": "qwen3_5_moe",
                 "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                 "text_config": {"model_type": "qwen3_5_moe_text",
                                 "private": "SHOULD_NOT_SHOW"}}
        self.assertEqual(probe.target_class(valid),
                         "PUBLIC_QWEN35_MOE_ARCH_COMPATIBLE")
        self.assertEqual(probe.target_class({"model_type": "secret"}),
                         "NOT_VERIFIED_AS_PUBLIC_QWEN35_MOE")

    def test_private_draft_only_returns_layout_class(self):
        hidden = {"layer_types": ["sliding_attention"] * 5 + ["full_attention"],
                  "secret_weights": "NO_PUBLICATION_EXAMPLE"}
        self.assertEqual(probe.draft_layout(hidden), "MIXED_SLIDING_FULL")
        self.assertEqual(probe.draft_layout({"layer_types": ["private_op"]}),
                         "OTHER_OR_PRIVATE_LAYOUT")
        self.assertEqual(probe.draft_layout({"layer_types": []}), "UNDETERMINED")

    def test_bounded_config_read_fails_closed_without_path_echo(self):
        with tempfile.TemporaryDirectory(prefix="PRIVATE_NEVER_ECHO_") as temp:
            path = pathlib.Path(temp) / "config.json"
            path.write_text("{not valid!}", encoding="utf-8")
            config, status = probe.read_model_config(str(path))
            self.assertIsNone(config)
            self.assertEqual(status, "UNREADABLE_OR_INVALID")
            path.write_bytes(b"x" * (probe.MAX_CONFIG_BYTES + 1))
            self.assertEqual(probe.read_model_config(str(path))[1], "TOO_LARGE")
            self.assertEqual(probe.read_model_config(
                str(path.with_name("private.json")))[1], "NOT_CONFIG_JSON")

    def test_end_to_end_never_emits_private_strings_paths_or_configs(self):
        with tempfile.TemporaryDirectory(prefix="SECRET_DIR_") as temp:
            root = pathlib.Path(temp)
            target_path = root / "target" / "config.json"
            draft_path = root / "draft" / "config.json"
            target_path.parent.mkdir()
            draft_path.parent.mkdir()
            target_path.write_text(json.dumps({
                "model_type": "qwen3_5_moe",
                "text_config": {"model_type": "qwen3_5_moe_text",
                                "danger": "PRIVATE_TARGET_MARKER_NEVER_ECHO"},
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "private_id": "PRIVATE_TARGET_MARKER_NEVER_ECHO"
            }), encoding="utf-8")
            draft_path.write_text(json.dumps({
                "layer_types": ["sliding_attention", "full_attention"],
                "internal_details": "SUPER_SECRET_DRAFT_NEVER_ECHO",
                "architecture": "UNPUBLISHED_SECRET_NAME"
            }), encoding="utf-8")
            # Source fixture uses only public-symbol marker strings.
            patch = root / "vllm_ascend/patch/worker/patch_idex_310.py"
            proposer = (root /
                "vllm_ascend/_310p/spec_decode/dflash_proposer_310.py")
            v2 = root / "vllm_ascend/_310p/v2/dflash_speculator.py"
            worker = root / "vllm_ascend/_310p/worker_310p.py"
            gpu = root / "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
            ms = root / "vllm/v1/worker/gpu/model_states/__init__.py"
            hybrid = root / "vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
            for file, text in (
                (patch, "AscendDflashProposer.set_inputs_first_pass = "
                        "AscendDflashProposer310.set_inputs_first_pass"),
                (proposer, "npu_copy_and_expand_dflash_inputs"),
                (v2, 'inspect.unwrap(DFlashSpeculator.propose)\n'
                     '"prepare_dflash_inputs": prepare_dflash_inputs'),
                (worker, "Triton-free V2 NPUWorker/ModelRunner is NOT_IMPLEMENTED"),
                (gpu, "def propose(self): prepare_dflash_inputs(...)\n"
                      "_prepare_dflash_inputs_kernel[(1,1)]"),
                (ms, "if is_hybrid: return MambaHybridModelState"),
                (hybrid, "preprocess_mamba_align_fused_kernel[grid]\n"
                         "_scatter_num_accepted_kernel[grid]"),
            ):
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text(text, encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(probe.main([
                    "--target-config", str(target_path),
                    "--draft-config", str(draft_path),
                    "--repo-root", str(root),
                    "--vllm-root", str(root),
                    "--mamba-cache-mode", "align",
                ]), 0)
            raw = output.getvalue()
            for secret in ("PRIVATE_TARGET_MARKER_NEVER_ECHO",
                           "SUPER_SECRET_DRAFT_NEVER_ECHO",
                           "UNPUBLISHED_SECRET_NAME", str(root),
                           "SECRET_DIR_", "config.json"):
                self.assertNotIn(secret, raw)
            obj = json.loads(raw)
            self.assertEqual(obj["draft"]["attention_layout"],
                             "MIXED_SLIDING_FULL")
            self.assertEqual(obj["target"]["architecture_compatibility"],
                             "PUBLIC_QWEN35_MOE_ARCH_COMPATIBLE")
            self.assertEqual(obj["routes"]["frozen_gpu_v2_dflash_input"],
                             "DIRECT_TRITON_SOURCE_PATH_PRESENT")
            self.assertEqual(obj["routes"]["legacy_310p_dflash_input"],
                             "ASCENDC_SOURCE_BINDING_PRESENT")
            self.assertEqual(obj["routes"]["effective_runtime_runner"],
                             "NOT_OBSERVED")
            self.assertEqual(obj["routes"]["actual_device_kernel"], "NOT_RUN")

    def test_real_checkout_reports_known_patch_without_npu_import(self):
        evidence = probe.inspect_routes(ROOT, None)
        self.assertEqual(evidence["legacy_310p_dflash_input"],
                         "ASCENDC_SOURCE_BINDING_PRESENT")
        self.assertEqual(evidence["v2_candidate_dflash_input"],
                         "SCOPED_NON_TRITON_CANDIDATE_PRESENT")
        self.assertEqual(evidence["v2_worker"],
                         "NOT_IMPLEMENTED_GUARD_PRESENT")
        self.assertEqual(evidence["frozen_gpu_v2_dflash_input"], "UNVERIFIED")


if __name__ == "__main__":
    unittest.main()

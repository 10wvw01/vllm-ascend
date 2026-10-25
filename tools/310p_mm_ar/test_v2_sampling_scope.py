"""No-Torch contract: 310P V2 greedy-only input guard is fail-closed."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / "vllm_ascend/_310p/v2/sampling_scope.py"
spec = importlib.util.spec_from_file_location("_r2_sampling_scope", P)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def params(**updates):
    values = dict(
        temperature=0.0, top_p=1.0, min_p=0.0, top_k=0,
        presence_penalty=0.0, frequency_penalty=0.0,
        repetition_penalty=1.0, n=1, min_tokens=0,
        max_tokens=2048, logprobs=None, prompt_logprobs=None,
        logprob_token_ids=None, logit_bias=None,
        allowed_token_ids=None, bad_words=None,
        structured_outputs=None, stop=None, stop_token_ids=None,
    )
    values.update(updates)
    return SimpleNamespace(**values)


class GreedyContract(unittest.TestCase):
    def test_explicit_frozen_greedy(self):
        module.validate_greedy_310p_sampling_params(params(ignore_eos=True))
        module.validate_greedy_310p_sampling_params(params(top_k=-1, max_tokens=1))

    def test_no_silent_conversion_of_default_temperature_one(self):
        for changes in [
            {"temperature": 1.0}, {"temperature": 0.7},
            {"top_p": 0.9}, {"min_p": 0.1}, {"top_k": 50},
            {"presence_penalty": 0.3}, {"frequency_penalty": 0.2},
            {"repetition_penalty": 1.1}, {"n": 2}, {"min_tokens": 2},
            {"logprobs": 1}, {"prompt_logprobs": 1},
            {"logit_bias": {1: 10}}, {"bad_words": [[3]]},
            {"stop": ["stop"]}, {"stop_token_ids": [100]},
            {"flat_logprobs": True},
        ]:
            with self.subTest(changes=changes):
                with self.assertRaises(NotImplementedError):
                    module.validate_greedy_310p_sampling_params(params(**changes))

    def test_missing_or_invalid_fails_closed(self):
        with self.assertRaises(NotImplementedError):
            module.validate_greedy_310p_sampling_params(None)
        with self.assertRaises(ValueError):
            module.validate_greedy_310p_sampling_params(params(max_tokens=0))

    def test_validation_is_before_any_inherited_request_mutation(self):
        src=(P.parent / "model_runner.py").read_text()
        method=src[src.index("    def add_requests(self, scheduler_output):"):
                   src.index("    def prepare_inputs(self, scheduler_output, batch_desc):")]
        self.assertLess(method.index("validate_greedy_310p_sampling_params"),
                        method.index("return super().add_requests"))
        self.assertIn("has_structured_output_requests", method)


if __name__ == "__main__":
    unittest.main()

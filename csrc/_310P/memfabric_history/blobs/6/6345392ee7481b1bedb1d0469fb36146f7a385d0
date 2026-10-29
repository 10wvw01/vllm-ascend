"""Fail-closed greedy-only sampling subset for 310P3 V2 Eager.

Never silently change sampling parameters. 310P3 stochastic rejection,
grammar/logprobs, penalties and top-p must receive genuine non-Triton
backends before they can be enabled.
"""


def validate_greedy_310p_sampling_params(params):
    if params is None:
        raise NotImplementedError("310P3 V2 requires generation sampling parameters")
    expected = {
        "temperature": 0.0, "top_p": 1.0, "min_p": 0.0,
        "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "repetition_penalty": 1.0, "n": 1, "min_tokens": 0,
    }
    for key, required in expected.items():
        got = getattr(params, key, object())
        if got != required:
            raise NotImplementedError(f"310P3 V2 {key}={got!r} is unsupported")
    if getattr(params, "top_k", None) not in (0, -1):
        raise NotImplementedError("310P3 V2 top_k must include full vocabulary")
    for key in (
        "logprobs", "prompt_logprobs", "logprob_token_ids",
        "logit_bias", "allowed_token_ids", "bad_words",
        "structured_outputs", "guided_decoding", "stop", "stop_token_ids",
        "logits_processors",
    ):
        got = getattr(params, key, None)
        if got not in (None, (), [], {}):
            raise NotImplementedError(f"310P3 V2 {key} needs a non-Triton backend")
    if getattr(params, "flat_logprobs", False):
        raise NotImplementedError("310P3 V2 flat_logprobs unsupported")
    mt = getattr(params, "max_tokens", None)
    if mt is None or mt < 1:
        raise ValueError("310P3 V2 requires positive max_tokens")

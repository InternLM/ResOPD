# Copyright 2026 Penghui Yang and the ResOPD authors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the production vLLM response parser without loading a model server."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_path = Path(__file__).resolve().parents[2] / "verl/workers/rollout/vllm_rollout/prompt_logprobs.py"
_spec = importlib.util.spec_from_file_location("resopd_prompt_logprobs", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
extract_prompt_logprobs = _module.extract_prompt_logprobs


def entry(logprob, rank):
    return SimpleNamespace(logprob=logprob, rank=rank)


@pytest.mark.parametrize("key_type", [str, int])
def test_head_and_tail_actions_preserve_causal_alignment(key_type):
    output = SimpleNamespace(
        prompt_token_ids=[7, 20, 30],
        prompt_logprobs=[
            None,
            {key_type(20): entry(-4.2, 7), key_type(12): entry(-0.3, 2), key_type(10): entry(-0.1, 1)},
            {key_type(31): entry(-0.4, 2), key_type(30): entry(-0.2, 1)},
        ],
    )
    result = {}
    extract_prompt_logprobs(output, 2, result)
    assert result["prompt_ids"] == [[10, 12], [30, 31], [0, 0]]
    assert result["prompt_logprobs"] == [[-0.1, -0.3], [-0.2, -0.4], [0.0, 0.0]]
    assert result["sampled_prompt_ids"] == [[20], [30], [0]]
    assert result["sampled_prompt_logprobs"] == [[-4.2], [-0.2], [0.0]]


def test_sampled_only_and_disabled_requests():
    output = SimpleNamespace(prompt_token_ids=[7, 20], prompt_logprobs=[None, {20: entry(-4.2, 7)}])
    result = {}
    extract_prompt_logprobs(output, None, result)
    assert not result
    extract_prompt_logprobs(output, 0, result)
    assert result["prompt_ids"] == result["sampled_prompt_ids"] == [[20], [0]]
    assert result["prompt_logprobs"] == result["sampled_prompt_logprobs"] == [[-4.2], [0.0]]


def test_token_logprob_length_mismatch_fails():
    output = SimpleNamespace(prompt_token_ids=[7, 20, 30], prompt_logprobs=[None, {20: entry(-4.2, 1)}])
    with pytest.raises(ValueError):
        extract_prompt_logprobs(output, 1, {})

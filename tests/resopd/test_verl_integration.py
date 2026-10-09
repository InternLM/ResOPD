# Copyright 2026 Penghui Yang and the ResOPD authors
# SPDX-License-Identifier: Apache-2.0
"""Small CPU integration checks in an installed verl runtime."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import torch
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
from verl.trainer.distillation.losses import DISTILLATION_LOSS_REGISTRY, collect_resopd, compute_topk_loss
from verl.trainer.distillation.resopd import resopd_loss
from verl.trainer.ppo.padding_utils import construct_minimal_padding_template
from verl.workers.config import DistillationLossConfig
from verl.workers.utils.padding import left_right_2_no_padding


def nested(rows, dtype=torch.float32):
    return torch.nested.as_nested_tensor([torch.tensor(row, dtype=dtype) for row in rows], layout=torch.jagged)


def loss_config(**overrides):
    args = dict(loss_mode="resopd", use_policy_gradient=False, loss_max_clamp=None, log_prob_min_clamp=None, topk=2)
    return DistillationLossConfig(**(args | overrides))


def test_resopd_has_only_canonical_registration():
    names = [name for name, fn in DISTILLATION_LOSS_REGISTRY.items() if fn is collect_resopd]
    assert names == ["resopd"]


def test_registration_and_nested_logits_dispatch():
    cfg = loss_config()
    assert cfg.loss_settings.use_topk and not cfg.loss_settings.use_estimator
    data = {
        "teacher_ids": nested([[[0, 1], [1, 2]]], torch.long),
        "teacher_logprobs": nested([[[-1.0, -1.2], [-0.7, -1.1]]]),
        "teacher_sampled_ids": nested([[[2], [0]]], torch.long),
        "teacher_sampled_logprobs": nested([[[-2.0], [-2.3]]]),
    }
    z = torch.tensor([[[1.2, 0.3, -0.4], [0.4, 0.2, 0.1]]], requires_grad=True)
    with patch("verl.trainer.distillation.fsdp.losses.get_ulysses_sequence_parallel_world_size", return_value=1):
        output = compute_topk_loss(
            SimpleNamespace(strategy="fsdp2"), SimpleNamespace(distillation_loss=cfg), data, z, "thd"
        )
    expected = resopd_loss(
        z,
        *[
            data[key].values().unsqueeze(0)
            for key in ("teacher_logprobs", "teacher_ids", "teacher_sampled_logprobs", "teacher_sampled_ids")
        ],
    )
    torch.testing.assert_close(output["distillation_losses"], expected["distillation_losses"])
    output["distillation_losses"].sum().backward()
    assert torch.isfinite(z.grad).all()


@pytest.mark.parametrize(
    "bad",
    [
        dict(use_policy_gradient=True),
        dict(loss_max_clamp=10),
        dict(log_prob_min_clamp=-10),
        dict(use_chunked_topk=True),
        dict(topk=0),
    ],
)
def test_invalid_resopd_configuration_fails(bad):
    with pytest.raises(ValueError, match="ResOPD"):
        loss_config(**bad)


@pytest.mark.parametrize("mode", ["resopd", "k1"])
def test_teacher_manager_requests_once_and_preserves_legacy_contract(mode):
    manager = object.__new__(AsyncTeacherLLMServerManager)
    manager.distillation_loss_config = SimpleNamespace(
        loss_mode=mode, topk=2, loss_settings=SimpleNamespace(use_topk=mode == "resopd")
    )
    manager.teacher_model_configs = {
        "teacher": SimpleNamespace(inference=SimpleNamespace(name="vllm", temperature=1.0))
    }
    payload = dict(
        prompt_ids=[[0, 1], [1, 2], [0, 0]],
        prompt_logprobs=[[-0.5, -1.2], [-1.1, -0.9], [0, 0]],
        sampled_prompt_ids=[[2], [0], [0]],
        sampled_prompt_logprobs=[[-2.2], [-2.3], [0]],
    )
    client = SimpleNamespace(generate=AsyncMock(return_value=SimpleNamespace(extra_fields=payload)))
    manager.teacher_client = {"teacher": client}
    result = asyncio.run(manager.compute_teacher_logprobs_single([8, 2, 0]))
    assert len(result) == (4 if mode == "resopd" else 2)
    client.generate.assert_awaited_once()
    assert client.generate.call_args.kwargs["sampling_params"]["prompt_logprobs"] == (2 if mode == "resopd" else 0)
    if mode == "resopd":
        assert result[2].tolist() == [[2], [0], [0]]


def test_agent_output_keeps_sampled_channel():
    output = AgentLoopOutput(prompt_ids=[8], response_ids=[2], response_mask=[1], metrics=AgentLoopMetrics())
    output.extra_fields["teacher_sampled_ids"] = torch.tensor([[2], [0]])
    output.extra_fields["teacher_sampled_logprobs"] = torch.tensor([[-2.2], [0.0]])
    assert output.as_dict()["teacher_sampled_ids"].tolist() == [[2], [0]]


def test_padding_removal_keeps_both_channels_aligned():
    data = TensorDict(
        {
            "input_ids": torch.tensor([[0, 7, 8, 9], [4, 5, 6, 0]]),
            "position_ids": torch.tensor([[0, 0, 1, 2], [0, 1, 2, 0]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0]]),
            "response_mask": torch.tensor([[1, 1], [1, 0]]),
            "teacher_ids": torch.arange(16).reshape(2, 4, 2),
            "teacher_logprobs": -torch.arange(16).float().reshape(2, 4, 2),
            "teacher_sampled_ids": torch.arange(8).reshape(2, 4, 1),
            "teacher_sampled_logprobs": -torch.arange(8).float().reshape(2, 4, 1),
        },
        batch_size=2,
    )
    result = left_right_2_no_padding(data)
    assert result["teacher_sampled_ids"].values().squeeze(-1).tolist() == [1, 2, 3, 4, 5, 6]
    assert result["teacher_ids"].values()[:, 0].tolist() == [2, 4, 6, 8, 10, 12]
    torch.testing.assert_close(
        result["teacher_sampled_logprobs"].values(), -result["teacher_sampled_ids"].values().float()
    )


def test_synthetic_padding_shortens_teacher_channels():
    source = {
        key: torch.ones(7, width)
        for key, width in (
            ("teacher_ids", 2),
            ("teacher_logprobs", 2),
            ("teacher_sampled_ids", 1),
            ("teacher_sampled_logprobs", 1),
        )
    }
    output, tag = construct_minimal_padding_template(source, {}, eos_token_id=0)
    assert tag["is_padding"] and output["response_mask"].sum() == 0
    for key in source:
        assert output[key].shape == (2, source[key].shape[1]) and output[key].eq(0).all()


def test_collector_applies_causal_response_shift_and_masks_padding():
    data = TensorDict(
        {
            "prompts": nested([[7, 8], [9]], torch.long),
            "responses": nested([[1, 2], [3]], torch.long),
            "response_mask": torch.tensor([[1, 1], [1, 0]]),
        },
        batch_size=2,
    )
    model_output = {
        name: nested([[10.0, 20.0, 30.0, 999.0], [40.0, 999.0]])
        for name in (
            "distillation_losses",
            "student_mass",
            "teacher_mass",
            "resopd_head_hit",
            "resopd_centered_event",
            "resopd_tail_log_ratio",
        )
    }
    losses, metrics = collect_resopd(None, None, model_output, data)
    assert losses.tolist() == [[20.0, 30.0], [40.0, 0.0]]
    assert metrics["distillation/student_mass"].aggregate() == 30.0

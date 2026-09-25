from __future__ import annotations

import torch

from fare.models.bsconv import BayarStammConv2d
from fare.models.fare import FAREVerifier
from fare.training.fare import _resolve_max_steps, _resolve_warmup_steps


def test_fare_verifier_uses_bayar_stamm_front_end_without_relu_directly_after() -> None:
    verifier = FAREVerifier(patch_size=32, top_k=10)
    backbone = verifier.network.backbone
    assert isinstance(backbone.front_end, BayarStammConv2d)
    assert isinstance(backbone.channel_mixer[0], torch.nn.Conv2d)


def test_fare_verifier_scores_images_with_topk_mean() -> None:
    verifier = FAREVerifier(patch_size=16, top_k=2)
    scores = verifier.score_images(torch.rand(2, 3, 32, 32))
    assert scores.shape == (2,)


def test_step_schedule_prefers_explicit_max_and_warmup_steps() -> None:
    config = {"max_steps": 100, "warmup_steps": 7}
    assert _resolve_max_steps(config, steps_per_epoch=4) == 100
    assert _resolve_warmup_steps(config, max_steps=100) == 7


def test_step_schedule_supports_epoch_and_fraction_aliases() -> None:
    config = {"epochs": 3, "warmup_fraction": 0.25}
    assert _resolve_max_steps(config, steps_per_epoch=4) == 12
    assert _resolve_warmup_steps(config, max_steps=12) == 3

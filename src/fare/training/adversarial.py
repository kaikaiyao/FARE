from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import torch
from torch.nn import functional as F


class ImageScorer(Protocol):
    def score_images(self, images: torch.Tensor) -> torch.Tensor:
        ...


@dataclass
class ForgeryAttackResult:
    adversarial_images: torch.Tensor
    scores: torch.Tensor
    restart_index: torch.Tensor
    steps_taken: torch.Tensor
    success: torch.Tensor
    lpips: torch.Tensor


def project_to_l2_shell(delta: torch.Tensor, radius: float, gamma: float) -> torch.Tensor:
    flat = delta.flatten(start_dim=1)
    norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
    projected = flat / norms
    clipped_norms = norms.clamp(min=radius, max=gamma * radius)
    projected = projected * clipped_norms
    return projected.view_as(delta)


def generate_near_boundary_positives(
    model: torch.nn.Module,
    patches: torch.Tensor,
    radius: float,
    gamma: float,
    steps: int,
    step_size: float,
) -> torch.Tensor:
    delta = torch.randn_like(patches)
    delta = project_to_l2_shell(delta, radius=radius, gamma=gamma)
    delta.requires_grad_(True)
    for _ in range(steps):
        adv_patches = (patches + delta).clamp(0.0, 1.0)
        logits = model(adv_patches)
        loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
        grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]
        delta = (delta - step_size * grad).detach()
        delta = project_to_l2_shell(delta, radius=radius, gamma=gamma)
        delta.requires_grad_(True)
    return (patches + delta.detach()).clamp(0.0, 1.0)


def _project_to_lp_ball(
    candidate: torch.Tensor,
    original: torch.Tensor,
    epsilon: float,
    norm: str,
) -> torch.Tensor:
    if norm == "linf":
        projected = torch.max(torch.min(candidate, original + epsilon), original - epsilon)
        return projected.clamp(0.0, 1.0)
    if norm == "l2":
        delta = candidate - original
        flat = delta.flatten(start_dim=1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        scale = (epsilon / norms).clamp(max=1.0)
        projected = original + (flat * scale).view_as(delta)
        return projected.clamp(0.0, 1.0)
    raise ValueError(f"Unsupported attack norm: {norm}")


def _random_start(images: torch.Tensor, epsilon: float, norm: str) -> torch.Tensor:
    if norm == "linf":
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    elif norm == "l2":
        delta = torch.randn_like(images)
        flat = delta.flatten(start_dim=1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        radii = torch.rand(images.size(0), 1, device=images.device, dtype=images.dtype) * epsilon
        delta = (flat / norms * radii).view_as(images)
    else:
        raise ValueError(f"Unsupported attack norm: {norm}")
    return (images + delta).clamp(0.0, 1.0)


def _step_direction(grad: torch.Tensor, norm: str) -> torch.Tensor:
    if norm == "linf":
        return grad.sign()
    if norm == "l2":
        flat = grad.flatten(start_dim=1)
        denom = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)
        return (flat / denom).view_as(grad)
    raise ValueError(f"Unsupported attack norm: {norm}")


def pgd_forgery_attack(
    scorer: ImageScorer,
    images: torch.Tensor,
    epsilon: float,
    steps: int,
    step_size: float,
    threshold: float | None = None,
    random_restarts: int = 5,
    early_stop: bool = True,
    lpips_max: float | None = None,
    norm: str = "linf",
) -> ForgeryAttackResult:
    original = images.detach()
    best_images = original.clone()
    best_scores = scorer.score_images(original).detach()
    best_restart = torch.full((images.size(0),), -1, dtype=torch.long, device=images.device)
    best_steps = torch.zeros(images.size(0), dtype=torch.long, device=images.device)
    best_lpips = torch.zeros(images.size(0), dtype=images.dtype, device=images.device)
    best_success = best_scores <= threshold if threshold is not None else torch.zeros(images.size(0), dtype=torch.bool, device=images.device)

    for restart in range(max(1, random_restarts)):
        candidate = _random_start(original, epsilon=epsilon, norm=norm)
        restart_best_images = candidate.clone()
        restart_best_scores = scorer.score_images(candidate).detach()
        restart_steps = torch.zeros(images.size(0), dtype=torch.long, device=images.device)
        restart_lpips = torch.zeros(images.size(0), dtype=images.dtype, device=images.device)
        restart_success = restart_best_scores <= threshold if threshold is not None else torch.zeros(images.size(0), dtype=torch.bool, device=images.device)
        active = torch.ones(images.size(0), dtype=torch.bool, device=images.device)
        if threshold is not None:
            succeeded = restart_best_scores <= threshold
            restart_steps[succeeded] = 0
            if early_stop:
                active &= ~succeeded
        for step in range(1, steps + 1):
            if early_stop and threshold is not None and not torch.any(active):
                break
            candidate.requires_grad_(True)
            scores = scorer.score_images(candidate)
            objective = scores[active].sum() if torch.any(active) else scores.sum() * 0.0
            grad = torch.autograd.grad(objective, candidate, only_inputs=True)[0]
            with torch.no_grad():
                candidate = candidate - step_size * _step_direction(grad, norm=norm)
                candidate = _project_to_lp_ball(candidate, original=original, epsilon=epsilon, norm=norm)
                current_scores = scorer.score_images(candidate).detach()
                improved = current_scores < restart_best_scores
                restart_best_images[improved] = candidate[improved]
                restart_best_scores[improved] = current_scores[improved]
                first_improvement = improved & (restart_steps == 0)
                restart_steps[first_improvement] = step
                if threshold is not None:
                    succeeded = current_scores <= threshold
                    restart_success |= succeeded
                    first_success = succeeded & (restart_steps == 0)
                    restart_steps[first_success] = step
                    if early_stop:
                        active &= ~succeeded
        if lpips_max is not None:
            restart_lpips = compute_lpips(original, restart_best_images)
            valid = restart_lpips <= lpips_max
            invalid = ~valid
            restart_best_scores[invalid] = best_scores[invalid]
            restart_success[invalid] = False
        improved_global = restart_best_scores < best_scores
        best_images[improved_global] = restart_best_images[improved_global]
        best_scores[improved_global] = restart_best_scores[improved_global]
        best_restart[improved_global] = restart
        best_steps[improved_global] = restart_steps[improved_global]
        best_lpips[improved_global] = restart_lpips[improved_global]
        best_success |= restart_success
    return ForgeryAttackResult(
        adversarial_images=best_images.detach(),
        scores=best_scores.detach(),
        restart_index=best_restart.detach(),
        steps_taken=best_steps.detach(),
        success=best_success.detach(),
        lpips=best_lpips.detach(),
    )


@lru_cache(maxsize=2)
def _lpips_model(device_name: str, net: str = "alex"):
    import lpips

    model = lpips.LPIPS(net=net).to(torch.device(device_name))
    model.eval()
    return model


def compute_lpips(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    net: str = "alex",
) -> torch.Tensor:
    model = _lpips_model(str(original.device), net=net)
    with torch.no_grad():
        values = model((original * 2.0) - 1.0, (adversarial * 2.0) - 1.0)
    return values.view(-1)


def maybe_load_lpips(device: torch.device, net: str = "alex"):
    return _lpips_model(str(device), net=net)


def lpips_distance(
    original: torch.Tensor,
    adversarial: torch.Tensor,
    model,
) -> torch.Tensor | None:
    if model is None:
        return None
    with torch.no_grad():
        values = model((original * 2.0) - 1.0, (adversarial * 2.0) - 1.0)
    return values.view(-1)

from __future__ import annotations

import math

import torch


def _tile_single(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, height, width = image.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Image dimensions must be divisible by patch_size for non-overlapping tiling.")
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous()
    return patches.view(-1, image.size(0), patch_size, patch_size)


def tile_patches(image: torch.Tensor, patch_size: int):
    """Split CHW or BCHW tensors into non-overlapping patches."""
    if image.ndim == 3:
        return _tile_single(image, patch_size)
    if image.ndim == 4:
        patch_sets = [_tile_single(sample, patch_size) for sample in image]
        patches_per_image = [patches.size(0) for patches in patch_sets]
        return torch.cat(patch_sets, dim=0), patches_per_image
    raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(image.shape)}")


def aggregate_topk_mean(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    if scores.ndim != 1:
        raise ValueError(f"Expected 1D scores, got {tuple(scores.shape)}")
    if scores.numel() == 0:
        raise ValueError("Cannot aggregate an empty score vector.")
    k = min(max(1, top_k), scores.numel())
    values = torch.topk(scores, k=k, largest=True).values
    return values.mean()


def aggregate_patch_scores(
    patch_scores: torch.Tensor,
    patches_per_image: list[int],
    top_k: int,
) -> torch.Tensor:
    outputs = []
    start = 0
    for count in patches_per_image:
        image_scores = patch_scores[start : start + count]
        outputs.append(aggregate_topk_mean(image_scores, top_k))
        start += count
    return torch.stack(outputs)


def infer_default_top_k(image_shape: tuple[int, int], patch_size: int, fraction: float = 0.125) -> int:
    height, width = image_shape
    count = (height // patch_size) * (width // patch_size)
    return max(1, math.ceil(count * fraction))

from __future__ import annotations

import torch

from fare.data.patches import aggregate_patch_scores, aggregate_topk_mean, tile_patches


def test_tile_patches_single_image() -> None:
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).view(3, 8, 8)
    patches = tile_patches(image, 4)
    assert patches.shape == (4, 3, 4, 4)


def test_tile_patches_batch_and_aggregate() -> None:
    images = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).view(2, 3, 8, 8)
    patches, counts = tile_patches(images, 4)
    assert patches.shape == (8, 3, 4, 4)
    assert counts == [4, 4]
    patch_scores = torch.tensor([0.1, 0.9, 0.2, 0.3, 1.0, 0.0, 0.5, 0.4])
    aggregated = aggregate_patch_scores(patch_scores, counts, top_k=2)
    assert torch.allclose(aggregated, torch.tensor([0.6, 0.75]))


def test_aggregate_topk_mean_uses_highest_scores() -> None:
    scores = torch.tensor([0.1, 1.5, 0.3, 2.0])
    assert torch.isclose(aggregate_topk_mean(scores, 2), torch.tensor(1.75))


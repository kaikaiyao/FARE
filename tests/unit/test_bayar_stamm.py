from __future__ import annotations

import torch

from fare.models.bsconv import BayarStammConv2d, project_bayar_stamm_weight


def test_bayar_stamm_projection_preserves_constraints() -> None:
    weight = torch.randn(3, 1, 5, 5)
    projected = project_bayar_stamm_weight(weight)
    center = projected[:, 0, 2, 2]
    neighbors = projected.view(3, -1)
    neighbor_sum = neighbors.sum(dim=1) - center
    assert torch.allclose(center, torch.full_like(center, -1.0))
    assert torch.allclose(neighbor_sum, torch.ones_like(neighbor_sum), atol=1e-5)


def test_bayar_stamm_depthwise_layer_expands_channels_per_rgb_input() -> None:
    layer = BayarStammConv2d(in_channels=3, kernel_size=5, filters_per_channel=8)
    outputs = layer(torch.rand(2, 3, 32, 32))
    assert outputs.shape == (2, 24, 32, 32)

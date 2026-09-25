from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def project_bayar_stamm_weight(weight: torch.Tensor, constraint_scale: float = 1.0) -> torch.Tensor:
    projected = weight.clone()
    kernel_size = projected.shape[-1]
    center = kernel_size // 2
    flat = projected.view(projected.shape[0], -1)
    center_index = center * kernel_size + center
    neighbors = flat.clone()
    neighbors[:, center_index] = 0.0
    mask = torch.ones_like(neighbors, dtype=torch.bool)
    mask[:, center_index] = False
    neighbor_values = neighbors[mask].view(projected.shape[0], -1)
    neighbor_values = neighbor_values - neighbor_values.mean(dim=1, keepdim=True)
    neighbor_values = neighbor_values + constraint_scale / neighbor_values.shape[1]
    flat.zero_()
    flat[mask] = neighbor_values.reshape(-1)
    flat[:, center_index] = -constraint_scale
    return flat.view_as(projected)


class BayarStammConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        kernel_size: int = 5,
        filters_per_channel: int = 8,
        constraint_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Bayar-Stamm kernel_size must be odd")
        if filters_per_channel <= 0:
            raise ValueError("filters_per_channel must be positive")
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.filters_per_channel = filters_per_channel
        self.out_channels = in_channels * filters_per_channel
        self.constraint_scale = constraint_scale
        weight = torch.randn(self.out_channels, 1, kernel_size, kernel_size) * 0.01
        self.weight = nn.Parameter(project_bayar_stamm_weight(weight, constraint_scale=constraint_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, stride=1, padding=self.kernel_size // 2, groups=self.in_channels)

    @torch.no_grad()
    def project_(self) -> None:
        self.weight.copy_(
            project_bayar_stamm_weight(self.weight, constraint_scale=self.constraint_scale)
        )

from __future__ import annotations

import torch
from torch import nn

from fare.data.patches import aggregate_patch_scores, tile_patches
from fare.models.bsconv import BayarStammConv2d


def _group_count(channels: int) -> int:
    for candidate in (32, 16, 8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class ConvGNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchFeatureBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        constrained_conv: bool = True,
        filters_per_channel: int = 8,
        mix_width: int = 64,
        backbone_widths: tuple[int, ...] = (64, 128, 128, 256, 256, 256),
        backbone_strides: tuple[int, ...] = (1, 2, 1, 2, 1, 1),
    ) -> None:
        super().__init__()
        if len(backbone_widths) != len(backbone_strides):
            raise ValueError("backbone_widths and backbone_strides must have the same length")
        front_end_channels = in_channels * filters_per_channel
        if constrained_conv:
            self.front_end = BayarStammConv2d(
                in_channels=in_channels,
                kernel_size=5,
                filters_per_channel=filters_per_channel,
                constraint_scale=1.0,
            )
        else:
            self.front_end = nn.Conv2d(
                in_channels,
                front_end_channels,
                kernel_size=5,
                padding=2,
                bias=False,
            )
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(front_end_channels, mix_width, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        current_channels = mix_width
        blocks: list[nn.Module] = []
        for out_channels, stride in zip(backbone_widths, backbone_strides):
            blocks.append(ConvGNBlock(current_channels, out_channels, stride=stride))
            current_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = current_channels

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = self.front_end(patches)
        x = self.channel_mixer(x)
        x = self.features(x)
        return self.pool(x).flatten(1)

    @torch.no_grad()
    def project_constraints_(self) -> None:
        if isinstance(self.front_end, BayarStammConv2d):
            self.front_end.project_()


class PatchAnomalyCNN(nn.Module):
    def __init__(self, in_channels: int = 3, constrained_conv: bool = True) -> None:
        super().__init__()
        self.backbone = PatchFeatureBackbone(
            in_channels=in_channels,
            constrained_conv=constrained_conv,
        )
        self.head = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        features = self.backbone(patches)
        return self.head(features).squeeze(-1)

    @torch.no_grad()
    def project_constraints_(self) -> None:
        self.backbone.project_constraints_()


class FAREVerifier(nn.Module):
    def __init__(self, patch_size: int, top_k: int, constrained_conv: bool = True) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.top_k = top_k
        self.network = PatchAnomalyCNN(constrained_conv=constrained_conv)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.network(patches)

    def score_images(self, images: torch.Tensor) -> torch.Tensor:
        patches, patches_per_image = tile_patches(images, self.patch_size)
        patch_scores = self.forward(patches)
        return aggregate_patch_scores(patch_scores, patches_per_image=patches_per_image, top_k=self.top_k)

    @torch.no_grad()
    def project_constraints_(self) -> None:
        self.network.project_constraints_()

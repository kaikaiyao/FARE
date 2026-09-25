from __future__ import annotations

import torch

from fare.data.transforms import crop_resize, gaussian_blur, gaussian_noise, jpeg_compress


def test_contradiction_transforms_preserve_shape() -> None:
    image = torch.rand(3, 32, 32)
    assert gaussian_blur(image, sigma=1.0, kernel_size=5).shape == image.shape
    assert gaussian_noise(image, sigma=2.0 / 255.0).shape == image.shape
    assert jpeg_compress(image, quality=75).shape == image.shape
    assert crop_resize(image, keep_area=0.9).shape == image.shape

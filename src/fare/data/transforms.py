from __future__ import annotations

import io
from typing import Callable

import torch
from PIL import Image
from torchvision.transforms import functional as F


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3:
        raise ValueError("Expected CHW tensor")
    return F.to_pil_image(image.clamp(0.0, 1.0))


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    return F.pil_to_tensor(image).float() / 255.0


def gaussian_blur(
    image: torch.Tensor,
    sigma: float = 1.0,
    kernel_size: int = 5,
) -> torch.Tensor:
    return F.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


def gaussian_noise(image: torch.Tensor, sigma: float = 2.0 / 255.0) -> torch.Tensor:
    return (image + torch.randn_like(image) * sigma).clamp(0.0, 1.0)


def jpeg_compress(image: torch.Tensor, quality: int = 75) -> torch.Tensor:
    buffer = io.BytesIO()
    tensor_to_pil(image).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return pil_to_tensor(Image.open(buffer).convert("RGB"))


def crop_resize(
    image: torch.Tensor,
    keep_area: float = 0.9,
    crop_fraction: float | None = None,
) -> torch.Tensor:
    image = image.cpu()
    _, height, width = image.shape
    if crop_fraction is not None:
        keep_area = crop_fraction * crop_fraction
    keep_area = min(max(float(keep_area), 1e-6), 1.0)
    linear_fraction = keep_area ** 0.5
    target_height = max(8, int(height * linear_fraction))
    target_width = max(8, int(width * linear_fraction))
    top = max(0, (height - target_height) // 2)
    left = max(0, (width - target_width) // 2)
    cropped = F.crop(image, top, left, target_height, target_width)
    return F.resize(cropped, size=[height, width], antialias=True)


def contradiction_transform_map() -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    return {
        "gaussian_blur": gaussian_blur,
        "gaussian_noise": gaussian_noise,
        "jpeg_compress": jpeg_compress,
        "crop_resize": crop_resize,
    }


DEFAULT_TRANSFORMS = contradiction_transform_map()


def apply_named_transform(
    images: torch.Tensor,
    name: str,
    params: dict[str, float | int] | None = None,
) -> torch.Tensor:
    params = params or {}
    if name not in DEFAULT_TRANSFORMS:
        raise KeyError(f"Unsupported transform: {name}")
    transform = DEFAULT_TRANSFORMS[name]
    if images.ndim == 3:
        return transform(images, **params).to(device=images.device, dtype=images.dtype)
    if images.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW tensor, got {tuple(images.shape)}")
    return torch.stack([transform(image, **params) for image in images]).to(
        device=images.device,
        dtype=images.dtype,
    )

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F

from fare.types import CorpusManifest

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def image_to_tensor(path: str | Path, image_size: int | None = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = F.pil_to_tensor(image).float() / 255.0
    if image_size is not None:
        tensor = F.resize(tensor, [image_size, image_size], antialias=True)
    return tensor


class ImageFolderDataset(Dataset[dict[str, Any]]):
    def __init__(self, root: str | Path, image_size: int | None = None):
        self.root = Path(root)
        self.image_size = image_size
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        return {"image": image_to_tensor(path, self.image_size), "path": str(path)}


class ManifestImageDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest_path: str | Path, image_size: int | None = None, split: str | None = None):
        self.manifest = CorpusManifest.load(manifest_path)
        self.root = Path(self.manifest.root)
        self.image_size = image_size
        items = self.manifest.items
        if split is not None:
            items = [item for item in items if item.split == split]
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        return {
            "image": image_to_tensor(self.root / item.path, self.image_size),
            "path": item.path,
            "seed": item.seed,
            "prompt": item.prompt,
            "metadata": item.metadata,
        }


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if "image" in samples[0]:
        return {
            "images": torch.stack([sample["image"] for sample in samples], dim=0),
            "paths": [sample["path"] for sample in samples],
            "seeds": [sample.get("seed") for sample in samples],
            "prompts": [sample.get("prompt") for sample in samples],
            "metadata": [sample.get("metadata", {}) for sample in samples],
        }
    raise ValueError("Unsupported dataset sample shape")


def build_dataloader(
    manifest_path: str | Path,
    split: str | None,
    batch_size: int,
    image_size: int | None = None,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = ManifestImageDataset(manifest_path, image_size=image_size, split=split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=_collate)


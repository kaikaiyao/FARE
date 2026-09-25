from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fare.types import VerifierArtifact


class AnomalyScorer(ABC):
    method_name: str

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device

    @abstractmethod
    def fit(self, manifest_path: str, split: str = "train") -> None:
        raise NotImplementedError

    @abstractmethod
    def score_images(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def save(self, artifact_dir: str | Path, thresholds: dict[str, Any] | None = None) -> VerifierArtifact:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, artifact: VerifierArtifact, device: torch.device) -> "AnomalyScorer":
        raise NotImplementedError

    def score_manifest(self, loader: Any) -> np.ndarray:
        scores: list[np.ndarray] = []
        for batch in loader:
            images = batch["images"].to(self.device)
            scores.append(self.score_images(images).detach().cpu().numpy())
        if not scores:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(scores, axis=0)


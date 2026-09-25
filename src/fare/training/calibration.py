from __future__ import annotations

from typing import Iterable

import numpy as np


def quantile_threshold(scores: np.ndarray, alpha: float) -> float:
    if scores.size == 0:
        raise ValueError("Cannot calibrate threshold on empty score array")
    return float(np.quantile(scores.astype(np.float64), 1.0 - alpha, method="higher"))


def calibrate_threshold(scores: Iterable[float], alpha: float) -> float:
    return quantile_threshold(np.asarray(list(scores), dtype=np.float32), alpha)


def batched_scores(scores: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    usable = len(scores) - (len(scores) % batch_size)
    if usable == 0:
        return []
    trimmed = scores[:usable]
    return trimmed.reshape(-1, batch_size)


def calibrate_batch_thresholds(scores: np.ndarray, batch_sizes: list[int], alpha: float) -> tuple[dict[int, float], dict[int, float]]:
    mean_thresholds: dict[int, float] = {}
    max_thresholds: dict[int, float] = {}
    for batch_size in batch_sizes:
        grouped = batched_scores(scores, batch_size)
        grouped = np.asarray(list(grouped))
        if grouped.size == 0:
            continue
        mean_thresholds[batch_size] = quantile_threshold(grouped.mean(axis=1), alpha)
        max_thresholds[batch_size] = quantile_threshold(grouped.max(axis=1), alpha)
    return mean_thresholds, max_thresholds


def calibrate_thresholds(scores: dict[str, Iterable[float]], alpha: float) -> dict[str, float]:
    return {name: calibrate_threshold(values, alpha) for name, values in scores.items()}


def warmup_steps(total_steps: int, fraction: float = 0.1) -> int:
    return max(1, int(np.ceil(total_steps * fraction)))

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def tpr_at_threshold(positive_scores: Iterable[float], threshold: float) -> float:
    positives = np.asarray(list(positive_scores), dtype=np.float32)
    if positives.size == 0:
        return 0.0
    return float((positives > threshold).mean() * 100.0)


def evaluate_fixed_fpr(
    certified_scores: Iterable[float],
    candidate_scores: Iterable[float],
    alpha: float,
) -> dict[str, float]:
    from .calibration import calibrate_threshold

    threshold = calibrate_threshold(certified_scores, alpha)
    tpr = tpr_at_threshold(candidate_scores, threshold)
    return {"threshold": threshold, "tpr": tpr, "alpha": alpha}


def batch_scores(scores: list[float], batch_size: int, rule: str) -> list[float]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(scores) % batch_size != 0:
        raise ValueError("scores length must be divisible by batch_size")
    batches: list[float] = []
    for start in range(0, len(scores), batch_size):
        chunk = scores[start : start + batch_size]
        if rule == "mean":
            batches.append(float(np.mean(chunk)))
        elif rule == "max":
            batches.append(float(np.max(chunk)))
        else:
            raise ValueError(f"Unsupported batch rule: {rule}")
    return batches


def group_scores(rows: list[dict[str, float]], key: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(float(row["score"]))
    return dict(grouped)

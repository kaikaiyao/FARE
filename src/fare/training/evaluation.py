from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScoreSummary:
    threshold: float
    tpr: float
    fpr: float
    mean_score: float
    std_score: float


def summarize_scores(negative_scores: np.ndarray, positive_scores: np.ndarray, threshold: float) -> ScoreSummary:
    negative_pred = negative_scores > threshold
    positive_pred = positive_scores > threshold
    return ScoreSummary(
        threshold=float(threshold),
        tpr=float(positive_pred.mean() * 100.0) if positive_scores.size else 0.0,
        fpr=float(negative_pred.mean() * 100.0) if negative_scores.size else 0.0,
        mean_score=float(positive_scores.mean()) if positive_scores.size else 0.0,
        std_score=float(positive_scores.std()) if positive_scores.size else 0.0,
    )


def batch_decisions(scores: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    usable = len(scores) - (len(scores) % batch_size)
    trimmed = scores[:usable].reshape(-1, batch_size)
    return trimmed.mean(axis=1), trimmed.max(axis=1)


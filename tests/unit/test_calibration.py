from __future__ import annotations

import numpy as np
import pytest

from fare.training.calibration import calibrate_batch_thresholds, quantile_threshold
from fare.training.evaluation import batch_decisions


def test_quantile_threshold_uses_upper_tail() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9], dtype=np.float32)
    assert quantile_threshold(scores, alpha=0.2) == pytest.approx(0.9)


def test_batch_calibration_and_decisions() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    mean_thresholds, max_thresholds = calibrate_batch_thresholds(scores, batch_sizes=[2], alpha=0.5)
    assert 2 in mean_thresholds
    means, maxes = batch_decisions(scores, batch_size=2)
    assert means.shape == (3,)
    assert maxes.shape == (3,)
    assert mean_thresholds[2] >= means.min()
    assert max_thresholds[2] >= maxes.min()

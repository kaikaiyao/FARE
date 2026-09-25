from fare.training.metrics import batch_scores, tpr_at_threshold


def test_batch_scores_mean_and_max() -> None:
    scores = [1.0, 3.0, 2.0, 4.0]
    assert batch_scores(scores, batch_size=2, rule="mean") == [2.0, 3.0]
    assert batch_scores(scores, batch_size=2, rule="max") == [3.0, 4.0]


def test_tpr_at_threshold() -> None:
    assert tpr_at_threshold([0.6, 0.7, 0.4], threshold=0.5) == (2 / 3) * 100.0

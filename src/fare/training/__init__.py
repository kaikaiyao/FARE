"""Training, calibration, evaluation, and attacks."""

from .adversarial import compute_lpips, generate_near_boundary_positives, pgd_forgery_attack, project_to_l2_shell
from .calibration import calibrate_batch_thresholds, quantile_threshold
from .evaluation import batch_decisions, summarize_scores
from .fare import FAREScorer

__all__ = [
    "FAREScorer",
    "batch_decisions",
    "calibrate_batch_thresholds",
    "compute_lpips",
    "generate_near_boundary_positives",
    "pgd_forgery_attack",
    "project_to_l2_shell",
    "quantile_threshold",
    "summarize_scores",
]

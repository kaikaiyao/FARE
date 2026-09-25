from __future__ import annotations

import torch

from fare.training.adversarial import project_to_l2_shell


def test_project_to_l2_shell_clips_norms_into_shell() -> None:
    delta = torch.randn(4, 3, 8, 8)
    projected = project_to_l2_shell(delta, radius=0.5, gamma=1.5)
    norms = projected.flatten(start_dim=1).norm(p=2, dim=1)
    assert torch.all(norms >= 0.5 - 1e-5)
    assert torch.all(norms <= 0.75 + 1e-5)


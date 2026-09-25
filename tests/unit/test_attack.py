import torch

from fare.training.adversarial import pgd_forgery_attack, project_to_l2_shell


def test_project_to_l2_shell_respects_bounds() -> None:
    delta = torch.randn(5, 3, 8, 8)
    projected = project_to_l2_shell(delta, radius=0.5, gamma=1.5)
    flat = projected.view(projected.size(0), -1)
    norms = flat.norm(dim=1)
    assert torch.all(norms >= 0.5 - 1e-5)
    assert torch.all(norms <= 0.75 + 1e-5)


class DummyScorer:
    def score_images(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(1, 2, 3))


def test_pgd_forgery_attack_returns_result_object() -> None:
    scorer = DummyScorer()
    images = torch.rand(2, 3, 64, 64)
    result = pgd_forgery_attack(
        scorer,
        images,
        epsilon=0.05,
        steps=2,
        step_size=0.01,
        threshold=0.4,
        random_restarts=2,
        early_stop=True,
    )
    assert result.adversarial_images.shape == images.shape
    assert result.scores.shape == (2,)
    assert result.restart_index.shape == (2,)
    assert result.steps_taken.shape == (2,)


def test_pgd_forgery_attack_can_stop_immediately_when_threshold_is_already_met() -> None:
    scorer = DummyScorer()
    images = torch.rand(2, 3, 16, 16)
    result = pgd_forgery_attack(
        scorer,
        images,
        epsilon=0.01,
        steps=5,
        step_size=0.005,
        threshold=1.0,
        random_restarts=1,
        early_stop=True,
    )
    assert torch.all(result.scores <= 1.0)
    assert torch.all(result.steps_taken == 0)

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from fare.baselines.base import AnomalyScorer
from fare.data.datasets import build_dataloader
from fare.data.patches import tile_patches
from fare.data.transforms import DEFAULT_TRANSFORMS, apply_named_transform
from fare.models.fare import FAREVerifier
from fare.training.adversarial import generate_near_boundary_positives
from fare.types import CalibrationThresholds, VerifierArtifact
from fare.utils import ensure_dir, json_dump

FARE_ARCHITECTURE_VERSION = "fare-paper-2026-v1"
DEFAULT_CONTRADICTION_PARAMS: dict[str, dict[str, float | int]] = {
    "gaussian_blur": {"sigma": 1.0, "kernel_size": 5},
    "gaussian_noise": {"sigma": 2.0 / 255.0},
    "jpeg_compress": {"quality": 75},
    "crop_resize": {"keep_area": 0.9},
}
DEFAULT_ADV_CONFIG: dict[str, float | int] = {
    "radius": 0.03,
    "gamma": 1.2,
    "steps": 5,
    "step_size": 0.005,
}


def _merge_nested_dicts(
    base: dict[str, dict[str, float | int]],
    update: dict[str, dict[str, float | int]] | None,
) -> dict[str, dict[str, float | int]]:
    merged = {key: dict(value) for key, value in base.items()}
    if not update:
        return merged
    for key, value in update.items():
        if isinstance(value, dict) and key in merged:
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = dict(value)
    return merged


def _resolve_optimizer(config: dict[str, Any], parameters) -> tuple[torch.optim.Optimizer, str, float]:
    optimizer_name = str(config.get("optimizer", "adamw")).lower()
    lr = float(config.get("lr", 1e-3))
    if optimizer_name == "adamw":
        weight_decay = float(config.get("weight_decay", 0.01))
        optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
        return optimizer, optimizer_name, weight_decay
    if optimizer_name == "adam":
        weight_decay = float(config.get("weight_decay", 0.0))
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        return optimizer, optimizer_name, weight_decay
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _resolve_max_steps(config: dict[str, Any], steps_per_epoch: int) -> int:
    if "max_steps" in config:
        return max(1, int(config["max_steps"]))
    epochs = int(config.get("epochs", 1))
    return max(1, epochs * max(1, steps_per_epoch))


def _resolve_warmup_steps(config: dict[str, Any], max_steps: int) -> int:
    if "warmup_steps" in config:
        return max(0, int(config["warmup_steps"]))
    if "warmup_fraction" in config:
        return max(0, int(max_steps * float(config["warmup_fraction"])))
    if "max_steps" in config or "epochs" not in config:
        return 2000
    return max(0, int(max_steps * 0.1))


def _resolve_history_stride(config: dict[str, Any], max_steps: int) -> int:
    return max(1, int(config.get("training_history_stride", max(1, max_steps // 1000))))


class FAREScorer(AnomalyScorer):
    method_name = "fare"

    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        super().__init__(config, device)
        self.patch_size = int(config.get("patch_size", 32))
        self.top_k = int(config.get("top_k", 10))
        self.image_size = config.get("image_size")
        self.model = FAREVerifier(
            patch_size=self.patch_size,
            top_k=self.top_k,
            constrained_conv=config.get("use_constrained_conv", True),
        ).to(device)
        self.training_history: list[dict[str, Any]] = []
        self.training_summary: dict[str, Any] = {}
        self.effective_config: dict[str, Any] = {}

    def fit(self, manifest_path: str, split: str = "train") -> None:
        batch_size = int(self.config.get("batch_size", 32))
        loader = build_dataloader(
            manifest_path,
            split=split,
            batch_size=batch_size,
            image_size=self.image_size,
            shuffle=True,
            num_workers=self.config.get("num_workers", 0),
        )
        optimizer, optimizer_name, weight_decay = _resolve_optimizer(self.config, self.model.parameters())
        max_steps = _resolve_max_steps(self.config, len(loader))
        warmup_steps = _resolve_warmup_steps(self.config, max_steps)
        history_stride = _resolve_history_stride(self.config, max_steps)
        record_history = bool(self.config.get("record_training_history", True))
        contradiction_weight = float(self.config.get("lambda_con", 0.1))
        adversarial_weight = float(self.config.get("mu_adv", 0.1))
        contradiction_transforms = list(
            self.config.get("contradiction_transforms", list(DEFAULT_TRANSFORMS.keys()))
        )
        transform_params = _merge_nested_dicts(
            DEFAULT_CONTRADICTION_PARAMS,
            self.config.get("transform_params"),
        )
        adv_config = {**DEFAULT_ADV_CONFIG, **self.config.get("adv", {})}
        self.effective_config = {
            "architecture_version": FARE_ARCHITECTURE_VERSION,
            "optimizer": optimizer_name,
            "lr": float(self.config.get("lr", 1e-3)),
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "patch_size": self.patch_size,
            "top_k": self.top_k,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "mu_adv": adversarial_weight,
            "lambda_con": contradiction_weight,
            "contradiction_transforms": contradiction_transforms,
            "transform_params": transform_params,
            "adv": {
                "radius": float(adv_config["radius"]),
                "gamma": float(adv_config["gamma"]),
                "steps": int(adv_config["steps"]),
                "step_size": float(adv_config["step_size"]),
            },
        }

        self.model.train()
        self.training_history = []
        step_records: list[dict[str, Any]] = []
        loader_iter = iter(loader)
        for global_step in range(1, max_steps + 1):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            images = batch["images"].to(self.device)
            patches, _ = tile_patches(images, self.patch_size)
            cert_logits = self.model(patches)
            cert_loss = F.binary_cross_entropy_with_logits(cert_logits, torch.zeros_like(cert_logits))
            loss = cert_loss
            adv_loss_value = 0.0
            contradiction_loss_value = 0.0
            if global_step > warmup_steps:
                if self.config.get("use_adv_positives", True):
                    adv_patches = generate_near_boundary_positives(
                        self.model,
                        patches,
                        radius=float(adv_config["radius"]),
                        gamma=float(adv_config["gamma"]),
                        steps=int(adv_config["steps"]),
                        step_size=float(adv_config["step_size"]),
                    )
                    adv_logits = self.model(adv_patches)
                    adv_loss = F.binary_cross_entropy_with_logits(adv_logits, torch.ones_like(adv_logits))
                    adv_loss_value = float(adv_loss.detach().cpu().item())
                    loss = loss + adversarial_weight * adv_loss
                if self.config.get("use_contradiction_positives", True):
                    contradiction_losses = []
                    for name in contradiction_transforms:
                        transformed = apply_named_transform(images, name, transform_params.get(name, {}))
                        transformed_patches, _ = tile_patches(transformed, self.patch_size)
                        logits = self.model(transformed_patches)
                        contradiction_losses.append(
                            F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
                        )
                    if contradiction_losses:
                        contradiction_loss = torch.stack(contradiction_losses).mean()
                        contradiction_loss_value = float(contradiction_loss.detach().cpu().item())
                        loss = loss + contradiction_weight * contradiction_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self.model.project_constraints_()

            step_record = {
                "epoch": ((global_step - 1) // max(1, len(loader))) + 1,
                "global_step": global_step,
                "warmup_active": global_step <= warmup_steps,
                "cert_loss": float(cert_loss.detach().cpu().item()),
                "adv_loss": adv_loss_value,
                "contradiction_loss": contradiction_loss_value,
                "total_loss": float(loss.detach().cpu().item()),
            }
            step_records.append(step_record)
            if record_history and (global_step == 1 or global_step % history_stride == 0 or global_step == max_steps):
                self.training_history.append(step_record)

        self.training_summary = {
            "architecture_version": FARE_ARCHITECTURE_VERSION,
            "steps_per_epoch": len(loader),
            "total_steps": max_steps,
            "warmup_steps": warmup_steps,
            "history_stride": history_stride,
            "record_training_history": record_history,
            "effective_hyperparameters": self.effective_config,
            "final_step": step_records[-1] if step_records else {},
            "mean_cert_loss": sum(record["cert_loss"] for record in step_records) / max(1, len(step_records)),
            "mean_adv_loss": sum(record["adv_loss"] for record in step_records) / max(1, len(step_records)),
            "mean_contradiction_loss": sum(record["contradiction_loss"] for record in step_records) / max(1, len(step_records)),
            "mean_total_loss": sum(record["total_loss"] for record in step_records) / max(1, len(step_records)),
        }
        self.model.eval()

    def score_images(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model.score_images(images)

    def save(self, artifact_dir: str | Path, thresholds: dict[str, Any] | None = None) -> VerifierArtifact:
        artifact_root = ensure_dir(artifact_dir)
        weights_path = artifact_root / "model.pt"
        torch.save({"model": self.model.state_dict()}, weights_path)
        threshold_obj = None
        if thresholds is not None:
            threshold_obj = CalibrationThresholds(
                alpha=thresholds["alpha"],
                single_image=thresholds["single_image"],
                batch_mean=thresholds.get("batch_mean", {}),
                batch_max=thresholds.get("batch_max", {}),
            )
        artifact = VerifierArtifact(
            method=self.method_name,
            config=self.config,
            weights_path=str(weights_path),
            patch_size=self.patch_size,
            top_k=self.top_k,
            contradiction_transforms=self.config.get(
                "contradiction_transforms",
                list(DEFAULT_TRANSFORMS.keys()),
            ),
            score_stats={},
            thresholds=threshold_obj,
            metadata={
                "architecture_version": FARE_ARCHITECTURE_VERSION,
                "effective_hyperparameters": self.effective_config,
                "training_summary": self.training_summary,
                "training_history_files": {
                    "json": "training_history.json",
                    "csv": "training_history.csv",
                    "summary": "training_summary.json",
                },
            },
        )
        if self.training_history:
            json_dump(self.training_history, artifact_root / "training_history.json")
            with (artifact_root / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "epoch",
                        "global_step",
                        "warmup_active",
                        "cert_loss",
                        "adv_loss",
                        "contradiction_loss",
                        "total_loss",
                    ],
                )
                writer.writeheader()
                writer.writerows(self.training_history)
        if self.training_summary:
            json_dump(self.training_summary, artifact_root / "training_summary.json")
        artifact.save(artifact_root / "artifact.json")
        return artifact

    @classmethod
    def load(cls, artifact: VerifierArtifact, device: torch.device) -> "FAREScorer":
        scorer = cls(dict(artifact.config), device=device)
        state = torch.load(artifact.weights_path, map_location=device)
        scorer.model.load_state_dict(state["model"])
        scorer.model.eval()
        scorer.effective_config = dict(artifact.metadata.get("effective_hyperparameters", {}))
        scorer.training_summary = dict(artifact.metadata.get("training_summary", {}))
        return scorer

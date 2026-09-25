from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fare.utils import json_dump, json_load


@dataclass
class GeneratorSpec:
    name: str
    family: str
    checkpoint: str
    adapter: str
    variant: str = "certified"
    source_id: str | None = None
    resolved_source: str | None = None
    auth_required: bool = False
    license_gate: str | None = None
    variant_recipe: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSource:
    id: str
    family: str
    source_type: str
    locator: str
    adapter: str
    checkpoint: str | None = None
    auth_required: bool = False
    license_gate: str | None = None
    sha256: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    provenance_notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VariantSpec:
    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PromptSourceConfig:
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PoolMember:
    member_id: str
    source: ModelSource
    splits: dict[str, int]
    seed_offset: int = 0
    variant: VariantSpec | None = None
    prompt_source: PromptSourceConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "source": self.source.to_dict(),
            "splits": self.splits,
            "seed_offset": self.seed_offset,
            "variant": None if self.variant is None else self.variant.to_dict(),
            "prompt_source": None if self.prompt_source is None else self.prompt_source.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class ScenarioPair:
    certified_member_id: str
    candidate_member_id: str
    variant: str = "certified"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScenarioExpansion:
    name: str
    scenario: str
    output_root: str
    members: list[PoolMember]
    pairs: list[ScenarioPair]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scenario": self.scenario,
            "output_root": self.output_root,
            "members": [member.to_dict() for member in self.members],
            "pairs": [pair.to_dict() for pair in self.pairs],
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        json_dump(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioExpansion":
        raw = json_load(path)
        members = []
        for item in raw.get("members", []):
            variant = item.get("variant")
            prompt_source = item.get("prompt_source")
            members.append(
                PoolMember(
                    member_id=item["member_id"],
                    source=ModelSource(**item["source"]),
                    splits=dict(item.get("splits", {})),
                    seed_offset=item.get("seed_offset", 0),
                    variant=None if variant is None else VariantSpec(**variant),
                    prompt_source=None if prompt_source is None else PromptSourceConfig(**prompt_source),
                    metadata=item.get("metadata", {}),
                )
            )
        pairs = [ScenarioPair(**item) for item in raw.get("pairs", [])]
        return cls(
            name=raw["name"],
            scenario=raw["scenario"],
            output_root=raw["output_root"],
            members=members,
            pairs=pairs,
            metadata=raw.get("metadata", {}),
        )


@dataclass
class CorpusItem:
    path: str
    split: str
    generator: GeneratorSpec
    seed: int | None = None
    prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusManifest:
    root: str
    items: list[CorpusItem]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "items": [
                {
                    **asdict(item),
                    "generator": asdict(item.generator),
                }
                for item in self.items
            ],
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        json_dump(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "CorpusManifest":
        raw = json_load(path)
        items = [
            CorpusItem(
                path=item["path"],
                split=item["split"],
                generator=GeneratorSpec(**item["generator"]),
                seed=item.get("seed"),
                prompt=item.get("prompt"),
                metadata=item.get("metadata", {}),
            )
            for item in raw.get("items", [])
        ]
        return cls(root=raw["root"], items=items, metadata=raw.get("metadata", {}))


@dataclass
class CalibrationThresholds:
    alpha: float
    single_image: float
    batch_mean: dict[int, float] = field(default_factory=dict)
    batch_max: dict[int, float] = field(default_factory=dict)


@dataclass
class VerifierArtifact:
    method: str
    config: dict[str, Any]
    weights_path: str | None
    patch_size: int | None
    top_k: int | None
    contradiction_transforms: list[str]
    score_stats: dict[str, float]
    thresholds: CalibrationThresholds | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "config": self.config,
            "weights_path": self.weights_path,
            "patch_size": self.patch_size,
            "top_k": self.top_k,
            "contradiction_transforms": self.contradiction_transforms,
            "score_stats": self.score_stats,
            "thresholds": None
            if self.thresholds is None
            else {
                "alpha": self.thresholds.alpha,
                "single_image": self.thresholds.single_image,
                "batch_mean": self.thresholds.batch_mean,
                "batch_max": self.thresholds.batch_max,
            },
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        json_dump(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "VerifierArtifact":
        raw = json_load(path)
        thresholds = raw.get("thresholds")
        threshold_obj = None
        if thresholds is not None:
            threshold_obj = CalibrationThresholds(
                alpha=thresholds["alpha"],
                single_image=thresholds["single_image"],
                batch_mean={int(k): v for k, v in thresholds.get("batch_mean", {}).items()},
                batch_max={int(k): v for k, v in thresholds.get("batch_max", {}).items()},
            )
        return cls(
            method=raw["method"],
            config=raw["config"],
            weights_path=raw.get("weights_path"),
            patch_size=raw.get("patch_size"),
            top_k=raw.get("top_k"),
            contradiction_transforms=raw.get("contradiction_transforms", []),
            score_stats=raw.get("score_stats", {}),
            thresholds=threshold_obj,
            metadata=raw.get("metadata", {}),
        )


@dataclass
class AttackConfig:
    epsilon: float = 0.025
    steps: int = 50
    step_size: float = 0.001
    random_restarts: int = 5
    early_stop: bool = True
    use_lpips: bool = True
    norm: str = "linf"
    lpips_max: float = 0.05


@dataclass
class ExperimentManifest:
    name: str
    certified_manifest: str
    calibration_manifest: str
    evaluation_manifest: str | None
    method_config: str
    artifact_dir: str
    alpha: float = 0.01
    batch_sizes: list[int] = field(default_factory=lambda: [5, 10])
    tags: dict[str, str] = field(default_factory=dict)
    attack: AttackConfig = field(default_factory=AttackConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentManifest":
        attack = AttackConfig(**data.get("attack", {}))
        return cls(
            name=data["name"],
            certified_manifest=data["certified_manifest"],
            calibration_manifest=data["calibration_manifest"],
            evaluation_manifest=data.get("evaluation_manifest"),
            method_config=data["method_config"],
            artifact_dir=data["artifact_dir"],
            alpha=data.get("alpha", 0.01),
            batch_sizes=list(data.get("batch_sizes", [5, 10])),
            tags=data.get("tags", {}),
            attack=attack,
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        json_dump(
            {
                "name": self.name,
                "certified_manifest": self.certified_manifest,
                "calibration_manifest": self.calibration_manifest,
                "evaluation_manifest": self.evaluation_manifest,
                "method_config": self.method_config,
                "artifact_dir": self.artifact_dir,
                "alpha": self.alpha,
                "batch_sizes": self.batch_sizes,
                "tags": self.tags,
                "attack": asdict(self.attack),
                "metadata": self.metadata,
            },
            path,
        )

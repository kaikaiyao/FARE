"""Train and calibrate FARE on certified images, then score a query folder."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fare.config import load_config
from fare.data.datasets import IMAGE_EXTENSIONS, build_dataloader
from fare.training.calibration import quantile_threshold
from fare.training.fare import FAREScorer
from fare.types import CorpusItem, CorpusManifest, GeneratorSpec
from fare.utils import json_dump, resolve_device, set_seed


def make_manifest(folder: Path, split: str, output: Path) -> Path:
    folder = folder.resolve()
    paths = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError(f"No images found in {folder}")
    spec = GeneratorSpec(name="user-supplied", family="external", checkpoint="external", adapter="external_command")
    items = [CorpusItem(path=str(p.relative_to(folder)), split=split, generator=spec) for p in paths]
    CorpusManifest(root=str(folder), items=items).save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="Certified training image folder")
    parser.add_argument("--calibration", type=Path, required=True, help="Independent certified calibration image folder")
    parser.add_argument("--query", type=Path, required=True, help="Query image folder")
    parser.add_argument("--output", type=Path, default=Path("outputs/fare"))
    parser.add_argument("--config", type=Path, default=Path("configs/methods/fare.yaml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, help="Optional override for a short run")
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.train.resolve() == args.calibration.resolve():
        parser.error("Training and calibration must use separate image sets")
    set_seed(args.seed)
    config = load_config(args.config)
    config["image_size"] = 256
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = {
        split: make_manifest(folder, split, args.output / f"{split}_manifest.json")
        for split, folder in [("train", args.train), ("calibration", args.calibration), ("query", args.query)]
    }
    scorer = FAREScorer(config, resolve_device(args.device))
    print(f"Training FARE for {config['max_steps']} steps on {scorer.device}", flush=True)
    scorer.fit(str(manifests["train"]))

    def score(split: str):
        loader = build_dataloader(manifests[split], split, batch_size=config["batch_size"], image_size=config["image_size"])
        with torch.no_grad():
            return scorer.score_manifest(loader)

    alpha = config.get("alpha", 0.01)
    threshold = quantile_threshold(score("calibration"), alpha)
    scorer.save(args.output, thresholds={"alpha": alpha, "single_image": threshold})
    scores = score("query")
    items = CorpusManifest.load(manifests["query"]).items
    results = [{"path": item.path, "score": float(value), "accepted": bool(value <= threshold)} for item, value in zip(items, scores)]
    json_dump({"alpha": alpha, "threshold": threshold, "seed": args.seed, "images": results}, args.output / "scores.json")
    print(f"Saved model and {len(results)} query decisions to {args.output}")


if __name__ == "__main__":
    main()

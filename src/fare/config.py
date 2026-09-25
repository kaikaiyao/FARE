from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_config_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    return _merge_dicts(base, update)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    data = load_yaml(source)
    parents = data.pop("extends", [])
    if isinstance(parents, (str, Path)):
        parents = [parents]
    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = source.parent / parent_path
        merged = _merge_dicts(merged, load_config(parent_path))
    return _merge_dicts(merged, data)


def dump_yaml(data: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

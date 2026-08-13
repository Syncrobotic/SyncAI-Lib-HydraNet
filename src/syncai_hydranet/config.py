"""Minimal config system: YAML to a nested dict with dot-path overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that also supports attribute access, e.g. ``cfg.model.backbone.name``."""

    def __getattr__(self, key: str) -> Any:
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, path: str, value: Any) -> None:
        """Set a nested key from a dotted path.

        List indices are NOT supported: ``data.datasets[0].root`` silently creates a
        key literally named ``datasets[0]``. Override the whole list instead, or copy
        the config file.
        """
        parts = path.split(".")
        node: dict = self
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_value(value)

    def clone(self) -> Config:
        return Config(copy.deepcopy(dict(self)))


def _parse_value(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    try:
        # Reuse the YAML scalar parser: "1e-4" -> float, "true" -> bool, "[1,2]" -> list.
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config.

    ``overrides`` looks like ``["train.lr=1e-4", "model.neck.name=fpn"]``.
    """
    with open(path, encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f))
    for ov in overrides or []:
        key, sep, val = ov.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value, got: {ov}")
        cfg.set_path(key.strip(), val.strip())
    return cfg

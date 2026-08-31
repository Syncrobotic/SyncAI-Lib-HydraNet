"""Minimal config system: YAML to a nested dict with dot-path overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .config_schema import check_config


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


def diff_config(old: Any, new: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """Every leaf that differs between two configs, as ``(dotted.path, old, new)``.

    Used when resuming: the checkpoint carries the config it was trained under, and a
    resume applies whatever the config file says now. Changing epochs is a normal thing
    to do; changing the learning rate or the class count by accident is not, and
    without this the run's meta.json would describe settings the weights never saw.
    """
    out: list[tuple[str, Any, Any]] = []
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new)):
            here = f"{path}.{key}" if path else str(key)
            if key not in old:
                out.append((here, None, new[key]))
            elif key not in new:
                out.append((here, old[key], None))
            else:
                out.extend(diff_config(old[key], new[key], here))
    elif old != new:
        out.append((path, old, new))
    return out


def _parse_value(v: Any) -> Any:
    """Turn a command-line string into the value it obviously means."""
    if not isinstance(v, str):
        return v
    for cast in (int, float):
        try:
            # Numbers first: YAML 1.1 only recognises floats written with a decimal
            # point, so "1e-4" would come back as the string "1e-4". It survived
            # because build_optimizer calls float() on it, but the same value reached
            # meta.json and the checkpoint as a string, and anything comparing it
            # numerically was comparing text.
            return cast(v)
        except ValueError:
            pass
    try:
        # Then the YAML scalar parser: "true" -> bool, "[1,2]" -> list, else a string.
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


BASE_KEY = "_base_"
_MAX_BASE_DEPTH = 8


def merge_config(base: dict, over: dict) -> dict:
    """Recursively overlay ``over`` on ``base``; ``over`` wins at every leaf.

    Dicts merge, everything else replaces. Lists in particular replace rather than
    concatenate: ``data.datasets`` is the case that matters, and a config that meant to
    *change* the dataset list would otherwise silently get both.
    """
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_with_bases(path: Path, seen: list[Path] | None = None) -> dict:
    """Load one YAML file and everything it inherits from.

    ``_base_`` names a file, or a list of files, relative to the config that declares it.
    Later entries win over earlier ones, and the declaring file wins over all of them.
    """
    seen = seen or []
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(p.name for p in [*seen, path])
        raise ValueError(f"circular _base_ chain: {chain}")
    if len(seen) >= _MAX_BASE_DEPTH:
        raise ValueError(f"_base_ nested more than {_MAX_BASE_DEPTH} deep, at {path.name}")

    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    bases = raw.pop(BASE_KEY, None)
    if not bases:
        return raw
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        merged = merge_config(merged, _load_with_bases(path.parent / b, [*seen, path]))
    return merge_config(merged, raw)


def _drop_disabled_heads(cfg: dict) -> list[str]:
    """``model.heads.<name>: null`` in a child config removes that head. Returns them.

    Inheritance can add and it can change, and until this it could not subtract. A
    config that wanted the base's backbone, neck, schedule and two of its three heads
    had no way to say so -- ``merge_config`` merges dicts key by key, so the base's head
    survived every attempt to override it away, and the only route was to stop
    inheriting and copy 90 lines.

    Deleting here rather than skipping at build time is deliberate: it happens once,
    before validation, so a removed head is removed for *everything* downstream --
    the schema, ``unsupervised_heads``, the trainer's loss balancer, the exporter and
    ``meta.json``. Skipping it in HydraNet alone would leave a config whose recorded
    lineage claims a head the checkpoint does not have.

    Only ``None`` removes. An empty dict is a different thing -- a head declared with no
    settings -- and stays an error about a missing ``type``, which is what it is.
    """
    heads = (cfg.get("model") or {}).get("heads")
    if not isinstance(heads, dict):
        return []
    disabled = [name for name, hcfg in heads.items() if hcfg is None]
    for name in disabled:
        del heads[name]
    return disabled


def load_config(
    path: str | Path, overrides: list[str] | None = None, validate: bool = True
) -> Config:
    """Load a YAML config, resolving ``_base_`` inheritance and dot-path overrides.

    ``overrides`` looks like ``["train.lr=1e-4", "model.neck.name=fpn"]``.

    Validation runs after the overrides, because a typo on the command line is exactly
    as expensive as one in the file: the setting silently falls back to its default and
    the run looks legitimate. Pass ``validate=False`` only to inspect a broken config.

    Inheritance exists because the three shipped configs were 390 lines of which the
    optimiser, schedule and export blocks were repeated verbatim three times. The cost of
    that is not the duplication, it is that changing ``warmup_iters`` in two files out of
    three produces an experiment whose difference nobody intended. What a config file
    should show is what makes this run *different*; the rest belongs in a base.

    Note the run's ``meta.json`` records the fully merged result, so lineage is unaffected
    by how the file was assembled.
    """
    cfg = Config(_load_with_bases(Path(path)))
    for ov in overrides or []:
        key, sep, val = ov.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value, got: {ov}")
        cfg.set_path(key.strip(), val.strip())
    _drop_disabled_heads(cfg)
    if validate:
        check_config(cfg)
    return cfg

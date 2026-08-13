"""Config validation.

The most expensive mistake in a training pipeline is a setting that never took effect:
a mistyped key falls back to its default, the run completes, and the result looks like
a genuine negative. Every key is therefore declared here, and anything unrecognised
stops the run before a GPU-hour is spent.

Kept dependency-free on purpose -- this package is installed on robots, and a schema
library is not worth a wheel on a Jetson. Every problem in the file is reported at
once, so one round trip fixes them all.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any

from .data.label_maps import SCHEMES

NUMBER = (int, float)


@dataclass(frozen=True)
class Spec:
    types: tuple[type, ...] = (object,)
    required: bool = False
    choices: tuple[Any, ...] | None = None
    note: str = ""


@dataclass
class _Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ConfigError(ValueError):
    """Raised with every problem found in the config, not just the first."""


ROOT = {
    "experiment": Spec((str,), required=True),
    "output_dir": Spec((str,), required=True),
    "seed": Spec((int,)),
    "device": Spec((str, type(None)), choices=("cuda", "mps", "cpu", None)),
    "model": Spec((dict,), required=True),
    "data": Spec((dict,), required=True),
    "train": Spec((dict,), required=True),
    "export": Spec((dict,)),
}

BACKBONE = {
    "name": Spec((str,), required=True),
    "pretrained": Spec((bool,)),
    "frozen_stages": Spec((int,)),
}

NECK = {
    "name": Spec((str,), choices=("bifpn", "fpn")),
    "out_channels": Spec((int,)),
    "num_repeats": Spec((int,)),
    "num_levels": Spec((int,)),
}

MODEL = {
    "backbone": Spec((dict,), required=True),
    "neck": Spec((dict,), required=True),
    "heads": Spec((dict,), required=True),
    "loss_balancing": Spec((str,), choices=("uncertainty", "fixed")),
    "fixed_weights": Spec((dict,)),
}

HEAD_COMMON = {
    "type": Spec((str,), required=True, choices=("semantic_fpn", "fcos")),
    "num_classes": Spec((int,), required=True),
    "in_levels": Spec((list,)),
    "channels": Spec((int,)),
    "loss": Spec((dict,)),
}
HEAD_BY_TYPE = {
    "semantic_fpn": {**HEAD_COMMON, "dropout": Spec(NUMBER)},
    "fcos": {**HEAD_COMMON, "num_convs": Spec((int,)), "strides": Spec((list,))},
}
LOSS_BY_TYPE = {
    "semantic_fpn": {
        "ce_weight": Spec(NUMBER),
        "dice_weight": Spec(NUMBER),
        "ignore_index": Spec((int,)),
    },
    "fcos": {
        "cls_weight": Spec(NUMBER),
        "reg_weight": Spec(NUMBER),
        "centerness_weight": Spec(NUMBER),
    },
}

DATA = {
    "input_size": Spec((list,), required=True),
    "letterbox": Spec((bool,)),
    "terrain_classes": Spec((list,)),
    "datasets": Spec((list,), required=True),
    "workers": Spec((int,)),
}

DATASET = {
    "name": Spec((str,), required=True),
    "type": Spec((str,), required=True, choices=("seg_folder", "coco")),
    "root": Spec((str,), required=True),
    "split_train": Spec((str,), required=True),
    "split_val": Spec((str,), required=True),
    "supervises": Spec((list,), required=True),
    "label_map": Spec((str,)),
    "label_format": Spec((str,), choices=("auto", "color", "id", "rugd_color")),
    "sample_ratio": Spec(NUMBER),
}

TRAIN = {
    "epochs": Spec((int,), required=True),
    "batch_size": Spec((int,), required=True),
    "optimizer": Spec((str,), choices=("adamw", "sgd")),
    "lr": Spec(NUMBER, required=True),
    "backbone_lr_mult": Spec(NUMBER),
    "weight_decay": Spec(NUMBER),
    "warmup_iters": Spec((int,)),
    # Declared so the key is legal and checked, but only cosine is implemented; a
    # config asking for anything else used to be accepted and silently ignored.
    "scheduler": Spec((str,), choices=("cosine",)),
    "amp": Spec((bool,)),
    "grad_clip": Spec(NUMBER),
    "ema": Spec((bool,)),
    "ema_decay": Spec(NUMBER),
    "log_interval": Spec((int,)),
    "val_interval": Spec((int,)),
    "primary_metric": Spec((str,)),
}

EXPORT = {
    "opset": Spec((int,)),
    "input_size": Spec((list,)),
}


def _type_name(types: tuple[type, ...]) -> str:
    return "/".join(t.__name__ for t in types)


def _check_section(rep: _Report, node: Any, spec: dict[str, Spec], path: str) -> None:
    if not isinstance(node, dict):
        rep.errors.append(f"{path}: expected a mapping, got {type(node).__name__}")
        return
    for key, value in node.items():
        if key not in spec:
            hint = difflib.get_close_matches(str(key), spec, n=1)
            suggestion = f" (did you mean {hint[0]!r}?)" if hint else ""
            rep.errors.append(
                f"{path}.{key}: unknown setting{suggestion}. Known: {', '.join(sorted(spec))}"
            )
            continue
        s = spec[key]
        if s.types != (object,) and not isinstance(value, s.types):
            # bool is an int subclass; catching it here keeps amp: 1 from passing.
            rep.errors.append(
                f"{path}.{key}: expected {_type_name(s.types)}, "
                f"got {type(value).__name__} ({value!r})"
            )
            continue
        if isinstance(value, bool) and bool not in s.types and s.types != (object,):
            rep.errors.append(f"{path}.{key}: expected {_type_name(s.types)}, got bool")
            continue
        if s.choices is not None and value not in s.choices:
            rep.errors.append(
                f"{path}.{key}: {value!r} is not one of {', '.join(repr(c) for c in s.choices)}"
            )
    for key, s in spec.items():
        if s.required and key not in node:
            rep.errors.append(f"{path}.{key}: required setting is missing")


def _check_size(rep: _Report, value: Any, path: str) -> None:
    if not (isinstance(value, list) and len(value) == 2):
        rep.errors.append(f"{path}: expected [height, width]")
        return
    if not all(isinstance(v, int) and v > 0 for v in value):
        rep.errors.append(f"{path}: expected two positive integers, got {value!r}")


def _check_heads(rep: _Report, heads: Any) -> None:
    if not isinstance(heads, dict) or not heads:
        rep.errors.append("model.heads: expected at least one head")
        return
    for name, hcfg in heads.items():
        path = f"model.heads.{name}"
        if not isinstance(hcfg, dict) or "type" not in hcfg:
            rep.errors.append(f"{path}.type: required setting is missing")
            continue
        htype = hcfg["type"]
        if htype not in HEAD_BY_TYPE:
            rep.errors.append(f"{path}.type: {htype!r} is not one of {', '.join(HEAD_BY_TYPE)}")
            continue
        _check_section(rep, hcfg, HEAD_BY_TYPE[htype], path)
        if isinstance(hcfg.get("loss"), dict):
            _check_section(rep, hcfg["loss"], LOSS_BY_TYPE[htype], f"{path}.loss")


def _check_datasets(rep: _Report, dcfg: dict, head_names: set[str]) -> None:
    datasets = dcfg.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        rep.errors.append("data.datasets: expected at least one dataset")
        return
    supervised: set[str] = set()
    seen: set[str] = set()
    for i, ds in enumerate(datasets):
        path = f"data.datasets[{i}]"
        _check_section(rep, ds, DATASET, path)
        if not isinstance(ds, dict):
            continue
        name = ds.get("name")
        if name in seen:
            rep.errors.append(f"{path}.name: {name!r} is used by an earlier dataset")
        seen.add(name)
        for head in ds.get("supervises", []) or []:
            # A typo here is invisible at runtime: the head simply never gets a loss.
            if head not in head_names:
                rep.errors.append(
                    f"{path}.supervises: {head!r} is not a declared head. "
                    f"Declared: {', '.join(sorted(head_names))}"
                )
            supervised.add(head)
        label_map = ds.get("label_map")
        if label_map is not None and label_map not in SCHEMES:
            rep.errors.append(
                f"{path}.label_map: {label_map!r} is not a known scheme. "
                f"Available: {', '.join(sorted(SCHEMES))}"
            )
        ratio = ds.get("sample_ratio")
        if isinstance(ratio, NUMBER) and ratio <= 0:
            rep.errors.append(f"{path}.sample_ratio: must be > 0, got {ratio}")

    for head in sorted(head_names - supervised):
        rep.warnings.append(
            f"head {head!r} is not supervised by any dataset: it will be built, "
            f"exported and left at its initial weights"
        )


def _check_class_counts(rep: _Report, cfg: dict) -> None:
    """data.terrain_classes names the terrain head's outputs; a mismatch means every
    per-class metric is labelled with the wrong class."""
    names = cfg.get("data", {}).get("terrain_classes")
    head = cfg.get("model", {}).get("heads", {}).get("terrain")
    if not isinstance(names, list) or not isinstance(head, dict):
        return
    n = head.get("num_classes")
    if isinstance(n, int) and n != len(names):
        rep.errors.append(
            f"data.terrain_classes has {len(names)} entries but "
            f"model.heads.terrain.num_classes is {n}"
        )


def check_config(cfg: dict) -> list[str]:
    """Validate a config. Returns non-fatal warnings; raises ConfigError otherwise."""
    rep = _Report()
    _check_section(rep, cfg, ROOT, "config")

    model = cfg.get("model")
    if isinstance(model, dict):
        _check_section(rep, model, MODEL, "model")
        if isinstance(model.get("backbone"), dict):
            _check_section(rep, model["backbone"], BACKBONE, "model.backbone")
        if isinstance(model.get("neck"), dict):
            _check_section(rep, model["neck"], NECK, "model.neck")
        _check_heads(rep, model.get("heads"))

    data = cfg.get("data")
    if isinstance(data, dict):
        _check_section(rep, data, DATA, "data")
        _check_size(rep, data.get("input_size"), "data.input_size")
        heads = model.get("heads", {}) if isinstance(model, dict) else {}
        _check_datasets(rep, data, set(heads) if isinstance(heads, dict) else set())

    if isinstance(cfg.get("train"), dict):
        _check_section(rep, cfg["train"], TRAIN, "train")
    if isinstance(cfg.get("export"), dict):
        _check_section(rep, cfg["export"], EXPORT, "export")
        if "input_size" in cfg["export"]:
            _check_size(rep, cfg["export"]["input_size"], "export.input_size")

    _check_class_counts(rep, cfg)

    if rep.errors:
        listing = "\n".join(f"  - {e}" for e in rep.errors)
        raise ConfigError(f"{len(rep.errors)} problem(s) in the config:\n{listing}")
    return rep.warnings

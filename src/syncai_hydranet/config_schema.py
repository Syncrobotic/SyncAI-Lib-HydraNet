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
    # Augmentation strength is part of what produced a checkpoint, so it is declared
    # here and therefore recorded in meta.json. Omitted keys fall back to
    # transforms.AUGMENT_DEFAULTS, which are the values this project trained on before
    # they were configurable.
    "augment": Spec((dict,)),
    "terrain_classes": Spec((list,)),
    "datasets": Spec((list,), required=True),
    "workers": Spec((int,)),
}

AUGMENT = {
    "scale_range": Spec((list, tuple)),
    "flip_p": Spec(NUMBER),
    "brightness": Spec(NUMBER),
    "contrast": Spec(NUMBER),
    "saturation": Spec(NUMBER),
}

DATASET = {
    "name": Spec((str,), required=True),
    "type": Spec((str,), required=True, choices=("seg_folder", "coco")),
    "root": Spec((str,), required=True),
    "split_train": Spec((str,), required=True),
    # Not required: a dataset may be trained on without joining checkpoint selection.
    # The evaluator accumulates one confusion matrix per head across every val set, so
    # listing a second domain here puts it into the number that picks best.pt -- and a
    # split that selects the answer cannot also measure it. Trainer raises if *no*
    # dataset declares one.
    "split_val": Spec((str,)),
    # Optional and unset by default: a held-out split only means anything if
    # nothing in training reads it, so it must be created deliberately.
    "split_test": Spec((str,)),
    "supervises": Spec((list,), required=True),
    "label_map": Spec((str,)),
    "label_format": Spec((str,), choices=("auto", "color", "id", "rugd_color")),
    "sample_ratio": Spec(NUMBER),
    # COCO only: train the head on this subset of category names instead of all 80.
    # The head's num_classes must match the length, and the check below enforces it --
    # a mismatch trains every box against the wrong channel and still converges.
    "classes": Spec((list,)),
    # COCO only, evaluation: score mAP over these categories while leaving the head and
    # the label mapping alone. Separate from `classes` because they are different jobs:
    # narrowing what a head learns changes its output space, narrowing what is scored
    # does not. Using `classes` for scoring would renumber the labels under a checkpoint
    # that was trained with the old numbering, and report a confident wrong number.
    "score_classes": Spec((list,)),
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
    "amp_dtype": Spec((str,), choices=("float16", "bfloat16")),
    "grad_accum_steps": Spec((int,)),
    # Backend knobs. deterministic trades a real speed penalty for bit-reproducibility
    # and so is off unless asked for.
    "deterministic": Spec((bool,)),
    "cudnn_benchmark": Spec((bool,)),
    "tf32": Spec((bool,)),
    # NHWC weights and activations. Off by default so that resuming an in-flight run
    # after a code change does not silently move it to a different kernel set.
    "channels_last": Spec((bool,)),
    # torch.compile. Off by default, and the reason is not caution about correctness:
    # this project's step is small (batch 8 at 512x640 measured 46-63% GPU on an RTX
    # PRO 6000, host-bound rather than compute-bound), which is exactly the regime
    # where graph capture pays off -- and also exactly the regime where a first-epoch
    # compile of several minutes shows up as a mysterious stall. Opt in per run, and
    # expect the gain to shrink as batch size rises.
    #
    # It is a *training* knob only. Export goes through torch.onnx with dynamo=False
    # and must see the eager module, and EMA deep-copies the model before any of this.
    "compile": Spec((bool,)),
    "compile_mode": Spec((str,), choices=("default", "reduce-overhead", "max-autotune")),
    "grad_clip": Spec(NUMBER),
    "ema": Spec((bool,)),
    "ema_decay": Spec(NUMBER),
    "ema_warmup_steps": Spec((int,)),
    "log_interval": Spec((int,)),
    "val_interval": Spec((int,)),
    # Score the detection head every Nth validation instead of every one. 1 keeps the
    # old behaviour. The last epoch always scores it whatever this says, so a run never
    # ends reporting an mAP from several epochs earlier.
    "detection_val_interval": Spec((int,)),
    # Stop after this many validations with no new best `primary_metric`. 0 disables.
    "early_stop_patience": Spec((int,)),
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


def unsupervised_heads(cfg: dict) -> set[str]:
    """Heads that no dataset supervises, so training never gives them a loss.

    Such a head is still built, still exported and still emits numbers at inference --
    they are just its initial random weights. Training only warns, because an
    unsupervised head is legitimate while a dataset is still being assembled. Export is
    where it stops being legitimate, so ``hydranet-export-onnx`` refuses on it.
    """
    heads = (cfg.get("model") or {}).get("heads")
    if not isinstance(heads, dict):
        return set()
    supervised: set[str] = set()
    for ds in (cfg.get("data") or {}).get("datasets") or []:
        if isinstance(ds, dict):
            supervised.update(ds.get("supervises") or [])
    return set(heads) - supervised


def unsourced_terrain_classes(cfg: dict) -> dict[str, tuple[str, ...]]:
    """Terrain classes this config declares that no dataset in it can ever produce.

    The config-time counterpart to ``unsupervised_heads``. That one catches a whole head
    with no gradient; this one catches a single *channel* with no gradient, which is the
    quieter of the two and has cost this project more. ``wet_slippery``, ``floor_metal``
    and ``threshold_ramp`` have been in the retail taxonomy since the beginning, are all
    still IoU 0.000, and every run that trained them looked completely normal --
    ``label_maps_retail_objects.unsourced_classes`` says as much in its docstring, and
    then only a test called it. An empty output channel is invisible at training time:
    the loss falls, the mean is taken over the classes that do have data, and the run
    reports a plausible number for sixty epochs.

    Returns ``{scheme names joined: classes}``, keyed per taxonomy because a config may
    legitimately mix schemes that do not share one. Datasets are grouped by their
    scheme's ``classes`` tuple and their mappings unioned, since two datasets under one
    taxonomy supply it together -- ``hydranet_retail_cctv`` pairs ADE20K with site masks
    precisely so that each covers what the other cannot.

    What it cannot do, and the reason it warns rather than raises: a *native* scheme is
    an identity map, so it claims every id by construction. That is the truth about
    expressibility -- a site annotator may draw any class in the taxonomy -- but not
    about the pixels on disk. "The scheme allows `product` and the export contains none"
    is a count over the masks, not a fact about the config, and nothing here can see it.

    The stronger version of that limit, measured, so nobody reads this gate as more than
    it is: `column` has an ADE20K source, passes this check, and scored val IoU 0.40-0.51
    on the 60-epoch retail-objects run -- and predicts **0.00% of pixels** across 240
    frames of four daytime shop-floor cameras, including one with a clad pillar bearing a
    shop sign dead centre of frame, which it calls `wall`. A sourced class can still be
    identically absent in deployment. Only site evaluation catches that; a config cannot.
    """
    by_taxonomy: dict[tuple, list] = {}
    for ds in (cfg.get("data") or {}).get("datasets") or []:
        if not isinstance(ds, dict):
            continue
        scheme = SCHEMES.get(ds.get("label_map"))
        if scheme is not None:
            by_taxonomy.setdefault(scheme.classes, []).append(scheme)

    out: dict[str, tuple[str, ...]] = {}
    for classes, schemes in by_taxonomy.items():
        produced = {v for s in schemes for v in s.mapping.values() if v != 255}
        # Class 0 is `void` in every scheme and is not a class anyone trains, so a
        # mapping that never emits it is correct rather than incomplete.
        missing = tuple(name for tid, name in enumerate(classes) if tid and tid not in produced)
        if missing:
            names = "+".join(sorted({s.name for s in schemes}))
            out[names] = missing
    return out


def _check_fixed_weights(rep: _Report, cfg: dict) -> None:
    """``fixed_weights`` keys have to name declared heads.

    ``FixedWeighting.forward`` does ``self.weights.get(name, 1.0)``, so a key naming no
    head is dropped and a head named by no key silently trains at 1.0. Neither shows up
    anywhere: the loss falls, the run completes, and the reweighting the config asked for
    never happened. That is this file's opening paragraph -- a setting that never took
    effect -- and ``data.datasets[].supervises`` is already checked against the declared
    heads for exactly the same reason. This closes the other half.

    Checked only under ``loss_balancing: fixed``, and that restriction is the point.
    ``fixed_weights`` is set once in ``_base/hydranet.yaml`` for all three heads, so a
    derived config that removes a head -- ``hydranet_retail_objects.yaml`` sets
    ``traversability: null`` -- inherits a weight naming a head it does not have. Under
    ``uncertainty`` the table is never read, so that stray key is the *normal* result of
    a correct config and warning about it would train people to ignore warnings. Under
    ``fixed`` the same key means a number the author wrote is silently discarded, which
    is worth refusing the run over.
    """
    model = cfg.get("model")
    if not isinstance(model, dict) or model.get("loss_balancing") != "fixed":
        return
    if not isinstance(model.get("fixed_weights"), dict):
        return
    heads = model.get("heads")
    if not isinstance(heads, dict):
        return
    weights = model["fixed_weights"]
    for name in sorted(set(weights) - set(heads)):
        rep.errors.append(
            f"model.fixed_weights.{name}: names no declared head, so FixedWeighting "
            f"drops it and the weight never applies. Declared: {', '.join(sorted(heads))}"
        )
    for name in sorted(set(heads) - set(weights)):
        rep.warnings.append(
            f"model.fixed_weights: head {name!r} has no weight and will train at 1.0, "
            f"because FixedWeighting defaults rather than raising. State it explicitly "
            f"if that is what you want."
        )


def _check_unsourced_classes(rep: _Report, cfg: dict) -> None:
    for scheme_names, missing in sorted(unsourced_terrain_classes(cfg).items()):
        rep.warnings.append(
            f"label_map {scheme_names!r} can never produce {', '.join(missing)}: "
            f"{'they are' if len(missing) > 1 else 'it is'} an empty output channel, "
            f"trained on nothing and reported as IoU 0.000. Add a dataset that supplies "
            f"{'them' if len(missing) > 1 else 'it'}, or drop "
            f"{'the classes' if len(missing) > 1 else 'the class'} from the taxonomy."
        )


def _check_one_dataset(rep: _Report, ds: dict, path: str, head_names: set[str]) -> set[str]:
    """Everything checkable about one dataset entry. Returns the heads it claims to train.

    All four failures here are silent at runtime, which is why they are worth a schema
    pass at all: an empty `supervises` reaches compute_losses with no loss to build, a
    misspelled head simply never gets one, an unknown `label_map` would resolve to
    nothing, and a non-positive `sample_ratio` removes the dataset from the mix without
    removing it from the config.
    """
    if isinstance(ds.get("supervises"), list) and not ds["supervises"]:
        # Every batch this dataset contributes reaches compute_losses with no loss to
        # build, which is a crash rather than a slow run -- and it is caught here so it
        # happens before the first step rather than during it.
        rep.errors.append(
            f"{path}.supervises: empty. A dataset that supervises no head costs "
            f"optimiser steps and contributes no gradient to anything; drop the "
            f"dataset, or name the heads its labels train."
        )
    supervised = set()
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
    return supervised


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
        supervised |= _check_one_dataset(rep, ds, path, head_names)

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


def _check_detection_subset(rep: _Report, cfg: dict) -> None:
    """A COCO subset has to match the detection head's width.

    Nothing downstream would complain: the head keeps 80 channels, every box trains
    against the channel its contiguous index happens to land on, and the loss falls.
    The run only looks wrong at the point someone reads a class name off a prediction.
    """
    head = cfg.get("model", {}).get("heads", {}).get("detection")
    if not isinstance(head, dict) or not isinstance(head.get("num_classes"), int):
        return
    for ds in cfg.get("data", {}).get("datasets") or []:
        if not isinstance(ds, dict) or ds.get("type") != "coco":
            continue
        subset = ds.get("classes")
        if isinstance(subset, list) and len(subset) != head["num_classes"]:
            rep.errors.append(
                f"dataset {ds.get('name')!r} lists {len(subset)} classes but "
                f"model.heads.detection.num_classes is {head['num_classes']}"
            )
        if isinstance(subset, list) and len(subset) != len(set(subset)):
            rep.errors.append(f"dataset {ds.get('name')!r}: duplicate entries in classes")


def _check_channels_last(rep: _Report, cfg: dict) -> None:
    """NHWC pays only under autocast.

    Measured on an RTX PRO 6000, batch 48 at 512x640: bf16 153.2 -> 121.1 ms (-21%),
    fp32 219.6 -> 220.9 ms (+0.6%). NHWC is the layout tensor cores want; fp32
    convolutions run on CUDA cores and are indifferent. Asking
    for it without AMP is not an error, it just buys nothing, and someone reading a
    benchmark later deserves to know which of the two knobs did the work.
    """
    train = cfg.get("train")
    if not isinstance(train, dict) or not train.get("channels_last"):
        return
    if not train.get("amp", True):
        rep.warnings.append(
            "train.channels_last is set but train.amp is false: NHWC was measured at "
            "+0.6% (i.e. nothing) without autocast, because fp32 convolutions do not "
            "use tensor cores"
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
        if isinstance(data.get("augment"), dict):
            _check_section(rep, data["augment"], AUGMENT, "data.augment")
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
    _check_detection_subset(rep, cfg)
    _check_channels_last(rep, cfg)
    _check_fixed_weights(rep, cfg)
    _check_unsourced_classes(rep, cfg)

    if rep.errors:
        listing = "\n".join(f"  - {e}" for e in rep.errors)
        raise ConfigError(f"{len(rep.errors)} problem(s) in the config:\n{listing}")
    return rep.warnings

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
from .data.label_maps_retail_security import get_det_vocab
from .labels import IGNORE

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
    "type": Spec(
        (str,), required=True, choices=("semantic_fpn", "fcos", "depth_fpn", "pose_p3")
    ),
    "num_classes": Spec((int,), required=True),
    "in_levels": Spec((list,)),
    "channels": Spec((int,)),
    "loss": Spec((dict,)),
}
# `num_classes` is required in HEAD_COMMON because both original head families are
# classifiers. Depth is a regressor with exactly one output channel, so requiring it would
# force every depth config to carry a `num_classes: 1` that means nothing and that
# `_check_class_counts`-style checks would then have to learn to ignore. Dropped rather
# than defaulted, so a config that sets it is told it is meaningless instead of obeyed.
HEAD_DEPTH_COMMON = {k: v for k, v in HEAD_COMMON.items() if k != "num_classes"}
HEAD_BY_TYPE = {
    "semantic_fpn": {**HEAD_COMMON, "dropout": Spec(NUMBER)},
    "depth_fpn": {
        **HEAD_DEPTH_COMMON,
        "dropout": Spec(NUMBER),
        # The ceiling in metres, on both the head's output clamp and the loss's validity
        # predicate. Indoor default 10.0 matches NYUv2's evaluation cap, so a run's
        # numbers are comparable to published ones without a footnote.
        "max_depth": Spec(NUMBER),
    },
    "fcos": {
        **HEAD_COMMON,
        "num_convs": Spec((int,)),
        "strides": Spec((list,)),
        # `linear` is the shipped classifier: one learned vector per class, class list
        # frozen into the weights. `text_embedding` scores against a matrix of class-name
        # embeddings, so the vocabulary is data a config points at rather than a retrain --
        # which is the only available answer to the cam08 audit's 1,683 `book` and no
        # `laptop`. See models/heads/text_classifier.py.
        "cls_head": Spec((str,), choices=("linear", "text_embedding")),
        # The text encoder's dimension. Ignored under `linear`, and deliberately not
        # defaulted per encoder: a matrix whose width disagrees with the head is refused at
        # load rather than broadcast, because a silent mismatch renames every class.
        "embed_dim": Spec((int,)),
        # Path to the matrix `scripts/make_text_embeddings.py` writes.
        #
        # **This belongs to the training run, not to export.** The buffer ships as a random
        # orthogonal placeholder and `embed_pred` learns to align with whatever matrix was
        # installed while the gradient flowed. Train on the placeholder and install real
        # embeddings afterwards and the visual projection is aimed at random directions --
        # confident output, no error. So the config that trains the head is the config that
        # names the matrix.
        "text_embeddings": Spec((str,)),
        # The class names this head's channels mean, in head order. Optional, and the only
        # thing that can catch a right-sized wrong-ordered matrix: `load_text_embeddings`
        # checks the shape, and a permuted matrix has the correct shape. Same argument as
        # `coco_subsets.head_order`, one level down.
        "classes": Spec((list,)),
    },
    "pose_p3": {
        **HEAD_COMMON,
        # num_classes is the keypoint count (17): the field is common and required, and
        # the keypoint count is the honest value for "how many output channels".
        "num_convs": Spec((int,)),
        # Gaussian radius in stride-8 cells, and the teacher-confidence floor below
        # which a keypoint renders no target at all (heads/pose.py).
        "sigma_cells": Spec(NUMBER),
        "teacher_min_conf": Spec(NUMBER),
    },
}
LOSS_BY_TYPE = {
    "semantic_fpn": {
        "ce_weight": Spec(NUMBER),
        "dice_weight": Spec(NUMBER),
        "ignore_index": Spec((int,)),
        # Per-class multipliers on the CE half, one per class in id order. Off unless
        # given, because passing them changes what a checkpoint means and a default here
        # would silently reinterpret every run that came before.
        #
        # What it is for, measured on batch02 as agreement with SAM 3 over six held-out
        # cameras: `fixture` reached 92.6% recall at 56.1% precision while every other
        # class sat at 91-96% precision and 21-69% recall. One class absorbing and the
        # rest under-firing is frequency bias, and `dice_weight` 1.5 did not settle it --
        # Dice is already class-balanced, and the unweighted CE beside it is what pulls.
        #
        # The length is checked against the head's num_classes rather than broadcast: a
        # short list would reweight the wrong classes and still train.
        "class_weights": Spec((list,)),
    },
    "fcos": {
        "cls_weight": Spec(NUMBER),
        "reg_weight": Spec(NUMBER),
        "centerness_weight": Spec(NUMBER),
    },
    "pose_p3": {},
    "depth_fpn": {
        # SILog's scale-invariance dial. 1.0 makes the loss blind to absolute scale, which
        # is exactly the failure measured in the public teacher (a flat 15% over-prediction
        # with sound geometry), so the default sits at 0.85 and a config that raises it to
        # 1.0 is opting out of metric supervision on purpose.
        "lam": Spec(NUMBER),
        "alpha": Spec(NUMBER),
        # Deliberately no `ignore_index`. `_check_ignore_index` requires 255 wherever the
        # key appears, and 255 is the 8-bit mask-file contract for a class id. Depth is
        # continuous metres and its no-return is 0.0, which validity handles as a
        # predicate; a sentinel here would be the same number meaning something else.
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
    "type": Spec(
        (str,),
        required=True,
        choices=("seg_folder", "coco", "nyu_depth", "rendered_depth", "pose_keypoints"),
    ),
    "root": Spec((str,), required=True),
    "split_train": Spec((str,), required=True),
    # Not required: a dataset may be trained on without joining checkpoint selection.
    # The evaluator accumulates one confusion matrix per head across every val set, so
    # listing a second domain here puts it into the number that picks best.pt -- and a
    # split that selects the answer cannot also measure it. Trainer raises if *no*
    # dataset declares one.
    # nyu_depth only: which head this dataset's depth target is addressed to. The target
    # dict is keyed by head name, so a config that renames the head must say so here or
    # the head is declared supervised and receives nothing.
    "head_name": Spec((str,)),
    # nyu_depth only: returns beyond this are zeroed as invalid rather than clipped.
    # Must match the head's `max_depth` or the dataset masks away depths the head is
    # still being asked to predict.
    "max_depth": Spec(NUMBER),
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
    # COCO only: share one head with another detection dataset by mapping both files'
    # category *names* into one vocabulary, instead of numbering each file's categories
    # from zero. Without it two detection sources both emit label 0 and one head is asked
    # to mean `person` and `boxed_stock` at once. See data/label_maps_retail_security.py.
    "det_vocab": Spec((str,)),
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
    out: dict[str, tuple[str, ...]] = {}
    for classes, members in _taxonomy_groups(cfg).items():
        produced = {v for _n, s, _r in members for v in s.mapping.values() if v != IGNORE}
        # Class 0 is `void` in every scheme and is not a class anyone trains, so a
        # mapping that never emits it is correct rather than incomplete.
        missing = tuple(name for tid, name in enumerate(classes) if tid and tid not in produced)
        if missing:
            out["+".join(sorted({s.name for _n, s, _r in members}))] = missing
    return out


def _taxonomy_groups(cfg: dict) -> dict[tuple, list[tuple[str, Any, float]]]:
    """Segmentation datasets grouped by the taxonomy they label into.

    ``(dataset name, scheme, sample_ratio)`` per member. Datasets with no ``label_map``
    are omitted deliberately: a COCO detection set contributes no segmentation target at
    all, so it neither supplies a terrain class nor supplies a negative for one, and
    counting it either way would misdescribe the mix.
    """
    groups: dict[tuple, list[tuple[str, Any, float]]] = {}
    for ds in (cfg.get("data") or {}).get("datasets") or []:
        if not isinstance(ds, dict):
            continue
        scheme = SCHEMES.get(ds.get("label_map"))
        if scheme is None:
            continue
        ratio = ds.get("sample_ratio")
        groups.setdefault(scheme.classes, []).append(
            (str(ds.get("name")), scheme, float(ratio) if isinstance(ratio, NUMBER) else 1.0)
        )
    return groups


def minority_sourced_terrain_classes(cfg: dict) -> dict[str, tuple[list, list]]:
    """Classes some segmentation dataset can produce and another cannot.

    ``unsourced_terrain_classes`` above asks "can *any* dataset produce this class" -- a
    yes/no over the union. That is the right question only while the answer is no. The
    moment one dataset supplies a class and a larger one cannot, the union says yes and
    goes quiet, and a harder failure begins.

    **A class absent from the dominant dataset is suppressed, not merely unlearned**, and
    the difference is sign rather than dilution. Measured on the retail-objects site run:
    ADE20K was 90.2% of segmentation steps and contains zero `product` pixels, so in nine
    batches out of ten every pixel was a *negative* for that channel. `product` sat at
    0.000 for 22 epochs while the run looked entirely normal. The gate above had said
    "`ade20k_retail_objects` can never produce product" all along and stopped saying it
    the instant a 5% site dataset was added -- which is exactly when it started to matter.

    Returns ``{class name: (producers, non_producers)}`` where each side is a list of
    ``(dataset name, sample_ratio, from_identity_map)``.

    **The producing side is a claim, and the flag is how far it can be trusted.** This
    reasons from ``label_map``, so "can produce" means "the scheme can express this id",
    which is not "the masks contain this class". An *identity* map -- ``retail_native``,
    ``indoor_native``, ``retail_objects_native`` -- expresses every class in its taxonomy
    by construction, so it claims all of them and evidences none.

    That distinction cost a wrong conclusion on 2026-08-17 and the correction is worth
    stating here rather than in a commit. This function reported `floor_metal`,
    `wet_slippery` and `threshold_ramp` as merely outvoted in ``hydranet_retail_cctv``,
    contradicting a months-old finding that they need annotation. Counting the pixels:
    **zero, in 0 of 408 masks.** `site_cctv_pseudo` is an identity map over pseudo-labels
    from a model that scores 0.000 on those three -- so it cannot contain them, and the
    months-old finding was right. The *lacking* side is sound (an explicit map that never
    emits an id genuinely cannot produce it); only the producing side needs the count.

    **What it deliberately does not compute: the step share.** `sample_ratio` is in the
    config; the number of images behind it is not, so a percentage derived from ratios
    alone would be an invented figure with a confident shape. The real share is known in
    ``MultiTaskLoader``, after the datasets are built, and belongs there. This reports the
    partition and the declared ratios and points at the measurement.
    """
    out: dict[str, tuple[list, list]] = {}
    for classes, members in _taxonomy_groups(cfg).items():
        if len(members) < 2:
            continue  # one dataset cannot be a minority within its own taxonomy
        for tid, name in enumerate(classes):
            if not tid:
                continue
            produces = [
                (n, r, _is_identity(s)) for n, s, r in members if tid in s.mapping.values()
            ]
            lacks = [
                (n, r, _is_identity(s)) for n, s, r in members if tid not in s.mapping.values()
            ]
            if produces and lacks:
                out[name] = (produces, lacks)
    return out


def _is_identity(scheme: Any) -> bool:
    """A scheme whose mapping is ``{id: id}`` claims every class and evidences none."""
    return all(k == v for k, v in scheme.mapping.items() if v != IGNORE)


# Deliberately NOT reported here: which class the dominant dataset gives those pixels
# instead. `274a4a08`'s mechanism for `product` is sharper than "absent" and is correct --
# `ADE20K_ID_TO_RETAIL_OBJECTS` sends 15 source ids to `fixture` and 0 to `product`, so a
# shelf of goods is one `fixture` region under ADE20K and a `fixture` + `product` split
# under the site masks: the same pixels, two contradictory targets, and the louder dataset
# wins. It also predicts something the "absent" reading does not, that `fixture` comes out
# inflated rather than merely unharmed.
#
# It was implemented here and removed, because the only config-time proxy for "where do
# those pixels go" is the source-id count, and that proxy does not support the claim. Under
# `ade20k_retail` the class with the most ids is `obstacle_furniture` at 45 -- a catch-all
# -- so the check would have announced that `floor_metal`'s pixels are labelled
# `obstacle_furniture`, which is nonsense: `floor_metal` is a floor surface. The
# product/fixture mechanism is true because someone looked at a shelf, not because 15 was
# the largest number. Asserting a pixel-level mechanism from a scheme-level count is the
# same error this function shipped with once already, and once was enough.


def _check_minority_sourced(rep: _Report, cfg: dict) -> None:
    def _fmt(rows: list) -> str:
        return ", ".join(f"{n} (sample_ratio {r:g})" for n, r, _i in rows)

    for name, (produces, lacks) in sorted(minority_sourced_terrain_classes(cfg).items()):
        unconfirmed = all(ident for _n, _r, ident in produces)
        caveat = (
            " CONFIRM WITH A PIXEL COUNT FIRST: the only dataset claiming this class uses "
            "an identity label map, which expresses every class in the taxonomy whether or "
            "not a single pixel of it was ever drawn. This check reasons from the scheme, "
            "not from the masks."
            if unconfirmed
            else ""
        )
        rep.warnings.append(
            f"class {name!r} is produced only by {_fmt(produces)}; "
            f"{_fmt(lacks)} supplies it as a negative in every batch.{caveat} If the count "
            f"confirms it, the class is suppressed rather than merely unlearned -- `product` "
            f"sat at IoU 0.000 for 22 epochs this way -- and the fix is to lower the abundant "
            f"dataset's sample_ratio rather than raise the scarce one, because raising the "
            f"scarce one reaches the same balance by showing the same few images many times "
            f"an epoch and trades a suppressed channel for a memorised one."
        )


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
        if ds.get("det_vocab"):
            # Under a vocabulary `classes` selects which *source* categories to read, and
            # several of them can share a channel -- `backpack`, `handbag` and `suitcase`
            # are one `bag`. The head's width is the vocabulary's, checked below.
            continue
        if isinstance(subset, list) and len(subset) != head["num_classes"]:
            rep.errors.append(
                f"dataset {ds.get('name')!r} lists {len(subset)} classes but "
                f"model.heads.detection.num_classes is {head['num_classes']}"
            )
        if isinstance(subset, list) and len(subset) != len(set(subset)):
            rep.errors.append(f"dataset {ds.get('name')!r}: duplicate entries in classes")


def _check_detection_vocab(rep: _Report, cfg: dict) -> None:
    """Two detection sources in one config, and the three ways that goes silently wrong.

    1. **No vocabulary at all.** `CocoDetDataset` numbers each file's categories from
       zero, so COCO's `person` and the site's `boxed_stock` both train channel 0. The
       loss falls, the run completes, and the head means two things. This is the failure
       `hydranet_retail_products.yaml` avoided by refusing to train COCO alongside the
       site boxes; a config that puts them together without `det_vocab` is asking for it.
    2. **Two different vocabularies.** Same collision, one layer up.
    3. **A head whose width is not the vocabulary's.** Every box then trains against
       whichever channel its id happens to land on, exactly as `_check_detection_subset`
       describes for a COCO subset.
    """
    heads = cfg.get("model", {}).get("heads", {})
    head = heads.get("detection") if isinstance(heads, dict) else None
    det_sets = [
        ds
        for ds in cfg.get("data", {}).get("datasets") or []
        if isinstance(ds, dict) and "detection" in (ds.get("supervises") or [])
    ]
    vocabs = {ds.get("det_vocab") for ds in det_sets}
    if len(det_sets) > 1 and None in vocabs:
        unnamed = ", ".join(
            sorted(str(ds.get("name")) for ds in det_sets if not ds.get("det_vocab"))
        )
        rep.errors.append(
            f"{len(det_sets)} datasets supervise `detection` and {unnamed} declare no "
            "`det_vocab`: each numbers its own categories from zero, so both train "
            "channel 0 with different meanings. Give every detection dataset the same "
            "det_vocab, or train them in separate runs."
        )
    named = {v for v in vocabs if v}
    if len(named) > 1:
        rep.errors.append(
            f"detection datasets declare more than one det_vocab ({', '.join(sorted(named))}); "
            "one head holds one vocabulary."
        )
    for name in sorted(named):
        try:
            vocab = get_det_vocab(name)
        except ValueError as exc:
            rep.errors.append(str(exc))
            continue
        if (
            isinstance(head, dict)
            and isinstance(head.get("num_classes"), int)
            and head["num_classes"] != len(vocab.classes)
        ):
            rep.errors.append(
                f"det_vocab {name!r} has {len(vocab.classes)} classes "
                f"({', '.join(vocab.classes)}) but model.heads.detection.num_classes "
                f"is {head['num_classes']}"
            )


def _check_detection_head_classes(rep: _Report, cfg: dict) -> None:
    """A detection head's `classes` list has to be as long as the head is wide.

    Caught the day it was introduced, by a renderer refusing rather than by anything in
    training. `hydranet_retail_surfaces.yaml` gained `classes: [boxed_stock, device]` so
    its panels would stop naming boxes from COCO's table; `hydranet_retail_security.yaml`
    inherits from it and widened the head to four without restating the list, and `_base_`
    merges keys rather than replacing the block. So a four-channel head carried two names.

    Training never reads `classes` unless a text-embedding matrix is installed, which is
    what made it invisible: the run converges, the checkpoint is fine, and only the panel
    is wrong -- naming a `person` box `boxed_stock`, which is the shape of error
    `detection_class_names` was written to prevent.
    """
    heads = (cfg.get("model") or {}).get("heads") or {}
    for name, head in heads.items():
        if not isinstance(head, dict) or head.get("type") != "fcos":
            continue
        declared = head.get("classes")
        width = head.get("num_classes")
        if isinstance(declared, list) and isinstance(width, int) and len(declared) != width:
            rep.errors.append(
                f"model.heads.{name}.classes has {len(declared)} names "
                f"({', '.join(map(str, declared))}) for a {width}-channel head. `_base_` "
                f"merges, so a widened head keeps its parent's shorter list and every "
                f"rendered box is named from it."
            )


def _check_detection_val_interval(rep: _Report, cfg: dict) -> None:
    """A detection `primary_metric` and a `detection_val_interval` above 1 kill the run.

    `Trainer.val_subset` drops every val set whose only supervised head is detection on
    the epochs between detection validations, so the metric that selects `best.pt` is not
    produced -- and `select_metric` raises `KeyError` rather than guessing. The run dies at
    the end of epoch 1, after the data is loaded and the GPU is warm, which is the most
    expensive moment to discover a config problem short of finishing.

    It is caught here rather than fixed there because both behaviours are right on their
    own: skipping detection validation is what the interval is *for*, and refusing an
    absent selection metric is what stops a run selecting on the wrong number. Only the
    combination is wrong, and a config is where a combination lives.

    Narrow on purpose. A detection metric whose dataset also supervises a segmentation
    head survives the subset, so this only fires when the metric's source is
    detection-only -- which is the case for every retail config that scores merchandise
    against a boxes-only export.
    """
    train = cfg.get("train")
    if not isinstance(train, dict):
        return
    metric = str(train.get("primary_metric", ""))
    interval = train.get("detection_val_interval", 1)
    if not metric.startswith("detection_") or not isinstance(interval, int) or interval <= 1:
        return
    _, _, source = metric.partition("/")
    for ds in (cfg.get("data") or {}).get("datasets") or []:
        if not isinstance(ds, dict) or (source and ds.get("name") != source):
            continue
        sup = set(ds.get("supervises") or [])
        if sup and sup <= {"detection"} and ds.get("split_val"):
            rep.errors.append(
                f"train.primary_metric={metric!r} is produced by dataset "
                f"{ds.get('name')!r}, which supervises detection only, while "
                f"train.detection_val_interval={interval}. Trainer.val_subset drops that "
                f"val set on the {interval - 1} epochs out of {interval} that skip "
                f"detection validation, so the selection metric is absent and the run "
                f"raises KeyError at the end of epoch 1. Set detection_val_interval to 1, "
                f"or select on a metric that is produced every epoch."
            )


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


def _check_ignore_index(rep: _Report, cfg: dict) -> None:
    """Every head's `ignore_index` must be the mask-file contract, and this raises.

    `labels.IGNORE` is what an annotation PNG carries where the class is unknown, and
    `hydranet-annotation check` refuses a dataset that writes anything else. A loss told
    a different number does not fail: it treats every unlabelled pixel as a trainable
    class, the loss still falls, and the per-class IoUs stay plausible -- the same shape
    of silent failure the annotation gate exists to prevent, one layer further in.

    An error rather than a warning, because there is no legitimate reason to disagree.
    The value is fixed by the file format (single-channel 8-bit PNG) rather than chosen,
    so a config that names another one is a typo or a misunderstanding, and both are
    cheaper to fix here than after a run.
    """
    heads = (cfg.get("model") or {}).get("heads")
    if not isinstance(heads, dict):
        return
    for name, head in heads.items():
        loss = head.get("loss") if isinstance(head, dict) else None
        if not isinstance(loss, dict) or "ignore_index" not in loss:
            continue
        got = loss["ignore_index"]
        if got != IGNORE:
            rep.errors.append(
                f"model.heads.{name}.loss.ignore_index is {got!r}, not {IGNORE}. That is "
                "the value an annotation PNG carries where the class is unknown "
                "(`labels.IGNORE`); a loss told anything else trains on void as if it "
                "were a class and nothing downstream says so."
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
    _check_detection_vocab(rep, cfg)
    _check_detection_val_interval(rep, cfg)
    _check_detection_head_classes(rep, cfg)
    _check_channels_last(rep, cfg)
    _check_fixed_weights(rep, cfg)
    _check_unsourced_classes(rep, cfg)
    _check_minority_sourced(rep, cfg)
    _check_ignore_index(rep, cfg)

    if rep.errors:
        listing = "\n".join(f"  - {e}" for e in rep.errors)
        raise ConfigError(f"{len(rep.errors)} problem(s) in the config:\n{listing}")
    return rep.warnings

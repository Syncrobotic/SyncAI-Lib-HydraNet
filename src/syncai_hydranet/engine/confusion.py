"""One confusion matrix per image, for a finished run scored against a labelled split.

Moved out of `scripts/site_confusion.py`, where `scripts/val_sampling_error.py` reached
it through a `sys.path` insert. The function's own docstring had already named the reason
it must not be copied, which is the same reason it should not have been in a script:

    "the alternative was to copy this loop, whose upsample-before-argmax step is easy to
    get quietly wrong"

`tests/test_scripts_are_not_libraries.py` ratchets on exactly that shape and cites the
case where it already cost this project three seeds: four scripts kept their own copy of
one loop, the copies disagreed about lens correction, and the disagreement changed which
observations the tracker linked. Shared code under `scripts/` is outside the wheel, the
type ratchet and the coverage floor, so the loop two callers depend on is the loop
nothing checks.

The signature changed with the move. It took an argparse `Namespace` and read five
attributes off it, which is a script's calling convention rather than a library's -- a
caller with the five values and no parser had to build a fake namespace to pass them.
They are keyword-only parameters now, so a caller cannot silently swap `dataset` and
`split`, which are both strings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from ..data.label_maps import get_scheme
from ..labels import IGNORE
from ..models.hydranet import build_model
from ..utils.checkpoint import load_checkpoint, select_weights
from ..utils.device import pick_device
from ..utils.visualize import preprocess


def run_config(run: Path) -> dict:
    """Read a finished run's config as a record, without today's validator.

    `load_config` calls `check_config`, which is right for launching a run and wrong for
    reading one: a run from this morning cannot be expected to satisfy a check added this
    afternoon, and refusing to *analyse* it because of that loses the only evidence about
    what it did. `runs/hydranet_retail_security_b03/config.yaml` is exactly that case --
    it carries the two-name `classes` list whose check landed hours after it started.

    A saved run config is already fully merged, so no `_base_` resolution is needed.
    """
    return yaml.safe_load((run / "config.yaml").read_text())


def per_image_confusions(
    run: Path,
    *,
    checkpoint: str,
    weights: str,
    label_map: str,
    dataset: str,
    split: str,
    device: torch.device | str | None = None,
) -> tuple[np.ndarray, list[str], list[Path]]:
    """One confusion matrix per image, un-summed.

    Un-summed because the sum discards what a val set's own sampling error is made of.
    Anything that asks "how much would this number move on a different draw of images" --
    `scripts/val_sampling_error.py` -- needs the per-image terms.

    Returns `(mats, classes, kept)`: an `(N, C, C)` stack indexed `[image, truth,
    prediction]`, the run's terrain class names, and the image paths in the same order as
    the first axis. Images whose annotation PNG is missing are skipped rather than
    counted empty, so `kept` is what `mats` is actually about.
    """
    device = pick_device(device) if not isinstance(device, torch.device) else device
    cfg = run_config(run)
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(str(run / checkpoint)), weights))
    classes = list(cfg["data"]["terrain_classes"])
    mapping = get_scheme(label_map).mapping
    imgs = sorted((Path(dataset) / "images" / split).rglob("*.jpg"))
    if not imgs:
        raise SystemExit(f"no images under {dataset}/images/{split}")
    mats: list[np.ndarray] = []
    kept: list[Path] = []
    for jpg in imgs:
        png = Path(str(jpg).replace("/images/", "/annotations/")).with_suffix(".png")
        if not png.exists():
            continue
        raw = np.asarray(Image.open(png))
        gt = np.full(raw.shape, 255, dtype=np.uint8)
        for k, v in mapping.items():
            if v != IGNORE:
                gt[raw == k] = v
        x, _, region = preprocess(Image.open(jpg), cfg["data"]["input_size"])
        with torch.no_grad():
            logits = model(x.to(device))["terrain"]
        x0, y0, cw, ch = region
        # Upsample the logits and argmax after: interpolating class ids is meaningless
        # arithmetic, and the mask is full resolution while the head is not.
        pred = (
            torch.nn.functional.interpolate(
                logits[:, :, y0 : y0 + ch, x0 : x0 + cw],
                size=gt.shape,
                mode="bilinear",
                align_corners=False,
            )
            .argmax(1)[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        c = np.zeros((len(classes), len(classes)), dtype=np.int64)
        ok = gt != IGNORE
        np.add.at(c, (gt[ok], pred[ok]), 1)
        mats.append(c)
        kept.append(jpg)
    return np.stack(mats), classes, kept

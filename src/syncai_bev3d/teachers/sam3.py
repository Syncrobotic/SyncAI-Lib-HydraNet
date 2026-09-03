"""SAM 3 as a teacher: one frame in, an id mask and instance boxes out.

Written as `scripts/sam3_prelabel.py`, which is still the command line around it. What
lives here is the part three other callers had a reason to import, and the part whose
output is a training set -- so it belongs where the wheel, the type ratchet and the
coverage floor reach it. `teachers/__init__.py` says why that mattered enough to
move.

`transformers` is imported inside `load_sam3` and nowhere else, so this module stays
importable on a base install and an inference box never carries a 6.5 GB checkpoint it
will not run.

The two entry points a caller wants are `frame_masks` (semantic id mask, from the concept
table) and `frame_boxes` (instances, from the same forward passes). `consensus` is what
collapses a fixed camera's frames into the part they agree on, and is the whole answer to
"there is no annotation budget".
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from PIL import Image

from syncai_hydranet.data.sam3_prompts import DEFAULT_MIN_SCORE
from syncai_hydranet.labels import IGNORE

from .boxes import dedupe

CONTESTED = -1
# Below this an "instance" is a speck: SAM 3 emits a handful of stray pixels on a
# confident prompt, and a 3-pixel box trains a detector to fire on noise.
MIN_BOX_PIXELS = 40

# And above this it is not an instance either. `product box` occasionally fires on a
# whole display counter rather than on the items sitting along it -- the right idea at
# the wrong scale, and the resulting box teaches a detector that a counter is stock.
#
# 2% of the frame, chosen from the distribution rather than picked. Measured over 137
# `boxed_stock` and 34 `device` boxes on Taichung-cam01:
#
#     boxed_stock   median 0.256%   p90 0.560%   p98 0.78%   p99 3.53%   max 22.6%
#     device        median 0.073%   p90 0.493%   p98 1.04%   p99 1.06%   max  1.1%
#
# There is a 4.5x gap between p98 and p99 on `boxed_stock` and nothing in it. A cut
# anywhere in that gap removes the two bad boxes and no good ones; 2% sits in the
# middle of it and leaves `device` -- whose largest real instance is 1.1% -- untouched.
#
# One camera, so it is a default rather than a constant: a shop whose merchandise is
# genuinely large in frame (an iMac on a near table) wants --max-box-frac raised, and
# the summary prints what the cut discarded so that case is visible rather than silent.
MAX_BOX_FRAC = 0.02  # a pixel two classes on one layer both claim; resolved to IGNORE
MODEL_ID = "facebook/sam3"
# Pinned rather than floating, for the reason argued in full on
# `syncai_bev3d/teachers/gdino.py`'s revision pin: an unpinned `from_pretrained`
# silently changes what a teacher produces when upstream pushes.
MODEL_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"


def load_sam3(model_id: str, device: str, revision: str = MODEL_REVISION):
    """Import inside the call so the module stays importable without the extra."""
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise SystemExit(
            "SAM 3 needs the `annotate` extra: uv pip install -e '.[annotate]'"
        ) from exc
    proc = Sam3Processor.from_pretrained(model_id, revision=revision)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Sam3Model.from_pretrained(model_id, dtype=dtype, revision=revision)
    model = model.to(device).eval()
    return proc, model


def vision_features(proc, model, image: Image.Image, device: str):
    """Encode the image once, for every prompt that will be asked of it.

    **Measured, because 44 prompts a frame is a long time to spend on a hunch.** A full
    forward is 188 ms at 1920x1080 and 113 ms of it -- 60% -- is the vision encoder,
    which does not depend on the text at all. The taxonomy asks 6 concepts and 44 prompts
    of every frame, so the encoder was being recomputed 43 times on an image that had not
    changed:

        44 prompts, encoder each time   8.3 s a frame
        44 prompts, encoder once        3.4 s a frame

    `Sam3Model.forward` accepts `vision_embeds` in place of `pixel_values` and refuses
    both, so this is the API's own intent rather than a trick. Checked before use rather
    than assumed: over 11 prompts spanning all six concepts the two paths return
    bit-identical masks and scores to 1e-5.
    """
    inputs = proc(images=image, text="person", return_tensors="pt").to(device)
    with torch.inference_mode():
        return model.get_vision_features(pixel_values=inputs["pixel_values"])


def segment(
    proc,
    model,
    image: Image.Image,
    prompt: str,
    min_score: float,
    device: str,
    vision_embeds=None,
):
    """Every instance SAM 3 returns for one text prompt, as (mask, score) pairs.

    `vision_embeds` is this image's cached encoder output; see `vision_features`. Omitting
    it is correct and 2.4x slower.
    """
    inputs = proc(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        if vision_embeds is None:
            out = model(**inputs)
        else:
            inputs.pop("pixel_values")
            out = model(vision_embeds=vision_embeds, **inputs)
    res = proc.post_process_instance_segmentation(
        out,
        threshold=min_score,
        mask_threshold=0.5,
        target_sizes=[(image.height, image.width)],
    )[0]
    scores = res["scores"].float().cpu().numpy()
    return [
        (m.cpu().numpy().astype(bool), float(s))
        for m, s in zip(res["masks"], scores, strict=True)
    ]


def compose(claims: dict[str, np.ndarray], shape: tuple[int, int], concepts) -> np.ndarray:
    """Merge per-class score maps into one id mask, and refuse to guess the overlaps.

    Two kinds of overlap turn up, and treating them the same is how a pre-label becomes
    a confident mistake:

    **Different layers.** A person standing on a floor is a person. Nobody would
    annotate it otherwise, so the higher layer simply wins and nothing is flagged.

    **Same layer.** `table` and `display table` fire on the same pixels on 6 of 8
    frames, and which one is right is a judgement about what the shop uses that table
    for -- not something visible from the ceiling. `label_maps_retail.py` refuses to
    map ADE20K's `table` to `display_fixture` for exactly this reason. So these pixels
    become 255 and an annotator decides.

    Everything no prompt claimed is 255 too. That is the difference between "nobody
    looked here" and `void`, and `hydranet-annotation check` fails a dataset that
    confuses them -- unlabelled must never train as a class.
    """
    out = np.full(shape, IGNORE, dtype=np.int16)
    layer_of = np.full(shape, -1, dtype=np.int8)
    score_of = np.zeros(shape, dtype=np.float32)

    for concept in sorted(concepts, key=lambda c: c.layer):
        score = claims.get(concept.name)
        if score is None:
            continue
        hit = score > 0
        if not hit.any():
            continue
        tid = concept.terrain_id

        # Higher layer overwrites outright; same layer is a disagreement, not a race.
        higher = hit & (layer_of < concept.layer)
        same = hit & (layer_of == concept.layer) & (out != tid) & (out != CONTESTED)

        out[higher] = tid
        score_of[higher] = score[higher]
        layer_of[higher] = concept.layer
        out[same] = CONTESTED

        # A second prompt for the *same* class is a union, not a conflict: the eight
        # display_fixture prompts each find a different fixture and all mean id 12.
        agree = hit & (out == tid)
        score_of[agree] = np.maximum(score_of[agree], score[agree])

    out[out == CONTESTED] = IGNORE
    return out.astype(np.uint8)


def consensus(masks: list[np.ndarray], threshold: float) -> tuple[np.ndarray, float]:
    """Collapse one camera's per-frame static masks into the part they agree on.

    This is the whole answer to "there is no annotation budget", so it is worth being
    exact about why it works. The camera does not move. A shelf is therefore the same
    pixels in every frame, and any pixel SAM 3 labels differently between two frames
    is a pixel it was guessing at in at least one of them. **Disagreement across
    frames is an error signal that costs nothing to collect** -- no human has to say
    which frame was right, because the question is only whether the answer was stable.

    What it removes is SAM 3's *random* error: a prompt that fires on one frame and
    not the next, a boundary that breathes, an instance that splits in two. What it
    cannot remove is systematic error -- a fixture SAM 3 consistently misses, or the
    pavement it consistently calls glass, are both perfectly stable and survive
    untouched. So this raises precision and says nothing about recall.

    Measured over 10 frames per camera on the three pilot stores, at 0.9: 37-51% of
    each frame survives, and `display_fixture` is 19% of what survives.

    ``person`` must NOT be passed in here. People move, so their pixels agree with
    nothing and would simply vanish -- which is correct behaviour and the reason the
    caller composites a per-frame person layer over the result instead.
    """
    stack = np.stack(masks)
    n = len(masks)
    flat = stack.reshape(n, -1)
    modal = np.full(flat.shape[1], IGNORE, np.uint8)
    best = np.zeros(flat.shape[1], np.int32)
    for cid in np.unique(flat):
        count = (flat == cid).sum(0)
        win = count > best
        modal[win] = cid
        best[win] = count[win]
    agree = best / n
    keep = (agree >= threshold - 1e-9) & (modal != IGNORE)
    modal[~keep] = IGNORE
    labelled = float(keep.mean())
    return modal.reshape(stack.shape[1:]), labelled


def _probe_image(img: Image.Image, upscale: float) -> Image.Image:
    """The image the model actually sees.

    Upscaling is how a 352x240 camera gets a usable answer -- the "0 instances" verdict on
    `product` prompts was measured at that resolution and did not survive being asked
    again at 3x. Both entry points below must resize identically or the boxes one returns
    will not land on the mask the other returns.
    """
    if upscale == 1.0:
        return img
    return img.resize(
        (int(img.width * upscale), int(img.height * upscale)),
        Image.Resampling.LANCZOS,
    )


def frame_masks(
    proc,
    model,
    frame: np.ndarray,
    subset,
    *,
    device: str,
    upscale: float = 1.0,
) -> tuple[Image.Image, np.ndarray]:
    """One frame through one subset of the concept table: the image and its id mask.

    Takes the frame rather than an index into the caller's list, so it can be reused
    for the static pass and the per-frame `person` pass without closing over the
    clip loop's variables.
    """
    img = Image.fromarray(frame)
    probe_img = _probe_image(img, upscale)
    shape = (probe_img.height, probe_img.width)
    # Per class, the union over its prompts, keeping the best score per pixel.
    claims: dict[str, np.ndarray] = {}
    embeds = vision_features(proc, model, probe_img, device)
    for concept in subset:
        acc = np.zeros(shape, dtype=np.float32)
        for prompt in concept.prompts:
            for mask, score in segment(
                proc, model, probe_img, prompt, concept.min_score, device, embeds
            ):
                np.maximum(acc, mask * score, out=acc)
        if acc.any():
            claims[concept.name] = acc
    return img, compose(claims, shape, subset)


def frame_boxes(
    proc,
    model,
    frame: np.ndarray,
    det_classes,
    *,
    device: str,
    upscale: float = 1.0,
    max_box_frac: float = MAX_BOX_FRAC,
) -> tuple[list[dict], Counter]:
    """Instance boxes for one frame, in the saved image's pixel coordinates.

    Read off the same forward passes `frame_masks` uses -- SAM 3 returns one mask per
    detection and `compose` discards that identity to fit a single-channel id map. This
    keeps it, which is the half of the answer a semantic mask cannot carry: thirty boxes
    on a shelf are one region to a per-pixel classifier and thirty rows here.

    Boxes are **never** put through the consensus vote. That vote exists to denoise
    *static structure* by asking whether a pixel keeps its class across a clip; an
    instance is not a pixel and has no identity across frames to vote on. Every box
    below belongs to the frame it was found in.

    Coordinates are scaled back from the upscaled probe image, so a caller running
    `--upscale 3` gets boxes that land on the JPEG it actually wrote.
    """
    img = Image.fromarray(frame)
    probe_img = _probe_image(img, upscale)
    sx, sy = img.width / probe_img.width, img.height / probe_img.height
    out: list[dict] = []
    dropped: Counter = Counter()
    embeds = vision_features(proc, model, probe_img, device)
    for cat_id, (name, prompts) in enumerate(det_classes, start=1):
        for prompt in prompts:
            for mask, score in segment(
                proc, model, probe_img, prompt, DEFAULT_MIN_SCORE, device, embeds
            ):
                ys, xs = np.nonzero(mask)
                if xs.size < MIN_BOX_PIXELS:
                    continue
                x0, x1 = float(xs.min()) * sx, float(xs.max() + 1) * sx
                y0, y1 = float(ys.min()) * sy, float(ys.max() + 1) * sy
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                if (x1 - x0) * (y1 - y0) > max_box_frac * img.width * img.height:
                    dropped[name] += 1
                    continue
                out.append(
                    {
                        "category_id": cat_id,
                        "category": name,
                        "prompt": prompt,
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "area": float(mask.sum()) * sx * sy,
                        "score": float(score),
                        "iscrowd": 0,
                    }
                )
    return dedupe(out), dropped

#!/usr/bin/env python3
"""Pre-label site clips with SAM 3, for the classes the model cannot see at all.

    python3 scripts/sam3_prelabel.py --out datasets/retail_sam3_batch01 \\
        --frames 40 /path/to/hls-record/cam*/archive_*.mp4

Sibling of `scripts/annotation_batch.py`, and deliberately not a replacement for it. The
two answer different questions and are best run over the same frames:

    annotation_batch.py   what the model already half-knows -- correct its mask
    sam3_prelabel.py      what the model has never seen    -- SAM 3 proposes one

The distinction matters more than it looks. `docs/RETAIL_SCOPE.md` measured self-training
on this exact footage: pseudo-labels moved `display_fixture` by **-0.0096**, because the
podiums came back unlabelled and the model then learned that shop fixtures are not
fixtures. Labels drawn from the model can only reinforce what it already believes. SAM 3
is an outside opinion, which is the one thing that can add a class -- and equally, is a
second model with its own errors rather than an oracle. Nothing here is ground truth.

Everything it writes lands in the `seg_folder` layout that
`hydranet-annotation check --scheme retail` gates, and the gate is not optional.

## `--consensus`, for when there is no annotation budget at all

Above assumes a human corrects the mask afterwards. Where nobody is available, that
assumption fails quietly and badly: a pre-label nobody checks is a label, and SAM 3's
mistakes go straight into the weights.

    python3 scripts/sam3_prelabel.py --out datasets/site --consensus 0.9 \\
        --frames 12 --upscale 1.0 /path/to/cam*/clip.mp4

**The camera does not move, so the answer should not either.** A shelf is the same
pixels in every frame of a fixed CCTV clip, and any pixel SAM 3 labels differently
between two frames is one it was guessing at. Voting across the clip and keeping only
what agrees turns that into a free denoiser -- no human has to say which frame was
right, because the question is only whether the answer was stable.

It buys precision and pays in coverage, which is the correct direction when nothing
downstream will catch a mistake. Measured over 10 frames per camera on the three
pilot stores at 0.9: 37-51% of each frame survives, `display_fixture` is 19% of it.

Two limits, both structural. It removes SAM 3's *random* error and leaves every
systematic one untouched -- a fixture it consistently misses, or pavement it
consistently calls glass, are perfectly stable. And it says nothing about recall.

`person` is excluded from the vote and composited per frame, since people move and
would otherwise vanish. That split is also what makes long runs affordable: the
static pass is paid once per camera, and additional frames cost one prompt each.

Requires the `annotate` extra:  uv pip install -e '.[annotate]'
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from syncai_hydranet.cli.infer_video import frames, probe  # noqa: E402
from syncai_hydranet.data import sam3_prompts as _retail_prompts  # noqa: E402
from syncai_hydranet.data import sam3_prompts_objects as _object_prompts  # noqa: E402
from syncai_hydranet.data.frame_selection import describe, farthest_first  # noqa: E402

# Two taxonomies, two prompt tables, and the pairing is not interchangeable: a concept
# resolves its class id against the taxonomy it belongs to, so running the retail table
# and writing the masks as an object dataset would put id 12 (display_fixture) into a
# 7-class label space where 12 does not exist. Selected together by --scheme, and the
# annotation gate named in the summary is selected with it.
PROMPT_SETS = {
    "retail": _retail_prompts,
    "retail_objects": _object_prompts,
}

IGNORE = 255
CONTESTED = -1  # a pixel two classes on one layer both claim; resolved to IGNORE
MODEL_ID = "facebook/sam3"


def load_sam3(model_id: str, device: str):
    """Import inside the call so the module stays importable without the extra."""
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise SystemExit(
            "SAM 3 needs the `annotate` extra: uv pip install -e '.[annotate]'"
        ) from exc
    proc = Sam3Processor.from_pretrained(model_id)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = Sam3Model.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return proc, model


def segment(proc, model, image: Image.Image, prompt: str, min_score: float, device: str):
    """Every instance SAM 3 returns for one text prompt, as (mask, score) pairs."""
    inputs = proc(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**inputs)
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


def frame_masks(proc, model, frame: np.ndarray, subset, args) -> tuple[Image.Image, np.ndarray]:
    """One frame through one subset of the concept table: the image and its id mask.

    Takes the frame rather than an index into the caller's list, so it can be reused
    for the static pass and the per-frame `person` pass without closing over the
    clip loop's variables.
    """
    img = Image.fromarray(frame)
    probe_img = (
        img.resize(
            (int(img.width * args.upscale), int(img.height * args.upscale)),
            Image.Resampling.LANCZOS,
        )
        if args.upscale != 1.0
        else img
    )
    shape = (probe_img.height, probe_img.width)
    # Per class, the union over its prompts, keeping the best score per pixel.
    claims: dict[str, np.ndarray] = {}
    for concept in subset:
        acc = np.zeros(shape, dtype=np.float32)
        for prompt in concept.prompts:
            for mask, score in segment(
                proc, model, probe_img, prompt, concept.min_score, args.device
            ):
                np.maximum(acc, mask * score, out=acc)
        if acc.any():
            claims[concept.name] = acc
    return img, compose(claims, shape, subset)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("clips", nargs="+", help="video files; one session directory per clip")
    ap.add_argument("--out", required=True, help="dataset root to write")
    ap.add_argument("--split", default="train")
    ap.add_argument("--frames", type=int, default=40, help="frames to keep per clip")
    ap.add_argument("--sample-fps", type=float, default=1.0, help="rate to consider frames at")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--upscale",
        type=float,
        default=3.0,
        help="resize before prompting. These cameras deliver 352x240; SAM 3 works at its "
        "own internal resolution and a 3x LANCZOS upscale measurably improves what it "
        "returns on them. Set 1.0 for footage that is already HD",
    )
    ap.add_argument(
        "--scheme",
        choices=sorted(PROMPT_SETS),
        default="retail",
        help="which taxonomy to write. retail is the robot's 13 classes; retail_objects "
        "is the 7-class object taxonomy, where display_fixture and obstacle_furniture "
        "are one `fixture` class and `column` and `product` exist. The masks are only "
        "readable under the matching label_map",
    )
    ap.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="CLASS",
        help="also pre-label a class that is off by default (see data/sam3_prompts.py "
        "for why each one is off -- the reasons are not the same)",
    )
    ap.add_argument("--exclude", action="append", default=[], metavar="CLASS")
    ap.add_argument(
        "--consensus",
        type=float,
        default=None,
        metavar="FRACTION",
        help="fixed-camera mode: keep a static pixel only where this fraction of the "
        "clip's frames agree on its class, and write 255 everywhere else. 0.9 is the "
        "measured setting. Costs coverage and buys precision, which is the right trade "
        "when nobody is available to correct a mistake -- see `consensus()`. `person` "
        "is excluded from the vote and composited per frame instead",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompts = PROMPT_SETS[args.scheme]
    concepts = prompts.resolve(args.include, args.exclude)
    print(f"scheme: {args.scheme}")
    print(f"classes: {', '.join(c.name for c in concepts)}")
    for c in concepts:
        if c.note:
            print(f"  {c.name:<20} {c.note}")

    proc, model = load_sam3(args.model, args.device)
    root = Path(args.out)
    totals = Counter()
    manifest = {
        "model": args.model,
        "upscale": args.upscale,
        "clips": [],
        "prompts": {
            c.name: {"id": c.terrain_id, "prompts": list(c.prompts), "min_score": c.min_score}
            for c in concepts
        },
    }

    for clip in args.clips:
        session = Path(clip).stem
        img_dir = root / "images" / args.split / session
        ann_dir = root / "annotations" / args.split / session
        img_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)

        w, h, _ = probe(clip)
        kept, descs = [], []
        for frame in frames(clip, w, h, args.sample_fps):
            kept.append(frame.copy())
            descs.append(describe(frame))
        if not kept:
            print(f"{session}: no frames decoded, skipped")
            continue
        picks = farthest_first(descs, args.frames)
        print(f"{session}: {len(kept)} candidates -> {len(picks)} kept")

        # In consensus mode the two halves of the frame are computed differently:
        # everything that cannot move is voted on across the clip, and everything that
        # can is left per frame. Splitting them here rather than in `compose` keeps the
        # rule visible. The prompt table names which classes move (`MOVING`) rather than
        # this being read off the layer: they coincide today -- people are foreground in
        # both taxonomies -- but "paints over everything" and "changes between frames"
        # are different properties, and a class that gained the first would silently
        # acquire the second.
        moving = [c for c in concepts if c.name in prompts.MOVING] if args.consensus else []
        static = [c for c in concepts if c not in moving]

        per_frame: list[tuple[Image.Image, np.ndarray]] = [
            frame_masks(proc, model, kept[idx], static, args) for idx in picks
        ]
        shared = None
        if args.consensus:
            shared, kept_share = consensus([m for _, m in per_frame], args.consensus)
            print(
                f"  consensus @{args.consensus:.2f} over {len(per_frame)} frames: "
                f"{100 * kept_share:.1f}% of the frame survives as static structure"
            )

        for n, (img, own) in enumerate(per_frame):
            mask = own if shared is None else shared.copy()
            if moving:
                # Painted after the vote, not into it: LAYER_PERSON overwrites, which
                # is what a person standing in front of a shelf should do.
                _, mv = frame_masks(proc, model, kept[picks[n]], moving, args)
                mask = mask.copy()
                mask[mv != IGNORE] = mv[mv != IGNORE]
            mask_img = Image.fromarray(mask).resize(img.size, Image.NEAREST)
            stem = f"{n:04d}"
            img.save(img_dir / f"{stem}.jpg", quality=92)
            mask_img.save(ann_dir / f"{stem}.png")
            totals.update(np.asarray(mask_img).ravel().tolist())

        entry = {"session": session, "source": clip, "frames": len(picks)}
        if args.consensus:
            entry["consensus"] = {
                "threshold": args.consensus,
                "static_share": round(kept_share, 4),
            }
        manifest["clips"].append(entry)

    total_px = sum(totals.values()) or 1
    labelled = total_px - totals[IGNORE]
    print(f"\nwrote {root}")
    print("pre-label composition (share of labelled pixels)")
    for name, concept in prompts.BY_NAME.items():
        px = totals[concept.terrain_id]
        if px:
            print(f"  {concept.terrain_id:>3} {name:<20} {100 * px / max(labelled, 1):6.2f}%")
    print(
        f"  ignore (unclaimed or contested): "
        f"{100 * totals[IGNORE] / total_px:.1f}% of all pixels"
    )
    # A class that was asked for and came back with nothing. Printed rather than left to
    # be noticed in the table above, where a class simply does not appear -- which reads
    # like the footage not containing one and is indistinguishable from a prompt that
    # does not work. `product` under --scheme retail_objects is the live case: it has no
    # prior measurement at all, so its first run is the measurement.
    empty = [c.name for c in concepts if not totals[c.terrain_id]]
    if empty:
        print(f"  FOUND NOTHING: {', '.join(empty)} -- prompts, threshold, or footage?")
    manifest["pre_label_share"] = {
        name: round(totals[c.terrain_id] / max(labelled, 1), 4)
        for name, c in prompts.BY_NAME.items()
        if totals[c.terrain_id]
    }
    manifest["scheme"] = args.scheme
    manifest["found_nothing"] = empty
    (root / "sam3_batch.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # The gate has to be named with the same scheme the masks were written under.
    # Checking object masks under --scheme retail would read id 4 as `wet_slippery`
    # instead of `fixture` and pass, because both are valid ids in their own taxonomy.
    gate = args.scheme
    print("\nNext:")
    print(f"  hydranet-annotation labels --scheme {gate} --out cvat_labels_{gate}.json")
    print(f"  hydranet-annotation check {root} --scheme {gate}")
    if args.scheme == "retail":
        print(
            "\nCorrect every mask before training on any of it. Two to look at hardest:\n"
            "  glass    SAM 3 segments what is *behind* the pane along with the pane\n"
            "  ignore   holes are contested table/display_fixture pixels, and they are the\n"
            "           judgement calls -- that is the work, not a gap in it"
        )
    else:
        print(
            "\nCorrect every mask before training on any of it. Three to look at hardest:\n"
            "  product  no prior measurement exists for this class -- the first batch is\n"
            "           an experiment, and the share above is its result\n"
            "  column   check it did not take the wall with it; it paints over wall by\n"
            "           design, so an over-wide column mask is not visible as a conflict\n"
            "  wall     absorbs glass here, and a pane still segments the pavement behind\n"
            "           it -- contained within one class now, but still wrong pixels"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

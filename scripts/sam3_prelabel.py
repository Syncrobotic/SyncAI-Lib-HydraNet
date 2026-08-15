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
from syncai_hydranet.data.frame_selection import describe, farthest_first  # noqa: E402
from syncai_hydranet.data.sam3_prompts import BY_NAME, resolve  # noqa: E402

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
        "--include",
        action="append",
        default=[],
        metavar="CLASS",
        help="also pre-label a class that is off by default (see data/sam3_prompts.py "
        "for why each one is off -- the reasons are not the same)",
    )
    ap.add_argument("--exclude", action="append", default=[], metavar="CLASS")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    concepts = resolve(args.include, args.exclude)
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

        for n, idx in enumerate(picks):
            img = Image.fromarray(kept[idx])
            probe_img = (
                img.resize(
                    (int(img.width * args.upscale), int(img.height * args.upscale)),
                    Image.LANCZOS,
                )
                if args.upscale != 1.0
                else img
            )
            shape = (probe_img.height, probe_img.width)

            # Per class, the union over its prompts, keeping the best score per pixel.
            claims: dict[str, np.ndarray] = {}
            for concept in concepts:
                acc = np.zeros(shape, dtype=np.float32)
                for prompt in concept.prompts:
                    for mask, score in segment(
                        proc, model, probe_img, prompt, concept.min_score, args.device
                    ):
                        np.maximum(acc, mask * score, out=acc)
                if acc.any():
                    claims[concept.name] = acc

            mask = compose(claims, shape, concepts)
            mask_img = Image.fromarray(mask).resize(img.size, Image.NEAREST)

            stem = f"{n:04d}"
            img.save(img_dir / f"{stem}.jpg", quality=92)
            mask_img.save(ann_dir / f"{stem}.png")
            totals.update(np.asarray(mask_img).ravel().tolist())

        manifest["clips"].append({"session": session, "source": clip, "frames": len(picks)})

    total_px = sum(totals.values()) or 1
    labelled = total_px - totals[IGNORE]
    print(f"\nwrote {root}")
    print("pre-label composition (share of labelled pixels)")
    for name, concept in BY_NAME.items():
        px = totals[concept.terrain_id]
        if px:
            print(f"  {concept.terrain_id:>3} {name:<20} {100 * px / max(labelled, 1):6.2f}%")
    print(
        f"  ignore (unclaimed or contested): "
        f"{100 * totals[IGNORE] / total_px:.1f}% of all pixels"
    )
    manifest["pre_label_share"] = {
        name: round(totals[c.terrain_id] / max(labelled, 1), 4)
        for name, c in BY_NAME.items()
        if totals[c.terrain_id]
    }
    (root / "sam3_batch.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\nNext:")
    print("  hydranet-annotation labels --scheme retail --out cvat_labels_retail.json")
    print(f"  hydranet-annotation check {root} --scheme retail")
    print(
        "\nCorrect every mask before training on any of it. Two to look at hardest:\n"
        "  glass    SAM 3 segments what is *behind* the pane along with the pane\n"
        "  ignore   holes are contested table/display_fixture pixels, and they are the\n"
        "           judgement calls -- that is the work, not a gap in it"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

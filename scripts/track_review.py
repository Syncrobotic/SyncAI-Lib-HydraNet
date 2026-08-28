#!/usr/bin/env python3
"""Turn track ground truth from hours of frame-by-frame work into minutes of judgement.

    # 1. propose
    python3 scripts/track_review.py propose \\
        --config runs/hydranet_retail_objects_site_batch02/config.yaml \\
        --checkpoint runs/hydranet_retail_objects_site_batch02/best.pt \\
        --clip datasets/studioa_clips/Kaohsiung-cam04/archive_20260816-112757*.mp4 \\
        --fps 5 --max-frames 300 --out runs/gt_cam04

    # 2. a human reads runs/gt_cam04/review_*.png and writes runs/gt_cam04/merges.json
    #    [[3, 17, 41], [8, 22]]   <- each inner list is one shopper

    # 3. apply
    python3 scripts/track_review.py apply --out runs/gt_cam04

`analytics/reid_metrics.idf1` has been armed since it was written and has never run on a
site clip, because no site clip is labelled. `scripts/retail_flow.py` names the same
blocker in its own docstring: *"this project has no hand-labelled site data at all.
Sample an hour, count it by eye, compare. That is the acceptance test and there is no
substitute for it."*

There is no substitute for the judgement. There is one for most of the labour.

**What is expensive about track ground truth is not the boxes.** The detection head
already draws them, and person boxes on these cameras are 244-336 px tall with 85-98%
above 128 px -- comfortably good enough. What is expensive is deciding *which fragments
are the same person*, and that is a merge decision over a few dozen rows rather than a
drawing task over a few hundred frames.

So this proposes tracks, lays them out as one row of crops per track, and applies the
merges a human writes back. It stops exactly where judgement starts and does not guess.

**The output format is the metric's input format, deliberately.** `tracks.json` is
`{track_id: {frame: [x0, y0, x1, y1]}}` because that is what `idf1` and `id_switches`
consume. A format invented here and converted later is a second place for a frame index
to go wrong.

**It refuses any detection head whose label `PERSON` does not mean a person**, and it
decides that by reading the head's declared class names rather than by counting them.
`PERSON` here is `COCO_NAMES.index("person")`, which is **0**, and a label of 0 exists in
every detection head this repo trains. `configs/hydranet_retail_openvocab.yaml` is the
case that matters: its two classes are `boxed_stock` and `device`, so label 0 is shelf
stock. Pointing this script at that checkpoint would not error -- it would track stock,
lay it out as shoppers, and hand back a ground-truth file. The pairing in this docstring's
own example was one such mistake: a 2-class config against an 80-class checkpoint, which
at least fails loudly on a shape mismatch. The refusal below covers the case that does not.

**This refusal used to be `n_classes != 80`, and that was wrong in the direction that
costs work.** Every checkpoint in this repo that detects people is the 4-class retail head
`(person, bag, boxed_stock, device)` -- pose01, pose02 and the security runs -- where
label 0 *is* person, and `analytics/clip_tracks.py` has been tracking shoppers with the
same `PERSON` constant on those checkpoints all along. So the count test refused the only
checkpoints this script could ever be pointed at, while its error message asserted that
label 0 meant `boxed_stock` on a head where it does not. Found 2026-08-27, when the whole
reason to run this -- the IDF1 ground truth decision 1 needs (PLAN §7.18) -- ran into it.
Reading the names is also strictly the safer test: it still refuses the openvocab head,
which counting to 80 would have refused only by accident of arity.

**What this does not do: it does not add missing people.** A shopper the detector never
saw is absent from the proposal and stays absent from the ground truth, so recall against
this file is recall against the detector, not against reality. That makes it valid for
measuring *association* -- which is the binding constraint -- and invalid for measuring
detection. Anyone quoting a number from it owes that sentence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from syncai_hydranet.analytics import Tracker
from syncai_hydranet.config import load_config
from syncai_hydranet.data.coco_subsets import COCO_NAMES
from syncai_hydranet.data.label_maps_retail_security import get_det_vocab
from syncai_hydranet.data.transforms import invert_geom
from syncai_hydranet.data.video import frames, probe
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.device import pick_device
from syncai_hydranet.utils.visualize import preprocess

PERSON = COCO_NAMES.index("person")
CROP_W, CROP_H = 48, 96
MAX_CROPS = 12


def head_class_names(cfg: dict) -> tuple[str, ...] | None:
    """What the detection head's channels are called, or `None` if the config cannot say.

    Three sources, in the order a config states them: the head's own `classes` list, a
    `det_vocab` shared with the datasets, and -- only for a head of exactly 80 channels --
    the COCO space, which this repo's 80-class configs name by omission rather than by
    listing. `None` means "unknown", which the caller must treat as a refusal: a head whose
    channels have no names is exactly the head nobody can check.
    """
    head = cfg.get("model", {}).get("heads", {}).get("detection", {})
    if head.get("classes"):
        return tuple(head["classes"])
    datasets = cfg.get("data", {}).get("datasets", [])
    vocabs = {d["det_vocab"] for d in datasets if d.get("det_vocab")}
    if len(vocabs) == 1:
        return tuple(get_det_vocab(next(iter(vocabs))).classes)
    if head.get("num_classes") == len(COCO_NAMES):
        return tuple(COCO_NAMES)
    return None


def check_person_class_space(n_classes: int, names: tuple[str, ...] | None = None) -> None:
    """Refuse a detection head in which label `PERSON` does not mean a person.

    Raises rather than warns. A warning is the wrong instrument here: the failure produces
    a complete, well-formed `tracks.json` and a review sheet full of crops, so nothing
    downstream can tell the difference and the reviewer is looking at the crops rather
    than at the scrollback.

    `names` defaults to the COCO space so the historical single-argument call still means
    what it did: "this head has `n_classes` channels and they are COCO's".
    """
    names = tuple(COCO_NAMES) if names is None else names
    if len(names) != n_classes:
        raise SystemExit(
            f"The config names {len(names)} detection classes and the checkpoint's head "
            f"has {n_classes} channels. One of them is not the model you meant; the "
            f"tracks would be labelled by a taxonomy the weights were not trained on."
        )
    if len(names) <= PERSON or names[PERSON] != "person":
        got = names[PERSON] if len(names) > PERSON else "nothing"
        raise SystemExit(
            f"In this detection head label {PERSON} is {got!r}, not 'person'. Nothing "
            f"would error; the tracks would just not be people. Its classes are "
            f"{', '.join(names)} -- use a checkpoint whose head detects people."
        )


def tracks_to_json(tracks) -> dict[str, dict[str, list[float]]]:
    """`Track` objects -> the shape `reid_metrics` consumes.

    Reads `frames` and `boxes`, which `Track` already carries -- the metric needs no
    change to the tracker, which is worth stating because it is the only reason these two
    pieces fit together without a contract being invented for them.
    """
    out: dict[str, dict[str, list[float]]] = {}
    for t in tracks:
        out[str(t.track_id)] = {
            str(int(f)): [float(v) for v in b] for f, b in zip(t.frames, t.boxes, strict=False)
        }
    return out


def apply_merges(tracks: dict, merges: list[list]) -> dict:
    """Fold each group of track ids into one identity, keeping the lowest id.

    A frame claimed by two tracks in the same group keeps the first -- that is a
    detector duplicate rather than a judgement, and dropping it silently is wrong the
    other way, so `merge_collisions` is reported by the caller.

    **An id that is not in `tracks` is refused rather than ignored.** `merges.json`
    records ids and nothing else, so it carries no evidence of which proposal it was
    written against, and a re-run renumbers freely: this directory has already held a
    114-track proposal and a 24-track one. Under `lookup.get(tid, tid)` a group naming
    ids from the wrong file is simply absent from the output -- the merge does not
    happen, no key is missing, no count changes, and the ground truth is silently the
    unmerged proposal. That is the one error a reviewer cannot see in the result, because
    what it produces is exactly what "the tracker fragmented nothing" produces.
    """
    known = set(tracks)
    missing = sorted({str(g) for group in merges for g in group} - known, key=int)
    if missing:
        raise SystemExit(
            f"merges.json names {len(missing)} track id(s) that tracks.json does not "
            f"contain: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}. "
            f"The proposal has {len(known)} tracks. This is what a merge file written "
            f"against an earlier proposal looks like -- re-review against the current "
            f"tracks.json rather than editing the ids to fit."
        )
    lookup = {}
    for group in merges:
        keep = str(sorted(int(g) for g in group)[0])
        for g in group:
            lookup[str(g)] = keep
    merged: dict[str, dict[str, list[float]]] = {}
    collisions = 0
    for tid, byframe in tracks.items():
        target = lookup.get(tid, tid)
        dest = merged.setdefault(target, {})
        for f, box in byframe.items():
            if f in dest:
                collisions += 1
                continue
            dest[f] = box
    return {"tracks": merged, "merge_collisions": collisions}


ROWS_PER_PAGE = 20


def _sheet(tracks: dict, crops: dict, out: Path) -> tuple[list[Path], int]:
    """Pages of rows, one row per fragment, **sorted by first appearance**.

    Both of those are the difference between a tool and a 12,000-pixel strip, and the
    first version was the strip: 117 fragments in one image 794 x 12176, which is not
    something a person reads.

    Sorted by time rather than by length because of what the job actually is. The
    fragments belong to a small number of shoppers -- cam04 shows eight people in a frame
    -- so this is ~117 assignment decisions against maybe fifteen identities, not 6,786
    pairwise comparisons. Time ordering puts one shopper's fragments near each other,
    which is what makes an assignment answerable by looking rather than by searching.

    Returns the pages **and the number of crops actually pasted**, because the six pages
    this replaced were 794 x 2088 each, correct in every dimension, and empty. The page
    geometry was checked and reported; the page contents were not. A caller that only
    counts pages cannot tell a sheet from a set of labels.
    """
    # Clear the previous run's pages before writing this one's. A re-run that yields fewer
    # tracks writes fewer pages, and the surplus survives: this directory held
    # `review_03..06.png` from an earlier pass alongside `review_01..02.png` from a later
    # one, six pages of which four were stale. Empty ones merely waste a reviewer's time;
    # the dangerous case is stale pages that *have* crops, because their track ids are
    # from a `tracks.json` that no longer exists and every grouping made from them is
    # silently against the wrong file. `merges.json` records ids and nothing else, so
    # there is no later step at which that could be caught.
    for old in out.glob("review_*.png"):
        old.unlink()

    ids = sorted(tracks, key=lambda t: min(int(f) for f in tracks[t]))
    row_h = CROP_H + 8
    pages = []
    drawn = 0
    for start in range(0, max(1, len(ids)), ROWS_PER_PAGE):
        chunk = ids[start : start + ROWS_PER_PAGE]
        img = Image.new(
            "RGB",
            (170 + MAX_CROPS * (CROP_W + 4), max(1, len(chunk)) * row_h + 8),
            (24, 24, 23),
        )
        d = ImageDraw.Draw(img)
        for r, tid in enumerate(chunk):
            y = 4 + r * row_h
            fr = sorted(int(f) for f in tracks[tid])
            d.text((6, y + 4), f"#{tid}", fill=(255, 255, 255))
            d.text((6, y + 20), f"f{fr[0]}-{fr[-1]}", fill=(170, 170, 165))
            d.text((6, y + 36), f"{len(fr)} fr", fill=(170, 170, 165))
            for c, im in enumerate(crops.get(tid, [])[:MAX_CROPS]):
                img.paste(im, (170 + c * (CROP_W + 4), y))
                drawn += 1
        path = out / f"review_{start // ROWS_PER_PAGE + 1:02d}.png"
        img.save(path)
        pages.append(path)
    return pages, drawn


def _copresent(tracks: dict) -> dict:
    """Pairs of tracks alive at the same time -- the merges no judgement is needed for.

    Two tracks whose frame spans overlap are two people, and that is arithmetic over frame
    indices: no model, no calibration, no threshold. It matters that it stays that way.
    A position or speed test would exclude far more pairs, and it would exclude them using
    exactly the geometry and the association rules that the ground truth is being built to
    measure -- IDF1 against a label set derived from the tracker's own assumptions scores
    agreement with those assumptions.

    On the first sheet this ran against (Taichung-cam01, 10 tracks) it removes 13 of the
    45 pairs. It is written beside the sheet rather than drawn on it because it is a fact
    about the tracks, and the sheet is the crops.
    """
    span = {t: (min(int(f) for f in v), max(int(f) for f in v)) for t, v in tracks.items()}
    ids = sorted(span, key=lambda t: span[t][0])
    excluded, open_pairs = [], []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            (a0, a1), (b0, b1) = span[a], span[b]
            if a0 <= b1 and b0 <= a1:
                excluded.append([a, b])
            else:
                open_pairs.append([a, b, b0 - a1 if a1 < b0 else a0 - b1])
    return {
        "spans": {t: list(s) for t, s in span.items()},
        "cannot_merge_co_present": excluded,
        "open_pairs_gap_frames": sorted(open_pairs, key=lambda p: p[2]),
    }


def cmd_propose(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config, validate=False)
    device = pick_device(cfg.get("device"))
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(args.checkpoint), args.weights))
    check_person_class_space(int(model.det_head.cls_pred.bias.shape[0]), head_class_names(cfg))
    size = cfg["data"]["input_size"]

    src_w, src_h, _ = probe(args.clip)
    tracker = Tracker(iou_threshold=args.iou, max_age=args.max_age, min_hits=args.min_hits)
    keep: dict[str, list[Image.Image]] = {}
    n = 0
    for frame in frames(args.clip, src_w, src_h, args.fps):
        x, _, region = preprocess(Image.fromarray(frame), size)
        with torch.no_grad():
            det = model.predict(x.to(device), score_thr=args.score_thr)["detection"][0]
        if len(det.get("labels", [])):
            lab = det["labels"].cpu().numpy()
            box = det["boxes"].cpu().numpy()[lab == PERSON]
            x0, y0, cw, ch = region
            box = invert_geom(box.astype(float), (cw / src_w, ch / src_h, x0, y0))
        else:
            box = np.zeros((0, 4))
        tracker.update(box, n)
        _stash(frame, tracker, keep)
        n += 1
        if n % 50 == 0:
            print(f"  frame {n}", flush=True)
        if args.max_frames and n >= args.max_frames:
            break

    tracks = tracks_to_json([t for t in tracker.finished() if t.confirmed])
    (out / "tracks.json").write_text(json.dumps(tracks, indent=1) + "\n")
    pages, drawn = _sheet(tracks, keep, out)
    cop = _copresent(tracks)
    (out / "copresent.json").write_text(json.dumps(cop, indent=1) + "\n")
    n_pairs = len(cop["cannot_merge_co_present"]) + len(cop["open_pairs_gap_frames"])
    print(
        f"  {len(cop['cannot_merge_co_present'])} of {n_pairs} pairs are co-present and "
        f"cannot merge (copresent.json)"
    )
    lengths = sorted(len(v) for v in tracks.values())
    med = lengths[len(lengths) // 2] if lengths else 0
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "config": args.config,
                "checkpoint": args.checkpoint,
                "weights": args.weights,
                "clip": args.clip,
                "fps": args.fps,
                "max_frames": args.max_frames,
                "score_thr": args.score_thr,
                "tracker": {
                    "iou": args.iou,
                    "max_age": args.max_age,
                    "min_hits": args.min_hits,
                },
                "frames_read": n,
                "confirmed_tracks": len(tracks),
                "median_track_frames": med,
                "crops_drawn": drawn,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"\n{n} frames -> {len(tracks)} confirmed tracks, median length {med} frames")
    print(f"  {out / 'tracks.json'}")
    print(f"  {len(pages)} review pages: {pages[0].name} .. {pages[-1].name}, {drawn} crops")
    if not drawn:
        raise SystemExit(
            f"{len(pages)} pages were written and not one crop was pasted onto them. The "
            f"only thing a reviewer could group by is the frame ranges printed in the "
            f"margin, which are the tracker's own output -- so the ground truth would be "
            f"the tracker grading itself. tracks.json is kept; the sheets are not usable."
        )
    print(
        "\nWrite merges.json as [[id, id, ...], ...] -- one inner list per shopper -- "
        "then run `apply`."
    )
    return 0


def _stash(frame: np.ndarray, tracker: Tracker, keep: dict) -> None:
    """Keep a thumbnail per track, spread over its life rather than all from the start."""
    for t in tracker.tracks:
        if not t.confirmed or not len(t.boxes):
            continue
        row = keep.setdefault(str(t.track_id), [])
        if len(row) >= MAX_CROPS:
            continue
        x0, y0, x1, y1 = (int(v) for v in t.boxes[-1])
        h, w = frame.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 4 or y1 - y0 < 8:
            continue
        row.append(Image.fromarray(frame[y0:y1, x0:x1]).resize((CROP_W, CROP_H)))


def cmd_apply(args) -> int:
    out = Path(args.out)
    tracks = json.loads((out / "tracks.json").read_text())
    mpath = out / "merges.json"
    if not mpath.exists():
        raise SystemExit(
            f"{mpath} does not exist. Read review.png, decide which track ids are the "
            f"same shopper, and write them as [[3, 17], [8, 22, 40]]. An empty list is a "
            f"valid answer and means the tracker fragmented nothing -- write [] for that "
            f"rather than skipping the step, so the file records that somebody looked."
        )
    merges = json.loads(mpath.read_text())
    res = apply_merges(tracks, merges)
    (out / "ground_truth.json").write_text(json.dumps(res["tracks"], indent=1) + "\n")
    print(f"{len(tracks)} proposed -> {len(res['tracks'])} identities")
    if res["merge_collisions"]:
        print(f"  {res['merge_collisions']} frames claimed twice within a group (kept first)")
    print(f"  {out / 'ground_truth.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("propose")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--clip", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--max-frames", type=int, default=300, help="0 means the whole clip")
    p.add_argument("--score-thr", type=float, default=0.25)
    p.add_argument("--weights", choices=("ema", "model"), default="ema")
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--max-age", type=int, default=5)
    p.add_argument("--min-hits", type=int, default=3)
    p.set_defaults(fn=cmd_propose)

    a = sub.add_parser("apply")
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_apply)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())

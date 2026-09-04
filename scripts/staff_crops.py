#!/usr/bin/env python3
"""Person crops for a human to sort into staff and customer, one folder each.

    python3 scripts/staff_crops.py --out datasets/staff_customer_batch01

`staff/customer` is listed in PLAN section 4.3 as necessarily in-domain and it is the
missing piece under every retail output: `reach_to_shelf` fires 11.7 times a minute on
one clip and every person in it wears the shop's polo. This produces the material for it
with the smallest human cost the tree's own measurements allow.

---------------------------------------------------------------------------
WHY THE UNIT IS A TRACK AND NOT A FRAME

`analytics/track_attributes.py` measured it: on Kaohsiung-cam04 **the same staff member
is labelled `F` and `M` in adjacent frames**, and its conclusion is that a shopper is not
a per-frame quantity. The same applies to the labelling. One decision per person is what
a human should be asked for, so this writes a handful of crops per track and puts the
track id in the filename -- the label attaches to the person, and the train/test split
can then be **by track and by camera**, which is what stops the same shopper appearing on
both sides and inflating the number (section 4.4).

**The tracker is the shipped single-stage one on purpose.** `bytetrack`'s two-stage
association produces longer tracks -- 202 become 100, measured 2026-08-27 -- but its tails
coast onto a neighbour: of the three longest Taichung-cam01 tracks, one ends on a
different person and one on an empty counter. A merged track would hand a folder of two
people to the labeller under one decision. Fragments are cheap here; merges are not.

---------------------------------------------------------------------------
WHICH CROPS ARE WORTH A HUMAN'S TIME

Truncated and tiny crops are dropped, for the reason `track_attributes` gives rather than
the one it refuses: excluding them did **not** stabilise gender (16.2% -> 16.3% flip rate,
so that claim is not made here), but a crop cut by the frame edge is a torso, and trousers
and shoes are half of what tells a uniform from a customer. Crops are also taken from the
**middle** of a track rather than its ends, because the ends are where a box is entering,
leaving, or drifting.

The output is deliberately flat and boring: `unsorted/` to drag from, `staff/` and
`customer/` to drag into, and a `manifest.json` recording every crop's provenance so a
label can always be traced back to the frame it came from.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from syncai_hydranet.analytics.clip_tracks import PERSON, to_source_pixels
from syncai_hydranet.analytics.tracker import Tracker
from syncai_hydranet.config import load_config
from syncai_hydranet.data.video import frames as decode_frames
from syncai_hydranet.data.video import probe as probe_video
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.serving.camera import BIRTH_REF
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.visualize import preprocess

ROOT = Path(__file__).resolve().parent.parent


CLIPS = ROOT / "datasets/studioa_clips"
DEFAULT_CONFIG = "configs/hydranet_retail_pose02.yaml"
DEFAULT_CHECKPOINT = "runs/hydranet_retail_pose02/last.pt"

README = """# staff / customer, batch 01

Drag every file out of `unsorted/` into exactly one of:

    staff/      wearing the shop's uniform -- the STUDIO A polo, lanyard, name badge
    customer/   everybody else
    unclear/    you cannot tell. Leave these here rather than guessing: a wrong label
                is worse than a missing one, and this folder is counted, not ignored.

**One decision per person, not per picture.** The filename is

    <camera>__<clip>__t<track>__f<frame>.jpg

and every crop sharing a `<camera>__<clip>__t<track>` prefix is the **same person**, so
they all go the same way. Sorting one of them and moving its siblings with it is correct
and is the intended cost -- roughly {n_tracks} decisions for {n_crops} files.

Two things that matter for the number this trains:

* **Do not sort by what the crop looks like it should be.** If the polo is not visible,
  it is `unclear/`, even when the person is obviously staff from the context of the
  frame. The model only ever sees the crop.
* The split will be **by track and by camera**, so a person cannot appear on both sides
  of it. That is why the filenames carry both.

`manifest.json` records where every crop came from. Nothing here is deleted by re-running
the extractor into a different `--out`.
"""


def _read(clip: str, fps: float):
    """Frames at the clip's own resolution -- seven of the fleet's cameras are 704x480."""
    w, h, _ = probe_video(clip)
    return decode_frames(clip, w, h, fps)


def curate(out: Path, target: int) -> int:
    """Leave about `target` crops to sort and put the rest in `pool/`.

    Extraction over the whole corpus produces thousands of crops, and handing a person
    thousands of files is how a labelling task does not get done. This picks a set that
    is worth the first hour instead.

    Two rules, both about not biasing what gets labelled:

    * **Round-robin over cameras.** Taking the first N alphabetically, or the N with the
      most crops, would fill the set from the busiest counters -- which is exactly where
      staff outnumber customers, so the class balance would be an artefact of the sort
      order.
    * **A person is never split.** All crops of one `camera__clip__track` move together,
      so no shopper ends up half in the set being labelled and half in the pool, which is
      the same split discipline the filenames exist for.

    Nothing is deleted. `pool/` is where a second pass looks after the first model exists
    and can propose labels for a human to correct rather than make.
    """
    unsorted = out / "unsorted"
    pool = out / "pool"
    pool.mkdir(parents=True, exist_ok=True)
    people: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for f in sorted(unsorted.glob("*.jpg")):
        camera, clip, track, _frame = f.stem.split("__")
        people[(camera, f"{clip}__{track}")].append(f)

    by_camera: dict[str, list[tuple[str, list[Path]]]] = defaultdict(list)
    for (camera, key), files in people.items():
        by_camera[camera].append((key, files))
    for camera in by_camera:
        # longer tracks first within a camera: more usable crops means a clearer person
        by_camera[camera].sort(key=lambda kv: (-len(kv[1]), kv[0]))

    kept: set[Path] = set()
    cameras = sorted(by_camera)
    idx = dict.fromkeys(cameras, 0)
    n_people = 0
    while len(kept) < target:
        moved_any = False
        for camera in cameras:
            i = idx[camera]
            if i >= len(by_camera[camera]):
                continue
            kept.update(by_camera[camera][i][1])
            idx[camera] = i + 1
            n_people += 1
            moved_any = True
            if len(kept) >= target:
                break
        if not moved_any:
            break

    moved = 0
    for f in sorted(unsorted.glob("*.jpg")):
        if f not in kept:
            f.rename(pool / f.name)
            moved += 1
    print(
        f"unsorted: {len(kept)} crops from {n_people} people across {len(cameras)} cameras\n"
        f"pool:     {moved} crops held back"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="datasets/staff_customer_batch01")
    ap.add_argument("--cameras", nargs="*", help="default: every camera with clips")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--frames", type=int, default=300, help="per clip, at --fps")
    ap.add_argument("--score-thr", type=float, default=BIRTH_REF)
    ap.add_argument("--per-track", type=int, default=3)
    ap.add_argument("--min-obs", type=int, default=5, help="a shorter track is a fragment")
    ap.add_argument("--min-height", type=int, default=90, help="source px")
    ap.add_argument("--edge-px", type=float, default=6.0)
    ap.add_argument("--crop-height", type=int, default=320)
    ap.add_argument(
        "--curate",
        type=int,
        metavar="N",
        help="do not extract: split an existing --out so `unsorted/` holds about N crops "
        "spread evenly over the cameras and `pool/` holds the rest",
    )
    a = ap.parse_args()

    out = Path(a.out)
    if a.curate:
        return curate(out, a.curate)
    for sub in ("unsorted", "staff", "customer", "unclear"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    cfg = load_config(a.config, validate=False)
    model = build_model(cfg).to(a.device).eval()
    model.load_state_dict(select_weights(load_checkpoint(a.checkpoint), "ema"))
    size = cfg["data"]["input_size"]

    cameras = a.cameras or sorted(p.name for p in CLIPS.iterdir() if (p / ".").is_dir())
    manifest: list[dict] = []
    per_camera: dict[str, int] = defaultdict(int)
    n_tracks = 0

    for camera in cameras:
        for clip in sorted((CLIPS / camera).glob("archive_*.mp4")):
            try:
                src_w, src_h, _ = probe_video(str(clip))
            except Exception as exc:  # a camera whose clip will not probe is reported
                print(f"  {camera}/{clip.name}: unreadable ({type(exc).__name__})")
                continue
            tracker = Tracker()
            keep: dict[int, Image.Image] = {}
            n = 0
            try:
                for frame in _read(str(clip), a.fps):
                    if n >= a.frames:
                        break
                    img = Image.fromarray(frame)
                    x, _canvas, region = preprocess(img, size)
                    with torch.no_grad():
                        res = model.predict(x.to(a.device), score_thr=a.score_thr)
                    det = res["detection"][0]
                    if len(det.get("labels", [])):
                        lab = det["labels"].cpu().numpy()
                        m = lab == PERSON
                        boxes = to_source_pixels(
                            det["boxes"].cpu().numpy()[m], region, src_w, src_h
                        )
                        scores = det["scores"].cpu().numpy()[m]
                    else:
                        boxes, scores = np.zeros((0, 4)), np.zeros(0)
                    tracker.update(boxes, n, scores=scores)
                    keep[n] = img
                    n += 1
            except Exception as exc:
                print(f"  {camera}/{clip.name}: decode stopped ({type(exc).__name__}: {exc})")
                continue

            for track in tracker.finished():
                if len(track.frames) < a.min_obs:
                    continue
                # usable crops only, and from the middle of the life rather than the ends
                usable = []
                for f, box in zip(track.frames, track.boxes, strict=True):
                    x0, y0, x1, y1 = (float(v) for v in box)
                    if y1 - y0 < a.min_height:
                        continue
                    if (
                        x0 <= a.edge_px
                        or y0 <= a.edge_px
                        or x1 >= src_w - a.edge_px
                        or y1 >= src_h - a.edge_px
                    ):
                        continue
                    usable.append((f, (x0, y0, x1, y1)))
                if not usable:
                    continue
                lo, hi = len(usable) // 6, len(usable) - len(usable) // 6
                middle = usable[lo:hi] or usable
                idx = np.unique(
                    np.linspace(0, len(middle) - 1, min(a.per_track, len(middle))).astype(int)
                )
                n_tracks += 1
                for j in idx:
                    f, (x0, y0, x1, y1) = middle[int(j)]
                    img = keep.get(f)
                    if img is None:
                        continue
                    crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
                    w_new = max(1, round(crop.width * a.crop_height / max(crop.height, 1)))
                    crop = crop.resize((w_new, a.crop_height), Image.Resampling.LANCZOS)
                    name = (
                        f"{camera}__{clip.stem.replace('archive_', '')}"
                        f"__t{track.track_id:04d}__f{f:04d}.jpg"
                    )
                    crop.save(out / "unsorted" / name, quality=92)
                    manifest.append(
                        {
                            "file": name,
                            "camera": camera,
                            "clip": clip.name,
                            "track_id": int(track.track_id),
                            "frame": int(f),
                            "fps": a.fps,
                            "box_src_px": [x0, y0, x1, y1],
                        }
                    )
                    per_camera[camera] += 1
            print(f"  {camera}/{clip.name}: {per_camera[camera]} crops so far")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (out / "README.md").write_text(
        README.format(n_tracks=n_tracks, n_crops=len(manifest)), encoding="utf-8"
    )
    print(f"\n{len(manifest)} crops from {n_tracks} tracks -> {out}/unsorted")
    print(f"{'camera':22s} crops")
    for cam, k in sorted(per_camera.items(), key=lambda kv: -kv[1]):
        print(f"{cam:22s} {k:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

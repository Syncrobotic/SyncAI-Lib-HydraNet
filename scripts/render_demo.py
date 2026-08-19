#!/usr/bin/env python3
"""One clip, every stage on it at once, as a video you can watch.

    python3 scripts/render_demo.py --out assets/studioa_demo.mp4 \\
        datasets/studioa_clips/Taichung-cam01/archive_20260816-113024_20260816-113528.mp4

What is drawn, and which instrument drew it:

    terrain overlay        L0 segmentation -- floor, wall, column, fixture, person
    boxed_stock / device   L0 detection, the merchandise head
    person + track id      L0 detection (COCO head) into L1's IoU tracker
    zone polygon           L2, a square in **metres** on the floor, projected back
    event banner           L2, live as each event's span is entered
    F/M and age band       the second-stage crop encoder, per frame

**Two checkpoints, because no single one holds both.** `hydranet_retail_surfaces` has the
six-class terrain map and the two merchandise classes; `hydranet_retail_cctv` has the COCO
head that finds people. The retail+security config now being trained is what makes them
one model -- this render is what it replaces, and drawing them side by side is the
clearest statement of why that run exists.

**Nothing here is validated on site.** Terrain is trained on ADE20K plus 180 SAM 3
pre-labelled site frames, `person` on COCO alone, attributes on PA-100K, and the ground
plane on one camera's tile-grid fit. The video shows what the system says, which is not
the same as what is true, and every number behind it is in docs/RETAIL.md with
its support.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from syncai_hydranet.analytics import events as ev  # noqa: E402
from syncai_hydranet.analytics.clip_tracks import to_source_pixels  # noqa: E402
from syncai_hydranet.analytics.track_attributes import (  # noqa: E402
    age_band,
    track_attributes,
    usable_crops,
)
from syncai_hydranet.analytics.tracker import Tracker  # noqa: E402
from syncai_hydranet.cli.scene import detection_class_names  # noqa: E402
from syncai_hydranet.config import load_config  # noqa: E402
from syncai_hydranet.data.attributes import ATTRIBUTES  # noqa: E402
from syncai_hydranet.data.coco_subsets import COCO_NAMES  # noqa: E402
from syncai_hydranet.data.video import frames, probe  # noqa: E402
from syncai_hydranet.geometry.ground import Camera, GroundPlane, ground_to_pixel  # noqa: E402
from syncai_hydranet.models.crop_encoder import CropEncoder  # noqa: E402
from syncai_hydranet.models.hydranet import build_model  # noqa: E402
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights  # noqa: E402
from syncai_hydranet.utils.device import pick_device  # noqa: E402
from syncai_hydranet.utils.visualize import overlay, preprocess, terrain_palette  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("clip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seg-config", default="runs/hydranet_retail_surfaces/config.yaml")
    ap.add_argument("--seg-checkpoint", default="runs/hydranet_retail_surfaces/best.pt")
    ap.add_argument("--det-config", default="runs/hydranet_retail_cctv/config.yaml")
    ap.add_argument("--det-checkpoint", default="runs/hydranet_retail_cctv/best.pt")
    ap.add_argument("--encoder", default="runs/crop_encoder01/last.pt")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--max-frames", type=int, default=0)
    # Output size and quality. The first render was 1080p at crf 23 and came out 39 MB for
    # five minutes -- a file people cannot open rather than a video they can watch. The
    # overlay is drawn at full resolution and scaled on the way out, so nothing measured
    # changes; only the picture is smaller.
    # Segmentation input size, overriding the checkpoint's own. Looked at rather than
    # reasoned about: at the trained 512x640 a 16:9 frame becomes 640x360 of content, and
    # a person's mask bleeds a halo of `person` onto the floor around their feet. At
    # 768x1344 the same checkpoint tracks the trouser silhouette and the shoes separate.
    # Boundary pixels go 15,345 -> 21,728 while the class shares barely move (floor 24.6%
    # -> 24.5%), which is detail being resolved rather than noise being added.
    #
    # It is inference at a size the model was not trained at -- legitimate for a fully
    # convolutional net, out of distribution all the same, and 4.4x the pixels. The real
    # fix is to train at this size; this makes the gap visible before paying for it.
    ap.add_argument("--seg-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    ap.add_argument("--out-height", type=int, default=720, metavar="PX")
    ap.add_argument("--crf", type=int, default=28)
    ap.add_argument("--score-thr", type=float, default=0.25)
    ap.add_argument("--person-thr", type=float, default=0.20)
    # Taichung-cam01's tile-grid fit; an assumption on any other camera.
    ap.add_argument("--vfov", type=float, default=70.4)
    ap.add_argument("--pitch", type=float, default=50.2)
    ap.add_argument("--camera-height", type=float, default=2.38)
    ap.add_argument("--zone-centre", type=float, nargs=2, default=(0.88, 1.62), metavar="M")
    ap.add_argument("--zone-side", type=float, default=2.0)
    ap.add_argument("--loiter-seconds", type=float, default=30.0)
    return ap


def load(cfg_path: str, ckpt: str, device):
    cfg = load_config(cfg_path)
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(select_weights(load_checkpoint(ckpt), "ema"))
    return cfg, model


def crop_logits(enc, device, frame: np.ndarray, box) -> np.ndarray | None:
    """One frame's attribute logits for one box, or None if there is nothing to read."""
    x0, y0, x1, y1 = (int(v) for v in box)
    if x1 - x0 < 24 or y1 - y0 < 48:
        return None
    crop = Image.fromarray(frame[max(y0, 0) : y1, max(x0, 0) : x1]).resize((128, 256))
    x = ((np.asarray(crop, dtype=np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)
    with torch.no_grad():
        _, logits = enc(torch.from_numpy(x[None]).to(device))
    return logits[0].float().cpu().numpy()


BAND_TEXT = {"AgeLess18": "<18", "Age18-60": "18-60", "AgeOver60": ">60", "unknown": "?"}


def attribute_text(history: dict, track_id: int, w: int, h: int) -> str:
    """The track's pooled answer with its agreement, not this frame's guess.

    Per frame the gender decision flips on 16.2% of consecutive frame pairs on
    Kaohsiung-cam04, which is what makes the same staff member `F` and `M` in adjacent
    frames of the first render. Pooling removes that by construction; the agreement is
    printed because removing the flicker does not make the answer true, and roughly a
    third of tracks pool to something close to a coin flip.
    """
    rec = history.get(track_id)
    if rec is None or len(rec["logits"]) < 3:
        return ""
    lg = np.stack(rec["logits"])
    use = usable_crops(np.stack(rec["boxes"]), w, h)
    attrs = track_attributes(lg, ATTRIBUTES, use)
    if not attrs:
        return ""
    sex = attrs["Female"]
    band, _p = age_band(attrs)
    letter = "F" if sex.value else "M"
    return f"{letter} {sex.agreement:.0%} {BAND_TEXT.get(band, band)}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = pick_device(None)
    seg_cfg, seg = load(args.seg_config, args.seg_checkpoint, device)
    det_cfg, det = load(args.det_config, args.det_checkpoint, device)
    enc = CropEncoder(len(ATTRIBUTES)).to(device).eval()
    enc.load_state_dict(torch.load(args.encoder, map_location="cpu")["model"])

    if args.seg_size:
        seg_cfg["data"]["input_size"] = list(args.seg_size)
    classes = seg_cfg["data"].get("terrain_classes")
    palette = terrain_palette(classes)
    # Both vocabularies resolved from their configs. `person` happens to be channel 0 in
    # `retail_security` as in COCO, and relying on that coincidence is how a rename
    # becomes a silent relabel of every box. The merchandise head is two classes in the
    # surfaces line and four here, so the hardcoded pair raised IndexError on this
    # checkpoint -- which is the better of its two possible failures.
    det_names = detection_class_names(det_cfg) or tuple(COCO_NAMES)
    if "person" not in det_names:
        raise SystemExit(f"the detection head has no `person` channel: {det_names}")
    person_idx = det_names.index("person")
    seg_det_names = detection_class_names(seg_cfg) or tuple(COCO_NAMES)
    w, h, _ = probe(args.clip)
    cam = Camera.from_vfov(h, w, args.vfov)
    plane = GroundPlane(height=args.camera_height, pitch=math.radians(args.pitch))
    half = args.zone_side / 2
    cx, cz = args.zone_centre
    corners = np.array(
        [
            [cx - half, cz - half],
            [cx + half, cz - half],
            [cx + half, cz + half],
            [cx - half, cz + half],
        ]
    )
    zu, zv, _ = ground_to_pixel(corners[:, 0], corners[:, 1], cam, plane)
    zone_px = [(float(a), float(b)) for a, b in zip(zu, zv, strict=True)]
    zone = ev.Zone("counter", corners, loiter_seconds=args.loiter_seconds)

    tracker = Tracker()
    writer = None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    inside_since: dict[int, int] = {}
    # Per-track attribute evidence, accumulated as the clip plays. A live renderer pools
    # what it has seen so far, which is why the label firms up rather than flickering.
    history: dict[int, dict] = {}
    n = 0
    for frame in frames(args.clip, w, h, args.fps):
        img = Image.fromarray(frame)
        xs, _, region_s = preprocess(img, seg_cfg["data"]["input_size"])
        xd, _, region_d = preprocess(img, det_cfg["data"]["input_size"])
        with torch.no_grad():
            sraw = seg(xs.to(device))["terrain"]
            sres = seg.predict(xs.to(device), score_thr=args.score_thr)
            dres = det.predict(xd.to(device), score_thr=args.person_thr)

        # Upsample the logits and argmax after, not the other way round. Measured at only
        # 0.43% of pixels different from nearest-neighbour on the class map, so this is
        # correctness rather than the fix it was hoped to be -- interpolating class *ids*
        # is meaningless arithmetic even when it happens to look the same.
        x0, y0, cw, ch = region_s
        terrain = (
            torch.nn.functional.interpolate(
                sraw[:, :, y0 : y0 + ch, x0 : x0 + cw],
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            .argmax(1)[0]
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        canvas = overlay(img, terrain, palette, alpha=0.45)
        d = ImageDraw.Draw(canvas)
        d.polygon(zone_px, outline=(255, 220, 0))
        d.text(
            (zone_px[0][0], zone_px[0][1] - 14),
            f"zone {args.zone_side:g}x{args.zone_side:g} m",
            fill=(255, 220, 0),
        )

        sdet = sres["detection"][0]
        if len(sdet.get("labels", [])):
            boxes = to_source_pixels(sdet["boxes"].cpu().numpy(), region_s, w, h)
            for b, lab, sc in zip(
                boxes, sdet["labels"].cpu().numpy(), sdet["scores"].cpu().numpy(), strict=True
            ):
                d.rectangle(list(b), outline=(120, 200, 255))
                d.text(
                    (b[0], b[1] - 11),
                    f"{seg_det_names[int(lab)]} {sc:.2f}",
                    fill=(120, 200, 255),
                )

        ddet = dres["detection"][0]
        pboxes = np.zeros((0, 4))
        if len(ddet.get("labels", [])):
            lab = ddet["labels"].cpu().numpy()
            pboxes = to_source_pixels(
                ddet["boxes"].cpu().numpy()[lab == person_idx], region_d, w, h
            )
        live = tracker.update(pboxes, n)
        banner = []
        for t in live:
            if not t.confirmed:
                continue
            b = t.box
            d.rectangle(
                [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                outline=(255, 90, 90),
                width=2,
            )
            lg = crop_logits(enc, device, frame, b)
            if lg is not None:
                rec = history.setdefault(t.track_id, {"logits": [], "boxes": []})
                rec["logits"].append(lg)
                rec["boxes"].append(np.asarray(b, dtype=float))
            label = f"#{t.track_id} " + attribute_text(history, t.track_id, w, h)
            d.text((float(b[0]), float(b[1]) - 12), label, fill=(255, 200, 200))
            foot = np.array([[(b[0] + b[2]) / 2, b[3]]])
            from syncai_hydranet.geometry.ground import pixel_to_ground

            gx, gz = pixel_to_ground(foot[:, 0], foot[:, 1], cam, plane)
            if zone.contains(np.stack([gx, gz], axis=-1))[0]:
                inside_since.setdefault(t.track_id, n)
                held = (n - inside_since[t.track_id]) / args.fps
                if held >= args.loiter_seconds:
                    banner.append(
                        f"loitering #{t.track_id}  {held:.0f}s / {args.loiter_seconds:g}s"
                    )
            else:
                inside_since.pop(t.track_id, None)

        d.rectangle([0, 0, w, 26], fill=(0, 0, 0))
        d.text(
            (6, 7),
            f"frame {n}  |  L0 terrain+boxes  L1 tracks+attributes  L2 zone events",
            fill=(230, 230, 230),
        )
        for i, line in enumerate(banner[:3]):
            d.rectangle([0, 30 + i * 22, 420, 50 + i * 22], fill=(140, 0, 0))
            d.text((6, 33 + i * 22), line, fill=(255, 255, 255))

        if writer is None:
            writer = subprocess.Popen(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{w}x{h}",
                    "-r",
                    str(args.fps),
                    "-i",
                    "-",
                    "-an",
                    "-vcodec",
                    "libx264",
                    "-vf",
                    f"scale={int(round(w * (args.out_height or h) / h / 2) * 2)}:"
                    f"{args.out_height or h}",
                    "-preset",
                    "slow",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    str(args.crf),
                    "-movflags",
                    "+faststart",
                    args.out,
                ],
                stdin=subprocess.PIPE,
            )
        writer.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
    if writer is not None:
        writer.stdin.close()
        writer.wait()
    print(f"wrote {args.out}  ({n} frames at {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

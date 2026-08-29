#!/usr/bin/env python3
"""The GitHub social preview card, as code.

    uv run python tools/commissioning/social_card.py Kaohsiung-cam04

`assets/social_preview.png` was made by hand, which is the third figure in this repository
to have been -- after the README gif and the blur that goes on it, both of which are code
now for the same reason. **A card nobody can re-make is a card nobody can re-check when
the scene behind it moves**, and the scene moved four times on 2026-08-29 alone.

The previous card showed it. Its 3D panel was cut from the ragged world-axis scene path,
before `main()` moved to the store frame, so it carried the staircase artefact a reviewer
reads as "cabinets at 45 degrees"; the furniture ran off the bottom of its own frame; and
its eye was the fixed (+x, -z) diagonal with a wall standing in it. All three are fixed
here by construction rather than by re-cropping: the scene comes from
`build_scene_regular`, the eye is chosen, and the crop is computed from the content.

1280x640 is GitHub's own size for the field.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from syncai_bev3d import scene_mesh

ROOT = Path("/home/paul/SyncAI-Lib-HydraNet")
CARD = (1280, 640)
PANEL_W = 660  # the scene half; the text needs the rest at thumbnail size
CAPTION_H = 62  # `scene_mesh.render`'s own caption band, rendered then cut
BG = (13, 16, 22)
INK = (233, 238, 246)
DIM = (150, 162, 180)
BULLETS = (
    ((110, 210, 130), "Person detection and 17-point pose"),
    ((90, 160, 240), "Tracks in metres, not pixels"),
    ((200, 150, 235), "A metric 3D scene per camera, from one plate"),
)


def _font(size: int, bold: bool = False):
    for name in (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""),
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # **Taichung-cam11, chosen by looking at the alternatives rather than by convenience.**
    # Taichung-cam01 has the most furniture and the *least faithful* reconstruction --
    # `scene_overlay` puts nine meshes on it and most miss what is underneath -- and a
    # front page should not lead with the camera the projection test likes least.
    # Kaohsiung-cam04 is the most faithful but its scene is one counter and three units,
    # and one wall pane runs the full height of the panel and cuts the card in two.
    # cam11 reads as a room -- two tables, a shelving run with its levels, wall behind --
    # and its overlay is middling rather than poor.
    ap.add_argument("camera", nargs="?", default="Taichung-cam11")
    ap.add_argument("--out", default=str(ROOT / "assets/social_preview.png"))
    a = ap.parse_args()

    built = scene_mesh.build_scene_regular(a.camera)
    _cf, items, heights = built[:3]
    solid = [m for m, name, _al, _s in items if name != "floor"] or [m for m, *_ in items]
    xs = np.concatenate([m[0][:, 0] for m in solid])
    zs = np.concatenate([m[0][:, 2] for m in solid])
    span = max(float(xs.max() - xs.min()), float(zs.max() - zs.min()), 2.0)
    cx = float((xs.min() + xs.max()) / 2)
    cz = float((zs.min() + zs.max()) / 2)
    # The (-x, -z) corner, chosen by looking at all four: it sees into the room rather
    # than through the wall the default diagonal stands behind. Whatever still blocks it
    # fades, because `scene_mesh.render` now scores occlusion.
    eye = [cx - 0.62 * span, 0.46 * span, cz - 0.54 * span]
    tmp = ROOT / "assets/dev/_social_scene_mesh.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    # **Render at the panel's own aspect rather than cropping to it afterwards.** The
    # renderer frames the furniture to fill whatever canvas it is given, so asking it for
    # a 620x640 picture composes the room for that shape; cropping a 1180x700 render down
    # to the same shape threw away half the width and left a corner nobody could read.
    w0, h0 = scene_mesh.W, scene_mesh.H
    scene_mesh.W, scene_mesh.H = PANEL_W, CARD[1] + CAPTION_H
    try:
        scene_mesh.render(a.camera, items, heights, tmp, eye=eye, target=[cx, 0.6, cz])
    finally:
        scene_mesh.W, scene_mesh.H = w0, h0

    # `render` writes its own two-line commissioning caption at the top. That belongs on a
    # diagnostic, not on a card, so the canvas is asked for those rows and they are cut --
    # rather than painting over them, which would leave a bar the caption used to be in.
    panel_img = (
        Image.open(tmp).convert("RGB").crop((0, CAPTION_H, PANEL_W, CARD[1] + CAPTION_H))
    )
    card = Image.new("RGB", CARD, BG)
    # Full bleed on the right half. Fitting the scene *inside* a box left a band of card
    # background above and below it, and a preview card is read at thumbnail size where
    # empty space is the first thing that registers. The scene is cropped to the panel's
    # aspect around its own centre of mass rather than letterboxed into it.
    panel = (CARD[0] - PANEL_W, 0, CARD[0], CARD[1])
    card.paste(panel_img.resize((PANEL_W, CARD[1]), Image.Resampling.LANCZOS), (panel[0], 0))
    # a soft edge so the panel does not read as a pasted rectangle
    fade = Image.new("L", (56, CARD[1]))
    fd = ImageDraw.Draw(fade)
    for i in range(56):
        fd.line([(i, 0), (i, CARD[1])], fill=int(255 * (1 - i / 56)))
    card.paste(Image.new("RGB", (56, CARD[1]), BG), (panel[0], 0), fade)

    d = ImageDraw.Draw(card)
    d.text((68, 150), "SyncAI-Lib-HydraNet", font=_font(46, True), fill=INK)
    d.text((68, 214), "Security & retail analytics", font=_font(26), fill=DIM)
    d.text((68, 250), "for the CCTV already on the ceiling", font=_font(26), fill=DIM)
    y = 320
    for colour, text in BULLETS:
        d.rectangle([68, y + 6, 80, y + 18], fill=colour)
        d.text((94, y), text, font=_font(21), fill=INK)
        y += 42
    d.text(
        (68, 480),
        "One shared trunk, one forward pass. No LiDAR, no new hardware.",
        font=_font(19),
        fill=DIM,
    )
    d.text((68, 510), "PyTorch  ·  ONNX  ·  TensorRT", font=_font(19), fill=DIM)

    out = Path(a.out)
    card.save(out)
    tmp.unlink(missing_ok=True)
    print(f"wrote {out} ({CARD[0]}x{CARD[1]}, scene from {a.camera})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

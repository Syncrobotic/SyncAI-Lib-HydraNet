"""Count a false positive whose answer is known, over the whole clip.

Not a quality metric on the target domain -- there are no labels there and none is being
invented. This counts one thing only: how often each model paints `caution` on a fixed
structural column, in a hand-drawn box, on a camera that never moves. A structural column
is never `caution`, so every such pixel is wrong by construction, and 1830 frames is a
better sample than the four that were eyeballed.

Read off the rendered overlays rather than re-running inference: the renders are what was
actually inspected, so the count and the picture cannot drift apart. That also means the
inputs are `bev_video.py` outputs, which `.gitignore` keeps out of the repository -- so
this reproduces a published number rather than shipping the evidence for it. Re-render
the three clips first if they are not on disk.

The decode is `data/video.frames`, not a copy of it. There was a copy here, and its own
comment said so -- "same shape as `cli/infer_video.frames`" -- which did not stop it
carrying the same defect the original had: a short read ended the loop and ffmpeg's exit
status went unread. Every number below is `hits / total`, so a clip that half-decoded
would have moved the denominator and the percentage with it, silently, in a figure this
file exists to publish.

    python3 scripts/count_false_caution.py
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for candidate in (HERE.parent / "src", HERE / "src"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from syncai_hydranet.data.video import frames  # noqa: E402

CAM_FRAC = 0.65
ROI = (0.035, 0.29, 0.135, 0.58)  # same box drawn on the figures
# Sizes differ because the baseline runs at 512x640 and the other two at 512x896; the
# raw stream has no header, so the reader has to be told.
RUNS = [
    ("baseline", "assets/cmp_clip3_base_bev.mp4", 986, 360),
    ("restart only", "assets/cmp_clip3_noself_bev.mp4", 1378, 504),
    ("restart + site", "assets/cmp_clip3_cctv_bev.mp4", 1378, 504),
]


def is_caution(px):
    """The overlay blends a yellow onto the frame, so match by hue, not exact RGB:
    red and green both high and close, blue clearly below them."""
    r = px[..., 0].astype(np.int16)
    g = px[..., 1].astype(np.int16)
    b = px[..., 2].astype(np.int16)
    return (r > 120) & (g > 110) & (abs(r - g) < 45) & (r - b > 55) & (g - b > 45)


# the baseline renders at 512x640 and the other two at 512x896, so the box holds a
# different number of pixels in each -- report area as a fraction of the box, never raw px
missing = [p for _, p, _, _ in RUNS if not Path(p).is_file()]
if missing:
    sys.exit("re-render these with hydranet-scene first:\n  " + "\n  ".join(missing))

header = f"{'model':16}{'frames with the column marked caution':>40}"
print(header + f"{'mean % of box':>16}")
for name, path, w, h in RUNS:
    cw = int(w * CAM_FRAC)
    x0, y0 = int(ROI[0] * cw), int(ROI[1] * h)
    x1, y1 = int(ROI[2] * cw), int(ROI[3] * h)
    area = (x1 - x0) * (y1 - y0)
    hits, total, px_sum = 0, 0, 0
    for f in frames(path, w, h, None):
        roi = f[y0:y1, x0:x1]
        n = int(is_caution(roi).sum())
        total += 1
        px_sum += n
        if n > 0.02 * area:  # 2% of the box, i.e. a visible patch not a speckle
            hits += 1
    frac = 100 * hits / total
    mean_area = 100 * px_sum / total / area
    print(f"{name:16}{hits:>10} / {total:<6} ({frac:5.1f}%){'':>10}{mean_area:15.2f}")

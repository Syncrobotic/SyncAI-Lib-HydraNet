"""Blurring the faces out of a frame before it leaves the building.

This lived in `tools/commissioning/demo_video.py` until 2026-08-28, when `demo_gif.py`
needed to *audit* it -- "was this face inside a region the render blurred" -- and the only
honest way to answer that is to call the same arithmetic rather than a copy, since a copy
answers the question about itself. `tests/test_scripts_are_not_libraries.py` refused the
tool-to-tool import that made, and it was right to: `tools/` is outside the wheel, outside
the type ratchet and outside the coverage floor, and it resolves at all only because
Python puts the entry script's directory on `sys.path`. Of everything under `tools/`, the
code that decides whether a customer's face is published is the last thing that should
live outside all three.

---------------------------------------------------------------------------
TWO INSTRUMENTS, AND WHY NEITHER IS THE LAST WORD

`blur_rect` covers what the **detector** finds, at `BLUR_THR`. `plate_person_boxes` covers
what **changed against the empty shop**, which is nothing to a detector and so cannot miss
a person for the reason a detector does. They exist as a pair because a detector that
misses a shopper costs a track and a blur that misses one publishes a face, and the two
errors are not comparable.

**Both of them missed the same people on 2026-08-28.** `demo_gif.py`'s audit re-runs the
detector at 0.03 and requires every person to fall inside a blurred rectangle; on
Tao-Hsin-cam04 it found **132 of 954 person boxes over 120 frames with a readable head**,
scoring 0.08 against a threshold of 0.10 -- and one of them, read off the source frame by
eye, is two shoppers at a shelf with a man's profile plainly recognisable. The static
plate did not catch them either. So: two instruments, an audit over both, and a person
still looking at contact sheets. Each of those has now failed at least once.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

# The blur threshold is deliberately BELOW the shipped detection one. See the module
# docstring for the failure that set it: 0.10 was below the shipped 0.35 and still not low
# enough, and the cost of fixing it is measured rather than assumed. Over the same windows
# 0.10 -> 0.07 takes Tao-Hsin-cam04's readable count to 0 while the blurred fraction of the
# frame goes 7.0% -> 7.2%, and changes nothing at all on Kaohsiung-cam04 (0 readable either
# way, 32.4% both). 0.03 costs no more again and is deliberately NOT taken: `demo_gif`'s
# audit runs at 0.03, and a render that blurs at exactly the threshold its auditor inspects
# at is checking itself with its own answer. 0.07 leaves 0.03-0.07 a question still asked.
BLUR_THR = 0.07

BLUR_PAD = 0.12  # outward margin on each side, as a fraction of the box
BLUR_TOP_FRACTION = 0.45  # head and shoulders; below this is torso, which stays readable

# What counts as a person-shaped change against the static plate. Deliberately loose --
# a false positive costs a blurred shelf.
PLATE_DIFF = 34  # 0-255 on grey; below this a pixel is lighting, not a person
PLATE_MIN_PX = 1200
PLATE_ASPECT = (1.0, 7.0)  # height / width of a standing person, generously


def blur_rect(
    w_img: int, h_img: int, x0: float, y0: float, x1: float, y1: float
) -> tuple[int, int, int, int] | None:
    """The rectangle `blur_region` would blur for this box, or None if it blurs nothing.

    Separated from the blurring so an auditor can ask "was this face inside a blurred
    region" without a second copy of the arithmetic.

    `float()` on the corners is not decoration, and it has been removed twice. The
    detector's boxes arrive as numpy scalars, and `ImageFilter.GaussianBlur` compares its
    radius against a tuple, which on a numpy scalar raises "truth value of an array is
    ambiguous" rather than blurring -- twelve minutes into a render, on its first person.
    """
    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    w, h = x1 - x0, y1 - y0
    if w < 4 or h < 4:
        return None
    px, py = w * BLUR_PAD, h * BLUR_PAD
    bx0 = max(0, int(x0 - px))
    by0 = max(0, int(y0 - py))
    bx1 = min(w_img, int(x1 + px))
    by1 = min(h_img, int(y0 + h * BLUR_TOP_FRACTION + py))
    if bx1 - bx0 < 3 or by1 - by0 < 3:
        return None
    return bx0, by0, bx1, by1


def blur_region(img: Image.Image, x0: float, y0: float, x1: float, y1: float) -> None:
    """Blur the head-and-shoulders of one box, in place, at a radius set by its width."""
    x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
    rect = blur_rect(img.width, img.height, x0, y0, x1, y1)
    if rect is None:
        return
    bx0, by0, bx1, by1 = rect
    crop = img.crop((bx0, by0, bx1, by1))
    img.paste(crop.filter(ImageFilter.GaussianBlur(max(6.0, (x1 - x0) / 5.0))), (bx0, by0))


def plate_person_boxes(frame: np.ndarray, plate: np.ndarray) -> list[tuple]:
    """Person-shaped regions that changed against the static plate.

    The second instrument, and the reason there are two: this one is nothing to a
    detector, so it cannot miss a person for the reason a detector does -- a shopper the
    model scores 0.04 on is still a region of the frame that is not the empty shop. It
    over-triggers on trolleys and opened doors, which costs a blurred trolley.
    """
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - the extra is installed in CI
        raise RuntimeError(
            "the static-plate blur instrument needs scipy and it is not installed, so "
            "only ONE of the two blur instruments can run. Install the extra "
            "(`uv sync --extra commission`) and try again. This refuses rather than "
            "silently publishing frames checked by half the pipeline."
        ) from exc

    d = np.abs(frame.astype(np.int16).mean(2) - plate.astype(np.int16).mean(2))
    m = ndimage.binary_opening(d > PLATE_DIFF, np.ones((5, 5)))
    lab, _n = ndimage.label(m)
    out = []
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if h * w < PLATE_MIN_PX or w < 1:
            continue
        if not (PLATE_ASPECT[0] <= h / w <= PLATE_ASPECT[1]):
            continue
        out.append((sl[1].start, sl[0].start, sl[1].stop, sl[0].stop))
    return out

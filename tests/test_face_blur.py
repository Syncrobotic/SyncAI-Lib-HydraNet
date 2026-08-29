"""The face blur's rectangle, and the numpy-scalar trap that has now bitten twice.

`blur_region`'s docstring has always said that `float()` on the corners "is not
decoration": the detector's boxes arrive as numpy scalars, and
`ImageFilter.GaussianBlur` compares its radius against a tuple, which on a numpy scalar
raises "truth value of an array is ambiguous" rather than blurring. On 2026-08-28 the
rectangle was factored out into `blur_rect` so `demo_gif.py` could audit against the same
arithmetic instead of a copy — and the factoring moved the `float()` calls with it,
leaving the radius on the raw scalars. The next render died on its first person, twelve
minutes in.

A docstring said it and nothing checked it, which is the pattern both 2026-08-28 handoffs
name. These tests are the check.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from syncai_hydranet.utils import face_blur


def _blank(w: int = 200, h: int = 200) -> Image.Image:
    return Image.fromarray(np.zeros((h, w, 3), np.uint8))


def test_a_numpy_box_blurs_rather_than_raising():
    """The regression, in the exact shape the detector hands over: numpy float64."""
    box = np.array([10.0, 10.0, 60.0, 160.0])
    face_blur.blur_region(_blank(), *box)  # must not raise


def test_a_python_box_and_a_numpy_box_blur_the_same_pixels():
    a, b = _blank(), _blank()
    face_blur.blur_region(a, 10.0, 10.0, 60.0, 160.0)
    face_blur.blur_region(b, *np.array([10.0, 10.0, 60.0, 160.0]))
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_the_rectangle_covers_the_head_and_not_the_whole_person():
    """`BLUR_TOP_FRACTION` is the claim: heads go, torsos stay -- and the torso staying is
    what `analytics.staff` reads, so this boundary has two consumers, not one."""
    x0, y0, x1, y1 = 40.0, 20.0, 80.0, 180.0
    bx0, by0, bx1, by1 = face_blur.blur_rect(200, 200, x0, y0, x1, y1)
    assert bx0 <= x0 and bx1 >= x1  # padded outwards, never inside the box
    assert by0 <= y0
    assert y0 < by1 < y1  # the head, not the person


def test_the_rectangle_is_clipped_to_the_image():
    bx0, by0, bx1, by1 = face_blur.blur_rect(200, 200, -50.0, -50.0, 260.0, 300.0)
    assert (bx0, by0) == (0, 0)
    assert bx1 <= 200 and by1 <= 200


def test_a_box_too_small_to_carry_a_face_is_declined_by_both():
    """`blur_rect` returning None and `_blur_region` doing nothing have to agree, or the
    auditor treats a face as uncovered that the render never had to cover."""
    # 3 px wide and 40 tall: a sliver the detector produces at a frame edge. The width
    # test is the one that has to catch it -- an earlier version of this test used a
    # 2x2 box, which the *rectangle*-size test below the width test also rejects, so
    # deleting the width test left it green. Verified by deleting each in turn.
    sliver = (10.0, 10.0, 13.0, 50.0)
    assert face_blur.blur_rect(200, 200, *sliver) is None
    before = _blank()
    after = before.copy()
    face_blur.blur_region(after, *sliver)
    assert np.array_equal(np.asarray(before), np.asarray(after))

"""Unit tests for letterbox preprocessing and the label scheme registry.

pytest tests/test_transforms.py -v
"""

import numpy as np
import pytest
from PIL import Image

from syncai_hydranet.data import label_maps
from syncai_hydranet.data.transforms import (
    PAD_LABEL,
    LetterboxResize,
    LetterboxScaleCrop,
    Resize,
    Sample,
    build_transforms,
)
from syncai_hydranet.utils.visualize import crop_box, letterbox

SIZE = (384, 512)  # H, W -> 1.333:1


def _sample(w, h, with_boxes=False):
    """Half-red, half-blue image: any geometric corruption is easy to see."""
    arr = np.zeros((h, w, 3), np.uint8)
    arr[:, : w // 2] = (255, 0, 0)
    arr[:, w // 2 :] = (0, 0, 255)
    s = Sample(image=Image.fromarray(arr))
    if with_boxes:
        s["boxes"] = np.array([[0.25 * w, 0.25 * h, 0.75 * w, 0.75 * h]], np.float32)
        s["labels"] = np.array([3], np.int64)
    else:
        s["masks"] = {"terrain": np.full((h, w), 5, np.uint8)}
    return s


# ---------------------------------------------------------------- letterbox


@pytest.mark.parametrize(("w", "h"), [(1920, 1080), (1080, 1920), (640, 512), (800, 800)])
def test_letterbox_preserves_aspect(w, h):
    img = Image.new("RGB", (w, h))
    _, (x0, y0, cw, ch) = letterbox(img, SIZE)
    assert abs(cw / ch - w / h) < 0.02
    assert cw <= SIZE[1] and ch <= SIZE[0]
    assert x0 >= 0 and y0 >= 0


def test_plain_resize_distorts_but_letterbox_does_not():
    """A portrait 9:16 frame: plain Resize squeezes it hard, letterbox does not."""
    w, h = 1080, 1920
    src_aspect = w / h  # 0.5625

    squeezed = Resize(SIZE)(_sample(w, h))["image"]
    assert abs(squeezed.width / squeezed.height - src_aspect) > 0.7

    _, region = letterbox(Image.new("RGB", (w, h)), SIZE)
    assert abs(region[2] / region[3] - src_aspect) < 0.02


def test_letterbox_resize_pads_masks_with_ignore():
    """Padding must be 255, otherwise it would be treated as a real label."""
    s = LetterboxResize(SIZE)(_sample(1080, 1920))
    m = s["masks"]["terrain"]
    assert m.shape == SIZE
    assert (m == PAD_LABEL).any(), "expected padding"
    assert (m == 5).any(), "content labels must survive"
    assert (m[:, 0] == PAD_LABEL).all(), "portrait source pads on the left and right"


def test_letterbox_scale_crop_keeps_shape_and_valid_labels():
    for _ in range(20):  # covers both the crop and the pad branches
        s = LetterboxScaleCrop(SIZE, scale_range=(0.5, 2.0))(_sample(1920, 1080))
        assert s["image"].size == (SIZE[1], SIZE[0])
        m = s["masks"]["terrain"]
        assert m.shape == SIZE
        assert set(np.unique(m)).issubset({5, PAD_LABEL})


def test_letterbox_boxes_stay_inside_canvas():
    for _ in range(20):
        s = LetterboxScaleCrop(SIZE, scale_range=(0.5, 2.0))(_sample(1920, 1080, True))
        b = s["boxes"]
        if len(b) == 0:
            continue  # cropping the box away entirely is a legal outcome
        assert (b[:, 0] >= 0).all() and (b[:, 1] >= 0).all()
        assert (b[:, 2] <= SIZE[1]).all() and (b[:, 3] <= SIZE[0]).all()
        assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()


def test_crop_box_roundtrip():
    _, region = letterbox(Image.new("RGB", (1080, 1920)), SIZE)
    out = crop_box(np.zeros(SIZE, np.uint8), region)
    assert out.shape == (region[3], region[2])


def test_build_transforms_flag_selects_pipeline():
    assert isinstance(build_transforms(SIZE, False, letterbox=True).ts[0], LetterboxResize)
    assert isinstance(build_transforms(SIZE, False, letterbox=False).ts[0], Resize)
    assert isinstance(build_transforms(SIZE, True, letterbox=True).ts[0], LetterboxScaleCrop)


# ------------------------------------------------------------ label schemes


def test_schemes_are_self_consistent():
    """Every mapped terrain id must be in range and have a traversability entry;
    otherwise training silently produces all-ignore labels."""
    for name, sc in label_maps.SCHEMES.items():
        assert sc.num_classes == 12, name
        for t in set(sc.mapping.values()):
            assert 0 <= t < sc.num_classes, f"{name}: terrain id {t} out of range"
            assert t in sc.trav, f"{name}: terrain id {t} has no traversability mapping"


def test_indoor_scheme_marks_glass_and_stairs():
    """The two highest-consequence indoor classes must never be labelled walkable."""
    sc = label_maps.get_scheme("ade20k_indoor")
    glass = sc.classes.index("glass")
    stairs = sc.classes.index("stairs")
    assert sc.trav[glass] == 0
    assert sc.trav[stairs] != 2
    assert glass in sc.mapping.values(), "ADE20K should map some source class to glass"


def test_outdoor_scheme_unchanged():
    """The off-road scheme must not be affected by the indoor additions."""
    sc = label_maps.get_scheme("rugd")
    assert sc.fmt == "color"
    assert sc.classes[2] == "grass"
    assert sc.trav[sc.classes.index("grass")] == 1  # grass -> caution


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unknown label_map"):
        label_maps.get_scheme("nope")

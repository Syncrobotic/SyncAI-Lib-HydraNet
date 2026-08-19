"""The Jetson scripts duplicate on purpose. This is what keeps the duplicates honest.

`bench_camera_orin.py` and `live_view_orin.py` run on a board with numpy and TensorRT
and no syncai_hydranet installed, so they carry their own MEAN, STD, letterbox and
palettes. That is a real constraint, not laziness, and importing is not an option.

What is not acceptable is a copy that drifts. `live_view_orin.py` carried a comment
saying its palettes matched the training-time grids; the package meanwhile held the
off-road palette, so the same class was a different colour in the live view and in
TensorBoard, and the comment is precisely why nobody looked. A normalisation constant
that drifted would be worse and quieter: the board would feed the network inputs it was
never trained on and the only symptom would be predictions that are a bit off.

These run on the dev box, where both sides are importable. That is the only place the
check can exist, which is exactly why it has to exist here rather than being left to
whoever next reads both files.

pytest tests/test_orin_standalone_copies.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "robot"


@pytest.fixture(scope="module")
def orin():
    """The two standalone modules, imported the way they import each other."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        import bench_camera_orin
        import live_view_orin
    finally:
        sys.path.remove(str(SCRIPTS))
    return bench_camera_orin, live_view_orin


def test_normalisation_constants_match_the_package(orin):
    """Different numbers here mean the board preprocesses differently from training."""
    from syncai_hydranet.preprocessing import IMAGENET_MEAN, IMAGENET_STD

    bench, _ = orin
    assert np.array_equal(bench.MEAN, IMAGENET_MEAN)
    assert np.array_equal(bench.STD, IMAGENET_STD)


def test_traversability_palette_matches(orin):
    from syncai_hydranet.utils.visualize import TRAV_COLORS

    _, live = orin
    assert np.array_equal(live.TRAV_COLORS, TRAV_COLORS)


def test_terrain_palette_matches_the_indoor_one(orin):
    """The regression. These were equal for the first five classes and unrelated after
    it, which is the worst shape a mismatch can take: it looks right at a glance."""
    from syncai_hydranet.utils.visualize import TERRAIN_COLORS_INDOOR

    _, live = orin
    assert np.array_equal(live.TERRAIN_COLORS, TERRAIN_COLORS_INDOOR)


def test_class_names_match_the_package_modulo_the_hyphen(orin):
    """The board draws category names on frames; the package scores against them.

    The two spellings differ by one documented rule -- the board hyphenates so a label
    stays one token on an overlay -- and by nothing else. Order matters as much as
    spelling: index i is class i out of the head, so a list that drifts by one entry
    renames every detection after it and still looks like a working system.
    """
    from syncai_hydranet.data.coco_subsets import COCO_NAMES

    _, live = orin
    assert [n.replace("-", " ") for n in live.COCO_NAMES] == COCO_NAMES


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("cv2") is None,
    reason="bench_camera_orin.letterbox needs cv2, which the board has and this box "
    "does not; the constants and palettes above are checked either way",
)
def test_letterbox_produces_the_same_pixels(orin):
    """Two implementations -- PIL in the package, numpy on the board -- so equality is
    checked by running them, not by reading them.

    Bilinear resampling differs slightly between the two libraries, so this asks that
    the geometry agrees exactly (padding colour, content placement and size) and that
    the content matches closely, which is what a layout bug would break and a resampling
    difference would not.
    """
    from syncai_hydranet.utils.visualize import letterbox as pil_letterbox

    bench, _ = orin
    h, w = 64, 96
    frame = np.zeros((50, 100, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(0, 255, 100, dtype=np.uint8)  # a horizontal ramp
    frame[:, :, 1] = 128

    board = np.asarray(bench.letterbox(frame, h, w), dtype=np.uint8)
    package = np.asarray(pil_letterbox(Image.fromarray(frame), (h, w)), dtype=np.uint8)
    assert board.shape == package.shape

    # Padding is the unambiguous part: identical fill, in identical places.
    pad = np.all(package == 114, axis=-1)
    assert pad.any(), "this fixture is meant to need padding"
    assert np.array_equal(np.all(board == 114, axis=-1), pad)
    assert np.abs(board[~pad].astype(int) - package[~pad].astype(int)).mean() < 8


# --- binding the CUDA runtime ----------------------------------------------
#
# Not a package-vs-board equality check like the rest of this file, but the same
# category of defect: a constant in a standalone copy that is right on one machine and
# silently wrong on the next. `libcudart.so.12` was hardcoded, a GB10 carries only
# `.so.13`, and `Cudart()` is built at the top of both main loops -- so neither script
# started at all, and the error named a missing file rather than a CUDA version.


def test_sonames_are_tried_newest_first_with_the_unversioned_one_last(orin):
    """The unversioned name usually exists only where the CUDA *development* package is
    installed, which a deployment board need not have -- so it is a fallback, not the
    first guess."""
    bench, _ = orin
    majors = [s for s in bench.CUDART_SONAMES if s != "libcudart.so"]
    assert bench.CUDART_SONAMES[-1] == "libcudart.so"
    assert majors == sorted(majors, key=lambda s: -int(s.rsplit(".", 1)[1]))
    assert "libcudart.so.13" in majors, "CUDA 13 boards exist; this list has to know"


def test_a_newer_cuda_is_bound_in_preference_to_an_older_one(orin, monkeypatch):
    bench, _ = orin
    loaded = {}

    class _Lib:
        def __init__(self, name):
            self.cudaGetErrorString = type("F", (), {"restype": None})()
            loaded["name"] = name

    def _cdll(name):
        if name != "libcudart.so.12":
            raise OSError(f"{name}: cannot open shared object file")
        return _Lib(name)

    monkeypatch.setattr(bench.ctypes, "CDLL", _cdll)
    cuda = bench.Cudart()
    assert cuda.soname == "libcudart.so.12"  # .13 was tried first and refused
    assert loaded["name"] == "libcudart.so.12"


def test_no_runtime_at_all_names_every_attempt_and_says_what_to_edit(orin, monkeypatch):
    """The failure this replaces printed one filename. A board with an unknown CUDA major
    needs to be told which names were tried and where to add its own."""
    bench, _ = orin

    def _cdll(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(bench.ctypes, "CDLL", _cdll)
    with pytest.raises(SystemExit) as exc:
        bench.Cudart()
    message = str(exc.value)
    for soname in bench.CUDART_SONAMES:
        assert soname in message
    assert "CUDART_SONAMES" in message

"""hydranet-prepare-cocostuff: the class resolution, the refusal, and the manifest.

This command had no tests at all. It also had no `build_parser`, alone among the ten
CLIs, which is why: its whole argument surface lived inside `main` and could only be
reached by running the thing against a 39 GB dataset.

Everything here runs on a synthetic tree of 20x20 PNGs. What is worth pinning is not the
selection arithmetic -- `_scan` is small and obvious -- but the two behaviours whose
failure is silent: names resolved against the dataset's own labels.txt rather than
transcribed, and the refusal to merge into a split someone else already wrote.

The synthetic tree is built out of the module's own class tuples, which leaves it blind to
a typo in them -- a misspelling would be written into the fixture's labels.txt and resolve
against its own reflection. `test_every_declared_name_is_a_real_cocostuff_class` is the
one test here that steps outside that circle.

pytest tests/test_prepare_cocostuff.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from syncai_hydranet.cli.prepare_cocostuff import (
    DEFAULT_CLASSES,
    INDOOR_EVIDENCE,
    OUTDOOR_VETO,
    VEHICLE_VETO,
    _load_png_values,
    _resolve,
    build_parser,
    main,
)

ALL_NAMES = sorted(
    set(DEFAULT_CLASSES) | set(INDOOR_EVIDENCE) | set(OUTDOOR_VETO) | set(VEHICLE_VETO)
)
VALUE = {n: i for i, n in enumerate(ALL_NAMES)}

# The real COCO-Stuff labels.txt ids for every name the four tuples above declare,
# transcribed from the dataset rather than generated from the tuples. `source` builds its
# labels.txt out of `ALL_NAMES`, so it cannot catch a typo in the tuples -- the typo would
# be written into the synthetic file and resolve against itself. This is the one place the
# names are checked against something that did not come from them.
REAL_IDS = {
    "airplane": 5,
    "boat": 9,
    "building-other": 96,
    "bus": 6,
    "bush": 97,
    "car": 3,
    "carpet": 101,
    "ceiling-other": 102,
    "ceiling-tile": 103,
    "clouds": 106,
    "door-stuff": 112,
    "floor-marble": 114,
    "floor-other": 115,
    "floor-stone": 116,
    "floor-tile": 117,
    "floor-wood": 118,
    "grass": 124,
    "mat": 131,
    "mirror-stuff": 133,
    "motorcycle": 4,
    "mountain": 135,
    "pavement": 140,
    "river": 148,
    "road": 149,
    "roof": 151,
    "rug": 152,
    "sand": 154,
    "sea": 155,
    "sky-other": 157,
    "snow": 159,
    "stairs": 161,
    "train": 7,
    "tree": 169,
    "truck": 8,
    "wall-other": 173,
    "wall-panel": 174,
    "wall-tile": 176,
    "wall-wood": 177,
    "window-other": 181,
}


def _ann(path: Path, rows: dict[str, int]) -> None:
    """A 20-row annotation, each named class occupying that many rows from the top.

    The rows must add up to 20. Value 0 is a real class here -- `ALL_NAMES` is sorted, so
    it is `airplane`, a vehicle veto -- and a couple of unassigned rows at the bottom are
    enough to veto a frame the test meant to keep.
    """
    assert sum(rows.values()) == 20, "leftover rows would fall through to a veto class"
    a = np.zeros((20, 20), np.uint8)
    r = 0
    for name, n in rows.items():
        a[r : r + n] = VALUE[name]
        r += n
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


@pytest.fixture
def source(tmp_path):
    """Per split: one indoor keeper, one outdoor frame carrying the same class, one
    frame carrying none of them, and one annotation whose image was never unpacked."""
    stuff, coco = tmp_path / "cocostuff", tmp_path / "coco"
    stuff.mkdir(parents=True)
    # 1-based on purpose: labels.txt id N is stored as N-1 in the PNGs, and this file is
    # the only thing that knows it. Transcribing ids by hand is the documented way to get
    # this wrong by one and filter on the neighbouring class.
    (stuff / "labels.txt").write_text(
        "\n".join(f"{i + 1}: {n}" for i, n in enumerate(ALL_NAMES))
    )
    for split in ("train2017", "val2017"):
        _ann(
            stuff / split / f"{split}_keep.png", {DEFAULT_CLASSES[0]: 8, INDOOR_EVIDENCE[0]: 12}
        )
        _ann(
            stuff / split / f"{split}_street.png", {DEFAULT_CLASSES[0]: 8, OUTDOOR_VETO[0]: 12}
        )
        _ann(stuff / split / f"{split}_none.png", {INDOOR_EVIDENCE[0]: 20})
        _ann(
            stuff / split / f"{split}_orphan.png",
            {DEFAULT_CLASSES[0]: 8, INDOOR_EVIDENCE[0]: 12},
        )
        d = coco / split
        d.mkdir(parents=True, exist_ok=True)
        for stem in (f"{split}_keep", f"{split}_street", f"{split}_none"):
            Image.new("RGB", (20, 20)).save(d / f"{stem}.jpg")
    return stuff, coco


def _run(source, out: Path, *extra):
    stuff, coco = source
    return main(
        [
            "--coco",
            str(coco),
            "--stuff",
            str(stuff),
            "--out",
            str(out),
            "--workers",
            "1",
            *extra,
        ]
    )


def _names(out: Path, kind: str, split: str) -> set[str]:
    d = out / kind / split
    return {p.name for p in d.iterdir()} if d.is_dir() else set()


def test_parser_defaults_are_the_documented_ones():
    args = build_parser().parse_args([])
    assert args.classes == list(DEFAULT_CLASSES)
    assert args.min_fraction == 0.01
    assert args.no_indoor_filter is False
    assert args.force is False


def test_a_class_name_that_is_not_in_labels_txt_is_refused(source, tmp_path):
    """Names are resolved against the dataset rather than transcribed, so a typo has to
    fail loudly. A hand-written id would not: it silently filters on the neighbouring
    class and the split still looks reasonable, which is the mistake the module docstring
    says was made once already."""
    with pytest.raises(SystemExit, match="not COCO-Stuff class names"):
        _run(source, tmp_path / "out", "--classes", "not-a-coco-class")


def test_train_becomes_train_and_val_becomes_test(source, tmp_path):
    """No val split by design: selection stays on ADE20K, so this one stays clean."""
    out = tmp_path / "out"
    assert _run(source, out) == 0
    assert (out / "images" / "train").is_dir()
    assert (out / "images" / "test").is_dir()
    assert not (out / "images" / "val").exists()


def test_the_indoor_filter_drops_a_frame_that_carries_the_class_outdoors(source, tmp_path):
    """A street frame with stairs in it teaches a shop robot the wrong thing about its
    highest-consequence class, and no summary statistic shows it."""
    out = tmp_path / "out"
    _run(source, out)
    assert _names(out, "images", "train") == {"train2017_keep.jpg"}
    assert (
        json.loads((out / "manifest.json").read_text())["splits"]["train"]["dropped_not_indoor"]
        == 1
    )


def test_disabling_the_indoor_filter_keeps_the_street_frame(source, tmp_path):
    out = tmp_path / "out"
    _run(source, out, "--no-indoor-filter")
    assert _names(out, "images", "train") == {"train2017_keep.jpg", "train2017_street.jpg"}


def test_an_annotation_with_no_image_is_counted_not_silently_skipped(source, tmp_path):
    """It means the two archives came from different releases, and the count is the only
    signal of that."""
    out = tmp_path / "out"
    _run(source, out)
    assert (
        json.loads((out / "manifest.json").read_text())["splits"]["train"]["missing_images"]
        == 1
    )


def test_outputs_are_symlinks_that_resolve(source, tmp_path):
    out = tmp_path / "out"
    _run(source, out)
    for p in (out / "images" / "train").iterdir():
        assert p.is_symlink() and p.resolve().is_file()


def test_it_refuses_to_merge_into_a_split_someone_already_wrote(source, tmp_path):
    """A half-rebuilt split is indistinguishable from a good one from the outside."""
    out = tmp_path / "out"
    _run(source, out)
    with pytest.raises(SystemExit, match="Refusing to merge"):
        _run(source, out)


def test_force_overwrites_instead_of_refusing(source, tmp_path):
    out = tmp_path / "out"
    _run(source, out)
    assert _run(source, out, "--force") == 0


def test_the_manifest_records_what_the_filter_was(source, tmp_path):
    """Which frames are in a split is a function of these settings, and the split
    outlives the shell history that produced it."""
    out = tmp_path / "out"
    _run(source, out, "--min-fraction", "0.25")
    m = json.loads((out / "manifest.json").read_text())
    assert m["min_fraction"] == 0.25
    assert m["indoor_filter"] is True
    assert m["classes"] == list(DEFAULT_CLASSES)


def test_every_declared_name_is_a_real_cocostuff_class(tmp_path):
    """The four tuples are the whole selection policy, and a typo in one of them does not
    raise: `stairs-other` matches nothing and quietly narrows the split, while a name that
    happens to exist filters on the wrong class. Every other test here builds its
    labels.txt from these same tuples, so only a transcribed one can disagree."""
    assert set(REAL_IDS) == set(ALL_NAMES), "REAL_IDS has drifted from the declared names"
    stuff = tmp_path / "real"
    stuff.mkdir()
    (stuff / "labels.txt").write_text(
        "\n".join(f"{i}: {n}" for n, i in sorted(REAL_IDS.items(), key=lambda kv: kv[1]))
    )
    png = _load_png_values(stuff)
    for names in (DEFAULT_CLASSES, INDOOR_EVIDENCE, OUTDOOR_VETO, VEHICLE_VETO):
        assert len(_resolve(png, names)) == len(set(names))

"""COCO-Stuff's PNG values sit one below its own labels.txt, and `person` is value 0.

Both facts were established by counting pixels, not by reading the dataset's label file,
and both fail silently if they regress: an off-by-one relabels every class at once while
every metric stays plausible, and treating 0 as void deletes every person in 118,287
images without an error.

The first three tests pin the invariants with no data on disk, so CI keeps them. The last
one re-derives the offset from the shipped annotations and skips when they are absent.

pytest tests/test_cocostuff_scheme.py -v
"""

from pathlib import Path

import numpy as np
import pytest

from syncai_hydranet.data.label_maps import get_scheme
from syncai_hydranet.data.label_maps_cocostuff import (
    COCOSTUFF_ID_TO_INDOOR,
    COCOSTUFF_ID_TO_RETAIL,
)
from syncai_hydranet.data.label_maps_indoor import INDOOR_TERRAIN
from syncai_hydranet.data.label_maps_retail import RETAIL_TERRAIN

DATA = Path(__file__).resolve().parents[1] / "datasets" / "cocostuff"

# labels.txt id -> the PNG value that id is stored as. Written out rather than computed,
# so a change to the shift has to disagree with a literal instead of following it.
LABELS_TXT = {"person": 1, "stairs": 161, "door-stuff": 112, "shelf": 156, "table": 165}


def test_png_values_are_one_below_labels_txt():
    assert COCOSTUFF_ID_TO_INDOOR[LABELS_TXT["stairs"] - 1] == INDOOR_TERRAIN["stairs"]
    assert COCOSTUFF_ID_TO_INDOOR[LABELS_TXT["door-stuff"] - 1] == INDOOR_TERRAIN["door"]
    # the unshifted ids must NOT be mapped, or both readings would appear to work
    assert LABELS_TXT["stairs"] not in COCOSTUFF_ID_TO_INDOOR
    assert LABELS_TXT["door-stuff"] not in COCOSTUFF_ID_TO_INDOOR


def test_person_is_png_value_zero():
    """The inverse of the in-house convention, where 0 means "nobody labelled this"."""
    assert COCOSTUFF_ID_TO_INDOOR[0] == INDOOR_TERRAIN["person"]
    assert COCOSTUFF_ID_TO_RETAIL[0] == INDOOR_TERRAIN["person"]


def test_dead_thing_ids_are_not_mapped():
    """mirror(66), window(68) and door(71) were dropped from COCO and carry no pixels.

    Mapping one costs nothing at runtime and quietly removes a class from the mIoU
    denominator, which is the failure `git show b7457c2:docs/METHODOLOGY.md` existed to prevent.
    """
    for dead in (66, 68, 71):
        assert dead - 1 not in COCOSTUFF_ID_TO_INDOOR, f"labels.txt {dead} is a dead class"


def test_fixtures_match_the_ade20k_scheme_and_exclude_table():
    fixture = RETAIL_TERRAIN["display_fixture"]
    assert COCOSTUFF_ID_TO_RETAIL[LABELS_TXT["shelf"] - 1] == fixture
    # `table` is the near-miss proxy ade20k_retail deliberately refuses
    assert COCOSTUFF_ID_TO_RETAIL[LABELS_TXT["table"] - 1] != fixture
    # and the indoor scheme must not invent the class at all
    assert fixture not in COCOSTUFF_ID_TO_INDOOR.values()


def test_schemes_are_registered():
    for name in ("cocostuff_indoor", "cocostuff_retail"):
        assert get_scheme(name).fmt == "id"


@pytest.mark.skipif(not (DATA / "train2017").is_dir(), reason="COCO-Stuff not downloaded")
def test_offset_still_holds_against_the_shipped_annotations():
    """Re-derive the offset from the data instead of trusting the constant.

    `person` is in roughly half of COCO and nothing else comes close at that frequency,
    so whichever value dominates the image count identifies the offset on its own.
    """
    from PIL import Image

    files = sorted((DATA / "train2017").iterdir())[:1500]
    counts = np.zeros(256, dtype=np.int64)
    for f in files:
        counts[np.unique(np.asarray(Image.open(f)))] += 1

    non_ignore = counts.copy()
    non_ignore[255] = 0  # 255 is unlabelled and is in nearly every image
    assert int(np.argmax(non_ignore)) == 0, "person should be the most common PNG value, 0"
    assert 0.4 < counts[0] / len(files) < 0.7, "person should appear in roughly half of COCO"

    # and the class this dataset was downloaded for is actually present
    assert counts[LABELS_TXT["stairs"] - 1] > 0

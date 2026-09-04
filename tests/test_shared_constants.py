"""Numbers this project states once and restated in several places.

Each entry here is a constant whose second copy would not raise -- it would move a
measurement quietly. The audit of 2026-09-04 found the 1.70 m adult prior in four places
under a docstring saying it "must exist exactly once", and the 0.35 person threshold as
three named constants, with two files in one directory reading it from two different
modules. The imports now converge; these hold that they still do.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    """A script by path, the way it is run rather than imported."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_person_threshold_has_one_source():
    """0.35 is the measured day/night gap -- night IR tops out at 0.326, day clears 0.35.

    It belongs to the Grounding DINO teacher that emits the boxes. `night_person`
    describes the population that threshold produces, and `campaign_site30k` trains on it;
    both stated it independently, and `tools/site30k/boxes.py` read the campaign's copy
    while its neighbour `box_pass.py` read the teacher's -- one directory, two routes to
    one number.
    """
    from syncai_bev3d.teachers.gdino import PERSON_THRESHOLD
    from syncai_hydranet.data.night_person import PERSON_SCORE

    assert PERSON_SCORE is PERSON_THRESHOLD
    assert _load_script("campaign_site30k").PERSON_TRAIN_THR is PERSON_THRESHOLD

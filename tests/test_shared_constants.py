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


def test_the_fleet_lens_has_one_source():
    """`k1 = -0.225` was a named constant in `scripts/onboard_camera.py` plus two argparse
    defaults that wrote the number out again.

    A script cannot be imported (`test_scripts_are_not_libraries.py`), so the other two
    had no way to read the name even if they had wanted to -- which is why it moved into
    the package. A lens that drifts between three statements returns metres rather than
    an error, and the metres look ordinary.
    """
    from syncai_bev3d.plate_calibration import K1_FLEET

    assert _load_script("onboard_camera").K1_FLEET is K1_FLEET
    for name in ("site_events", "mine_fall_candidates"):
        parser = _load_script(name).build_parser()
        default = next(a.default for a in parser._actions if a.dest == "k1")
        assert default == K1_FLEET, f"{name}'s --k1 default drifted from the fleet value"


def test_the_imagenet_normalisation_has_one_source():
    """Three files rebuilt the arrays -- one inside `src/`, on every sample.

    `preprocessing`'s docstring is the argument: a value that drifts between two of them
    does not raise, it feeds the network inputs it was never trained on, and the only
    symptom is predictions that are slightly worse.
    """
    import numpy as np

    from syncai_hydranet.preprocessing import IMAGENET_MEAN, IMAGENET_STD

    for name in ("staff_probe", "offline_tracks"):
        mod = _load_script(name)
        if hasattr(mod, "MEAN"):
            assert np.array_equal(mod.MEAN, IMAGENET_MEAN)
            assert np.array_equal(mod.STD, IMAGENET_STD)

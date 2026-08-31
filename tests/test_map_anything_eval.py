"""The negative control is the instrument, and this is what stops it becoming a footnote.

`tools/commissioning/map_anything_eval.py` asks whether an external metric model can put
two cameras in one frame. It can: three cameras in one room came back 2.46-3.86 m apart.

**It also placed three cameras in three different cities 2.13-5.91 m apart**, and those
numbers look just as reasonable. So the poses prove nothing on their own, and the only
thing separating the two cases is confidence -- 1.0 across the control against 4.33 / 2.03
/ 2.04 for the real group.

That makes the control load-bearing rather than diligent. A run without one measures
nothing while appearing to measure everything, which is why `registration_verdict` refuses
instead of scoring, and why the case these tests spend most of their length on is the one
where the control scores *high*.

Loaded by path rather than imported as `tools.commissioning.map_anything_eval`, for the
reason `tests/test_retention_policy.py` records: `python -m pytest` puts the working
directory on `sys.path` and CI's `uv run pytest` does not.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load():
    name = "_map_anything_eval"
    spec = importlib.util.spec_from_file_location(
        name, REPO / "tools" / "commissioning" / "map_anything_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # `@dataclass` resolves string annotations through this
    spec.loader.exec_module(module)
    return module


mae = _load()
GroupReading = mae.GroupReading


def _reading(label, confs, control=False):
    return GroupReading(
        label=label,
        cameras=("a", "b", "c"),
        confidences=tuple(confs),
        pair_distances_m={"a~b": 3.1},
        is_control=control,
    )


# ---------------------------------------------------------------------------
# the refusals


def test_a_run_without_a_negative_control_measures_nothing():
    """The failure this whole design exists to prevent, stated as a refusal.

    Without a set that physically cannot register, every group returns plausible
    distances and a reader concludes the method works. The control is what turned that
    from an assumption into a measurement, so its absence is not a missing nicety -- the
    remaining numbers do not mean what they appear to.
    """
    v = mae.registration_verdict([_reading("same room", [4.3, 2.0, 2.0])])
    assert v.usable is False
    assert "no negative control" in v.reason


def test_a_control_that_scores_as_high_as_the_real_group_makes_it_unusable():
    """The disaster case, and the reason confidence is not taken on trust.

    If cameras in different buildings score like cameras in one room, the signal cannot
    gate anything -- and a store frame built on it would silently span three cities. The
    verdict has to be a refusal here even though every distance in the run looks fine.
    """
    v = mae.registration_verdict(
        [
            _reading("same room", [2.0, 2.1, 2.2]),
            _reading("CONTROL", [2.4, 1.0, 1.0], control=True),
        ]
    )
    assert v.usable is False
    assert "different buildings" in v.reason
    assert v.separation < 0


def test_an_exact_tie_is_also_a_refusal():
    """Equal is not separated. A gate needs somewhere to sit."""
    v = mae.registration_verdict(
        [_reading("same room", [2.0, 2.0]), _reading("CONTROL", [2.0, 1.0], control=True)]
    )
    assert v.usable is False
    assert v.separation == 0


def test_nothing_to_compare_the_control_against_is_a_refusal():
    v = mae.registration_verdict([_reading("CONTROL", [1.0, 1.0], control=True)])
    assert v.usable is False
    assert "nothing to compare" in v.reason


# ---------------------------------------------------------------------------
# the measured case


def test_the_readings_taken_on_2026_08_30_separate():
    """The actual numbers, so a later change that breaks the separation is visible here.

    Same room 4.33 / 2.03 / 2.04, same store far apart 1.88 / 1.04 / 1.10, control
    1.0 / 1.0 / 1.0. The tightest gap is the far-apart group's 1.04 against the control's
    1.0 -- a separation of 0.04, which is why this tool reports the number and refuses to
    invent a threshold from three groups.
    """
    v = mae.registration_verdict(
        [
            _reading("same room", [4.328, 2.029, 2.043]),
            _reading("same store, far apart", [1.877, 1.037, 1.096]),
            _reading("CONTROL: three different stores", [1.0, 1.0, 1.0], control=True),
        ]
    )
    assert v.usable is True
    assert v.control_max == 1.0
    assert v.overlapping_min == pytest.approx(1.037)
    assert v.separation == pytest.approx(0.037, abs=1e-3)


# ---------------------------------------------------------------------------
# the disagreement, reported in the unit that costs something


def test_the_anchor_disagreement_is_reported_as_focal_not_as_degrees():
    """70.4 against 38.26 reads as "a bit under half"; the focal ratio reads as 2x.

    `Camera.from_vfov` turns the angle into `fy` immediately and every metre downstream is
    divided by it, so degrees understate what the disagreement costs.
    """
    ratio = mae.focal_disagreement(mae.ANCHOR_VFOV_DEG, 38.26)
    assert ratio == pytest.approx(2.03, abs=0.02)


def test_agreement_is_a_ratio_of_one():
    assert mae.focal_disagreement(70.4, 70.4) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# the baseline it compares against, and what that baseline is not


def test_the_commissioned_baseline_is_readable_and_plausible():
    """Eight cameras, heights inside the shop-ceiling range the fit itself uses.

    This does not check they are *right*. Every one is fitted from the 1.70 m person prior
    (PLAN 7.19), so the comparison this tool runs is two estimates meeting -- which the
    tool prints in words on every run rather than leaving to be inferred from a table.

    **It reads `runs/commission01/`, which is gitignored, so on a fresh checkout there is
    nothing to read.** It asserted `len == 8` against that and turned `dev` red on
    2026-08-30 for three Python rows -- passing on the box that made the artefacts and
    failing everywhere else, which is the same `runs/`-dependence this repository has
    written about, in the form where it fails instead of skipping.

    The skip is scoped to *nothing at all*, not to "fewer than eight". An empty directory
    means this test could not run; a directory holding three camera.json means the shipped
    set has lost five, and those must not look alike. `tests/test_camera_json.py` skips on
    the same condition and in the same words.
    """
    base = mae.commissioned()
    if not base:
        pytest.skip("no commissioned cameras in this checkout")
    assert len(base) == 8, "the eight shipped camera.json"
    assert mae.ANCHOR_CAMERA in base, "the only camera with a measured vfov must be there"
    for cam, m in base.items():
        assert 2.0 <= m["height_m"] <= 3.6, f"{cam} outside any shop ceiling"
        assert m["k1"] is not None, f"{cam} has no lens; the plate cannot be undistorted"

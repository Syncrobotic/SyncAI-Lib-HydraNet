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
    assert spec is not None and spec.loader is not None
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
    """The original baseline remains readable as additional cameras are commissioned.

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
    original = {
        "Kaohsiung-cam04",
        "Taichung-cam01",
        "Taichung-cam04",
        "Taichung-cam07",
        "Taichung-cam10",
        "Taichung-cam11",
        "Tao-Hsin-cam03",
        "Tao-Hsin-cam04",
    }
    assert original <= base.keys(), (
        f"original commissioned cameras missing: {original - base.keys()}"
    )
    assert mae.ANCHOR_CAMERA in base, "the only camera with a measured vfov must be there"
    for cam, m in base.items():
        assert 2.0 <= m["height_m"] <= 3.6, f"{cam} outside any shop ceiling"
        assert m["k1"] is not None, f"{cam} has no lens; the plate cannot be undistorted"


# ---------------------------------------------------------------------------
# the backends the intrinsics question can now be put to


def test_an_external_backend_without_a_revision_is_refused_before_anything_loads():
    """`--backend da3` names a model this repository does not depend on; without a commit
    id the run would measure whatever upstream `main` is today, which is not a reading."""
    for backend in ("da3", "vggt"):
        with pytest.raises(ValueError, match="--revision"):
            mae.intrinsics_backend(backend, None)
        assert callable(mae.intrinsics_backend(backend, "0" * 40))
    assert mae.intrinsics_backend("mapanything", None) is mae._vfov_mapanything
    with pytest.raises(ValueError, match="unknown backend"):
        mae.intrinsics_backend("moge", "0" * 40)


def test_register_refuses_a_backend_it_does_not_have(tmp_path):
    """Only MapAnything registers here. A silently ignored `--backend` would let a reader
    file a VGGT registration that MapAnything actually produced."""
    assert mae.main(["register", "--out", str(tmp_path), "--backend", "vggt"]) == 2


def _raw_manifest(tmp_path):
    import hashlib
    import json

    from PIL import Image

    frames = {}
    for i, store in enumerate(("shop", "shop", "elsewhere")):
        path = tmp_path / f"frame-{i}.png"
        Image.new("RGB", (64, 48), (i * 40, 80, 120)).save(path)
        frames[f"frame-{i}"] = {
            "path": path.name,
            "camera_id": f"camera-{i}",
            "store_id": store,
            "image_size_px": [64, 48],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "frames": frames,
        "groups": {
            "same": {"frames": ["frame-0", "frame-1"], "is_control": False},
            "different": {"frames": ["frame-0", "frame-2"], "is_control": True},
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_explicit_raw_inputs_do_not_read_commissioned_data(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image

    def forbidden(*_args, **_kwargs):
        raise AssertionError("commissioned data is not an image-only input")

    monkeypatch.setattr(mae, "commissioned", forbidden)
    monkeypatch.setattr(mae, "undistorted_plates", forbidden)
    manifest, _ = _raw_manifest(tmp_path)
    plates, groups, provenance = mae.explicit_frames(manifest, tmp_path / "prepared", 0)
    assert set(groups) == {"same", "different"}
    assert provenance["commissioned_inputs_used"] is False
    assert provenance["pixel_space"] == "raw"
    assert provenance["lens_verified"] is False
    for key, p in plates.items():
        np.testing.assert_array_equal(Image.open(p), Image.open(tmp_path / f"{key}.png"))


@pytest.mark.parametrize(
    "failure", ["hash", "size", "repeated_camera", "store", "missing_control"]
)
def test_explicit_manifest_rejects_corrupted_or_misrepresented_evidence(tmp_path, failure):
    import json

    path, manifest = _raw_manifest(tmp_path)
    if failure == "hash":
        manifest["frames"]["frame-0"]["sha256"] = "0" * 64
    elif failure == "size":
        manifest["frames"]["frame-0"]["image_size_px"] = [48, 64]
    elif failure == "repeated_camera":
        manifest["frames"]["frame-1"]["camera_id"] = "camera-0"
    elif failure == "store":
        manifest["frames"]["frame-2"]["store_id"] = "shop"
    else:
        del manifest["groups"]["different"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        mae.explicit_frames(path, tmp_path / "prepared", 0)
    assert not (tmp_path / "prepared").exists()


def test_model_pixel_transform_accounts_for_resize_rounding_and_crop():
    import numpy as np

    # The pinned loader resizes 1920x1080 to 522x294, then cuts two columns each side.
    a = mae.model_pixel_transform((1920, 1080), (518, 294))
    np.testing.assert_allclose(a @ [-0.5, -0.5, 1], [-2.5, -0.5, 1])
    np.testing.assert_allclose(a @ [1919.5, 1079.5, 1], [519.5, 293.5, 1])
    np.testing.assert_allclose(a @ [959.5, 539.5, 1], [258.5, 146.5, 1])
    np.testing.assert_array_equal(mae.model_pixel_transform((518, 294), (518, 294)), np.eye(3))


def test_explicit_entry_does_not_promote_confidence_to_geometry(tmp_path, monkeypatch):
    import json

    manifest, _ = _raw_manifest(tmp_path)
    monkeypatch.setattr(
        mae,
        "run_register",
        lambda *_a, **_kw: [
            _reading("same", [4, 4]),
            _reading("control", [1, 1], control=True),
        ],
    )
    monkeypatch.setattr(mae, "undistorted_plates", lambda *_: pytest.fail("commissioned read"))
    out = tmp_path / "out"
    assert (
        mae.main(
            [
                "register",
                "--out",
                str(out),
                "--frames-manifest",
                str(manifest),
                "--lens-k1",
                "0",
            ]
        )
        == 0
    )
    payload = json.loads((out / "register.json").read_text())
    assert payload["verdict"]["confidence_separated"] is True
    assert payload["verdict"]["geometry_verified"] is False
    assert payload["verdict"]["deployment_ready"] is False
    assert "pair_distances_m" not in payload["groups"][0]


@pytest.mark.parametrize("args", [["--lens-k1", "0"], ["--frames-manifest", "missing.json"]])
def test_explicit_inputs_require_an_explicit_lens_choice(tmp_path, args):
    assert mae.main(["register", "--out", str(tmp_path), *args]) == 2

"""`run_depth` against the real Depth-Anything V2, on a box that has it.

The last uncovered function in `plate_calibration.py`, and the only one whose reason for
being uncovered was a dependency rather than a difficulty: it downloads a 1.3 GB
checkpoint and wants a CUDA device. Both are now present on the commissioning box, so the
call is checked here instead of being assumed.

**This file skips in CI and that is a compromise, not a design.** `test_figures_are_audited`
opens by warning about exactly this shape -- a gate that can only skip where nobody needs
it -- and the warning applies. What makes it acceptable *here* and not there is what the
skip costs: the figure audit is a privacy gate whose whole value is stopping a bad file
reaching the history, so one that skips in CI protects nothing. This one checks that a
pinned third-party checkpoint still loads and still returns metres, on the one machine
that will ever run it. Nothing ships behind it, and the arithmetic that does ship --
`floor_candidates`, `choose_floor`, the pose fit -- is held in `test_plate_calibration.py`
against synthetic depth, with no model and no GPU, precisely so that the part CI must
protect does not depend on a download.

`pytest.mark.slow` is applied for intent, but it is **not** what keeps this out of CI.
The marker is declared in `pyproject.toml` with the note "deselect with -m 'not slow'",
no test in the suite used it before this one, and `ci.yml` runs plain
`uv run pytest --cov` with no `-m` at all. So a marker alone would have let CI try to
download 1.3 GB and fail. The skips below are the mechanism; the marker is the label.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cached_model() -> str:
    """The pinned snapshot, from the local cache only -- a test must never download.

    `local_files_only=True` is the whole guard: without it a miss becomes a 1.3 GB fetch
    inside a test run, which is a different thing from a test.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    hub = pytest.importorskip("huggingface_hub")
    import torch

    from syncai_bev3d.plate_calibration import MODEL, MODEL_REVISION

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device; run_depth is the commissioning box's path")
    try:
        return hub.snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True)
    except Exception as exc:  # not cached, offline, or the revision is absent
        pytest.skip(f"{MODEL}@{MODEL_REVISION[:7]} is not in the local cache ({exc})")


def test_the_pinned_revision_is_the_one_on_disk(cached_model):
    """The pin is a claim about which weights produced every depth number in PLAN.

    `MODEL_REVISION` says "pinned rather than floating", and nothing checked that the
    thing on disk is that revision. A cache holding the branch tip instead would load
    silently and shift every fitted plane by an unknown amount.
    """
    from syncai_bev3d.plate_calibration import MODEL_REVISION

    assert MODEL_REVISION in cached_model, cached_model


@pytest.mark.usefixtures("cached_model")
def test_run_depth_returns_metres_the_right_shape_and_no_holes():
    """One real forward pass: shape, finiteness, and a plausible indoor range.

    The range assertion is deliberately loose. This is the *Metric* Indoor checkpoint, so
    its output is metres rather than a relative disparity, and the failure worth catching
    is that swap -- a model returning 0-1 disparity would satisfy every other assertion
    here and quietly rescale every camera in the fleet.
    """
    from syncai_bev3d.plate_calibration import run_depth

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (270, 480, 3), dtype=np.uint8)

    depth = run_depth(rgb)

    assert depth.shape == rgb.shape[:2], "depth is returned at the frame's own resolution"
    assert np.isfinite(depth).all(), "a hole here becomes a hole in the floor fit"
    assert depth.dtype.kind == "f"
    assert 0.1 < float(np.nanmedian(depth)) < 30.0, (
        f"median {np.nanmedian(depth):.3f} is outside any indoor range in metres -- if "
        "this is around 0-1 the checkpoint has become a relative-disparity one"
    )


@pytest.mark.usefixtures("cached_model")
def test_two_runs_on_the_same_frame_agree():
    """Determinism, because a commissioning artefact is cached and never recomputed.

    `camera.json` is fitted once and read for the life of the camera, so a depth pass that
    varied between runs would put a camera's metres beyond reproduction -- and the
    disagreement would only ever surface as a re-commissioning that disagrees with the
    figures already published from the first one.
    """
    from syncai_bev3d.plate_calibration import run_depth

    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 255, (180, 240, 3), dtype=np.uint8)

    first = run_depth(rgb)
    second = run_depth(rgb)

    assert np.allclose(first, second, rtol=0, atol=1e-6)

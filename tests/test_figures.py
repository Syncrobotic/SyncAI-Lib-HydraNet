"""`syncai_bev3d.figures`: the conversions a render makes before it draws anything.

The frame a detector saw and the frame a camera was commissioned on are not the same
size, and every value in the 3D panel -- floor position, stature, the dwell cell a
figure lands in -- is downstream of getting that ratio right. It was a literal here
until 2026-09-04.
"""

from __future__ import annotations


def test_the_calibration_scale_comes_from_the_camera_file_not_a_literal():
    """`track_states` divided box pixels by a hard-coded 2.0.

    Correct exactly while a camera is commissioned at half the decode resolution, which
    is true of this fleet (960x540 against 1920x1080 clips) and is not a property of
    anything. Every other consumer of this conversion derives it from `image_size_px`;
    this one now does too, so a camera commissioned at full resolution stops silently
    placing every figure at half its floor distance.
    """
    from syncai_bev3d.figures import _calibration_scale

    class _CF:
        image_size_px = (960, 540)

    # the fleet today: unchanged from the literal it replaced
    assert _calibration_scale(_CF(), (1920, 1080)) == (0.5, 0.5)
    # the camera the literal would have been wrong for
    assert _calibration_scale(_CF(), (960, 540)) == (1.0, 1.0)


def test_omitting_the_source_size_keeps_the_old_ratio_but_says_so():
    """An un-updated caller must not change behaviour, and must not do it quietly --
    silence is what let a literal stand in for this for as long as it did."""
    import warnings

    from syncai_bev3d.figures import _calibration_scale

    class _CF:
        image_size_px = (960, 540)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _calibration_scale(_CF(), None) == (0.5, 0.5)
    assert len(caught) == 1
    assert "source_size_px" in str(caught[0].message)

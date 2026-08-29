"""What `best.pt` gave up on the heads it was not selected for."""

from __future__ import annotations

from syncai_hydranet.utils.runmeta import selection_report


def _row(ep, **kw):
    return {"epoch": ep, **kw}


def test_the_head_nobody_selected_for_is_reported_with_its_cost():
    rows = [
        _row(1, detection_mAP=0.05, terrain_mIoU=0.40),
        _row(2, detection_mAP=0.20, terrain_mIoU=0.42),  # selected
        _row(3, detection_mAP=0.19, terrain_mIoU=0.55),  # terrain's real peak
    ]
    summary, warnings = selection_report(rows, "detection_mAP")
    assert summary["terrain_mIoU"] == {
        "best": 0.55, "best_epoch": 3, "at_selected": 0.42, "selected_epoch": 2,
    }  # fmt: skip
    assert any("gave up 0.1300" in w and "terrain_mIoU" in w for w in warnings)
    assert not any(w.startswith("detection_mAP") for w in warnings), (
        "the primary metric \
        cannot have given anything up to itself"
    )


def test_a_head_that_barely_moved_raises_nothing():
    rows = [
        _row(1, detection_mAP=0.1, terrain_mIoU=0.500),
        _row(2, detection_mAP=0.2, terrain_mIoU=0.505),
    ]
    _, warnings = selection_report(rows, "detection_mAP")
    assert warnings == []


def test_a_peak_in_the_first_fifth_is_called_out_as_noise():
    """The real failure: `hydranet_retail_objects_site_balanced` selected on a rare
    class's IoU, which peaked at epoch 2 of 39, so best.pt predates the other heads."""
    rows = [_row(1, rare=0.1, detection_mAP=0.01)]
    rows += [_row(2, rare=0.48, detection_mAP=0.02)]
    rows += [_row(e, rare=0.2, detection_mAP=0.02) for e in range(3, 21)]
    _, warnings = selection_report(rows, "rare")
    assert any("first fifth" in w for w in warnings)


def test_a_short_run_is_not_accused_of_latching_early():
    """Nine validations cannot distinguish an early peak from a short run."""
    rows = [_row(1, rare=0.9)] + [_row(e, rare=0.1) for e in range(2, 10)]
    _, warnings = selection_report(rows, "rare")
    assert not any("first fifth" in w for w in warnings)


def test_a_metric_absent_from_every_row_is_skipped_not_invented():
    rows = [_row(1, detection_mAP=0.1), _row(2, detection_mAP=0.2)]
    summary, _ = selection_report(rows, "detection_mAP")
    assert "terrain_mIoU" not in summary

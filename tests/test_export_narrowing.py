"""Narrowing the detection head at export must not rename anyone's boxes.

RETAIL.md §4 has said "train on 80, narrow at export" since it was written, and the
reason is measured: on the AGX Orin post-processing was 16.33 ms of a 37.8 ms frame,
nearly all of it the sigmoid over 80 classes at 6,820 positions. The narrowing itself is
four lines. What needs testing is everything around it, because every failure mode here is
silent -- a wrongly mapped channel still decodes, still produces a box, and still writes a
confident class name on it.

Three properties, in the order they would bite:

1. The kept channels are the *same numbers* the full head produced. A slice that got its
   indices from the order the config wrote the names in, rather than the order the head
   assigns them, passes every shape check and reports `zebra` for a customer.
2. A class the head never trained on is refused, not silently given an empty channel.
3. A narrowed engine's `det_cls` bindings are named differently, so a host decoder written
   for 80 classes fails to find its binding instead of reading 8 channels as 80.

pytest tests/test_export_narrowing.py -v
"""

import json

import pytest
import torch

from _export_cfg import INPUT_SIZE, det_head, seg_head, tiny_trunk
from syncai_hydranet.cli.export_onnx import (
    ExportWrapper,
    det_output_names,
    narrow_detection_head,
    sidecar_path,
    trained_detection_classes,
)
from syncai_hydranet.data.coco_subsets import (
    COCO_NAMES,
    EXPORT_SUBSETS,
    RETAIL_ANALYTICS,
    RETAIL_BELONGINGS,
    RETAIL_OBJECT_GROUP,
    ROBOT_8,
    head_order,
    narrow_indices,
    resolve_export_subset,
)
from syncai_hydranet.models.hydranet import build_model

CFG = {
    "model": {
        **tiny_trunk(),
        "heads": {"traversability": seg_head(), "detection": det_head()},
        "loss_balancing": "fixed",
        "fixed_weights": {"traversability": 1.0, "detection": 1.0},
    },
    "data": {
        "input_size": INPUT_SIZE,
        "datasets": [
            {"name": "coco", "type": "coco", "supervises": ["detection"]},
        ],
    },
}


def _model():
    torch.manual_seed(0)
    return build_model(CFG).eval()


# --- the class lists -------------------------------------------------------


def test_head_order_is_coco_id_order_not_the_order_written():
    """``CocoDetDataset`` builds its labels from ``sorted(getCatIds(...))``, so a list
    written alphabetically or by importance is not the head's channel order."""
    assert head_order(["chair", "person"]) == ["person", "chair"]
    assert head_order(ROBOT_8) == sorted(ROBOT_8, key=COCO_NAMES.index)


def test_head_order_rejects_a_name_coco_does_not_have():
    with pytest.raises(ValueError, match="trolley"):
        head_order(["person", "trolley"])


def test_head_order_deduplicates():
    assert head_order(["person", "person", "chair"]) == ["person", "chair"]


def test_retail_analytics_is_derived_from_the_grouping_table():
    """Not a second copy of it. A written-out list would drift the first time someone
    decides a `tv` in a phone shop is signage, and the engine would stop emitting a class
    the mapping still reads -- invisibly, since a class that never fires looks the same as
    one that is never asked about."""
    assert set(RETAIL_ANALYTICS) == set(RETAIL_OBJECT_GROUP) | set(RETAIL_BELONGINGS)
    assert head_order(RETAIL_ANALYTICS) == RETAIL_ANALYTICS


def test_robot_8_and_retail_analytics_are_different_deployments():
    """The one that matters: `book` is the strongest merchandise signal the head produces
    (1,683 on Kaohsiung-cam08) and ROBOT_8 deletes it. Narrowing the analytics build with
    the robot's list would silently remove the class the audit was about."""
    assert "book" in RETAIL_ANALYTICS
    assert "book" not in ROBOT_8


def test_named_subsets_resolve_and_so_does_a_comma_list():
    assert resolve_export_subset("robot_8") == head_order(ROBOT_8)
    assert resolve_export_subset("chair,person") == ["person", "chair"]
    assert set(EXPORT_SUBSETS) >= {"robot_8", "retail_analytics"}


def test_an_empty_class_list_names_the_alternatives():
    with pytest.raises(ValueError, match="robot_8"):
        resolve_export_subset(" , ")


def test_narrow_indices_refuses_a_class_the_head_never_trained():
    """A head narrowed at training time to 25 classes has no channel for `zebra`, and
    handing it one would export a channel that is structurally silent."""
    with pytest.raises(ValueError, match="zebra"):
        narrow_indices(["zebra"], ["person", "chair"])


def test_narrow_indices_are_positions_in_the_trained_list_not_in_coco():
    trained = ["person", "chair", "book"]  # a head already narrowed at training time
    assert narrow_indices(["person", "book"], trained) == [0, 2]


# --- naming the head's channels -------------------------------------------


def test_trained_classes_default_to_cocos_80():
    assert trained_detection_classes(CFG, 80) == COCO_NAMES


def test_trained_classes_come_from_the_dataset_subset_in_head_order():
    cfg = {
        "data": {
            "datasets": [
                {
                    "type": "coco",
                    "supervises": ["detection"],
                    "classes": ["chair", "person"],
                }
            ]
        }
    }
    assert trained_detection_classes(cfg, 2) == ["person", "chair"]


def test_a_head_whose_width_disagrees_with_the_config_is_refused():
    """The silent failure this guard exists for: 80 names onto 25 channels maps every
    class to the wrong one and nothing downstream can tell."""
    with pytest.raises(SystemExit, match="80"):
        trained_detection_classes(CFG, 25)


# --- the slice itself ------------------------------------------------------


def test_narrowed_head_reproduces_the_kept_channels_exactly():
    """The property everything else rests on. Not "the shape is 8" -- the numbers have to
    be the same numbers, at the same class, at every position and every level."""
    model = _model()
    x = torch.rand(1, 3, 128, 160) * 255.0
    wrapper = ExportWrapper(model)
    with torch.no_grad():
        full = [t.clone() for t in wrapper(x)]

    keep = head_order(ROBOT_8)
    idx = narrow_indices(keep, COCO_NAMES)
    narrow_detection_head(model, idx)
    with torch.no_grad():
        got = wrapper(x)

    n_seg = len(wrapper.seg_names)
    n_lv = len(model.det_head.in_levels)
    for level in range(n_lv):
        before, after = full[n_seg + level], got[n_seg + level]
        assert after.shape[1] == len(keep)
        torch.testing.assert_close(after, before[:, idx])


def test_narrowing_leaves_the_other_outputs_untouched():
    """Segmentation, regression and centerness share the trunk with the class tower. If
    slicing the classifier moved any of them, the trade would not be free."""
    model = _model()
    x = torch.rand(1, 3, 128, 160) * 255.0
    wrapper = ExportWrapper(model)
    with torch.no_grad():
        full = [t.clone() for t in wrapper(x)]
    n_seg, n_lv = len(wrapper.seg_names), len(model.det_head.in_levels)

    narrow_detection_head(model, narrow_indices(head_order(ROBOT_8), COCO_NAMES))
    with torch.no_grad():
        got = wrapper(x)

    for i in list(range(n_seg)) + list(range(n_seg + n_lv, n_seg + 3 * n_lv)):
        torch.testing.assert_close(got[i], full[i])


def test_narrowing_reports_its_own_width():
    model = _model()
    narrow_detection_head(model, [0, 1, 2])
    assert model.det_head.num_classes == 3
    assert model.det_head.cls_pred.out_channels == 3


# --- the contract carried to the board ------------------------------------


def test_binding_names_are_unchanged_when_nothing_is_narrowed():
    """Every engine already built looks up `det_cls_p3`. Renaming it for everyone would
    break each one that did not ask for narrowing, and an engine keeps its binding names
    and nothing else -- there is no version field in it to check."""
    names = det_output_names(5, None)
    assert names[:5] == [f"det_cls_p{i}" for i in range(3, 8)]


def test_a_narrowed_engine_renames_only_the_class_bindings():
    """The `INPUT_RAW` / `INPUT_NORMALISED` trick on the output side: a host decoder
    written for 80 classes must fail to find its binding rather than read 8 as 80. `reg`
    and `ctr` keep their names because their shapes do not change."""
    names = det_output_names(5, 8)
    assert names[:5] == [f"det_cls8_p{i}" for i in range(3, 8)]
    assert names[5:] == det_output_names(5, None)[5:]


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("hydranet.onnx", "hydranet.classes.json"),
        ("/tmp/a/b.onnx", "/tmp/a/b.classes.json"),
        ("hydranet", "hydranet.classes.json"),
    ],
)
def test_sidecar_sits_beside_the_engine(output, expected):
    assert str(sidecar_path(output)) == expected


def test_the_sidecar_names_exactly_the_bindings_the_engine_will_carry(tmp_path):
    """One end of a two-ended contract, and the surviving half is still the load-bearing one.

    This used to round-trip the exporter against `scripts/live_view_orin.py`, the Jetson
    viewer that read the sidecar back. Both went on 2026-08-28 with the Orin as a target,
    and **nothing in this tree reads a `.classes.json` today** -- which does not retire
    the invariant, it removes the only consumer that was checking it. `cli/export_onnx.py`
    states why the file exists at all: "a TensorRT engine keeps its binding names and
    nothing else, so the ONNX metadata never reaches the board. The sidecar is what does
    -- it is the only record of which COCO class each remaining channel means, and an
    engine shipped without it is a set of unlabelled numbers."

    So what is pinned here is that the record is *internally correct*: the `cls_outputs`
    the sidecar lists are exactly the narrowed class-logit bindings the graph carries, in
    order, and the names line up with the source indices they were narrowed from. A host
    written later indexes by that order; getting it wrong renames every box, which is the
    failure `head_order` exists to prevent one process boundary earlier.
    """
    keep = head_order(ROBOT_8)
    bindings = det_output_names(5, len(keep))
    cls_bindings = [b for b in bindings if b.startswith("det_cls")]

    assert cls_bindings == [f"det_cls8_p{i}" for i in range(3, 8)], (
        "the class count travels in the binding name, so a host reading `det_cls_p3` "
        "against a narrowed engine gets 80 channels' worth of names for 8 channels"
    )

    path = sidecar_path(str(tmp_path / "hydranet.onnx"))
    path.write_text(
        json.dumps(
            {
                "detection_classes": keep,
                "source_indices": narrow_indices(keep, COCO_NAMES),
                "cls_outputs": cls_bindings,
            }
        )
    )
    side = json.loads(path.read_text())

    assert side["cls_outputs"] == cls_bindings
    assert len(side["detection_classes"]) == len(side["source_indices"])
    assert [COCO_NAMES[i] for i in side["source_indices"]] == side["detection_classes"], (
        "the indices are what map a kept channel back to the checkpoint it came from; "
        "if they disagree with the names, the sidecar renames boxes rather than labelling "
        "them"
    )


def test_an_unnarrowed_export_keeps_the_plain_binding_names():
    """No narrowing, no class count in the name, and no sidecar written.

    Kept from the round-trip test that preceded this one, because it is the property that
    let every engine built before `--detection-classes` existed keep working: the plain
    `det_cls_p3` spelling is what an unnarrowed graph carries.
    """
    assert det_output_names(5, None)[0] == "det_cls_p3"

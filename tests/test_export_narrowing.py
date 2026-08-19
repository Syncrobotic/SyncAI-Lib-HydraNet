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
import sys
from pathlib import Path

import pytest
import torch

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
        "backbone": {"name": "resnet18", "pretrained": False},
        "neck": {"name": "fpn", "out_channels": 32, "num_repeats": 1, "num_levels": 5},
        "heads": {
            "traversability": {
                "type": "semantic_fpn",
                "num_classes": 3,
                "in_levels": [0, 1, 2],
                "channels": 32,
            },
            "detection": {"type": "fcos", "num_classes": 80, "channels": 32, "num_convs": 1},
        },
        "loss_balancing": "fixed",
        "fixed_weights": {"traversability": 1.0, "detection": 1.0},
    },
    "data": {
        "input_size": [128, 160],
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
    """Existing engines and the two Jetson scripts look up `det_cls_p3`. Renaming it for
    everyone would break every deployment that did not ask for narrowing."""
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


def test_the_board_reads_back_what_the_exporter_wrote(tmp_path):
    """The two ends of the contract, in one test: the exporter narrows and the Jetson
    viewer names the channels. They live in different files that cannot import each other
    -- `live_view_orin.py` runs on a board with no `syncai_hydranet` installed -- so this
    is the only place the round trip is checked at all."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    try:
        import live_view_orin
    finally:
        sys.path.pop(0)

    keep = head_order(ROBOT_8)
    path = sidecar_path(str(tmp_path / "hydranet.onnx"))
    path.write_text(
        json.dumps(
            {
                "detection_classes": keep,
                "source_indices": narrow_indices(keep, COCO_NAMES),
            }
        )
    )

    engine_outputs = ["traversability", "terrain", *det_output_names(5, len(keep))]
    found = live_view_orin.cls_bindings(engine_outputs)
    assert found == [f"det_cls8_p{i}" for i in range(3, 8)]
    assert live_view_orin.load_class_names(str(path), len(keep)) == keep

    # An unnarrowed engine still works with no sidecar at all, which is what keeps every
    # engine already on a board running.
    assert live_view_orin.cls_bindings(det_output_names(5, None))[0] == "det_cls_p3"
    assert live_view_orin.load_class_names(None, 80) == live_view_orin.COCO_NAMES

    # A narrowed engine with no sidecar refuses rather than guessing.
    with pytest.raises(SystemExit, match="narrowed at export"):
        live_view_orin.load_class_names(None, len(keep))

    # A sidecar from a different export refuses too: the count is the only thing that
    # can be checked, and it catches the mistake that actually happens.
    with pytest.raises(SystemExit, match="different export"):
        live_view_orin.load_class_names(str(path), 32)


def test_sidecar_round_trips_the_names_and_their_source_channels(tmp_path):
    """A TensorRT engine keeps binding names and nothing else, so the ONNX metadata never
    reaches the board. The sidecar is the only record of what each channel means."""
    keep = head_order(ROBOT_8)
    idx = narrow_indices(keep, COCO_NAMES)
    path = sidecar_path(str(tmp_path / "hydranet.onnx"))
    path.write_text(json.dumps({"detection_classes": keep, "source_indices": idx}))

    loaded = json.loads(path.read_text())
    assert loaded["detection_classes"] == keep
    assert [COCO_NAMES[i] for i in loaded["source_indices"]] == keep

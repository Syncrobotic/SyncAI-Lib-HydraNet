"""Box labels on the panel must come from the config, not from a hardcoded COCO table.

Found by rendering `configs/hydranet_retail_openvocab.yaml` -- a two-class merchandise
head -- and reading the output: channel 0 was drawn `person` and channel 1 `bicycle`.
They are `boxed_stock` and `device`. Nothing errored, and the frame looked *right*: a shop
contains people, so a box labelled `person` in a shop is the most convincing possible
wrong answer.

That is `class_names` in `analytics/stage.py` stated as a bug that already shipped rather
than one to avoid, and it is not exotic: `--detection-classes` narrows an exported head,
so any build that is not the full 80 has this available to it.

The rule these tests pin is that **the channel count is the evidence**. `COCO_NAMES` is
correct for a head with exactly as many channels as COCO has names and for nothing else --
a two-class head is a different vocabulary, not a narrowed COCO head whose names survive.
Where names cannot be established the panel falls back to the bare channel index, which is
deliberately worse to read and honest: `0` sends a viewer to look it up, `person` does not.

pytest tests/test_scene_det_names.py -v
"""

from __future__ import annotations

from syncai_hydranet.cli.scene import detection_class_names
from syncai_hydranet.data.coco_subsets import COCO_NAMES


def _cfg(det: dict | None, datasets: list | None = None) -> dict:
    heads: dict = {"terrain": {"type": "semantic_fpn", "num_classes": 7}}
    if det is not None:
        heads["detection"] = {"type": "fcos", **det}
    return {"model": {"heads": heads}, "data": {"datasets": datasets or []}}


def test_the_heads_own_class_list_wins():
    """Declared in head order, and already checked against the text matrix's names."""
    got = detection_class_names(_cfg({"num_classes": 2, "classes": ["boxed_stock", "device"]}))
    assert got == ("boxed_stock", "device")


def test_a_two_class_head_with_no_names_gets_none_rather_than_cocos_first_two():
    """The regression. `COCO_NAMES[:2]` is `person, bicycle`, both plausible in a shop and
    both wrong -- so the fallback must be no names at all, which the caller renders as the
    bare channel index."""
    assert detection_class_names(_cfg({"num_classes": 2})) is None


def test_a_full_width_head_still_gets_coco_names():
    """The count is the evidence, and it is also what keeps the shipped robot configs
    working: `hydranet_indoor` trains all 80 and its panel has always been correct."""
    got = detection_class_names(_cfg({"num_classes": len(COCO_NAMES)}))
    assert got == tuple(COCO_NAMES)


def test_a_coco_datasets_narrowing_list_comes_back_in_head_order():
    """`CocoDetDataset` builds its labels from `sorted(getCatIds(...))`, so a config's
    writing order is not the head's channel order. Taking the list verbatim would label
    every box with its neighbour's name -- the same failure one level up, which is what
    `coco_subsets.head_order` exists for."""
    got = detection_class_names(
        _cfg(
            {"num_classes": 3},
            [{"type": "coco", "classes": ["backpack", "person", "chair"]}],
        )
    )
    assert got == tuple(sorted(got, key=COCO_NAMES.index))
    assert got[0] == "person", "person is COCO id 1 and must lead whatever order was written"


def test_a_config_with_no_detection_head_asks_for_no_names():
    assert detection_class_names(_cfg(None)) is None


def test_the_shipped_retail_and_robot_configs_resolve_as_intended():
    """End to end on the real files, because every unit above uses a synthetic dict."""
    from syncai_hydranet.config import load_config

    assert detection_class_names(load_config("configs/hydranet_retail_openvocab.yaml")) == (
        "boxed_stock",
        "device",
    )
    # Declares no names and is not COCO-width, so the panel shows indices rather than lying.
    assert detection_class_names(load_config("configs/hydranet_retail_products.yaml")) is None
    assert detection_class_names(load_config("configs/hydranet_indoor.yaml")) == tuple(
        COCO_NAMES
    )

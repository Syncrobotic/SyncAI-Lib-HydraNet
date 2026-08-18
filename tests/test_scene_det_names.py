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


# ----------------------------------------------------- reading COCO as shop nouns


def test_retail_vocab_renames_a_coco_head_and_keeps_the_evidence():
    """The rename is not extra knowledge: it is the same head's same output read as shop
    nouns. The COCO word stays beside the group because it is the evidence for the group,
    and hiding it would make a wrong grouping unfalsifiable from the rendered frame."""
    from syncai_hydranet.cli.scene import apply_vocab
    from syncai_hydranet.data.coco_subsets import COCO_NAMES

    renamed = apply_vocab(tuple(COCO_NAMES), "retail")
    assert renamed[COCO_NAMES.index("oven")] == "fixture/oven"
    assert renamed[COCO_NAMES.index("book")] == "product/book"
    assert renamed[COCO_NAMES.index("person")] == "person"
    assert len(renamed) == len(COCO_NAMES)


def test_coco_vocab_is_the_default_and_changes_nothing():
    from syncai_hydranet.cli.scene import apply_vocab
    from syncai_hydranet.data.coco_subsets import COCO_NAMES

    assert apply_vocab(tuple(COCO_NAMES), "coco") == tuple(COCO_NAMES)
    assert apply_vocab(None, "coco") is None
    assert apply_vocab(("boxed_stock", "device"), "coco") == ("boxed_stock", "device")


def test_retail_vocab_refuses_a_head_it_cannot_address():
    """The mapping is addressed by COCO index. Applied to `hydranet_retail_openvocab`'s
    two-class head it would put `product/book` over a `boxed_stock` box -- the same failure
    `detection_class_names` exists to stop, wearing a shop's clothes. A plausible wrong
    noun is the one nobody checks, so this refuses rather than renames.
    """
    import pytest

    from syncai_hydranet.cli.scene import apply_vocab

    for names in (("boxed_stock", "device"), None):
        with pytest.raises(ValueError, match="COCO 80-class"):
            apply_vocab(names, "retail")


def test_a_renamed_label_still_finds_its_shape_and_its_colour():
    """The rename produces `fixture/chair`, and the shape table and the panel palette are
    keyed on `chair`. Without `detected_class` between them a `--vocab retail` run silently
    loses every mesh and every colour it had."""
    from syncai_hydranet.geometry.bev3d import OBJECT_RGB
    from syncai_hydranet.geometry.meshes import detected_class, for_object

    plain = for_object({"name": "chair", "width_m": 0.6, "height_m": 0.9})
    renamed = for_object({"name": "fixture/chair", "width_m": 0.6, "height_m": 0.9})
    assert len(renamed[1]) == len(plain[1]) > 12
    assert OBJECT_RGB.get(detected_class("person")) == OBJECT_RGB["person"]
    assert detected_class("potted plant") == "potted plant"

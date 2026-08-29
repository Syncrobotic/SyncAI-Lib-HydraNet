"""One detection head holding both the retail and the security question.

Two things are being pinned here and they fail in opposite ways.

**The vocabulary** stops two annotation files from meaning different things by the same
label. Without it COCO's `person` and the site's `boxed_stock` are both label 0 and the
run converges anyway -- `hydranet_retail_products.yaml` refuses to train the two together
for exactly that reason, and this is what lifts the refusal.

**The class mask** stops the shared head from teaching itself that shoppers are not
people. A site frame has people in it and no person boxes, so every point on every
shopper is a labelled negative unless something says otherwise. That is not dilution: it
is the mechanism that held `product` at IoU 0.000 for 22 consecutive epochs while the run
looked entirely normal, and a test that only checked the ids would pass through it.

pytest tests/test_retail_security_vocab.py -v
"""

from __future__ import annotations

import json

import pytest
import torch

from syncai_hydranet.config import load_config
from syncai_hydranet.config_schema import ConfigError, check_config
from syncai_hydranet.data.datasets import CocoDetDataset
from syncai_hydranet.data.label_maps_retail_security import (
    RETAIL_SECURITY_DET,
    get_det_vocab,
)
from syncai_hydranet.models.losses import sigmoid_focal_loss

COCO_CATS = [
    {"id": 1, "name": "person"},
    {"id": 27, "name": "backpack"},
    {"id": 31, "name": "handbag"},
    {"id": 24, "name": "zebra"},
]
SITE_CATS = [{"id": 1, "name": "boxed_stock"}, {"id": 2, "name": "device"}]


def write_coco(tmp_path, name, cats, plan, split="val"):
    from PIL import Image

    root = tmp_path / name
    (root / split).mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    images, anns, ann_id = [], [], 1
    for img_id, cat_ids in plan.items():
        fname = f"{img_id:012d}.jpg"
        Image.new("RGB", (64, 48), (110, 110, 110)).save(root / split / fname)
        images.append({"id": img_id, "file_name": fname, "width": 64, "height": 48})
        for c in cat_ids:
            anns.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": c,
                    "bbox": [4, 4, 20, 20],
                    "area": 400,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    (root / "annotations" / f"instances_{split}.json").write_text(
        json.dumps({"images": images, "annotations": anns, "categories": cats})
    )
    return str(root)


@pytest.fixture
def sources(tmp_path):
    coco = write_coco(tmp_path, "coco", COCO_CATS, {1: [1, 27], 2: [31], 3: [24]})
    site = write_coco(tmp_path, "site", SITE_CATS, {1: [1, 2], 2: [2]})
    return coco, site


def build(root, **kw):
    return CocoDetDataset(root, "val", (64, 64), train=False, det_vocab="retail_security", **kw)


def test_ids_are_frozen():
    """These are in checkpoints and in every box exported against them.

    Renumbering is the tidy-up that breaks nothing visibly and invalidates everything at
    once -- `tests/test_retail_scheme.py` exists for the same reason on the terrain side.
    """
    assert RETAIL_SECURITY_DET == {"person": 0, "bag": 1, "boxed_stock": 2, "device": 3}


def test_two_sources_agree_on_the_numbering(sources):
    coco, site = sources
    a, b = build(coco), build(site)
    assert a.cat_to_label[1] == 0, "COCO person is channel 0"
    assert b.cat_to_label[1] == 2, "site boxed_stock is channel 2, not channel 0"
    assert set(a.supplied_labels).isdisjoint(b.supplied_labels)


def test_three_bag_categories_become_one_channel(sources):
    coco, _ = sources
    ds = build(coco)
    assert ds.cat_to_label[27] == 1 and ds.label_to_cat[1] == 27
    names = {c["id"]: c["name"] for c in ds.coco.loadCats(ds.coco.getCatIds())}
    assert names == {1: "person", 27: "bag"}, "the ground truth is merged, not just the labels"
    bags = [a for a in ds.coco.dataset["annotations"] if a["category_id"] == 27]
    assert len(bags) == 2, "the handbag annotation was rewritten to the bag category"


def test_categories_outside_the_vocabulary_are_dropped_not_guessed(sources):
    coco, _ = sources
    ds = build(coco)
    assert 24 not in ds.cat_to_label, "zebra has no nearby retail class and must not get one"
    assert len(ds) == 2, "the zebra-only image carries no signal and is dropped"


def test_a_source_with_nothing_in_the_vocabulary_is_refused(tmp_path):
    root = write_coco(tmp_path, "zoo", [{"id": 24, "name": "zebra"}], {1: [24]})
    with pytest.raises(ValueError, match="maps none of this dataset's categories"):
        build(root)


def test_the_class_mask_says_what_this_dataset_can_answer_for(sources):
    coco, site = sources
    assert build(coco).class_mask.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert build(site).class_mask.tolist() == [0.0, 0.0, 1.0, 1.0]
    sample = build(site)[0]
    assert sample["targets"]["det_class_mask"].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_a_masked_channel_receives_no_gradient_at_all():
    """The whole point, and the reason a weight of zero is not the same thing.

    A masked channel is not "learned more slowly on this dataset". It gets exactly zero
    gradient, so an unlabelled shopper in a site frame is not evidence that shoppers are
    not people. The unmasked channels must still learn, or the mask has simply turned the
    loss off.
    """
    logits = torch.zeros(2, 5, 4, requires_grad=True)
    onehot = torch.zeros(2, 5, 4)
    onehot[:, 0, 2] = 1.0  # a boxed_stock positive, which is all a site batch can carry
    mask = torch.tensor([0.0, 0.0, 1.0, 1.0])[None, None, :]
    sigmoid_focal_loss(logits, onehot, channel_mask=mask).backward()
    assert torch.count_nonzero(logits.grad[..., :2]) == 0, "person and bag must be untouched"
    assert torch.count_nonzero(logits.grad[..., 2:]) > 0, "site classes must still learn"


def test_the_evaluator_can_ask_for_a_class_this_dataset_has_no_ground_truth_for(sources):
    """`person` predicted on a site frame. It is dropped, not a KeyError.

    And the consequence is the thing to remember rather than the lookup: each class is
    scored only on the dataset that labels it, so `person` mAP comes from COCO and never
    from a store camera.
    """
    _, site = sources
    ds = build(site)
    assert ds.label_to_cat.get(0) is None
    assert sorted(ds.label_to_cat) == [2, 3]


SHIPPED = "configs/hydranet_retail_security.yaml"


def test_the_shipped_config_validates():
    assert check_config(load_config(SHIPPED)) == []


def test_two_detection_sources_without_a_vocabulary_are_refused():
    cfg = load_config(SHIPPED)
    for ds in cfg["data"]["datasets"]:
        ds.pop("det_vocab", None)
    with pytest.raises(ConfigError, match="numbers its own categories from zero"):
        check_config(cfg)


def test_a_head_narrower_than_the_vocabulary_is_refused():
    cfg = load_config(SHIPPED)
    cfg["model"]["heads"]["detection"]["num_classes"] = 2
    with pytest.raises(ConfigError, match="det_vocab 'retail_security' has 4 classes"):
        check_config(cfg)


def test_the_vocabulary_is_named_in_code_not_invented_by_a_config():
    with pytest.raises(ValueError, match="unknown det_vocab"):
        get_det_vocab("whatever_the_store_asked_for")

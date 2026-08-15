"""Training the detection head on a subset of COCO, and scoring it comparably.

The subset exists because the head spends 57% of all head parameters on 80 classes of
which an indoor robot meets maybe fifteen. The scoring rule exists because fewer classes
makes mAP go up on its own -- a narrower model measured against a narrower denominator
looks better whether or not anything improved.
"""

import json

import numpy as np
import pytest

from syncai_hydranet.config_schema import ConfigError, check_config
from syncai_hydranet.data.datasets import CocoDetDataset

CATS = [
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 62, "name": "chair", "supercategory": "furniture"},
    {"id": 24, "name": "zebra", "supercategory": "animal"},
]


def write_coco(tmp_path, split="val2017"):
    """Three images: one person+chair, one zebra only, one with both kinds."""
    root = tmp_path / "coco"
    (root / split).mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    from PIL import Image

    images, anns, ann_id = [], [], 1
    plan = {1: [1, 62], 2: [24], 3: [1, 24]}
    for img_id, cats in plan.items():
        name = f"{img_id:012d}.jpg"
        Image.new("RGB", (64, 48), (120, 120, 120)).save(root / split / name)
        images.append({"id": img_id, "file_name": name, "width": 64, "height": 48})
        for c in cats:
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
        json.dumps({"images": images, "annotations": anns, "categories": CATS})
    )
    return root


def test_all_classes_by_default(tmp_path):
    ds = CocoDetDataset(str(write_coco(tmp_path)), "val2017", (64, 64), train=False)
    assert ds.cat_ids == [1, 24, 62]
    assert len(ds) == 3


def test_subset_keeps_only_its_categories(tmp_path):
    ds = CocoDetDataset(
        str(write_coco(tmp_path)), "val2017", (64, 64), train=False, classes=["person", "chair"]
    )
    assert ds.cat_ids == [1, 62]
    # image 2 is zebra-only, so it carries no signal for this subset and is dropped
    assert len(ds) == 2
    labels = np.concatenate([ds[i]["targets"]["labels"].numpy() for i in range(len(ds))])
    assert set(labels.tolist()) <= {0, 1}, "labels must be contiguous over the subset"


def test_labels_map_back_to_the_right_category(tmp_path):
    ds = CocoDetDataset(
        str(write_coco(tmp_path)), "val2017", (64, 64), train=False, classes=["person", "chair"]
    )
    assert ds.label_to_cat == {0: 1, 1: 62}, "contiguous label -> COCO id, for scoring"


def test_a_typo_in_the_class_list_is_an_error_not_a_silent_drop(tmp_path):
    """Otherwise the head is built one channel too wide and nothing ever says so."""
    with pytest.raises(ValueError, match="chairs"):
        CocoDetDataset(
            str(write_coco(tmp_path)),
            "val2017",
            (64, 64),
            train=False,
            classes=["person", "chairs"],
        )


def test_dropping_every_class_leaves_no_images(tmp_path):
    ds = CocoDetDataset(
        str(write_coco(tmp_path)), "val2017", (64, 64), train=False, classes=["zebra"]
    )
    assert ds.cat_ids == [24]
    assert len(ds) == 2  # images 2 and 3


# ------------------------------------------------------------------ config guard


def base_cfg(n_classes: int, subset: list[str] | None):
    ds = {
        "name": "coco",
        "type": "coco",
        "root": "datasets/coco",
        "split_train": "train2017",
        "split_val": "val2017",
        "supervises": ["detection"],
    }
    if subset is not None:
        ds["classes"] = subset
    return {
        "experiment": "t",
        "output_dir": "runs/t",
        "model": {
            "backbone": {"name": "regnet_x_800mf"},
            "neck": {"name": "bifpn", "out_channels": 96},
            "heads": {
                "detection": {
                    "type": "fcos",
                    "num_classes": n_classes,
                    "in_levels": [0, 1, 2, 3, 4],
                    "channels": 96,
                    "strides": [8, 16, 32, 64, 128],
                }
            },
        },
        "data": {"input_size": [512, 640], "datasets": [ds]},
        "train": {"epochs": 1, "batch_size": 1, "lr": 1e-4},
    }


def test_subset_length_must_match_the_head_width():
    with pytest.raises(ConfigError, match="lists 2 classes"):
        check_config(base_cfg(80, ["person", "chair"]))


def test_matching_subset_and_head_is_accepted():
    check_config(base_cfg(2, ["person", "chair"]))


def test_duplicate_class_names_are_rejected():
    with pytest.raises(ConfigError, match="duplicate"):
        check_config(base_cfg(2, ["person", "person"]))


def test_no_subset_still_means_all_eighty():
    check_config(base_cfg(80, None))


# ------------------------------------------------- scoring, separately from training


def test_score_classes_narrows_scoring_without_touching_the_labels(tmp_path):
    """The whole point: an 80-class checkpoint keeps its output space and its label
    numbering, and is scored over the same categories as a narrower model."""
    ds = CocoDetDataset(
        str(write_coco(tmp_path)),
        "val2017",
        (64, 64),
        train=False,
        score_classes=["person", "chair"],
    )
    assert ds.cat_ids == [1, 24, 62], "the head still covers every category"
    assert ds.label_to_cat == {0: 1, 1: 24, 2: 62}, "numbering must not shift"
    assert ds.score_cat_ids == [1, 62], "but mAP is taken over the subset"
    assert len(ds) == 3, "no images are dropped: scoring does not change the data"


def test_score_classes_defaults_to_everything_trained(tmp_path):
    ds = CocoDetDataset(str(write_coco(tmp_path)), "val2017", (64, 64), train=False)
    assert ds.score_cat_ids == ds.cat_ids


def test_scoring_a_class_the_head_never_learned_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="not trained by this dataset"):
        CocoDetDataset(
            str(write_coco(tmp_path)),
            "val2017",
            (64, 64),
            train=False,
            classes=["person", "chair"],
            score_classes=["person", "zebra"],
        )


def test_score_classes_may_be_a_subset_of_a_trained_subset(tmp_path):
    ds = CocoDetDataset(
        str(write_coco(tmp_path)),
        "val2017",
        (64, 64),
        train=False,
        classes=["person", "chair"],
        score_classes=["chair"],
    )
    assert ds.cat_ids == [1, 62] and ds.score_cat_ids == [62]


def test_indoor_25_is_spelled_the_way_coco_spells_it():
    """A name that COCO does not use scores nothing, and mAP over nothing is not zero --
    it is a shorter mean over the categories that did match. The 0.3246 baseline is
    defined by this list, so a spelling drift here moves what that number refers to
    without changing anything that looks like a measurement."""
    from syncai_hydranet.data.coco_subsets import COCO_NAMES, INDOOR_25

    assert len(COCO_NAMES) == 80 == len(set(COCO_NAMES))
    unknown = [n for n in INDOOR_25 if n not in COCO_NAMES]
    assert not unknown, f"not COCO category names: {unknown}"
    assert len(INDOOR_25) == 25 == len(set(INDOOR_25))

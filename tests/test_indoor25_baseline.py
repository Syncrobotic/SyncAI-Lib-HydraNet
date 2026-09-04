"""Pin the subset that defines the 0.3246 detection baseline.

The number was published as the bar a narrowed detection head has to clear. The 25
category names that produced it were never committed -- they existed in one shell
command, in one session's history, in `/tmp`. The number outlived its own definition by
most of a day, and a narrowed head measured against a reconstructed-from-memory list
would have produced a confident comparison between two different quantities.

So the list is pinned in three places that must agree, and this file is what makes
disagreement a failure instead of a silent drift.

pytest tests/test_indoor25_baseline.py -v
"""

from pathlib import Path

import pytest
import yaml

from syncai_hydranet.config import load_config
from syncai_hydranet.config_schema import check_config
from syncai_hydranet.data.coco_subsets import INDOOR_25

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
# Anchored like CONFIGS above, and it was not until 2026-09-04: a bare relative
# `datasets/...` resolves against the INVOCATION directory, so this guard skipped
# from anywhere but the repo root and the test simply never ran. `test_zone_bridge`
# cites this file as the pattern for an artefact the repository does not ship, which
# is the reason to make the citation true rather than to fix it quietly.
COCO_VAL_ANNOTATIONS = (
    Path(__file__).resolve().parents[1] / "datasets/coco/annotations/instances_val2017.json"
)
EVAL_CFG = CONFIGS / "eval_indoor25.yaml"

# From runs/hydranet_joint_coco10/best.pt on val2017, COCO block at sample_ratio 0.1.
# Full precision on purpose: mAP over a subset is the mean of its per-category APs, so
# a single category added or dropped moves this in the third decimal. Agreement to
# sixteen digits is proof the list is the original one, not a plausible substitute.
BASELINE_MAP = 0.3245647505782422
BASELINE_MAP50 = 0.49341168993493867


def _coco_block(cfg) -> dict:
    (block,) = [d for d in cfg["data"]["datasets"] if d["name"] == "coco"]
    return block


def test_the_subset_has_twenty_five_distinct_categories():
    assert len(INDOOR_25) == 25
    assert len(set(INDOOR_25)) == 25, "a duplicate would silently shrink the denominator"


def test_the_config_matches_the_canonical_list_exactly():
    """Order included. `score_classes` is sorted into category ids downstream, so order
    does not change the score -- but a diff that reorders is a diff someone edited, and
    this list is not supposed to be edited."""
    cfg = load_config(str(EVAL_CFG), [])
    assert _coco_block(cfg)["score_classes"] == INDOOR_25


def test_the_baseline_is_recorded_where_it_can_be_re_run():
    """The expected numbers have to be in the file you would run, not only in a journal
    entry -- otherwise reproducing the baseline means first finding out what it was."""
    text = EVAL_CFG.read_text()
    assert f"{BASELINE_MAP}" in text
    assert f"{BASELINE_MAP50}" in text


def test_the_sample_ratio_is_part_of_the_baseline():
    """0.1, not the 0.2 inherited from hydranet_indoor.yaml. Scoring runs over whichever
    images the split yields, so this value is as much a part of the published figure as
    the category list is."""
    cfg = load_config(str(EVAL_CFG), [])
    assert _coco_block(cfg)["sample_ratio"] == 0.1


def test_the_ade20k_block_still_matches_its_parent():
    """The COCO block had to be redeclared because list values replace rather than
    merge, which drags the ADE20K block along for the ride. If the parent's segmentation
    setup changes and this copy does not, the two configs quietly stop evaluating the
    same thing."""
    evaluated = load_config(str(EVAL_CFG), [])
    parent = load_config(str(CONFIGS / "hydranet_indoor.yaml"), [])

    def ade(cfg):
        (block,) = [d for d in cfg["data"]["datasets"] if d["name"] == "ade20k"]
        return block

    assert ade(evaluated) == ade(parent)


def test_the_config_is_valid():
    check_config(load_config(str(EVAL_CFG), []))


def test_score_classes_is_not_set_on_the_training_configs():
    """Narrowing what is *scored* on a training config would change every run's
    reported mAP without changing what the model learned -- the one thing `score_classes`
    must never be used for."""
    paths = sorted(CONFIGS.glob("hydranet_*.yaml"))
    assert paths, "no configs found; the loop below would silently pass"
    checked = 0
    for path in paths:
        raw = yaml.safe_load(path.read_text())
        for ds in (raw.get("data") or {}).get("datasets") or []:
            checked += 1
            assert "score_classes" not in ds, (
                f"{path.name} sets score_classes; put it in an eval-only config instead"
            )
    assert checked, "no config carried a datasets list; the assertion above never ran"


@pytest.mark.skipif(
    not COCO_VAL_ANNOTATIONS.exists(),
    reason="needs the real COCO annotations",
)
def test_every_name_is_a_real_coco_category():
    """A typo here does not raise until an evaluation is halfway through."""
    import contextlib
    import io

    from pycocotools.coco import COCO

    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(COCO_VAL_ANNOTATIONS))
    names = {c["name"] for c in coco.loadCats(coco.getCatIds())}
    assert not [n for n in INDOOR_25 if n not in names]

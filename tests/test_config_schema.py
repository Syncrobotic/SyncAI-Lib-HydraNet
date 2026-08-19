"""Config validation: a setting that never took effect is the expensive failure.

pytest tests/test_config_schema.py -v
"""

from pathlib import Path

import pytest

from syncai_hydranet.config import load_config
from syncai_hydranet.config_schema import (
    ConfigError,
    _check_minority_sourced,
    _Report,
    check_config,
    minority_sourced_terrain_classes,
    unsourced_terrain_classes,
)
from syncai_hydranet.data.label_maps import SCHEMES

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def _cfg():
    """A known-good config to mutate; validated on load, so it starts clean."""
    return load_config(CONFIG_DIR / "hydranet_indoor.yaml")


# ------------------------------------------------------- the shipped configs


# The empty output channels the shipped configs currently carry, pinned as data rather
# than allowed as a class of warning. Each of these is a class the taxonomy declares and
# no dataset in that config can produce: it trains on nothing and reports IoU 0.000 for
# the life of the run. They are recorded here so that they stay visible and so that a
# *new* one fails this suite instead of joining them quietly.
#
# `hydranet_regnet800mf` and `hydranet_retail_cctv` are absent because this check reports
# them clean -- but read the second one with the caveat below rather than as a result. It
# pairs ADE20K with site masks under a *native identity* scheme, and an identity scheme
# expresses every class in the taxonomy whether or not a pixel of it was ever drawn. So it
# silences this check by construction. It is silence, not evidence: those masks contain
# 0 pixels of `floor_metal`, `wet_slippery`, `threshold_ramp`, `floor_soft`, `stairs` and
# `glass`, counted over all 408 of them.
KNOWN_UNSOURCED = {
    "eval_indoor25.yaml": ("floor_metal", "wet_slippery", "threshold_ramp"),
    "hydranet_indoor.yaml": ("floor_metal", "wet_slippery", "threshold_ramp"),
    "hydranet_retail.yaml": ("floor_metal", "wet_slippery", "threshold_ramp"),
    "hydranet_retail_cocostuff.yaml": ("floor_metal", "wet_slippery", "threshold_ramp"),
    # Documented in label_maps_retail_objects.py: no public segmentation dataset labels
    # merchandise, so this one is filled from site annotation or it is not filled.
    "hydranet_retail_objects.yaml": ("product",),
    # Inherits the line above, and inherits its empty channel with it. Listed rather than
    # pattern-matched on the parent: a derived config is free to add a dataset that fills
    # the channel, and if it does, this entry has to change. The whole point of pinning
    # these as data is that a config's empty channels are stated per config.
    "hydranet_retail_objects_nc2.yaml": ("product",),
    # The robot's four-head config. Same twelve terrain classes as the three-head model
    # it derives from, so it inherits the same three that ADE20K cannot source -- adding
    # a depth head does not fill a segmentation channel. Listed rather than inferred for
    # the reason above: if a dataset is ever added that fills them, this line must change.
}


# Classes one segmentation dataset in the config produces and another cannot. Not empty
# channels -- these train on real pixels -- but the larger dataset supplies every one of
# those pixels as a *negative*, which suppresses rather than merely dilutes. Pinned for
# the same reason as the table above: a new one has to be noticed.
#
# `hydranet_retail_cctv`'s three entries are FALSE POSITIVES and are pinned as such. An
# earlier version of this comment claimed `site_cctv_pseudo` draws `floor_metal`,
# `wet_slippery` and `threshold_ramp` and that ADE20K merely outvotes them -- i.e. that a
# months-old finding was backwards. It is not. Counted over every mask in
# `datasets/retail_cctv_pseudo`: **0 pixels in 0 of 408 masks**, for all three, and for
# `floor_soft`, `stairs` and `glass` too.
#
# The check said otherwise because `retail_native` is an *identity* map, so it expresses
# all 13 classes by construction and claims every one of them. Expressible is not present.
# `site_cctv_pseudo` is pseudo-labels from a model scoring 0.000 on those three, so it
# cannot contain them: the labels are that model's opinion and its opinion is that they do
# not exist. The months-old conclusion -- they need site annotation -- stands.
#
# Kept rather than filtered out, because the warning now carries the identity-map caveat
# and a reader who sees these needs to see why they are here. Found by `26251130`, who
# counted the pixels instead of believing the label map.
KNOWN_MINORITY_SOURCED = {
    # UNVERIFIED. Both `rugd` and `rellis` are explicit maps, so unlike the entry below
    # this is at least scheme-grounded -- rugd emits `rock`, rellis does not. Neither
    # dataset is on this machine, so the pixels have not been counted. Do not act on it.
    "hydranet_regnet800mf.yaml": ("rock",),
    # FALSE POSITIVES, per the paragraph above. 0 pixels in 0 of 408 masks.
    "hydranet_retail_cctv.yaml": ("floor_metal", "threshold_ramp", "wet_slippery"),
    # VERIFIED by counting: batch01 is 19.28% `product`. Real suppression, and the run that
    # proved it sat at IoU 0.000 for 22 epochs.
    "hydranet_retail_objects_site.yaml": ("product",),
    "hydranet_retail_objects_site_balanced.yaml": ("product",),
    # VERIFIED by counting `datasets/retail_objects_batch02`: **12.88% `product` pixels in
    # 360 of 360 masks**. Counted rather than trusted, because the claimant is again the
    # `retail_objects_native` identity map and that is exactly what produced the false
    # positives above. Also worth knowing from the same count: batch02 carries `column` at
    # 3.58% in 201/360 masks, which is the first real data for the class that predicted
    # 0.00% on store cameras.
    "hydranet_retail_products.yaml": ("product",),
    # Inherited whole from the line above via `_base_`, and pinned per config rather than
    # derived from the parent on purpose -- a derived config is free to add a dataset that
    # changes this, so reading it off the parent would make the check weaker than the
    # thing it checks. `hydranet_retail_openvocab.yaml` changes only the detection
    # classifier (`linear` -> `text_embedding`); it touches no segmentation dataset, so
    # the same 12.88% count stands and no separate verification was done or is owed.
    "hydranet_retail_openvocab.yaml": ("product",),
    # Seed replicates of `hydranet_retail_products.yaml`, differing only in `seed` and
    # `output_dir`. They inherit the same segmentation datasets and therefore the same
    # 12.88% count; a replicate that needed its own verification would not be a replicate.
    "hydranet_retail_products_seed7.yaml": ("product",),
    "hydranet_retail_products_seed13.yaml": ("product",),
    # Same dataset as the line above and the same count: 12.88% `product` in 360/360 masks
    # of `datasets/retail_objects_batch02`. This is the config that was training when the
    # entry was added, and the run confirms the warning was right about the *direction* --
    # `product` went from 22 epochs pinned at 0.000 under the old ratios to 0.253 by
    # epoch 10 under ade20k 1.0 -> 0.15. Suppression, not absence.
    "hydranet_retail_objects_site_batch02.yaml": ("product",),
}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_raise_nothing_and_warn_only_about_pinned_class_problems(path):
    """This assertion used to be `== []`.

    That was true only because nothing measured the empty channels; it was not evidence
    that there were none. Keeping the form strict -- every warning must be one of the two
    pinned kinds -- means a config that grows any *other* warning still fails here.
    """
    warnings = check_config(load_config(path, validate=False))
    unpinned = [
        w
        for w in warnings
        if "empty output channel" not in w and "is produced only by" not in w
    ]
    assert unpinned == []


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_config_minority_sourced_classes_are_the_known_ones(path):
    """The check that would have named `product`'s 22 epochs at 0.000 before the run.

    Fails on a new one rather than after sixty epochs of a plausible-looking curve, which
    is the same bargain as the empty-channel table above and the reason both are data.
    """
    found = minority_sourced_terrain_classes(load_config(path, validate=False))
    assert tuple(sorted(found)) == KNOWN_MINORITY_SOURCED.get(path.name, ())


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_config_empty_channels_are_the_known_ones(path):
    """A new empty channel fails here rather than at epoch 60, which is the whole point."""
    found = unsourced_terrain_classes(load_config(path, validate=False))
    flat = tuple(name for classes in found.values() for name in classes)
    assert flat == KNOWN_UNSOURCED.get(path.name, ())


def test_configs_exist():
    assert CONFIGS, "no configs found; the parametrised test above would silently pass"


# ------------------------------------------------------------- unknown keys


def test_unknown_key_is_rejected_with_a_suggestion():
    cfg = _cfg()
    cfg["train"]["batchsize"] = 4
    with pytest.raises(ConfigError, match="batch_size"):
        check_config(cfg)


def test_dead_key_is_rejected():
    """save_interval was read by nothing for the whole life of the project."""
    cfg = _cfg()
    cfg["train"]["save_interval"] = 1
    with pytest.raises(ConfigError, match="unknown setting"):
        check_config(cfg)


def test_command_line_override_typo_is_caught():
    """--set applies before validation, so a typo there fails as loudly as one in the
    file rather than creating a key nothing reads."""
    with pytest.raises(ConfigError, match="learning_rate"):
        load_config(CONFIG_DIR / "hydranet_indoor.yaml", ["train.learning_rate=1e-4"])


def test_valid_override_still_loads():
    cfg = load_config(CONFIG_DIR / "hydranet_indoor.yaml", ["train.lr=1e-3"])
    assert cfg["train"]["lr"] == 1e-3


def test_validation_can_be_skipped():
    cfg = load_config(CONFIG_DIR / "hydranet_indoor.yaml", validate=False)
    assert cfg["experiment"] == "hydranet_indoor"


# -------------------------------------------------------------- value checks


def test_wrong_type_is_rejected():
    cfg = _cfg()
    cfg["train"]["epochs"] = "sixty"
    with pytest.raises(ConfigError, match="expected int"):
        check_config(cfg)


def test_bool_is_not_accepted_where_a_number_belongs():
    cfg = _cfg()
    cfg["train"]["lr"] = True
    with pytest.raises(ConfigError, match="got bool"):
        check_config(cfg)


def test_unimplemented_scheduler_is_rejected():
    """The key existed but nothing read it: asking for step decay got cosine anyway."""
    cfg = _cfg()
    cfg["train"]["scheduler"] = "step"
    with pytest.raises(ConfigError, match="cosine"):
        check_config(cfg)


def test_input_size_must_be_two_positive_integers():
    cfg = _cfg()
    cfg["data"]["input_size"] = [512]
    with pytest.raises(ConfigError, match=r"\[height, width\]"):
        check_config(cfg)


def test_missing_required_key_is_reported():
    cfg = _cfg()
    del cfg["train"]["lr"]
    with pytest.raises(ConfigError, match="required setting is missing"):
        check_config(cfg)


# ------------------------------------------------------------ cross-checks


def test_supervising_an_undeclared_head_is_rejected():
    """Silent in every other way: the head simply never receives a loss, and you find
    out at deployment."""
    cfg = _cfg()
    cfg["data"]["datasets"][0]["supervises"] = ["traversibility"]  # misspelt
    with pytest.raises(ConfigError, match="not a declared head"):
        check_config(cfg)


def test_unknown_label_map_is_rejected():
    cfg = _cfg()
    cfg["data"]["datasets"][0]["label_map"] = "ade20k"
    with pytest.raises(ConfigError, match="not a known scheme"):
        check_config(cfg)


def test_terrain_class_count_must_match_the_head():
    """Otherwise every per-class IoU is logged under the wrong class name."""
    cfg = _cfg()
    cfg["data"]["terrain_classes"] = cfg["data"]["terrain_classes"][:-1]
    with pytest.raises(ConfigError, match="terrain_classes has 11"):
        check_config(cfg)


def test_duplicate_dataset_names_are_rejected():
    cfg = _cfg()
    cfg["data"]["datasets"].append(dict(cfg["data"]["datasets"][0]))
    with pytest.raises(ConfigError, match="used by an earlier dataset"):
        check_config(cfg)


def test_unsupervised_head_warns_but_does_not_fail():
    """Dropping a dataset to get started is documented and supported; leaving a head at
    its initial weights is worth saying out loud, not worth refusing."""
    cfg = _cfg()
    cfg["data"]["datasets"] = [cfg["data"]["datasets"][0]]  # segmentation only
    warnings = check_config(cfg)
    assert any("detection" in w for w in warnings)


def test_every_problem_is_reported_in_one_pass():
    """One round trip should be enough to fix a config."""
    cfg = _cfg()
    cfg["train"]["epochs"] = "sixty"
    cfg["train"]["batchsize"] = 4
    cfg["data"]["input_size"] = [512]
    with pytest.raises(ConfigError) as e:
        check_config(cfg)
    assert str(e.value).startswith("3 problem(s)")


def test_a_dataset_that_supervises_nothing_is_an_error_not_a_warning():
    """Every batch it contributes reaches `compute_losses` with no loss to build.

    The balancer sums an empty dict to None and the step dies on `None.detach()`, deep in
    the training loop and hours from the config that caused it. A dataset supervising no
    head also cannot be what anyone meant: it costs optimiser steps and contributes no
    gradient to anything.
    """
    cfg = _cfg()
    cfg["data"]["datasets"][0]["supervises"] = []
    with pytest.raises(ConfigError) as exc:
        check_config(cfg)
    assert "supervises" in str(exc.value)


# --------------------------------------------------- classes nothing can produce


def test_an_empty_channel_warns_but_does_not_fail():
    """Same reasoning as the unsupervised head above: a class with no source is
    legitimate while a dataset is still being assembled, and worth saying out loud."""
    cfg = _cfg()
    warnings = check_config(cfg)
    assert any("wet_slippery" in w and "empty output channel" in w for w in warnings)


def test_a_native_scheme_silences_it_because_annotation_can_draw_any_class():
    """The mechanism that will fill `product` once site masks exist, pinned.

    A native scheme is an identity map, so it claims every id in the taxonomy. That is
    the honest answer at config time -- an annotator may draw any class the taxonomy
    has -- and it is also the limit of the check: it proves the class is expressible,
    never that a pixel of it was drawn. See `test_it_cannot_see_whether_any_pixel_exists`.
    """
    cfg = _cfg()
    assert unsourced_terrain_classes(cfg)  # ade20k_indoor alone cannot supply them
    site = dict(cfg["data"]["datasets"][0])
    site["name"] = "site_masks"
    site["label_map"] = "indoor_native"
    cfg["data"]["datasets"].append(site)
    assert unsourced_terrain_classes(cfg) == {}


def test_it_cannot_see_whether_any_pixel_exists():
    """The check reads the config, not the masks on disk.

    Stated as a test so the guarantee is not overread later: `retail_objects_native`
    silences `product` the moment it appears in a config, whether or not the prelabel
    pass that was supposed to produce it emitted a single instance. Counting instances
    is a job for whatever writes the masks.

    `column` is the measured case for the stronger limit. It is sourced from ADE20K id
    43, so it passes here and always has, and it still predicted 0.00% of pixels across
    four daytime shop-floor cameras while scoring val IoU 0.40-0.51. Passing this check
    is not evidence that a class works.
    """
    assert 5 not in SCHEMES["ade20k_retail_objects"].mapping.values()
    assert 5 in SCHEMES["retail_objects_native"].mapping.values()
    assert (
        3 in SCHEMES["ade20k_retail_objects"].mapping.values()
    )  # column: sourced, and 0% on site


def test_two_taxonomies_in_one_config_are_reported_separately():
    """A config may legitimately mix schemes that share no class list, and merging their
    mappings would let one taxonomy's ids vouch for another's classes."""
    cfg = _cfg()
    other = dict(cfg["data"]["datasets"][0])
    other["name"] = "offroad"
    other["label_map"] = "rugd"
    cfg["data"]["datasets"].append(other)
    found = unsourced_terrain_classes(cfg)
    assert list(found) == ["ade20k_indoor"]
    assert "wet_slippery" in found["ade20k_indoor"]


# ------------------------------------------------- fixed_weights names heads


def test_a_fixed_weight_naming_no_head_is_rejected():
    """`FixedWeighting.forward` does `weights.get(name, 1.0)`, so the key is dropped and
    the reweighting the config asked for silently never happens."""
    cfg = _cfg()
    cfg["model"]["loss_balancing"] = "fixed"
    cfg["model"]["fixed_weights"] = {"terrian": 0.5, **cfg["model"]["fixed_weights"]}
    with pytest.raises(ConfigError, match="names no declared head"):
        check_config(cfg)


def test_a_head_with_no_fixed_weight_warns_that_it_defaults():
    cfg = _cfg()
    cfg["model"]["loss_balancing"] = "fixed"
    cfg["model"]["fixed_weights"] = {"terrain": 0.5}
    warnings = check_config(cfg)
    assert any("detection" in w and "1.0" in w for w in warnings)


def test_an_inherited_stray_weight_is_silent_under_uncertainty():
    """The case that made this check conditional rather than universal.

    `_base/hydranet.yaml` weights all three heads, so every derived config that removes
    one -- `hydranet_retail_objects.yaml` sets `traversability: null` -- inherits a
    weight for a head it does not declare. Under `uncertainty` the table is never read,
    so that is a correct config and must not warn.
    """
    cfg = load_config(CONFIG_DIR / "hydranet_retail_objects.yaml", validate=False)
    assert cfg["model"]["loss_balancing"] == "uncertainty"
    assert "traversability" in cfg["model"]["fixed_weights"]
    assert "traversability" not in cfg["model"]["heads"]
    assert [w for w in check_config(cfg) if "fixed_weights" in w] == []


# ------------------------------------------ classes one dataset has and another lacks


def _two_source_cfg(second_map: str = "retail_objects_native"):
    return {
        "data": {
            "datasets": [
                {"name": "ade20k", "label_map": "ade20k_retail_objects", "sample_ratio": 1.0},
                {"name": "site", "label_map": second_map, "sample_ratio": 0.05},
            ]
        }
    }


def test_a_class_only_the_small_dataset_can_produce_is_reported():
    found = minority_sourced_terrain_classes(_two_source_cfg())
    assert "product" in found
    produces, lacks = found["product"]
    assert produces == [("site", 0.05, True)]  # identity map: a claim, not evidence
    assert lacks == [("ade20k", 1.0, False)]


def test_the_unsourced_gate_goes_quiet_on_exactly_this_case():
    """Why this function had to exist, stated as a test rather than as a docstring claim.

    `unsourced_terrain_classes` asks whether *any* dataset can produce the class. Adding
    one tiny source flips that to yes and silences it -- at the moment the harder failure
    begins, because the dominant dataset now supplies every pixel of that class as a
    negative. `product` sat at IoU 0.000 for 22 epochs in exactly this configuration.
    """
    cfg = _two_source_cfg()
    assert unsourced_terrain_classes(cfg) == {}, "the old gate says nothing here"
    assert "product" in minority_sourced_terrain_classes(cfg), "the new one has to"


def test_a_wholly_unsourced_class_is_not_also_reported_as_minority_sourced():
    """The two checks partition the problem; a class in both would be double-warned."""
    cfg = {"data": {"datasets": [{"name": "ade20k", "label_map": "ade20k_retail_objects"}]}}
    assert unsourced_terrain_classes(cfg) == {"ade20k_retail_objects": ("product",)}
    assert minority_sourced_terrain_classes(cfg) == {}


def test_a_detection_dataset_is_not_counted_as_supplying_negatives():
    """A COCO set has no `label_map` and contributes no segmentation target at all.

    Counting it among the datasets that "cannot produce" a terrain class would report
    every class in every multi-task config as minority-sourced, which is both wrong and
    the fastest way to get a warning ignored.
    """
    cfg = {
        "data": {
            "datasets": [
                {"name": "ade20k", "label_map": "ade20k_retail_objects", "sample_ratio": 1.0},
                {"name": "coco", "type": "coco", "supervises": ["detection"]},
            ]
        }
    }
    assert minority_sourced_terrain_classes(cfg) == {}


def test_the_advice_prefers_lowering_the_abundant_ratio():
    """Raising the scarce dataset reaches the same balance by repeating a few images many
    times an epoch -- it trades a suppressed channel for a memorised one. The message has
    to say which direction, or it pushes people toward the worse of the two fixes."""
    rep = _Report()
    _check_minority_sourced(rep, _two_source_cfg())
    assert len(rep.warnings) == 1
    assert "lower the abundant" in rep.warnings[0]
    assert "memorised" in rep.warnings[0]
    assert "ade20k (sample_ratio 1)" in rep.warnings[0]


def test_two_taxonomies_do_not_vouch_for_each_other():
    """A config may mix schemes sharing no class list; a class in one is not "missing"
    from a dataset that labels into the other."""
    cfg = _two_source_cfg()
    cfg["data"]["datasets"].append(
        {"name": "offroad", "label_map": "rugd", "sample_ratio": 1.0}
    )
    found = minority_sourced_terrain_classes(cfg)
    assert "product" in found
    assert [n for n, _r, _i in found["product"][1]] == ["ade20k"], "rugd must not be listed"


def test_an_identity_only_producer_is_flagged_as_unconfirmed():
    """The guard against the false positive this check shipped with on 2026-08-17.

    An identity map expresses every class in its taxonomy, so it claims all of them and
    evidences none. Reported as "merely outvoted", `floor_metal`/`wet_slippery`/
    `threshold_ramp` contradicted a months-old finding that they need annotation -- and
    the masks contain 0 pixels of all three, in 0 of 408. The warning has to say that the
    producing side is a claim, or it reads as a measurement.
    """
    rep = _Report()
    _check_minority_sourced(rep, _two_source_cfg())
    assert len(rep.warnings) == 1
    assert "CONFIRM WITH A PIXEL COUNT FIRST" in rep.warnings[0]
    assert "identity label map" in rep.warnings[0]


def test_a_real_producer_carries_no_such_caveat():
    """`retail_objects_migrated` is an explicit map, so what it emits it genuinely emits.

    Without this the caveat would be unconditional, which is the same failure one level
    up: a warning that always hedges tells a reader nothing about which case they have.
    """
    rep = _Report()
    _check_minority_sourced(rep, _two_source_cfg(second_map="retail_objects_migrated"))
    assert rep.warnings, "an explicit map that emits ids the other lacks must still warn"
    assert "CONFIRM WITH A PIXEL COUNT FIRST" not in rep.warnings[0]


def test_the_lacking_side_is_trustworthy_even_when_the_producing_side_is_not():
    """The asymmetry that makes the check worth keeping rather than reverting.

    "cannot produce" is sound: an explicit map that never emits an id genuinely cannot.
    Only "can produce" needs a pixel count. So the partition is half-evidence, and the
    warning is shaped around which half.
    """
    found = minority_sourced_terrain_classes(_two_source_cfg())
    _produces, lacks = found["product"]
    assert lacks == [("ade20k", 1.0, False)], "ADE20K genuinely emits no product id"

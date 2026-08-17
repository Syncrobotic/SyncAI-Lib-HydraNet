"""Config validation: a setting that never took effect is the expensive failure.

pytest tests/test_config_schema.py -v
"""

from pathlib import Path

import pytest

from syncai_hydranet.config import load_config
from syncai_hydranet.config_schema import (
    ConfigError,
    check_config,
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
# `hydranet_regnet800mf` and `hydranet_retail_cctv` are absent because they are clean,
# and the second one shows what clean costs: it pairs ADE20K with site masks under a
# native identity scheme, so annotation is what fills the channels ADE20K cannot.
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
}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_raise_nothing_and_warn_only_about_empty_channels(path):
    """This assertion used to be `== []`.

    That was true only because nothing measured the empty channels; it was not evidence
    that there were none. Keeping the form strict -- every warning must be one of the
    pinned ones -- means a config that grows any *other* warning still fails here.
    """
    warnings = check_config(load_config(path, validate=False))
    assert [w for w in warnings if "empty output channel" not in w] == []


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

"""Config validation: a setting that never took effect is the expensive failure.

pytest tests/test_config_schema.py -v
"""

from pathlib import Path

import pytest

from syncai_hydranet.config import load_config
from syncai_hydranet.config_schema import ConfigError, check_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def _cfg():
    """A known-good config to mutate; validated on load, so it starts clean."""
    return load_config(CONFIG_DIR / "hydranet_indoor.yaml")


# ------------------------------------------------------- the shipped configs


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_shipped_configs_are_valid(path):
    assert check_config(load_config(path, validate=False)) == []


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

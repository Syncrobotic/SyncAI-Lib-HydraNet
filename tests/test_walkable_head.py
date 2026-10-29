"""The binary walkable head: the `_walk` schemes, the width check, and a warm start that
tolerates a head the checkpoint never had.

pose02's three heads plus the four-head model's `traversability` (blocked / go) is one
network, warm-started from pose02's weights. Three things had to change for that and each
is pinned here: a scheme can be two-way, the config refuses a width that does not match
the head, and `--load` names the tensors it could not fill instead of refusing them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from syncai_hydranet.config import load_config
from syncai_hydranet.config_schema import ConfigError, check_config
from syncai_hydranet.data.label_maps import (
    SCHEMES,
    TRAV_CLASSES,
    WALK_CLASSES,
    WALK_GO,
    walkable_from_trav,
)
from syncai_hydranet.engine.evaluator import _trav_names
from syncai_hydranet.engine.trainer import Trainer
from syncai_hydranet.labels import IGNORE
from syncai_hydranet.models.hydranet import build_model
from syncai_hydranet.utils.visualize import TRAV_COLORS, WALK_COLORS, trav_colors

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

WALK_BASES = (
    "ade20k_retail_surfaces",
    "retail_surfaces_native",
    "retail_surfaces_from_objects",
    "retail_v3_native",
)


# ------------------------------------------------------------------ schemes


def test_walkable_from_trav_collapses_caution_into_blocked():
    assert walkable_from_trav({0: IGNORE, 1: 2, 2: 1, 3: 0}) == {0: IGNORE, 1: 1, 2: 0, 3: 0}


@pytest.mark.parametrize("base", WALK_BASES)
def test_walk_twin_shares_the_terrain_path_and_is_binary(base):
    src, walk = SCHEMES[base], SCHEMES[f"{base}_walk"]
    assert walk.mapping == src.mapping and walk.classes == src.classes
    assert src.trav_classes == TRAV_CLASSES and walk.trav_classes == WALK_CLASSES
    assert set(walk.trav.values()) <= {0, 1, IGNORE}
    floor = walk.classes.index("floor")
    assert walk.trav[floor] == WALK_GO
    # Only floor walks: every other terrain class the three-way table knew is blocked.
    assert [t for t, v in walk.trav.items() if v == WALK_GO] == [floor]


# --------------------------------------------------------------- the config


def _walk_cfg():
    return load_config(CONFIG_DIR / "hydranet_retail_walk01_smoke.yaml")


def test_shipped_walk_config_is_four_heads_and_validates():
    cfg = _walk_cfg()
    assert list(cfg["model"]["heads"]) == ["traversability", "terrain", "detection", "pose"]
    assert cfg["model"]["heads"]["traversability"]["num_classes"] == 2
    assert check_config(cfg) == []


def test_three_way_scheme_on_a_two_class_head_is_refused():
    cfg = _walk_cfg()
    cfg["data"]["datasets"][0]["label_map"] = "ade20k_retail_surfaces"
    with pytest.raises(ConfigError, match="whose traversability has 3"):
        check_config(cfg)


def test_walk_scheme_on_a_three_class_head_is_refused():
    cfg = _walk_cfg()
    cfg["model"]["heads"]["traversability"]["num_classes"] = 3
    with pytest.raises(ConfigError, match="whose traversability has 2"):
        check_config(cfg)


def test_two_widths_on_one_head_are_refused_together():
    cfg = _walk_cfg()
    cfg["model"]["heads"]["traversability"]["num_classes"] = 3
    cfg["data"]["datasets"][0]["label_map"] = "ade20k_retail_surfaces"
    with pytest.raises(ConfigError, match="different widths"):
        check_config(cfg)


# ------------------------------------------------------- names and colours


def test_metric_names_follow_the_head_width():
    assert _trav_names(_walk_cfg()) == WALK_CLASSES
    assert _trav_names({"model": {"heads": {"traversability": {"num_classes": 3}}}}) == (
        TRAV_CLASSES
    )
    assert _trav_names({"data": {}}) == TRAV_CLASSES


def test_walk_palette_keeps_go_green_and_drops_caution():
    assert np.array_equal(trav_colors(2), WALK_COLORS)
    assert np.array_equal(WALK_COLORS, TRAV_COLORS[[0, 2]])
    assert np.array_equal(trav_colors(3), TRAV_COLORS)
    with pytest.raises(ValueError, match="2 or 3"):
        trav_colors(4)


# ------------------------------------------------------------ warm start


def _model_cfg(heads: dict) -> dict:
    return {
        "model": {
            "backbone": {"name": "resnet18", "pretrained": False},
            "neck": {"name": "fpn", "out_channels": 16, "num_levels": 5},
            "heads": heads,
            "loss_balancing": "uncertainty",
        }
    }


SEG = {"type": "semantic_fpn", "in_levels": [0, 1, 2], "channels": 16}


def _load(module, state):
    stub = SimpleNamespace(logger=logging.getLogger("test_walkable_head"))
    return Trainer._load_weights(stub, module, state, "model")


def test_load_fills_what_it_can_and_names_the_new_head(caplog):
    three = build_model(_model_cfg({"terrain": {**SEG, "num_classes": 6}}))
    four = build_model(
        _model_cfg(
            {"traversability": {**SEG, "num_classes": 2}, "terrain": {**SEG, "num_classes": 6}}
        )
    )
    with caplog.at_level(logging.WARNING, logger="test_walkable_head"):
        missing = _load(four, three.state_dict())
    assert missing and all(
        k.startswith(("seg_heads.traversability.", "balancer.log_vars.traversability"))
        for k in missing
    )
    assert "seg_heads.traversability" in caplog.text
    assert "balancer.log_vars.traversability" in caplog.text
    # The shared tensors really were loaded, not merely tolerated.
    for k, v in three.state_dict().items():
        assert (four.state_dict()[k] == v).all(), k


def test_load_refuses_a_checkpoint_the_model_cannot_place():
    bigger = build_model(
        _model_cfg(
            {"traversability": {**SEG, "num_classes": 2}, "terrain": {**SEG, "num_classes": 6}}
        )
    )
    smaller = build_model(_model_cfg({"terrain": {**SEG, "num_classes": 6}}))
    with pytest.raises(ValueError, match="no place for"):
        _load(smaller, bigger.state_dict())


def test_load_refuses_a_class_count_that_changed():
    src = build_model(_model_cfg({"traversability": {**SEG, "num_classes": 3}}))
    dst = build_model(_model_cfg({"traversability": {**SEG, "num_classes": 2}}))
    with pytest.raises(ValueError, match="shape mismatch"):
        _load(dst, src.state_dict())

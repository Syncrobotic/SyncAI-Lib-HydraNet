"""A detection `primary_metric` plus a detection val interval above 1 is a run that dies.

Both halves are correct alone. `detection_val_interval` exists to skip expensive detection
validation, and `select_metric` refuses an absent selection metric rather than guessing --
so a config that combines them loses its selection metric on the epochs in between and
raises `KeyError` at the end of epoch 1, with the data loaded and the GPU warm.

Found while setting up the retail+security seed replicates, i.e. one command before it
would have taken out three runs. `runs/_crashed_20260817_surfaces_palette` is what that
looks like when nobody checks first.

pytest tests/test_val_interval_trap.py -v
"""

from __future__ import annotations

import pytest

from syncai_hydranet.config import load_config
from syncai_hydranet.config_schema import ConfigError, check_config

SHIPPED = "configs/hydranet_retail_security.yaml"


def test_the_shipped_pairing_is_allowed():
    cfg = load_config(SHIPPED)
    assert cfg["train"]["primary_metric"] == "detection_mAP/site_boxes"
    assert check_config(cfg) == []


def test_raising_the_interval_under_a_detection_metric_is_refused():
    cfg = load_config(SHIPPED)
    cfg["train"]["detection_val_interval"] = 5
    with pytest.raises(ConfigError, match="raises KeyError at the end of epoch 1"):
        check_config(cfg)


def test_a_metric_from_a_dataset_that_also_trains_segmentation_survives():
    """Narrow on purpose: that val set is kept, so its metric is produced every epoch."""
    cfg = load_config(SHIPPED)
    cfg["train"]["detection_val_interval"] = 5
    for ds in cfg["data"]["datasets"]:
        if ds["name"] == "site_boxes":
            ds["supervises"] = ["detection", "terrain"]
    check_config(cfg)  # no raise


def test_a_segmentation_metric_is_unaffected():
    cfg = load_config(SHIPPED)
    cfg["train"]["detection_val_interval"] = 5
    cfg["train"]["primary_metric"] = "terrain_mIoU"
    assert check_config(cfg) == []

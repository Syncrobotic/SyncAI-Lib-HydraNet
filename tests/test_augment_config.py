"""Augmentation is part of a run's lineage, or it is an unexplainable difference.

It was hard-coded in `transforms.py` until this project's own rule -- the most expensive
mistake is a setting that never took effect -- was noticed to apply to itself. Two runs
could differ only in jitter strength and nothing in `meta.json` would say so.

The defaults must not move: runs from before the change have to stay comparable to runs
from after it.

pytest tests/test_augment_config.py -v
"""

import pytest

from syncai_hydranet.config_schema import ConfigError, check_config
from syncai_hydranet.data.transforms import (
    AUGMENT_DEFAULTS,
    ColorJitter,
    LetterboxScaleCrop,
    RandomHorizontalFlip,
    RandomScaleCrop,
    build_transforms,
)

SIZE = (64, 80)


def _stages(compose):
    return {type(t).__name__: t for t in compose.ts}


def test_defaults_are_the_values_trained_on_before_this_was_configurable():
    """If these move, every run recorded before the change becomes incomparable."""
    assert AUGMENT_DEFAULTS == {
        "scale_range": (0.75, 1.5),
        "flip_p": 0.5,
        "brightness": 0.3,
        "contrast": 0.3,
        "saturation": 0.3,
    }


def test_unconfigured_pipeline_matches_the_defaults():
    s = _stages(build_transforms(SIZE, train=True))
    assert s["RandomScaleCrop"].scale_range == (0.75, 1.5)
    assert s["RandomHorizontalFlip"].p == 0.5
    assert (s["ColorJitter"].b, s["ColorJitter"].c, s["ColorJitter"].s) == (0.3, 0.3, 0.3)


def test_overrides_reach_every_stage():
    s = _stages(
        build_transforms(
            SIZE,
            train=True,
            augment={"scale_range": [0.5, 2.0], "flip_p": 0.0, "brightness": 0.1},
        )
    )
    assert s["RandomScaleCrop"].scale_range == (0.5, 2.0)
    assert s["RandomHorizontalFlip"].p == 0.0
    assert s["ColorJitter"].b == 0.1
    assert s["ColorJitter"].c == 0.3, "unspecified keys keep their default"


def test_letterbox_path_is_configured_too():
    """Two geometric pipelines exist; configuring only one is the kind of gap that
    shows up as 'it works off-road but not indoors'."""
    s = _stages(
        build_transforms(SIZE, train=True, letterbox=True, augment={"scale_range": [0.9, 1.1]})
    )
    assert isinstance(s["LetterboxScaleCrop"], LetterboxScaleCrop)
    assert s["LetterboxScaleCrop"].scale_range == (0.9, 1.1)


def test_validation_never_augments():
    names = set(_stages(build_transforms(SIZE, train=False)))
    assert not names & {"RandomScaleCrop", "RandomHorizontalFlip", "ColorJitter"}


# -------------------------------------------------------------------------- schema


def _cfg(augment):
    return {
        "experiment": "x",
        "output_dir": "runs/x",
        "model": {
            "backbone": {"name": "resnet18"},
            "neck": {"name": "fpn"},
            "heads": {
                "terrain": {"type": "semantic_fpn", "num_classes": 12},
            },
        },
        "data": {
            "input_size": [64, 80],
            "augment": augment,
            "datasets": [
                {
                    "name": "d",
                    "type": "seg_folder",
                    "root": "x",
                    "split_train": "train",
                    "split_val": "val",
                    "supervises": ["terrain"],
                }
            ],
        },
        "train": {"epochs": 1, "batch_size": 1, "lr": 1e-3},
    }


def test_schema_accepts_a_valid_augment_block():
    check_config(_cfg({"scale_range": [0.75, 1.5], "flip_p": 0.5}))


def test_schema_rejects_a_typo_in_the_augment_block():
    """The whole point: a mistyped key would otherwise fall back to the default and the
    run would report a setting it never used."""
    with pytest.raises(ConfigError, match="brightnes"):
        check_config(_cfg({"brightnes": 0.4}))


def test_the_stages_are_the_ones_the_config_names():
    """Guards against a stage being dropped from the pipeline while its config key
    lingers -- the setting would validate and do nothing."""
    compose = build_transforms(SIZE, train=True)
    kinds = {type(t) for t in compose.ts}
    assert {RandomScaleCrop, RandomHorizontalFlip, ColorJitter} <= kinds

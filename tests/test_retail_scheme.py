"""The retail scheme is the indoor scheme plus one class, and must stay that way.

That alignment is what lets a retail run warm-start from an indoor checkpoint and lets
indoor site masks be read under the retail scheme without re-export. It is also invisible
-- nothing fails at training time if someone renumbers the list while tidying it up. The
model simply learns different classes than the masks mean, and every metric stays
plausible. Hence these tests.

pytest tests/test_retail_scheme.py -v
"""

from pathlib import Path

import pytest

from syncai_hydranet.config import load_config
from syncai_hydranet.data.label_maps import get_scheme
from syncai_hydranet.data.label_maps_indoor import (
    ADE20K_ID_TO_INDOOR,
    INDOOR_TERRAIN,
    INDOOR_TERRAIN_TO_TRAV,
)
from syncai_hydranet.data.label_maps_retail import (
    ADE20K_ID_TO_RETAIL,
    RETAIL_NATIVE_ID,
    RETAIL_TERRAIN,
    RETAIL_TERRAIN_TO_TRAV,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "hydranet_retail.yaml"


def test_ids_0_to_11_are_identical_to_indoor():
    """Renumbering these silently invalidates every indoor mask and checkpoint."""
    for name, tid in INDOOR_TERRAIN.items():
        assert RETAIL_TERRAIN[name] == tid


def test_retail_adds_exactly_one_class():
    assert set(RETAIL_TERRAIN) - set(INDOOR_TERRAIN) == {"display_fixture"}
    assert RETAIL_TERRAIN["display_fixture"] == len(INDOOR_TERRAIN)


def test_the_traversability_policy_agrees_with_indoor_on_the_shared_classes():
    for tid, trav in INDOOR_TERRAIN_TO_TRAV.items():
        assert RETAIL_TERRAIN_TO_TRAV[tid] == trav
    assert RETAIL_TERRAIN_TO_TRAV[RETAIL_TERRAIN["display_fixture"]] == 0  # blocked


def test_unlabelled_background_is_ignored():
    """Same trap as indoor: annotation tools export background as 0, and the loss
    ignores 255 alone, so an identity map would train `void` over unlabelled pixels."""
    assert RETAIL_NATIVE_ID[0] == 255
    assert all(RETAIL_NATIVE_ID[v] == v for v in RETAIL_TERRAIN.values() if v)


# ------------------------------------------------------------------ ADE20K bootstrap


def test_ade20k_retail_only_moves_fixtures_out_of_furniture():
    """Every id the retail map treats differently must have been obstacle_furniture."""
    moved = {k for k, v in ADE20K_ID_TO_RETAIL.items() if ADE20K_ID_TO_INDOOR.get(k) != v}
    assert moved, "the retail map is identical to indoor -- display_fixture has no source"
    for ade_id in moved:
        assert ADE20K_ID_TO_INDOOR[ade_id] == INDOOR_TERRAIN["obstacle_furniture"]
        assert ADE20K_ID_TO_RETAIL[ade_id] == RETAIL_TERRAIN["display_fixture"]


def test_domestic_tables_are_not_display_fixtures():
    """ADE20K's `table` is a dining table. Mapping it would teach the model that a
    proxy which is nearly the right thing is the right thing -- the mistake this
    project has already paid for once with COCO."""
    assert ADE20K_ID_TO_RETAIL[16] == RETAIL_TERRAIN["obstacle_furniture"]


def test_no_ade20k_id_maps_to_void():
    """Unlisted ids fall through to 255 in the loader; an explicit 0 would be trained."""
    assert 0 not in set(ADE20K_ID_TO_RETAIL.values())


# ------------------------------------------------------------------------- the config


@pytest.mark.parametrize("scheme_name", ["ade20k_retail", "retail_native"])
def test_schemes_are_registered_and_sized(scheme_name):
    scheme = get_scheme(scheme_name)
    assert scheme.num_classes == len(RETAIL_TERRAIN) == 13
    assert scheme.classes[12] == "display_fixture"


def test_config_head_width_matches_the_taxonomy():
    """A head narrower than the class list silently drops the last class; wider, and one
    output channel is trained on nothing."""
    cfg = load_config(CONFIG)
    assert cfg["model"]["heads"]["terrain"]["num_classes"] == len(RETAIL_TERRAIN)
    assert cfg["data"]["terrain_classes"] == list(RETAIL_TERRAIN)

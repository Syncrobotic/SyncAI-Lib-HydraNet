"""Free space for a config that has no traversability head, derived rather than refused.

`hydranet-scene` builds every panel it draws from free space -- the floor polygon, the wall
it raises at the boundary, the ground projection of every box. The object taxonomies drop
the traversability head on purpose, because it was a lookup on terrain rather than a second
signal, so the command used to refuse them outright.

The lookup is exactly what the panel needs. `RETAIL_OBJECTS_TO_TRAV` exists for this and
its own comment says so: "A traversability policy for a taxonomy whose configs do not train
a traversability head. It is here anyway, and it is not dead code."

What must stay refused is the case where the config's scheme ships no table at all. Which
classes a robot may walk on is not something a renderer gets to guess, and a wrong guess
here draws a floor polygon over a wall -- confidently, in metres, with no error.

pytest tests/test_scene_derived_trav.py -v
"""

import numpy as np
import pytest

from syncai_hydranet.cli.scene import _trav_map_for
from syncai_hydranet.data.label_maps import terrain_to_traversability
from syncai_hydranet.data.label_maps_retail_objects import RETAIL_OBJECTS_TO_TRAV


def cfg_with(label_map: str | None, supervises=("terrain",)) -> dict:
    return {"data": {"datasets": [{"label_map": label_map, "supervises": list(supervises)}]}}


def test_the_object_taxonomy_supplies_a_table():
    assert _trav_map_for(cfg_with("retail_objects_native")) == dict(RETAIL_OBJECTS_TO_TRAV)


def test_a_config_with_no_terrain_dataset_yields_nothing():
    """`main` turns a falsy return into a refusal, so nothing may invent a default here."""
    assert _trav_map_for(cfg_with("retail_objects_native", supervises=("detection",))) is None


def test_a_dataset_without_a_label_map_is_skipped_not_guessed():
    assert _trav_map_for(cfg_with(None)) is None


def test_the_first_terrain_dataset_decides():
    """Every terrain source in one config shares a taxonomy, so the first table is the table.

    A config whose masks disagreed about what id 3 means would be broken long before it
    reached a renderer, and picking the first is what makes the read deterministic rather
    than dependent on dataset order having no meaning.
    """
    cfg = {
        "data": {
            "datasets": [
                {"label_map": "ade20k_retail_objects", "supervises": ["terrain"]},
                {"label_map": "retail_objects_native", "supervises": ["terrain"]},
            ]
        }
    }
    assert _trav_map_for(cfg) == dict(RETAIL_OBJECTS_TO_TRAV)


def test_only_the_floor_is_walkable_under_the_object_taxonomy():
    """The panel's whole geometry starts here, so the mapping is worth pinning by hand.

    `go` is 2 and `blocked` is 0. A shop's floor is the only walkable class; a column, a
    fixture and a shelf of product are all things to walk around, and `void` is ignored
    rather than trained.
    """
    terrain = np.array([[0, 1, 2], [3, 4, 5], [6, 1, 1]], dtype=np.uint8)
    trav = terrain_to_traversability(terrain, dict(RETAIL_OBJECTS_TO_TRAV))
    assert trav[0, 1] == 2 and trav[2, 1] == 2 and trav[2, 2] == 2  # floor -> go
    # wall, column, fixture, product and person are all things to walk around
    blocked = trav[[0, 1, 1, 1, 2], [2, 0, 1, 2, 0]]
    assert set(np.unique(blocked).tolist()) == {0}
    assert trav[0, 0] == 255  # void is ignored, not walkable and not an obstacle


@pytest.mark.parametrize("cid,expected", [(1, 2), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)])
def test_every_class_has_a_verdict(cid, expected):
    """A class missing from the table would come back 255 and silently vanish from the map."""
    out = terrain_to_traversability(
        np.full((2, 2), cid, np.uint8), dict(RETAIL_OBJECTS_TO_TRAV)
    )
    assert (out == expected).all()

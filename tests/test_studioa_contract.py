import numpy as np
import pytest

from syncai_hydranet.data.studioa_contract import (
    ENTITY_IDS,
    VIEWS,
    contract,
    entity_views,
    migrate_mask,
)


def test_site_migration_does_not_recover_distinctions_the_source_lost():
    raw = np.arange(11, dtype=np.uint8)[None]
    mapped, report = migrate_mask(raw, "site30k_native")
    assert mapped.tolist() == [
        [
            255,
            ENTITY_IDS["floor"],
            255,
            ENTITY_IDS["column"],
            ENTITY_IDS["display_table"],
            255,
            ENTITY_IDS["person"],
            ENTITY_IDS["laptop"],
            ENTITY_IDS["tablet"],
            ENTITY_IDS["phone"],
            ENTITY_IDS["boxed_stock"],
        ]
    ]
    assert report["requires_relabel"]["wall"]["pixels"] == 1
    assert report["requires_relabel"]["shelf"]["pixels"] == 1
    assert report["human_reviewed"] is False


@pytest.mark.parametrize(
    "scheme,person_id", [("retail_objects_native", 6), ("retail_surfaces_native", 5)]
)
def test_coarse_masks_preserve_person_without_inventing_fixture_subtypes(scheme, person_id):
    raw = np.array([[0, 1, 2, 3, 4, person_id, 255]], dtype=np.uint8)
    mapped, _ = migrate_mask(raw, scheme)
    assert mapped.tolist() == [
        [255, ENTITY_IDS["floor"], 255, ENTITY_IDS["column"], 255, ENTITY_IDS["person"], 255]
    ]


@pytest.mark.parametrize(
    "raw",
    [
        np.array([[42]]),
        np.array([[-1]]),
        np.zeros((3, 3, 3), dtype=np.uint8),
        np.array([[1.0]]),
    ],
)
def test_invalid_or_unrecognised_masks_fail_instead_of_silently_ignoring(raw):
    with pytest.raises(ValueError):
        migrate_mask(raw, "site30k_native")


def test_views_preserve_one_glass_door_and_require_evidence_for_checkout():
    assert entity_views(
        {"entity": "door", "glazed": True, "materials": ["glass", "metal"]}
    ) == {"door", "glass_door"}
    assert entity_views({"entity": "door"}) == {"door"}
    assert entity_views({"entity": "counter"}) == set()
    assert entity_views(
        {"entity": "counter", "role": "checkout", "role_evidence": "store plan"}
    ) == {"checkout_counter"}
    with pytest.raises(ValueError, match="role_evidence"):
        entity_views({"entity": "counter", "role": "checkout", "role_evidence": None})
    with pytest.raises(ValueError, match="glazed"):
        entity_views({"entity": "laptop", "glazed": True})
    with pytest.raises(ValueError, match="glass material"):
        entity_views({"entity": "door", "glazed": True})
    with pytest.raises(ValueError, match="unknown entity"):
        entity_views({"entity": "void"})


def test_all_requested_views_plus_floor_and_person_are_representable():
    assert set(VIEWS) == {
        "laptop",
        "phone",
        "tablet",
        "boxed_stock",
        "cardboard_box",
        "speaker",
        "poster",
        "fire_equipment",
        "chair",
        "wall",
        "ceiling",
        "door",
        "column",
        "glass",
        "glass_door",
        "display_cabinet",
        "display_table",
        "checkout_counter",
        "floor",
        "person",
    }
    assert contract()["ignore_id"] == 255

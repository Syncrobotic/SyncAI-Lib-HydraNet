"""The object taxonomy exists to merge two classes and split one. Pin both.

Every property here is invisible at training time, which is why it is a test rather than
a comment. Undo the merge and the run still trains, still converges, and reports two
fixture classes that each see half the data -- which is the state this taxonomy was
built to leave. Undo the split and `column` becomes an output channel nothing can fill,
reporting IoU 0.000 after sixty epochs with no error anywhere.

The sibling file tests/test_retail_scheme.py pins the *other* retail taxonomy to the
indoor one. The two sets of assertions are deliberately incompatible: that is the reason
this is a second scheme and not two more ids on the first.

pytest tests/test_retail_objects_scheme.py -v
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from syncai_hydranet.config import load_config
from syncai_hydranet.data.coco_subsets import (
    COCO_NAMES,
    RETAIL_BELONGINGS,
    RETAIL_OBJECT_GROUP,
    retail_box_label,
    retail_group,
)
from syncai_hydranet.data.label_maps import SCHEMES, get_scheme
from syncai_hydranet.data.label_maps_indoor import ADE20K_ID_TO_INDOOR, INDOOR_TERRAIN
from syncai_hydranet.data.label_maps_retail import RETAIL_TERRAIN
from syncai_hydranet.data.label_maps_retail_objects import (
    ADE20K_ID_TO_RETAIL_OBJECTS,
    RETAIL_ID_TO_OBJECTS,
    RETAIL_OBJECTS,
    RETAIL_OBJECTS_NATIVE_ID,
    RETAIL_OBJECTS_TO_TRAV,
    columns_from_migration_only,
    unsourced_classes,
)
from syncai_hydranet.utils.visualize import (
    TERRAIN_COLORS_INDOOR,
    TERRAIN_COLORS_RETAIL,
    TERRAIN_COLORS_RETAIL_OBJECTS,
    terrain_palette,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "hydranet_retail_objects.yaml"


# --------------------------------------------------------------------- the taxonomy


def test_the_taxonomy_is_the_six_classes_asked_for_plus_void():
    assert list(RETAIL_OBJECTS) == [
        "void",
        "floor",
        "wall",
        "column",
        "fixture",
        "product",
        "person",
    ]
    assert list(RETAIL_OBJECTS.values()) == list(range(7))


def test_unlabelled_background_is_ignored():
    """Third taxonomy, same trap: tools export unlabelled background as 0 and the loss
    ignores 255 alone, so an identity map trains `void` over everything nobody drew."""
    assert RETAIL_OBJECTS_NATIVE_ID[0] == 255
    assert all(RETAIL_OBJECTS_NATIVE_ID[v] == v for v in RETAIL_OBJECTS.values() if v)


def test_it_is_not_the_indoor_or_retail_taxonomy_renamed():
    assert len(RETAIL_OBJECTS) < len(INDOOR_TERRAIN) < len(RETAIL_TERRAIN)


def test_every_class_has_a_traversability_reading():
    """The head is not trained here, but SegFolderDataset still hands `scheme.trav` to
    terrain_to_traversability, which falls back to the OFF-ROAD policy on a falsy map --
    where id 2 is `grass`. A missing table would not disable anything, it would supply
    the wrong one."""
    assert set(RETAIL_OBJECTS_TO_TRAV) == set(RETAIL_OBJECTS.values())
    assert RETAIL_OBJECTS_TO_TRAV[RETAIL_OBJECTS["floor"]] == 2  # go
    assert RETAIL_OBJECTS_TO_TRAV[0] == 255  # void -> ignore


# ------------------------------------------------------------------------ the merge


def test_both_fixture_classes_migrate_to_one():
    """The finding this taxonomy is built on: an Apple-store frame labels the podium
    `obstacle_furniture` and the shelving three metres away `display_fixture`."""
    assert (
        RETAIL_ID_TO_OBJECTS[RETAIL_TERRAIN["obstacle_furniture"]]
        == RETAIL_ID_TO_OBJECTS[RETAIL_TERRAIN["display_fixture"]]
        == RETAIL_OBJECTS["fixture"]
    )


def test_ade20k_tables_are_fixtures_here_and_are_not_under_the_robot_taxonomy():
    """The exact reversal. label_maps_retail.py withholds ADE20K's `table` (16) from
    display_fixture so a dining table does not teach the model it is shop furniture --
    correct there, where a competing `obstacle_furniture` class takes it. Here there is
    no competing class and the dining table's shape is the wanted prior.

    tests/test_retail_scheme.py asserts the opposite about the same ADE20K id. Both are
    right, which is the argument for two taxonomies rather than one.
    """
    assert ADE20K_ID_TO_RETAIL_OBJECTS[16] == RETAIL_OBJECTS["fixture"]
    assert ADE20K_ID_TO_INDOOR[16] == INDOOR_TERRAIN["obstacle_furniture"]


def test_seating_is_not_a_fixture():
    """A shop has chairs and stools and none of them is what the question is about.
    They fall through to ignore rather than being forced into the nearest class."""
    for ade_id in (20, 24, 31, 32, 76, 111):  # chair, sofa, armchair, seat, swivel, stool
        assert ade_id not in ADE20K_ID_TO_RETAIL_OBJECTS


# ------------------------------------------------------------------------ the split


def test_column_is_its_own_class_and_takes_ade20k_43():
    assert ADE20K_ID_TO_RETAIL_OBJECTS[43] == RETAIL_OBJECTS["column"]
    assert ADE20K_ID_TO_RETAIL_OBJECTS[43] != RETAIL_OBJECTS["wall"]
    assert ADE20K_ID_TO_INDOOR[43] == INDOOR_TERRAIN["wall"], "the id this splits from"


def test_the_migration_cannot_produce_a_single_column_pixel():
    """The one thing that does not survive reading the old masks under the new
    taxonomy, and the reason it is stated three times in three places: those masks put
    columns inside `wall`, so there is no signal left to separate them."""
    assert RETAIL_OBJECTS["column"] not in set(RETAIL_ID_TO_OBJECTS.values())


def test_a_run_sourcing_columns_only_from_migration_is_told_so():
    assert columns_from_migration_only("retail_objects_migrated") is not None
    assert (
        columns_from_migration_only("retail_objects_migrated", "ade20k_retail_objects") is None
    )
    assert columns_from_migration_only("ade20k_retail_objects") is None


def test_get_scheme_warns_on_the_migration():
    """Same shape as the rellis warning, and for the same reason: what is lost is lost
    silently, so the warning fires where the scheme is used."""
    with pytest.warns(UserWarning, match="column"):
        get_scheme("retail_objects_migrated")


def test_product_has_no_public_source():
    """If this ever fails, someone found a dataset that labels merchandise -- which is
    good news and makes several comments in label_maps_retail_objects.py wrong."""
    assert unsourced_classes(ADE20K_ID_TO_RETAIL_OBJECTS) == ("product",)
    assert unsourced_classes(ADE20K_ID_TO_RETAIL_OBJECTS, RETAIL_ID_TO_OBJECTS) == ("product",)


def test_no_ade20k_id_maps_to_void():
    assert 0 not in set(ADE20K_ID_TO_RETAIL_OBJECTS.values())


def test_every_ade20k_id_used_is_one_the_indoor_table_verified():
    """The indoor table's ids were checked against objectInfo150.txt. Anything new here
    has not been, and a wrong id relabels a whole class without an error."""
    assert set(ADE20K_ID_TO_RETAIL_OBJECTS) <= set(ADE20K_ID_TO_INDOOR)


# ----------------------------------------------------------------------- registered


@pytest.mark.parametrize(
    "name", ["ade20k_retail_objects", "retail_objects_native", "retail_objects_migrated"]
)
def test_schemes_are_registered_and_sized(name):
    assert SCHEMES[name].num_classes == len(RETAIL_OBJECTS) == 7
    assert SCHEMES[name].classes[3] == "column"
    assert SCHEMES[name].classes[5] == "product"


# ------------------------------------------------------------------------- palette


def test_the_object_taxonomy_gets_its_own_palette():
    palette = terrain_palette(list(RETAIL_OBJECTS))
    assert np.array_equal(palette, TERRAIN_COLORS_RETAIL_OBJECTS)
    assert len(palette) == 7


def test_it_is_not_mistaken_for_indoor_or_retail():
    """It shares `wall` and `person` with both. Selecting on a marker either of them
    also owns would hand it the wrong palette and, at 12 or 13 entries against 7, would
    not even fail loudly -- it would simply draw `column` in `wet_slippery`'s grey."""
    palette = terrain_palette(list(RETAIL_OBJECTS))
    assert not np.array_equal(palette, TERRAIN_COLORS_INDOOR[:7])
    assert not np.array_equal(palette, TERRAIN_COLORS_RETAIL[:7])


def test_column_and_wall_are_visually_distinct():
    """The two classes this taxonomy separates must be separable in a rendered frame,
    or a review of the split cannot be done by looking at one."""
    wall = TERRAIN_COLORS_RETAIL_OBJECTS[RETAIL_OBJECTS["wall"]]
    column = TERRAIN_COLORS_RETAIL_OBJECTS[RETAIL_OBJECTS["column"]]
    assert int(np.abs(wall.astype(int) - column.astype(int)).sum()) > 60


def test_colours_within_the_palette_are_distinct():
    assert len({tuple(c) for c in TERRAIN_COLORS_RETAIL_OBJECTS}) == len(
        TERRAIN_COLORS_RETAIL_OBJECTS
    )


# -------------------------------------------------------------------------- config


def test_config_head_width_matches_the_taxonomy():
    cfg = load_config(CONFIG)
    assert cfg["model"]["heads"]["terrain"]["num_classes"] == len(RETAIL_OBJECTS)
    assert cfg["data"]["terrain_classes"] == list(RETAIL_OBJECTS)


def test_the_config_has_no_traversability_head():
    """It is a lookup on terrain, not a second signal: the 60-epoch retail run measured
    head_disagreement at 0.0091 because one head is a deterministic function of the
    other. Dropping it is the point of this config, and `traversability: null` in YAML
    is the only thing that expresses it -- merge_config merges, so without the deletion
    the base's head survives every attempt to override it away."""
    cfg = load_config(CONFIG)
    assert "traversability" not in cfg["model"]["heads"]
    assert set(cfg["model"]["heads"]) == {"terrain", "detection"}
    assert cfg["train"]["primary_metric"] == "terrain_mIoU"


def test_the_null_head_is_written_in_the_yaml_not_merely_absent():
    """The base declares three heads. If this config ever stops saying `null` -- because
    someone 'tidied' it -- the head silently returns and nothing else in this file
    fails: the assertions above would still pass on a config that inherited it back.
    """
    raw = yaml.safe_load(CONFIG.read_text())
    assert raw["model"]["heads"]["traversability"] is None


def test_a_null_head_is_removed_everywhere_not_just_at_build_time():
    """Deleted before validation, so the schema, unsupervised_heads, the exporter and
    meta.json all agree the head does not exist. A model-only skip would record a
    lineage claiming a head the checkpoint does not have."""
    from syncai_hydranet.config_schema import unsupervised_heads

    cfg = load_config(CONFIG)
    assert "traversability" not in unsupervised_heads(cfg)


def test_no_dataset_declares_a_head_the_model_does_not_have():
    cfg = load_config(CONFIG)
    heads = set(cfg["model"]["heads"])
    for ds in cfg["data"]["datasets"]:
        assert set(ds["supervises"]) <= heads, ds["name"]


# ---------------------------------------------------------- reading COCO as retail


def test_the_retail_vocabulary_only_renames_classes_the_head_has():
    assert set(RETAIL_OBJECT_GROUP) <= set(COCO_NAMES)
    assert set(RETAIL_OBJECT_GROUP.values()) <= {"product", "fixture", "person"}


def test_book_is_read_as_a_product():
    """The audit's strongest single signal: 1,683 `book` over 40 frames of an Apple
    store, which is boxed stock and not literature."""
    assert retail_group(COCO_NAMES.index("book")) == "product"
    assert retail_box_label(COCO_NAMES.index("book")) == "product/book"


def test_a_customers_bag_is_not_merchandise():
    for name in RETAIL_BELONGINGS:
        assert name not in RETAIL_OBJECT_GROUP
        assert retail_group(COCO_NAMES.index(name)) is None
        assert retail_box_label(COCO_NAMES.index(name)) == name


def test_person_keeps_its_own_name_rather_than_being_prefixed():
    assert retail_box_label(COCO_NAMES.index("person")) == "person"


def test_an_out_of_range_label_degrades_to_its_index():
    assert retail_box_label(999) == "999"
    assert retail_group(999) is None

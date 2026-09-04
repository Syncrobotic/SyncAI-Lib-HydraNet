"""A colour is a claim about which class a pixel is.

There used to be one TERRAIN_COLORS, holding the off-road taxonomy, and every indoor
and retail run was rendered through it. Indices line up regardless of meaning, so
nothing failed: `wall` came out in sky's blue, `glass` in water's blue, `person` in the
orange belonging to `stairs_log`. The one artefact meant to catch a model calling a
floor a wall was itself mislabelling the classes.

The second half is `overlay`, which clamped an out-of-range class into the last palette
entry. Retail has thirteen terrain classes and the palette had twelve, so every
`display_fixture` in every rendered frame was drawn as `person` -- in the class
`git show b7457c2:docs/RETAIL.md` called the reason the retail taxonomy exists at all.

pytest tests/test_palettes.py -v
"""

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from syncai_hydranet.utils.visualize import (
    TERRAIN_COLORS_INDOOR,
    TERRAIN_COLORS_RETAIL,
    TERRAIN_COLORS_RETAIL_OBJECTS,
    TERRAIN_COLORS_RETAIL_SURFACES,
    overlay,
    terrain_palette,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def classes_of(name):
    raw = yaml.safe_load((CONFIGS / name).read_text())
    return raw["data"]["terrain_classes"]


# ------------------------------------------------------- one palette per taxonomy


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ("hydranet_indoor.yaml", TERRAIN_COLORS_INDOOR),
        ("hydranet_retail.yaml", TERRAIN_COLORS_RETAIL),
        ("hydranet_retail_objects.yaml", TERRAIN_COLORS_RETAIL_OBJECTS),
        ("hydranet_retail_surfaces.yaml", TERRAIN_COLORS_RETAIL_SURFACES),
    ],
)
def test_each_config_gets_its_own_palette(config, expected):
    assert np.array_equal(terrain_palette(classes_of(config)), expected)


def test_retail_is_not_mistaken_for_indoor():
    """Retail is the indoor twelve plus one, so it contains every indoor class name.
    Selecting on the first marker that matches would hand it the indoor palette and
    put it one colour short -- which is the shape of the original bug, not a new one.
    """
    retail = terrain_palette(classes_of("hydranet_retail.yaml"))
    assert len(retail) == 13
    assert not np.array_equal(retail, TERRAIN_COLORS_INDOOR)


def test_selection_is_by_class_name_and_not_by_class_count():
    """Why the palette is chosen from a marker class rather than from `len(classes)`.

    The counter-example used to be `TERRAIN_COLORS_OFFROAD`, which was 12 long like
    `TERRAIN_COLORS_INDOOR` and entirely different; it went with the off-road taxonomy
    on 2026-09-04, and no two palettes share a length today. That does not make count
    selection safe -- it makes the collision one taxonomy away -- so the property is
    asserted directly instead of through a pair that happened to demonstrate it.
    """
    indoor = classes_of("hydranet_indoor.yaml")
    # A taxonomy of exactly indoor's size, named differently: a count-based selector would
    # hand it indoor's colours. This refuses instead, which is the property.
    disguised = tuple(f"class_{i}" for i in range(len(indoor)))
    assert len(disguised) == len(indoor)
    with pytest.raises(ValueError, match="no terrain palette"):
        terrain_palette(disguised)


def test_every_palette_covers_its_config():
    for name in ("hydranet_indoor.yaml", "hydranet_retail.yaml"):
        classes = classes_of(name)
        assert len(terrain_palette(classes)) >= len(classes), name


def test_colours_within_a_palette_are_distinct():
    """Two classes sharing a colour is the same failure as clamping, arrived at by
    typing rather than by indexing."""
    for palette in (TERRAIN_COLORS_INDOOR, TERRAIN_COLORS_RETAIL):
        assert len({tuple(c) for c in palette}) == len(palette)


def test_an_unknown_taxonomy_is_refused():
    with pytest.raises(ValueError, match="no terrain palette"):
        terrain_palette(["void", "lava", "quicksand"])


def test_a_taxonomy_that_outgrew_its_palette_is_refused():
    """Adding a class to a config and not to the palette must fail at selection, not
    silently at draw time or -- worse -- not at all."""
    with pytest.raises(ValueError, match="terrain classes but the matched palette"):
        terrain_palette([*classes_of("hydranet_retail.yaml"), "shopping_trolley"])


# ------------------------------------------------- the unnamed-taxonomy fallback


def test_unnamed_classes_get_generated_colours():
    """A config declaring no terrain_classes is not an error: with nothing named, no
    colour can be wrong. Refusing here would push callers into passing some other
    taxonomy's palette, which is the thing being prevented."""
    palette = terrain_palette(None, n_classes=5)
    assert len(palette) == 5
    assert tuple(palette[0]) == (0, 0, 0), "index 0 is void everywhere else too"
    assert len({tuple(c) for c in palette}) == 5


def test_unnamed_and_uncounted_is_still_an_error():
    with pytest.raises(ValueError, match="either class names or n_classes"):
        terrain_palette(None)


# ---------------------------------------------------------------- no clamping


def test_overlay_refuses_a_class_the_palette_does_not_cover():
    """The regression. Previously np.clip drew class 12 in class 11's colour, so a
    display fixture and a person were the same pixel value in every rendered frame.
    """
    base = Image.new("RGB", (2, 1), (0, 0, 0))
    mask = np.array([[11, 12]], dtype=np.int64)
    with pytest.raises(ValueError, match="palette has 12 colours"):
        overlay(base, mask, TERRAIN_COLORS_INDOOR)


def test_overlay_draws_the_thirteenth_class_when_given_the_right_palette():
    base = Image.new("RGB", (2, 1), (0, 0, 0))
    mask = np.array([[11, 12]], dtype=np.int64)
    out = np.asarray(overlay(base, mask, TERRAIN_COLORS_RETAIL, alpha=1.0))
    assert not np.array_equal(out[0, 0], out[0, 1])


def test_overlay_handles_an_empty_mask():
    base = Image.new("RGB", (0, 0), (0, 0, 0))
    mask = np.zeros((0, 0), dtype=np.int64)
    assert np.asarray(overlay(base, mask, TERRAIN_COLORS_INDOOR)).size == 0


# ------------------------------------------- the annotation tool draws the same


def test_cvat_labels_match_the_render_colours():
    """annotation.label_spec promises the colours the overlays use, so a mask drawn in
    CVAT and a prediction rendered next to it are comparable without a translation
    step. It used to keep its own EXTRA_COLORS for anything past the twelfth class,
    which matched nothing that ever drew a prediction."""
    from syncai_hydranet.cli.annotation import SCHEMES, label_spec

    for scheme_name, scheme in SCHEMES.items():
        palette = terrain_palette(list(scheme.terrain))
        for entry in label_spec(scheme):
            r, g, b = (int(c) for c in palette[entry["terrain_id"]])
            assert entry["color"] == f"#{r:02x}{g:02x}{b:02x}", f"{scheme_name}/{entry['name']}"


# ------------------------------------------- surfaces, which has no word of its own


def test_surfaces_is_not_mistaken_for_objects():
    """`retail_surfaces` is `retail_objects` minus `product`, so every name it carries
    appears there too and it has no distinctive marker. It is matched on `fixture` placed
    after `product`, and this is the test that keeps those two lines in that order: the
    object palette applied to surfaces classes is off by one from `person` onwards, and a
    wrong colour is indistinguishable from a wrong prediction.
    """
    surfaces = terrain_palette(classes_of("hydranet_retail_surfaces.yaml"))
    objects = terrain_palette(classes_of("hydranet_retail_objects.yaml"))
    assert np.array_equal(surfaces, TERRAIN_COLORS_RETAIL_SURFACES)
    assert np.array_equal(objects, TERRAIN_COLORS_RETAIL_OBJECTS)
    assert len(surfaces) == 6 and len(objects) == 7


def test_a_class_keeps_its_colour_across_the_split_that_removed_product():
    """The two taxonomies are read side by side in the comparison the split was made to
    settle. A class that changed colour between them would look like a class that changed
    behaviour, so the shared ids are copied rather than re-picked -- and `person`, which
    moves from id 6 to id 5, keeps its colour across the move.
    """
    surfaces = TERRAIN_COLORS_RETAIL_SURFACES
    objects = TERRAIN_COLORS_RETAIL_OBJECTS
    assert np.array_equal(surfaces[:5], objects[:5])
    assert np.array_equal(surfaces[5], objects[6])


def test_a_seed_replicate_gets_the_same_palette_as_its_parent():
    """The seed configs exist to measure variance between runs. A palette that differed
    between replicates would put a rendering difference inside that measurement.

    Resolved through `load_config` rather than raw YAML, because a seed replicate inherits
    its classes through `_base_` and declares none of its own -- which is exactly the file
    shape that reaches the trainer, so it is the one worth checking.
    """
    from syncai_hydranet.config import load_config

    parent = terrain_palette(classes_of("hydranet_retail_surfaces.yaml"))
    for seed in ("hydranet_retail_surfaces_seed7.yaml", "hydranet_retail_surfaces_seed13.yaml"):
        cfg = load_config(str(CONFIGS / seed), [])
        assert np.array_equal(terrain_palette(cfg["data"]["terrain_classes"]), parent), seed

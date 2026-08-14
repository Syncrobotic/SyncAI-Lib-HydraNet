"""A colour is a claim about which class a pixel is.

There used to be one TERRAIN_COLORS, holding the off-road taxonomy, and every indoor
and retail run was rendered through it. Indices line up regardless of meaning, so
nothing failed: `wall` came out in sky's blue, `glass` in water's blue, `person` in the
orange belonging to `stairs_log`. The one artefact meant to catch a model calling a
floor a wall was itself mislabelling the classes.

The second half is `overlay`, which clamped an out-of-range class into the last palette
entry. Retail has thirteen terrain classes and the palette had twelve, so every
`display_fixture` in every rendered frame was drawn as `person` -- in the class
docs/RETAIL_SCOPE.md calls the reason the retail taxonomy exists at all.

pytest tests/test_palettes.py -v
"""

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from syncai_hydranet.utils.visualize import (
    TERRAIN_COLORS_INDOOR,
    TERRAIN_COLORS_OFFROAD,
    TERRAIN_COLORS_RETAIL,
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
        ("hydranet_regnet800mf.yaml", TERRAIN_COLORS_OFFROAD),
        ("hydranet_indoor.yaml", TERRAIN_COLORS_INDOOR),
        ("hydranet_retail.yaml", TERRAIN_COLORS_RETAIL),
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


def test_indoor_and_offroad_are_the_same_length_and_different_colours():
    """Why selection cannot be by class count."""
    assert len(TERRAIN_COLORS_INDOOR) == len(TERRAIN_COLORS_OFFROAD) == 12
    assert not np.array_equal(TERRAIN_COLORS_INDOOR, TERRAIN_COLORS_OFFROAD)


def test_every_palette_covers_its_config():
    for name in ("hydranet_regnet800mf.yaml", "hydranet_indoor.yaml", "hydranet_retail.yaml"):
        classes = classes_of(name)
        assert len(terrain_palette(classes)) >= len(classes), name


def test_colours_within_a_palette_are_distinct():
    """Two classes sharing a colour is the same failure as clamping, arrived at by
    typing rather than by indexing."""
    for palette in (TERRAIN_COLORS_OFFROAD, TERRAIN_COLORS_INDOOR, TERRAIN_COLORS_RETAIL):
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

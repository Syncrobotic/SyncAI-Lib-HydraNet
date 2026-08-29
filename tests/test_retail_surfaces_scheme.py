"""Taking `product` out of segmentation must not move a single pixel of free space.

The object taxonomy carries `product` as a terrain class and the measurement says it
should not: merchandise is a median 17.7 x 28.7 px at the model's input scale, which at
stride 8 is two or three feature cells, and a dense head cannot draw an edge with too few
cells to put one in. FCOS needs no cells.

But these configs **drop the traversability head and derive free space from terrain**
(`cli/scene.py`), so every terrain class is load-bearing for the floor polygon and
removing one is not a free simplification. The test that decides it is not "does the
detection head resolve this class better" but:

    what does free space become where this class used to be?

    product -> falls back to `fixture`, which merchandise sits on. Blocked either way.
    person  -> would fall back to `floor`, which is `go`: a person-shaped hole of
               walkable floor, in metres, with no error anywhere.

So `product` may leave and `person` may not, and the asymmetry is a property of what each
class stands on rather than of how well it segments. These tests pin that, the
renumbering it causes, and the derivation that keeps the two taxonomies from drifting.

pytest tests/test_retail_surfaces_scheme.py -v
"""

from __future__ import annotations

import pytest

from syncai_hydranet.data.label_maps import get_scheme
from syncai_hydranet.data.label_maps_retail_objects import (
    ADE20K_ID_TO_RETAIL_SURFACES,
    RETAIL_OBJECTS,
    RETAIL_OBJECTS_ID_TO_SURFACES,
    RETAIL_OBJECTS_TO_TRAV,
    RETAIL_SURFACES,
    RETAIL_SURFACES_TO_TRAV,
)

SCHEMES = ("ade20k_retail_surfaces", "retail_surfaces_native", "retail_surfaces_from_objects")


def test_free_space_is_bit_for_bit_unchanged_by_dropping_product():
    """The invariant the whole change rests on, checked class by class rather than argued.

    If this ever fails, some class's go/blocked verdict moved when it was renumbered, and
    the floor polygon in every rendered panel moved with it -- silently, because a wrong
    floor polygon is drawn exactly as confidently as a right one.
    """
    for name, old in RETAIL_OBJECTS.items():
        if name == "void":
            continue
        new = RETAIL_OBJECTS_ID_TO_SURFACES[old]
        assert RETAIL_OBJECTS_TO_TRAV[old] == RETAIL_SURFACES_TO_TRAV[new], (
            f"{name} changes traversability when mapped into the surfaces taxonomy"
        )


def test_product_becomes_fixture_and_not_floor_or_void():
    """`fixture` is the only correct merge target: merchandise sits on the thing that
    holds it, so the pixels are blocked either way.

    `floor` would open free space where a stack of boxes is. 255 would leave 12.88% of
    `datasets/retail_objects_batch02` unsupervised -- a hole rather than a label, and the
    trunk would learn nothing there instead of learning the right thing.
    """
    assert (
        RETAIL_OBJECTS_ID_TO_SURFACES[RETAIL_OBJECTS["product"]] == RETAIL_SURFACES["fixture"]
    )


def test_person_survives_and_is_renumbered_rather_than_taken_verbatim():
    """`person` is 6 under the objects taxonomy and 5 here, because `product` was removed
    from the middle of the numbering.

    Reading the existing site masks verbatim would therefore label every shopper
    `fixture` -- the class that happens to occupy id 5 now. That is not a hypothetical:
    the masks on disk carry object ids, and this scheme is what reads them.
    """
    assert RETAIL_OBJECTS["person"] == 6
    assert RETAIL_SURFACES["person"] == 5
    assert RETAIL_OBJECTS_ID_TO_SURFACES[6] == RETAIL_SURFACES["person"]
    assert RETAIL_OBJECTS_ID_TO_SURFACES[6] != RETAIL_SURFACES["fixture"]


def test_the_surfaces_taxonomy_is_derived_so_the_two_cannot_drift():
    """Adding a class to `RETAIL_OBJECTS` and forgetting it here would otherwise produce
    a silently narrower scheme, which trains a head with a missing channel and reports
    nothing about it."""
    assert set(RETAIL_OBJECTS) - set(RETAIL_SURFACES) == {"product"}
    assert list(RETAIL_SURFACES) == [n for n in RETAIL_OBJECTS if n != "product"]
    assert list(RETAIL_SURFACES.values()) == list(range(len(RETAIL_SURFACES)))


def test_unlabelled_background_is_ignored_and_not_a_trainable_class():
    """Annotation tools export unlabelled as 0, and 0 is `void`. An identity table would
    make "nobody labelled this" a class the loss optimises -- the trap
    `RETAIL_OBJECTS_NATIVE_ID` documents, avoided a second time here."""
    assert RETAIL_OBJECTS_ID_TO_SURFACES[0] == 255


def test_no_ade20k_id_maps_to_a_class_this_taxonomy_does_not_have():
    """ADE20K has no merchandise, so nothing should have been targeting `product`
    anyway -- but the surfaces map is derived from the objects map, and a derivation is
    where an out-of-range id would enter without anyone writing it."""
    assert set(ADE20K_ID_TO_RETAIL_SURFACES.values()) <= set(RETAIL_SURFACES.values())
    assert max(ADE20K_ID_TO_RETAIL_SURFACES.values()) < len(RETAIL_SURFACES)


@pytest.mark.parametrize("name", SCHEMES)
def test_every_surfaces_scheme_declares_the_same_six_classes(name):
    """Three schemes, one taxonomy. A config that swaps its source dataset must not also
    swap its output space, and nothing else checks that they agree."""
    scheme = get_scheme(name)
    assert scheme.classes == tuple(RETAIL_SURFACES)
    assert len(scheme.classes) == 6


@pytest.mark.parametrize("name", SCHEMES)
def test_every_surfaces_scheme_ships_the_same_traversability_table(name):
    """`SegFolderDataset` hands `scheme.trav` to `terrain_to_traversability`, which falls
    back to the **off-road** policy on a falsy map -- where id 2 means `grass`. A scheme
    that shipped no table would not disable the derived free space, it would supervise it
    from RUGD's rules in a shop."""
    assert get_scheme(name).trav == RETAIL_SURFACES_TO_TRAV
    assert get_scheme(name).trav[RETAIL_SURFACES["floor"]] == 2
    assert get_scheme(name).trav[RETAIL_SURFACES["person"]] == 0

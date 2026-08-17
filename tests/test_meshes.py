"""Procedural meshes: the dimensions have to be the dimensions.

This module exists so a scene can be drawn from measurements, which makes one property
worth more than all the others: **a mesh asked for 1.70 m is 1.70 m tall.** It was not,
when first written -- the 7.5-head canon puts the crown at 1.013 of standing height, so
`human(1.70)` came back 1.722 m. Small, plausible, and wrong in exactly the quantity a
renderer would then print next to it.

The other tests here are about the shapes that come from a real footprint rather than a
made-up one. A gondola run with an end cap is L-shaped and a shelf bay is U-shaped, and a
triangle fan over either fills in the notch -- quietly, as a solid where the aisle is.

pytest tests/test_meshes.py -v
"""

import math

import numpy as np
import pytest

from syncai_hydranet.geometry.meshes import (
    Placement,
    box,
    cabinet,
    column,
    extrude,
    ground_disc,
    human,
    place,
    table,
    to_obj,
    wall,
)


def _bbox(mesh):
    v = mesh[0]
    return v.min(axis=0), v.max(axis=0)


# --- the property the module is for ---------------------------------------


@pytest.mark.parametrize("h", [1.45, 1.62, 1.70, 1.78, 1.95])
def test_a_human_is_exactly_the_height_it_was_asked_for(h):
    """The regression. The canon's landmarks overshoot by 1.3%, which reads as a correct
    figure and prints as a wrong measurement."""
    lo, hi = _bbox(human(h))
    assert hi[1] - lo[1] == pytest.approx(h, abs=1e-9)


@pytest.mark.parametrize("h", [1.45, 1.70, 1.95])
def test_a_human_stands_on_the_floor(h):
    """Feet at y=0, so placing one is a translation in x and z and nothing else."""
    lo, _ = _bbox(human(h))
    assert lo[1] == pytest.approx(0.0, abs=1e-6)


def test_shoulder_span_is_anthropometric():
    """0.44 m across at 1.70 m. Not decoration: a figure with narrow shoulders reads as a
    child, and these are drawn beside fixtures whose width is measured."""
    v = human(1.70)[0]
    assert 0.42 < v[:, 0].max() - v[:, 0].min() < 0.63  # joints 0.44, plus arm thickness


def test_a_human_has_a_facing():
    """The feet reach forward, so `heading_rad` renders as a visible change. A figure
    symmetric in z would make a measured heading look like no measurement at all."""
    v = human(1.70)[0]
    assert v[:, 2].max() > 0.10


def test_heights_scale_every_part_not_just_the_total():
    small, large = human(1.40)[0], human(2.00)[0]
    assert np.ptp(large[:, 0]) > np.ptp(small[:, 0])  # wider, not just taller


# --- footprints that are not convex ---------------------------------------

L_SHAPE = [[0, 0], [2, 0], [2, 0.6], [0.6, 0.6], [0.6, 2], [0, 2]]


def _cap_area(mesh, n_footprint):
    """Total area of the floor cap, which is what a fan would get wrong."""
    v, f = mesh
    caps = [t for t in f if all(v[i][1] < 1e-9 for i in t)]
    assert len(caps) == n_footprint - 2, "a simple polygon triangulates to n-2 triangles"
    total = 0.0
    for a, b, c in caps:
        pa, pb, pc = v[a], v[b], v[c]
        total += 0.5 * abs(
            (pb[0] - pa[0]) * (pc[2] - pa[2]) - (pb[2] - pa[2]) * (pc[0] - pa[0])
        )
    return total


def test_an_l_shaped_footprint_does_not_get_its_notch_filled_in():
    """The ear-clipping case, and the one that matters in a shop: an end-capped gondola
    run is L-shaped, and a fan over it puts a solid where the aisle is."""
    mesh = extrude(L_SHAPE, 1.5)
    # 2 x 2 minus the 1.4 x 1.4 notch = 2.04
    assert _cap_area(mesh, len(L_SHAPE)) == pytest.approx(2.04, abs=1e-6)


def test_winding_does_not_have_to_be_known_by_the_caller():
    """OpenCV contours and a hand-written walk disagree about winding often enough that
    requiring one would be a bug waiting for whichever caller guessed wrong."""
    ccw = extrude(L_SHAPE, 1.5)
    cw = extrude(L_SHAPE[::-1], 1.5)
    assert _cap_area(cw, len(L_SHAPE)) == pytest.approx(_cap_area(ccw, len(L_SHAPE)))


def test_extrusion_height_is_the_height():
    lo, hi = _bbox(extrude(L_SHAPE, 2.7))
    assert hi[1] - lo[1] == pytest.approx(2.7)
    assert lo[1] == pytest.approx(0.0)


# --- the named builders ----------------------------------------------------


def test_column_round_and_square_share_their_footprint_extent():
    sq = column(0.5, 0.4, 3.0)
    rd = column(0.5, 0.4, 3.0, round_=True)
    for m in (sq, rd):
        lo, hi = _bbox(m)
        assert hi[0] - lo[0] == pytest.approx(0.5, abs=1e-6)
        assert hi[1] == pytest.approx(3.0)


def test_a_table_has_a_top_and_legs_and_stands_at_its_stated_height():
    lo, hi = _bbox(table(1.4, 0.9, 0.9))
    assert hi[1] == pytest.approx(0.9)
    assert lo[1] == pytest.approx(0.0)
    # The top slab is the widest thing at the top; the legs are inset.
    v = table(1.4, 0.9, 0.9)[0]
    top = v[v[:, 1] > 0.85]
    assert top[:, 0].max() - top[:, 0].min() == pytest.approx(1.4, abs=1e-6)


def test_a_cabinet_has_one_slab_per_shelf_plus_the_base():
    """The shelves are the reason this is not `box()`: merchandise sits on horizontal
    planes at known heights, and that is where `product` lives."""
    few, many = cabinet(1.8, 0.5, 2.0, shelves=1), cabinet(1.8, 0.5, 2.0, shelves=5)
    assert len(many[1]) > len(few[1])


def test_a_wall_follows_its_polyline():
    m = wall([[0, 0], [3, 0], [3, 2]], 2.4)
    _, hi = _bbox(m)
    assert hi[1] == pytest.approx(2.4)
    assert hi[0] == pytest.approx(3.0, abs=0.07)  # plus half the thickness
    assert hi[2] == pytest.approx(2.0, abs=0.07)


def test_ground_disc_is_flat_and_the_radius_it_says():
    v = ground_disc(0.85)[0]
    assert np.allclose(v[:, 1], 0.0)
    assert np.hypot(v[:, 0], v[:, 2]).max() == pytest.approx(0.85)


# --- placement -------------------------------------------------------------


def test_place_translates_onto_the_floor():
    moved = place(human(1.70), Placement(x_m=2.0, z_m=-3.0))
    lo, hi = _bbox(moved)
    assert (lo[0] + hi[0]) / 2 == pytest.approx(2.0, abs=0.05)
    assert lo[1] == pytest.approx(0.0, abs=1e-6)  # still on the floor


def test_a_quarter_turn_swaps_the_extents():
    m = box(2.0, 1.0, 0.4)
    turned = place(m, Placement(heading_rad=math.pi / 2))
    lo, hi = _bbox(turned)
    assert hi[0] - lo[0] == pytest.approx(0.4, abs=1e-6)
    assert hi[2] - lo[2] == pytest.approx(2.0, abs=1e-6)


def test_no_heading_means_no_rotation():
    """A round footprint gets no arrow, in `bev_page.py`'s words, because nothing measured
    one. `None` has to mean that rather than zero."""
    a = place(box(2.0, 1.0, 0.4), Placement(heading_rad=None))[0]
    b = box(2.0, 1.0, 0.4)[0]
    assert np.allclose(a, b)


# --- export ----------------------------------------------------------------


def test_obj_is_one_indexed_and_complete():
    """OBJ is 1-indexed and every 3D tool reads it, which is what keeps this module free
    of a renderer dependency."""
    v, f = box(1.0, 2.0, 3.0)
    text = to_obj((v, f), name="fixture")
    assert text.startswith("o fixture\n")
    assert text.count("\nv ") == len(v)
    assert text.count("\nf ") == len(f)
    first = next(line for line in text.splitlines() if line.startswith("f "))
    assert min(int(i) for i in first.split()[1:]) >= 1


# --- refusals --------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: human(0), "height_m"),
        (lambda: human(-1.7), "height_m"),
        (lambda: box(1, 0, 1), "height_m"),
        (lambda: extrude([[0, 0], [1, 0]], 1.0), "N>=3"),
        (lambda: extrude([[0, 0], [1, 0], [1, 1]], 0), "height_m"),
        (lambda: wall([[0, 0]], 2.4), "N>=2"),
        (lambda: ground_disc(0), "radius_m"),
        (lambda: cabinet(1, 1, 1, shelves=-1), "shelves"),
    ],
)
def test_impossible_dimensions_are_refused(call, match):
    with pytest.raises(ValueError, match=match):
        call()


# --- what a scene payload gets drawn as ------------------------------------


def test_a_person_becomes_a_person():
    """The join `bev3d` uses. Before it existed the mesh library had no consumer at all
    and every real render drew cuboids, which is indistinguishable from the outside."""
    from syncai_hydranet.geometry.meshes import for_object

    verts, _ = for_object({"name": "person", "height_m": 1.63, "width_m": 0.5})
    assert np.ptp(verts[:, 1]) == pytest.approx(1.63)
    assert np.ptp(verts[:, 0]) < 0.7  # a figure, not a 0.5 m cube


def test_a_person_with_no_measured_height_falls_back_rather_than_vanishing():
    """`bev.scene` emits height_m None when the box could not be placed against the ground
    plane. 1.70 m is the same assumption `fit_camera_from_people.py` fits pose against."""
    from syncai_hydranet.geometry.meshes import for_object

    verts, _ = for_object({"name": "person", "height_m": None, "width_m": 0.5})
    assert np.ptp(verts[:, 1]) == pytest.approx(1.70)


def test_an_implausible_person_height_is_not_believed():
    """A box whose bottom landed on a counter edge gives a 0.3 m person. Drawing that
    would put the projection error into the figure's stature, where it reads as a child."""
    from syncai_hydranet.geometry.meshes import for_object

    verts, _ = for_object({"name": "person", "height_m": 0.31, "width_m": 0.4})
    assert np.ptp(verts[:, 1]) == pytest.approx(1.70)


def test_anything_else_stays_the_extruded_footprint():
    """Adding a shape is a claim about what the thing looks like. The only one currently
    earned is that a person is person-shaped."""
    from syncai_hydranet.geometry.meshes import for_object

    verts, faces = for_object({"name": "chair", "width_m": 0.6, "height_m": 0.9})
    assert len(faces) == 12  # a box
    assert np.ptp(verts[:, 1]) == pytest.approx(0.9)


def test_an_object_with_no_extent_gets_no_mesh():
    """`width_m`/`height_m` are None when they could not be measured, and `bev.scene` says
    so rather than substituting. A renderer must be able to say so too."""
    from syncai_hydranet.geometry.meshes import for_object

    assert for_object({"name": "chair", "width_m": None, "height_m": None}) is None


def test_smooth_normals_are_unit_and_one_per_face():
    from syncai_hydranet.geometry.meshes import smooth_normals

    verts, faces = human(1.70)
    n = smooth_normals(verts, faces)
    assert n.shape == (len(faces), 3)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)


def test_smooth_normals_differ_from_flat_ones_on_a_curved_surface():
    """The whole reason they exist: with per-face geometric normals a 10-sided tube reads
    as a prism."""
    from syncai_hydranet.geometry.meshes import smooth_normals

    verts, faces = human(1.70)
    flat = np.cross(
        verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]]
    )
    flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
    assert not np.allclose(smooth_normals(verts, faces), flat, atol=1e-3)

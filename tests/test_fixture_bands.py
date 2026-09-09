"""The floor beside a fixture, not the floor nearest to it.

The zones as commissioned tile the walkable floor -- Taichung-cam04's twelve sum to
38.9 m2 against a 26.2 m2 walkable polygon, 95.6% of it claimed -- so a zone visit has
meant "in that quarter of the room". A band is the fix in kind, and these hold it to being
a distance rather than a partition.
"""

from __future__ import annotations

import numpy as np
import pytest

from syncai_hydranet.geometry.bands import Band, contact_line


def test_a_band_is_a_distance_and_not_a_partition():
    """The point this module exists for: far floor is OUTSIDE, where a tiling would have
    assigned it to whichever fixture was nearest."""
    band = Band(np.array([[0.0, 0.0]]), width_m=1.0)
    got = band.contains(np.array([[0.0, 0.5], [0.0, 1.0], [0.0, 1.01], [0.0, 8.0]]))
    assert got.tolist() == [True, True, False, False]


def test_the_band_follows_every_contact_point_not_just_one():
    band = Band(np.array([[0.0, 0.0], [10.0, 0.0]]), width_m=1.0)
    assert band.contains(np.array([[10.5, 0.0], [5.0, 0.0]])).tolist() == [True, False]


def test_an_empty_band_engages_nobody_rather_than_everybody():
    """A fixture whose mask projected to nothing must read as "no engagement measured".
    True here would silently promote the whole floor to engaged, which is the failure the
    tiling already has and this module exists to end."""
    assert Band(np.zeros((0, 2)), width_m=1.0).contains(np.array([[0.0, 0.0]])).tolist() == [
        False
    ]


def test_a_non_positive_width_is_refused():
    with pytest.raises(ValueError, match="width_m must be positive"):
        Band(np.array([[0.0, 0.0]]), width_m=0.0)


def test_contains_accepts_an_empty_query():
    assert Band(np.array([[0.0, 0.0]]), width_m=1.0).contains(np.zeros((0, 2))).tolist() == []


class _FlatCamera:
    """A camera that projects a pixel to itself, in metres, so the geometry under test is
    the contact-line extraction and not `ground_points`, which has its own tests."""

    def ground_points(self, points_px, *, above_horizon, what):
        assert above_horizon == "drop", "a mask legitimately reaches past the horizon"
        assert what, "the projection is asked to name its subject, so errors point somewhere"
        return np.asarray(points_px, dtype=float)


def test_contact_line_takes_the_lowest_pixel_of_each_column():
    """The only part of a furniture mask whose ground projection means anything. Projecting
    the whole mask puts a shelf's top shelf several metres behind the shop."""
    mask = np.zeros((10, 4), dtype=np.uint8)
    mask[2:6, 0:4] = 255  # a block four columns wide, bottom row index 5
    got = contact_line(mask, _FlatCamera(), min_component_px=1)
    assert sorted(got[:, 0].tolist()) == [0.5, 1.5, 2.5, 3.5]
    assert set(got[:, 1].tolist()) == {5.5}


def test_contact_line_drops_speckle():
    """A stray pixel projected to the floor is a fixture that is not there."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[8, 8] = 255
    assert len(contact_line(mask, _FlatCamera(), min_component_px=200)) == 0


def test_contact_line_on_an_empty_mask_is_empty_not_an_error():
    assert contact_line(np.zeros((5, 5), np.uint8), _FlatCamera()).shape == (0, 2)


def test_contact_line_keeps_two_separate_fixtures_apart():
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:6, 0:3] = 255
    mask[3:8, 8:12] = 255
    got = contact_line(mask, _FlatCamera(), min_component_px=1)
    assert sorted(set(got[:, 1].tolist())) == [5.5, 7.5], "each component keeps its own floor"

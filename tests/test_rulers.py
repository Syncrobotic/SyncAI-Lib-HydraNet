"""The project finds its own scale reference: a catalogue tile chosen by the store's
cameras' consensus, a store's counter height as the relative ruler, and a decision that
needs two witnesses."""

import math

import pytest

from syncai_bev3d import rulers
from syncai_bev3d.rulers import (
    Ruler,
    combine,
    person_ruler,
    standard_tile,
    table_ruler,
    tile_ruler,
)


def test_taichung_agrees_on_the_60cm_tile():
    tile, err = standard_tile({"c01": 0.47, "c04": 0.62, "c10": 0.66, "c11": 0.57})
    assert tile == 0.60
    assert err < rulers.TILE_CONSENSUS_MAX


def test_one_camera_agrees_with_any_tile_so_it_is_no_consensus():
    assert standard_tile({"c01": 0.47}) == (None, math.inf)


def test_a_floor_off_every_standard_size_abstains():
    tile, err = standard_tile({"a": 0.70, "b": 0.70, "c": 0.70})  # between 0.60 and 0.80
    assert tile is None and err > rulers.TILE_CONSENSUS_MAX


def test_the_tile_ruler_is_the_real_over_the_read():
    r = tile_ruler(0.47, 0.60)
    assert r is not None and abs(r.factor - 0.60 / 0.47) < 1e-9
    assert tile_ruler(None, 0.60) is None and tile_ruler(0.47, None) is None


def test_the_table_ruler_moves_an_outlier_toward_the_store():
    store = [0.78, 0.84, 0.86, 0.96]
    r = table_ruler(0.78, store)
    assert r is not None and r.factor == pytest.approx(0.85 / 0.78)
    assert table_ruler(0.78, [0.78, 0.84]) is None  # too few to take a median of


def test_the_person_ruler_is_unity_with_the_calibrations_own_uncertainty():
    r = person_ruler(
        {
            "uncertainty": {
                "scale_frac_stat_person_mad": 0.087,
                "scale_frac_sys_person_prior": 0.11,
            }
        }
    )
    assert r.factor == 1.0 and r.sigma == pytest.approx(math.hypot(0.087, 0.11))


def test_two_agreeing_rulers_apply():
    v = combine(
        [Ruler("person", 1.0, 0.14), Ruler("tile", 1.28, 0.05), Ruler("table", 1.09, 0.08)]
    )
    assert v.apply and v.factor > 1.10
    assert "tile" in v.reason and "table" in v.reason


def test_one_ruler_alone_does_not_apply():
    v = combine([Ruler("person", 1.0, 0.14), Ruler("tile", 1.28, 0.05)])
    assert not v.apply and "only one ruler" in v.reason


def test_within_ten_percent_nothing_moves():
    v = combine(
        [Ruler("person", 1.0, 0.14), Ruler("tile", 0.97, 0.05), Ruler("table", 1.05, 0.08)]
    )
    assert not v.apply and abs(math.log(v.factor)) <= math.log(1.10)


def test_two_rulers_on_opposite_sides_do_not_apply():
    v = combine([Ruler("tile", 1.30, 0.05), Ruler("table", 0.80, 0.08)])
    assert not v.apply


def test_no_ruler_keeps():
    v = combine([])
    assert v.factor == 1.0 and not v.apply

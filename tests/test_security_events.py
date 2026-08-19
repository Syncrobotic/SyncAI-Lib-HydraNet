"""The security half's output: events over tracks and the floor, not model channels.

Every threshold here is an argument and every position is in metres, and both are the
same decision as `RETAIL.md`'s refusal to make "within 5 m" a class: a number that
belongs to a store belongs in a config, and a boundary drawn in pixels dies when the
mount is knocked. These tests pin the arithmetic that makes that possible, and pin the
refusals that stop an unbuilt producer from returning an empty list that reads as
"nothing happened".

The behaviour tiers are checked in the order they can be built: track geometry first
(nothing new required), then pose (a second-stage model that does not exist yet, so the
rules are exercised on synthetic keypoints), and `fight` not at all, on purpose.

pytest tests/test_security_events.py -v
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from syncai_hydranet.analytics import events as ev
from syncai_hydranet.analytics.tracker import Track
from syncai_hydranet.geometry.ground import Camera, GroundPlane, ground_to_pixel

CAM = Camera.from_vfov(1080, 1920, 50.0)
PLANE = GroundPlane(height=3.0, pitch=math.radians(35.0))
FPS = 5.0


def track_on_floor(track_id: int, xs, zs, frames=None, box_h=200.0, box_w=80.0) -> Track:
    """A track whose foot points land on given floor positions, in metres.

    Built by projecting *forward* through the same geometry the detectors invert. That is
    deliberate: a test that hand-wrote pixel boxes would be checking that two different
    guesses about the camera agree, which is not a property of the code under test.
    """
    xs, zs = np.asarray(xs, dtype=float), np.asarray(zs, dtype=float)
    u, v, _depth = ground_to_pixel(xs, zs, CAM, PLANE)
    boxes = [
        np.array([ui - box_w / 2, vi - box_h, ui + box_w / 2, vi], dtype=float)
        for ui, vi in zip(u, v, strict=True)
    ]
    idx = list(range(len(boxes))) if frames is None else list(frames)
    return Track(track_id=track_id, box=boxes[-1], frames=idx, boxes=boxes, confirmed=True)


SQUARE = np.array([[-1.0, 3.0], [1.0, 3.0], [1.0, 5.0], [-1.0, 5.0]])


def test_a_zone_is_metres_on_the_floor_and_a_track_walks_into_it():
    zone = ev.Zone("stockroom", SQUARE, restricted=True)
    inside = track_on_floor(1, xs=[0.0] * 10, zs=np.linspace(3.2, 4.8, 10))
    outside = track_on_floor(2, xs=[3.0] * 10, zs=np.linspace(3.2, 4.8, 10))
    got = ev.zone_events([inside, outside], [zone], CAM, PLANE, FPS, "cam01")
    assert [e.type for e in got] == ["zone_intrusion"]
    assert got[0].track_ids == (1,) and got[0].zone == "stockroom"


def test_a_position_the_geometry_declines_to_measure_is_not_inside():
    """NaN comes back from `pixel_to_ground` above the horizon. It must not count.

    Counting a refusal to measure as presence would put a shopper in a restricted zone on
    the strength of a missing number.
    """
    zone = ev.Zone("stockroom", SQUARE, restricted=True)
    points = np.array([[0.0, 4.0], [np.nan, np.nan]])
    assert zone.contains(points).tolist() == [True, False]


def test_loitering_needs_the_track_to_survive_and_says_how_long_it_was():
    zone = ev.Zone("counter", SQUARE, loiter_seconds=1.0)
    lingering = track_on_floor(1, xs=[0.0] * 10, zs=[4.0] * 10)
    passing = track_on_floor(2, xs=[0.0] * 2, zs=[4.0] * 2)
    got = ev.zone_events([lingering, passing], [zone], CAM, PLANE, FPS, "cam01")
    assert [e.type for e in got] == ["loitering"]
    assert got[0].value == pytest.approx(2.0) and got[0].threshold == 1.0
    assert got[0].extra["track_frames"] == 10, "the fragmentation this number depends on"


def test_occupancy_counts_tracks_and_refuses_a_zone_with_no_limit():
    zone = ev.Zone("queue", SQUARE, max_occupancy=1)
    a = track_on_floor(1, xs=[-0.5] * 20, zs=[4.0] * 20)
    b = track_on_floor(2, xs=[0.5] * 20, zs=[4.0] * 20)
    got = ev.occupancy_events([a, b], zone, CAM, PLANE, FPS, "cam01")
    assert [e.type for e in got] == ["occupancy_exceeded"]
    assert got[0].value == 2.0 and got[0].threshold == 1.0
    with pytest.raises(ValueError, match="no max_occupancy"):
        ev.occupancy_events([a], ev.Zone("queue", SQUARE), CAM, PLANE, FPS, "cam01")


def test_a_line_is_crossed_with_a_direction_and_only_when_it_is_actually_crossed():
    line = ev.CountingLine("door", np.array([-2.0, 4.0]), np.array([2.0, 4.0]))
    through = track_on_floor(1, xs=[0.0] * 6, zs=[3.0, 3.4, 3.8, 4.2, 4.6, 5.0])
    beside = track_on_floor(2, xs=[6.0] * 6, zs=[3.0, 3.4, 3.8, 4.2, 4.6, 5.0])
    got = ev.line_events([through, beside], line, CAM, PLANE, FPS, "cam01")
    assert len(got) == 1, "a track past the end of the segment does not cross it"
    assert got[0].track_ids == (1,) and got[0].extra["direction"] in {"in", "out"}
    back = track_on_floor(3, xs=[0.0] * 6, zs=[5.0, 4.6, 4.2, 3.8, 3.4, 3.0])
    other = ev.line_events([back], line, CAM, PLANE, FPS, "cam01")
    assert other[0].extra["direction"] != got[0].extra["direction"]


def test_running_is_net_displacement_and_pacing_on_the_spot_is_not_running():
    fast = track_on_floor(1, xs=np.linspace(0.0, 12.0, 15), zs=[6.0] * 15)
    pacing = track_on_floor(2, xs=[0.0, 0.6] * 8, zs=[6.0] * 16)
    got = ev.speed_events([fast, pacing], CAM, PLANE, FPS, "cam01", max_speed_mps=2.5)
    assert [e.track_ids for e in got] == [(1,)]
    assert got[0].value > 2.5 and got[0].type == "running"


def test_tailgating_is_two_tracks_one_line_and_one_direction():
    line = ev.CountingLine("door", np.array([-2.0, 4.0]), np.array([2.0, 4.0]))
    zs = [3.0, 3.5, 4.5, 5.0]
    first = track_on_floor(1, xs=[0.0] * 4, zs=zs, frames=[0, 1, 2, 3])
    close = track_on_floor(2, xs=[0.5] * 4, zs=zs, frames=[2, 3, 4, 5])
    late = track_on_floor(3, xs=[0.5] * 4, zs=zs, frames=[40, 41, 42, 43])
    got = ev.tailgating_events([first, close, late], line, CAM, PLANE, FPS, "cam01", 2.0)
    assert len(got) == 1 and set(got[0].track_ids) == {1, 2}
    assert got[0].value <= 2.0


def test_a_crowd_is_a_density_proxy_and_says_so():
    tracks = [track_on_floor(i, xs=[0.2 * i] * 40, zs=[6.0] * 40) for i in range(4)]
    got = ev.crowd_events(tracks, CAM, PLANE, FPS, "cam01", radius_m=1.5, min_people=4)
    assert [e.type for e in got] == ["crowd_forming"]
    assert got[0].value >= 4
    spread = [track_on_floor(i, xs=[4.0 * i] * 40, zs=[6.0] * 40) for i in range(4)]
    assert ev.crowd_events(spread, CAM, PLANE, FPS, "cam01", min_people=4) == []


def test_the_fall_proxy_is_named_a_candidate_and_fires_on_bending_too():
    """It cannot separate a fall from a shopper reaching a bottom shelf. That is the point.

    Both shapes below produce the same event, which is why the type is `fall_candidate`
    and why the tier-2 pose rule exists.
    """
    upright = [np.array([100.0, 100.0, 160.0, 300.0])] * 6
    down = [np.array([100.0, 250.0, 260.0, 300.0])] * 12
    boxes = upright + down
    track = Track(1, boxes[-1], frames=list(range(len(boxes))), boxes=boxes, confirmed=True)
    got = ev.fall_candidates([track], FPS, "cam01")
    assert [e.type for e in got] == ["fall_candidate"]
    assert "cannot separate a fall from bending" in got[0].basis


def keypoints(angle_deg: float, hip_ankle_px: float, score: float = 0.9) -> np.ndarray:
    """A synthetic person: torso at ``angle_deg`` from vertical, legs ``hip_ankle_px`` long."""
    kps = np.zeros((17, 3))
    kps[:, 2] = score
    hip = np.array([500.0, 600.0])
    torso = 150.0
    dx = torso * math.sin(math.radians(angle_deg))
    dy = torso * math.cos(math.radians(angle_deg))
    shoulder = hip - np.array([dx, dy])
    for i in (ev.KP["left_hip"], ev.KP["right_hip"]):
        kps[i, :2] = hip
    for i in (ev.KP["left_shoulder"], ev.KP["right_shoulder"]):
        kps[i, :2] = shoulder
    for i in (ev.KP["left_ankle"], ev.KP["right_ankle"]):
        kps[i, :2] = hip + np.array([0.0, hip_ankle_px])
    for i in (ev.KP["left_wrist"], ev.KP["right_wrist"]):
        kps[i, :2] = shoulder + np.array([60.0, 40.0])
    return kps


def posed(track_id: int, poses: list[np.ndarray]) -> Track:
    box = np.array([400.0, 400.0, 600.0, 800.0])
    return Track(
        track_id=track_id,
        box=box,
        frames=list(range(len(poses))),
        boxes=[box] * len(poses),
        confirmed=True,
        keypoints=poses,
    )


def test_pose_separates_a_fall_from_a_crouch_which_is_why_the_tier_exists():
    fallen = posed(1, [keypoints(80.0, 200.0)] * 10)
    crouching = posed(2, [keypoints(10.0, 60.0)] * 10)
    upright = posed(3, [keypoints(5.0, 200.0)] * 10)
    got = ev.pose_posture_events([fallen, crouching, upright], FPS, "cam01")
    by_track = {e.track_ids[0]: e.type for e in got}
    assert by_track == {1: "fall", 2: "crouch"}


def test_a_pose_event_on_a_track_with_no_pose_names_the_missing_model():
    """Not an empty list. An empty list is indistinguishable from "nobody fell"."""
    box = np.array([0.0, 0.0, 10.0, 20.0])
    bare = Track(1, box, frames=[0, 1], boxes=[box, box], confirmed=True)
    with pytest.raises(NotImplementedError, match="second-stage pose model"):
        ev.pose_posture_events([bare], FPS, "cam01")


def test_reach_to_shelf_needs_both_heads_and_refuses_a_frame_with_no_terrain():
    fixture_id = 4
    terrain = np.zeros((720, 1280), dtype=np.int64)
    terrain[600:700, 500:700] = fixture_id  # a shelf where the wrists land
    poses = [keypoints(5.0, 200.0)] * 6
    for kps in poses:
        kps[ev.KP["left_wrist"], :2] = np.array([560.0, 640.0])
        kps[ev.KP["right_wrist"], :2] = np.array([560.0, 640.0])
    track = posed(1, poses)
    frames = [
        {"frame_index": i, "terrain": terrain, "class_names": (), "boxes": None}
        for i in range(6)
    ]
    got = ev.reach_to_shelf_events([track], frames, FPS, "cam01", fixture_id=fixture_id)
    assert [e.type for e in got] == ["reach_to_shelf"]
    assert got[0].value == pytest.approx(1.0), "the whole disk is fixture here"
    bare = [{"frame_index": i, "terrain": None} for i in range(6)]
    with pytest.raises(ValueError, match="no frame carries a `terrain` map"):
        ev.reach_to_shelf_events([track], bare, FPS, "cam01", fixture_id=fixture_id)


# --------------------------------------------------------------------------
# The wrist-self-occlusion fix, pinned against the shapes the pose pilot measured.
#
# runs/pose_pilot01/REPORT.md section 3: the original rule tested the wrist *pixel*
# against fixture, and that pixel is covered by the person's own hand whenever the pose
# and the segmentation are both right -- so the rule could only fire on occlusion or
# pose error, the opposite of "touching the shelf". Rerun on the pilot's own data
# (keypoints.npz + the same checkpoint's terrain, display_fixture id 12, person id 11,
# defaults radius_px=25 / contact_ratio=0.35), old rule vs this one:
#
#   cam04 old: 9 spans, all track 4 (f22-107) -- the child whose hand the display
#     *occludes*. The bending adult (track 2), wrists at conf 0.89-0.97 visibly on the
#     table, produced zero spans: their wrist pixels read `person` (98-100% of the
#     25 px disk at the probe frames f114/f150).
#   cam04 new: track 4 still fires across f22-107 (same spans within a frame or two,
#     plus f92-94), and track 2 now fires exactly where it bends over the table
#     (f10-12, 104-110, 119-122, 126-128, 162-168, 181-187, 203-206), plus the clerk
#     and two children with hands at the display edge (tracks 5, 6, 9) -- each checked
#     against terrain-overlay renders: every firing wrist sits at a fixture region.
#   cam11 old and new: zero events at every radius/ratio swept (r 15-40, ratio
#     0.15-0.5). Nobody in that clip reaches, and the fix does not invent it.


def _shelf_scene(ring_class: int, fixture_id: int = 4, person_id: int = 7):
    """A hand covering its own wrist: person out to r=24, ``ring_class`` beyond.

    The geometry of the pilot's probe frame cam04 t2 f114 -- 98% of the wrist disk is
    the person's own body -- so the single-pixel rule reads `person` and stays silent
    no matter what the hand is over. Only the sliver past the fingertips says.
    """
    wrist = np.array([560.0, 640.0])
    terrain = np.zeros((720, 1280), dtype=np.int64)
    yy, xx = np.mgrid[0:720, 0:1280]
    d2 = (xx - wrist[0]) ** 2 + (yy - wrist[1]) ** 2
    terrain[d2 <= 27.0**2] = ring_class
    terrain[d2 <= 24.0**2] = person_id
    poses = [keypoints(5.0, 200.0)] * 6
    for kps in poses:
        kps[ev.KP["left_wrist"], :2] = wrist
        kps[ev.KP["right_wrist"], :2] = wrist
    frames = [{"frame_index": i, "terrain": terrain} for i in range(6)]
    return posed(1, poses), frames, fixture_id, person_id


def test_a_wrist_covered_by_its_own_hand_still_reads_what_the_hand_is_over():
    """The pilot's structural miss: pose right + segmentation right = wrist pixel is
    `person`, and the old rule could never fire. The neighbourhood rule must."""
    track, frames, fixture_id, person_id = _shelf_scene(ring_class=4)
    got = ev.reach_to_shelf_events(
        [track], frames, FPS, "cam01", fixture_id=fixture_id, person_id=person_id
    )
    assert [e.type for e in got] == ["reach_to_shelf"]
    assert got[0].value == pytest.approx(1.0), "every resolvable pixel is fixture"
    assert "non-person" in got[0].basis


def test_excluding_the_persons_own_pixels_is_the_mechanism_not_a_detail():
    """Same scene, person exclusion off: the hand dilutes the ratio below threshold.

    This is the difference between "what is the hand over" and "how much of the disk is
    shelf", and only the first is the event.
    """
    track, frames, fixture_id, _person_id = _shelf_scene(ring_class=4)
    got = ev.reach_to_shelf_events(
        [track], frames, FPS, "cam01", fixture_id=fixture_id, person_id=None
    )
    assert got == []


def test_a_hand_over_the_floor_beside_a_shelf_is_not_reaching():
    """The ring past the fingertips is floor, not fixture: no event, whatever the old
    single-pixel rule would have said about occlusion."""
    track, frames, fixture_id, person_id = _shelf_scene(ring_class=0)
    got = ev.reach_to_shelf_events(
        [track], frames, FPS, "cam01", fixture_id=fixture_id, person_id=person_id
    )
    assert got == []


def test_a_terrain_map_in_the_wrong_pixel_space_is_refused_not_silently_empty():
    """The pilot's section 4-2 failure: a canvas-resolution map (512x896) against
    1920x1080 keypoints put every wrist off the map's edge, and the old code skipped
    them -- zero events that read as "nobody reached". Now it raises, naming who
    resizes: the caller, nearest-neighbour, back to image resolution.
    """
    track, frames, fixture_id, person_id = _shelf_scene(ring_class=4)
    small = [
        {"frame_index": f["frame_index"], "terrain": f["terrain"][::2, ::2]} for f in frames
    ]
    with pytest.raises(ValueError, match="caller must upsample"):
        ev.reach_to_shelf_events(
            [track], small, FPS, "cam01", fixture_id=fixture_id, person_id=person_id,
            image_size=(720, 1280),
        )  # fmt: skip
    # A StageFrame-shaped payload carries `image`, so it is checked with no extra arg.
    with_image = [
        {**f, "terrain": f["terrain"][::2, ::2], "image": np.zeros((720, 1280, 3), np.uint8)}
        for f in frames
    ]
    with pytest.raises(ValueError, match="nearest-neighbour"):
        ev.reach_to_shelf_events(
            [track], with_image, FPS, "cam01", fixture_id=fixture_id, person_id=person_id
        )
    # And the matching size passes the same gate rather than being exempt from it.
    ok = ev.reach_to_shelf_events(
        [track], frames, FPS, "cam01", fixture_id=fixture_id, person_id=person_id,
        image_size=(720, 1280),
    )  # fmt: skip
    assert [e.type for e in ok] == ["reach_to_shelf"]


def test_stock_removed_ignores_a_shopper_standing_in_front_of_the_shelf():
    zone = ev.Zone("endcap", SQUARE)
    steady = dict.fromkeys(range(200), 8)
    occluded = dict(steady)
    occluded.update(dict.fromkeys(range(100, 110), 5))  # two seconds at 5 fps
    assert ev.stock_removed_events(occluded, zone, FPS, "cam01", sustained_seconds=10.0) == []
    removed = dict(steady)
    removed.update(dict.fromkeys(range(100, 200), 5))
    got = ev.stock_removed_events(removed, zone, FPS, "cam01", sustained_seconds=10.0)
    assert [e.type for e in got] == ["stock_removed"]
    assert got[0].value == 3.0


def test_an_object_left_is_a_still_bag_with_nobody_near_it():
    bag = track_on_floor(1, xs=[0.0] * 200, zs=[5.0] * 200, box_h=40.0, box_w=40.0)
    owner_far = track_on_floor(2, xs=[9.0] * 200, zs=[5.0] * 200)
    got = ev.object_left_events([bag], [owner_far], CAM, PLANE, FPS, "cam01")
    assert [e.type for e in got] == ["object_left"]
    owner_near = track_on_floor(3, xs=[0.5] * 200, zs=[5.0] * 200)
    assert ev.object_left_events([bag], [owner_near], CAM, PLANE, FPS, "cam01") == []


def test_every_event_type_declares_the_instrument_that_produced_it():
    assert set(ev.TIERS) == set(ev.EVENT_TYPES)
    assert set(ev.TIERS.values()) == {"track geometry", "pose", "temporal model"}


def test_an_unbuilt_type_refuses_with_its_blocker_rather_than_returning_nothing():
    with pytest.raises(NotImplementedError, match="association"):
        ev.require_buildable("fight")
    with pytest.raises(ValueError, match="unknown event type"):
        ev.require_buildable("shoplifting")
    ev.require_buildable("zone_intrusion")


def test_a_row_survives_json_with_no_numpy_left_in_it():
    import json

    zone = ev.Zone("stockroom", SQUARE, restricted=True)
    track = track_on_floor(1, xs=[0.0] * 10, zs=[4.0] * 10)
    row = ev.zone_events([track], [zone], CAM, PLANE, FPS, "cam01")[0].as_row()
    assert json.loads(json.dumps(row))["type"] == "zone_intrusion"
    assert row["basis"], "a row without a basis is a number nobody can attribute"


# -------------------------------------------------- the wall clock, at the edge

TAIPEI = timezone(timedelta(hours=8))
CLIP = "datasets/studioa_clips/Taichung-cam01/archive_20260816-113012_20260816-113518.mp4"


def _event(**kw) -> ev.SecurityEvent:
    base = {"type": "loitering", "camera": "Taichung-cam01", "frame_start": 0,
            "frame_end": 9, "fps": FPS}  # fmt: skip
    return ev.SecurityEvent(**{**base, **kw})


def test_the_filename_is_utc_and_the_store_is_not():
    """The failure pull_studioa.py records in capitals, in one assertion.

    11:30:12 UTC is 19:30:12 in a UTC+8 store. Asking for the local hour and getting the
    UTC one is how a request for "16:00, the busy hour" returned a closed shop.
    """
    assert ev.clip_start_from_name(CLIP, TAIPEI) == datetime(
        2026, 8, 16, 19, 30, 12, tzinfo=TAIPEI
    )
    assert ev.clip_start_from_name(CLIP) == datetime(
        2026, 8, 16, 11, 30, 12, tzinfo=timezone.utc
    )


def test_a_clip_with_no_time_in_its_name_is_refused_rather_than_guessed():
    """An invented start puts every row at a confidently wrong moment."""
    with pytest.raises(ValueError, match="does not carry a recording time"):
        ev.clip_start_from_name("clip.mp4")


def test_an_unstamped_event_says_none_rather_than_a_plausible_time():
    e = _event()
    assert e.started_at is None and e.ended_at is None
    assert e.as_row()["started_at"] is None


def test_a_stamped_event_locates_itself_in_the_footage():
    """`frame_start: 5312` does not locate anything in a DVR; this is what does."""
    start = ev.clip_start_from_name(CLIP, TAIPEI)
    (e,) = ev.with_clip_start([_event(frame_start=25, frame_end=49)], start)
    assert e.started_at == start + timedelta(seconds=5.0)  # 25 frames at 5 fps
    assert e.ended_at == start + timedelta(seconds=10.0)  # frame_end is inclusive
    assert e.ended_at - e.started_at == timedelta(seconds=e.seconds)


def test_the_row_carries_the_offset_so_nobody_has_to_ask_which_zone():
    (e,) = ev.with_clip_start([_event()], ev.clip_start_from_name(CLIP, TAIPEI))
    assert e.as_row()["started_at"] == "2026-08-16T19:30:12+08:00"


def test_a_naive_start_is_refused():
    """A time with no zone, in a report spanning a UTC bucket and a UTC+8 store, is a
    time nobody can act on."""
    with pytest.raises(ValueError, match="no timezone"):
        ev.with_clip_start([_event()], datetime(2026, 8, 16, 19, 30, 12))


def test_stamping_leaves_the_originals_alone():
    """SecurityEvent is frozen and the builders return shared lists; replace, not mutate."""
    original = _event()
    (stamped,) = ev.with_clip_start([original], ev.clip_start_from_name(CLIP, TAIPEI))
    assert original.clip_start is None
    assert stamped is not original


def test_no_start_is_a_no_op_rather_than_an_error():
    """A clip from outside the corpus still produces events, correct in frames."""
    rows = [_event(), _event(frame_start=10, frame_end=19)]
    assert [e.as_row() for e in ev.with_clip_start(rows, None)] == [e.as_row() for e in rows]

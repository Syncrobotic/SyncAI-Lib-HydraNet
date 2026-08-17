"""The fixed-camera stabiliser, on scenes whose right answer is written down.

This filter edits model output, and its failure mode is a panel that looks *better*: a
person voted away leaves a clean floor and no error. Nothing downstream would flag it,
and the eye reads a smoother map as a more confident one. So the tests here are not
about the voting arithmetic -- a per-pixel majority is small and obvious -- but about the
gate that decides where voting is allowed to happen at all.

Three properties carry the whole safety argument:

* where the image changed, the live prediction passes through untouched, so the filter
  can settle an argument about empty floor and can never average an occupant away;
* the background plate follows only what is already static, so someone who walks in and
  stands still never becomes background and never earns the right to be smoothed;
* with less history than `window`, nothing is smoothed at all.

The greyscale case is not hypothetical padding: the IR night clips under
`datasets/studioa_clips/` are single-channel, and they are the only footage where this
gate can be measured in isolation from the thing it exists to protect.

pytest tests/test_temporal.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from syncai_hydranet.utils.temporal import FixedCameraStabiliser, StabiliserSelfSealedError

H, W = 20, 20
FLOOR, PERSON = 100, 200  # grey levels, far enough apart to clear the default diff_thr
GO, BLOCKED, OTHER = 2, 1, 0


def _frame(person_rows: int = 0) -> np.ndarray:
    """A flat grey frame, optionally with a brighter block at the top."""
    a = np.full((H, W, 3), FLOOR, np.uint8)
    if person_rows:
        a[:person_rows] = PERSON
    return a


def _pred(value: int, top_rows: int = 0, top_value: int = 0) -> np.ndarray:
    a = np.full((H, W), value, np.uint8)
    if top_rows:
        a[:top_rows] = top_value
    return a


def _settle(stab: FixedCameraStabiliser, frame: np.ndarray, pred: np.ndarray, n: int):
    """Feed the same frame and prediction `n` times; return the last output."""
    out = None
    for _ in range(n):
        out = stab(frame, pred)
    return out


def test_a_window_that_is_not_full_yet_smooths_nothing():
    """Voting on two frames is not voting. Until there is history to outvote a frame
    with, the live prediction has to come back exactly as it went in."""
    stab = FixedCameraStabiliser(window=5, num_classes=3)
    frame = _frame()
    _settle(stab, frame, _pred(GO), 3)
    # the fourth call leaves the ring one short of `window`, so nothing may be outvoted
    dissent = _pred(BLOCKED)
    assert np.array_equal(stab(frame, dissent), dissent)


def test_a_static_scene_outvotes_a_single_flickering_frame():
    """The 16.7% this was built for: one frame disagreeing with four is noise on a
    constant scene, and the majority is the answer."""
    stab = FixedCameraStabiliser(window=5, num_classes=3)
    frame = _frame()
    _settle(stab, frame, _pred(GO), 4)
    out = stab(frame, _pred(BLOCKED))
    assert np.all(out == GO), "a lone dissenting frame must not flip a settled pixel"


def test_where_the_image_changed_the_live_prediction_passes_through():
    """The gate. Somebody has walked into a region that was settled floor; the vote
    there is stale by construction, so it must not be applied."""
    stab = FixedCameraStabiliser(window=3, num_classes=3)
    _settle(stab, _frame(), _pred(GO), 3)

    out = stab(_frame(person_rows=5), _pred(GO, top_rows=5, top_value=BLOCKED))
    assert np.all(out[:5] == BLOCKED), "the changed region must return the live prediction"
    assert np.all(out[5:] == GO), "the untouched floor must still be voted on"


def test_someone_who_walks_in_and_stands_still_never_becomes_background():
    """The plate follows only pixels that are already static, so an occupant who arrives
    after it is established is never absorbed into it however long they stay.

    Without this the filter would be safe for a walking shopper and unsafe for a queuing
    one, which is the wrong way round -- a person standing at a shelf is exactly the
    case a shop robot must not drive through.
    """
    stab = FixedCameraStabiliser(window=5, num_classes=3)
    _settle(stab, _frame(), _pred(GO), 10)  # empty floor first: the plate is the floor

    occupied = _frame(person_rows=5)
    _settle(stab, occupied, _pred(GO, top_rows=5, top_value=BLOCKED), 30)

    # 30 frames of agreement is far more than `window`; if the occupied rows had been
    # absorbed into the plate, they would now be voted on and this would come back BLOCKED.
    out = stab(occupied, _pred(GO, top_rows=5, top_value=OTHER))
    assert np.all(out[:5] == OTHER), "an occupied region must stay on the live prediction"


def test_anyone_already_in_the_very_first_frame_is_baked_into_the_background():
    """The limit of the guarantee above, pinned because it is not the one the module
    docstring states.

    The plate is seeded from the first frame outright (`self._plate = grey.copy()`), so
    whoever is standing there when the stabiliser is constructed *is* the background from
    then on, and their pixels are voted over like floor. The docstring's "can never
    average a person away" holds only for occupants who arrive after that first frame.

    This is not an edge case on retail footage: `cli/scene.py` builds one stabiliser per
    clip, and a shop clip that opens with nobody in shot is the exception. Anything that
    fixes it -- seeding the plate from a median over the first `window` frames, or
    refusing to smooth until a pixel has been stable for one -- should turn this test
    around rather than delete it.
    """
    stab = FixedCameraStabiliser(window=5, num_classes=3)
    occupied = _frame(person_rows=5)
    _settle(stab, occupied, _pred(GO, top_rows=5, top_value=BLOCKED), 30)

    out = stab(occupied, _pred(GO, top_rows=5, top_value=OTHER))
    assert np.all(out[:5] == BLOCKED), (
        "documents current behaviour: a person present at construction is background, "
        "so their region is voted on rather than passed through"
    )


def test_a_seed_time_occupant_who_leaves_never_gives_the_region_back():
    """The second consequence of the same root, and the one the module docstring gets
    actively wrong.

    The plate updates only where the scene is already static, so once a seed-time
    occupant walks away the vacated floor disagrees with a plate that still holds them,
    the region is non-static forever, and the plate cannot recover. The docstring puts
    the cost of an occupied-then-free pixel at "up to `window` frames … at most
    `window / fps` seconds". That is true of the ring, which ages out; it is not true of
    the plate, which does not. For a seed-time occupant the region is excluded from
    smoothing for the whole clip.

    Nothing unsafe happens -- passing the live prediction through is the conservative
    direction -- but the filter silently stops working in the region it was aimed at,
    and reports nothing. Verified out to 200 empty frames.
    """
    stab = FixedCameraStabiliser(window=5, num_classes=3)
    _settle(stab, _frame(person_rows=5), _pred(GO, top_rows=5, top_value=BLOCKED), 20)

    empty = _frame()
    _settle(stab, empty, _pred(GO, top_rows=5, top_value=BLOCKED), 200)

    # History says BLOCKED, the live frame flickers to OTHER. If the plate had recovered,
    # the settled vote would win and this would be BLOCKED.
    out = stab(empty, _pred(GO, top_rows=5, top_value=OTHER))
    assert np.all(out[:5] == OTHER), (
        "documents current behaviour: the plate never re-learns floor a seed-time "
        "occupant was standing on, so the region stays gated for the rest of the clip"
    )


def test_a_change_of_frame_size_starts_over_rather_than_crashing():
    """A stabiliser reused across clips of different sizes must drop its history, not
    broadcast a stale plate against a new shape."""
    stab = FixedCameraStabiliser(window=3, num_classes=3)
    _settle(stab, _frame(), _pred(GO), 5)

    bigger = np.full((H * 2, W * 2, 3), FLOOR, np.uint8)
    pred = np.full((H * 2, W * 2), BLOCKED, np.uint8)
    assert np.array_equal(stab(bigger, pred), pred), "history must not survive a reshape"


def test_reset_drops_the_plate_so_one_clip_cannot_contaminate_the_next():
    """Same size is the dangerous case: nothing about the array shapes says the footage
    changed, so a caller reusing one stabiliser across two clips gets a plate built from
    a scene that is no longer on screen. The filter would then read every pixel as
    changed and silently do nothing at all -- a no-op that looks exactly like a working
    filter on noisy footage.

    `cli/scene.py` constructs one per clip in `render_video` for this reason. This pins
    the escape hatch for anyone who reuses an instance instead.
    """
    stab = FixedCameraStabiliser(window=3, num_classes=3)
    frame = _frame()
    _settle(stab, frame, _pred(GO), 5)
    assert np.all(stab(frame, _pred(BLOCKED)) == GO)  # settled

    stab.reset()
    dissent = _pred(BLOCKED)
    assert np.array_equal(stab(frame, dissent), dissent), "reset must clear the history"


def test_single_channel_infrared_frames_are_accepted():
    """The night clips are greyscale. A filter that only understands RGB would fail on
    the one domain where its gate can be measured against an empty store."""
    stab = FixedCameraStabiliser(window=3, num_classes=3)
    grey = np.full((H, W), FLOOR, np.uint8)
    _settle(stab, grey, _pred(GO), 3)
    assert np.all(stab(grey, _pred(BLOCKED)) == GO)


def test_a_global_change_of_illumination_switches_the_whole_filter_off():
    """`static` compares raw grey levels against `diff_thr`, so nothing distinguishes
    "the scene changed" from "the exposure changed".

    A fixed camera is fixed in geometry, not in brightness: auto-exposure, a light
    switched on, or dusk through a shopfront moves every pixel at once. Past `diff_thr`
    the entire frame reads as changed, every pixel falls back to the live prediction, and
    the filter becomes a no-op that reports nothing -- indistinguishable from the outside
    from a filter that ran and found nothing to correct.

    The cliff is sharp, not gradual: 11 grey levels is fully smoothed, 12 is fully off.
    Anything that primes a plate from separate footage inherits this, which is why an IR
    night plate cannot be reused on a daytime clip however fixed the camera is.
    """
    settled = _pred(GO)
    dissent = _pred(BLOCKED)

    for shift, expect_smoothed in ((11, True), (12, False)):
        stab = FixedCameraStabiliser(window=3, num_classes=3)
        _settle(stab, _frame(), settled, 5)
        out = stab(np.full((H, W, 3), FLOOR + shift, np.uint8), dissent)
        if expect_smoothed:
            assert np.all(out == GO), f"a {shift}-level shift is under diff_thr"
        else:
            assert np.array_equal(out, dissent), (
                f"a {shift}-level shift gates every pixel and the filter does nothing"
            )


def _ramp_then_probe(stab: FixedCameraStabiliser, rate: float, frames: int = 400) -> float:
    """Drift the whole frame at `rate` grey levels per frame; return the static fraction.

    The probe frame's live prediction disagrees with the settled majority, so a returned
    `GO` means the vote won (the plate kept up) and `BLOCKED` means the pixel was gated.
    """
    for i in range(frames):
        level = np.clip(FLOOR + rate * i, 0, 255)
        stab(np.full((H, W, 3), level, np.uint8), _pred(GO))
    level = np.clip(FLOOR + rate * frames, 0, 255)
    out = stab(np.full((H, W, 3), level, np.uint8), _pred(BLOCKED))
    return float(np.mean(out == GO))


def test_the_drift_the_plate_can_follow_is_the_product_of_both_constants():
    """`plate_alpha` lets the plate follow a moving scene, but only where it is already
    static -- so a ramp it cannot keep up with accumulates lag until the lag crosses
    `diff_thr`, at which point the pixel stops being static and the plate stops updating
    at all. The filter seals itself off and does not come back.

    Steady state for a ramp of `r` levels/frame is a lag of `r / plate_alpha`, so the
    filter survives exactly while `r < diff_thr * plate_alpha`. With the values nothing
    can currently change -- 12.0 and 0.02 -- that is 0.24 levels/frame, or 1.44 per second
    at 6 fps. Slow dusk survives; a passing cloud or one auto-exposure correction does not.

    The threshold is read off the instance rather than hard-coded, because the number that
    has to be justified is the *product*: tuning `diff_thr` alone moves the illumination
    cliff and this drift tolerance together, which is not obvious from either signature.
    Neither constant is reachable from `--stabilise`, which passes only `window`.

    **Turned around, as this docstring originally asked.** The earlier version asserted
    that a ramp past the criterion returned a static share of exactly 0.0 -- the filter
    off for good, reporting nothing. That behaviour was the complaint, not the
    specification, so it is now a refusal. The *criterion itself is unchanged* and is
    still what this test pins: half of it tracks, twice it seals. Only what happens after
    it seals has changed, which is the distinction between pinning today's value and
    pinning the mechanism.
    """
    stab = FixedCameraStabiliser(window=3, num_classes=3)
    critical = stab.diff_thr * stab.plate_alpha

    assert _ramp_then_probe(stab, critical * 0.5) == 1.0, "the plate should track a slow ramp"

    faster = FixedCameraStabiliser(window=3, num_classes=3)
    with pytest.raises(StabiliserSelfSealedError, match="does not recover"):
        _ramp_then_probe(faster, critical * 2.0)


def test_a_brief_full_frame_occlusion_does_not_trip_the_refusal():
    """The refusal has to separate "sealed" from "something crossed the whole view".

    A delivery trolley passing a wall-mounted camera gates every pixel for a second or
    two. Refusing on that would make the filter unusable on exactly the footage it is for,
    so `seal_patience` is what distinguishes them -- and it is the reason the check counts
    *consecutive* frames rather than a rate.
    """
    stab = FixedCameraStabiliser(window=3, num_classes=3, seal_patience=10)
    _settle(stab, _frame(), _pred(GO), 5)
    for _ in range(9):  # one short of the patience, then the scene comes back
        stab(np.full((H, W, 3), FLOOR + 60, np.uint8), _pred(BLOCKED))
    assert stab.sealed_share() == 0.0, "nothing has been shut long enough to be stuck"

    stab(_frame(), _pred(GO))  # the trolley passes; the gate reopens
    assert stab.diagnostics()["static_share"] == 1.0
    assert stab.sealed_share() == 0.0, "the per-pixel counter resets rather than accruing"


def test_the_refusal_names_the_mechanism_and_not_just_the_symptom():
    """An operator who sees this needs to know it will not recover and what to do.

    Pinned because the value of raising over coasting is entirely in the message: a bare
    RuntimeError here would be the same no-op with an extra stack trace.
    """
    stab = FixedCameraStabiliser(window=3, num_classes=3, seal_patience=3)
    _settle(stab, _frame(), _pred(GO), 5)
    with pytest.raises(StabiliserSelfSealedError) as e:
        for _ in range(3):
            stab(np.full((H, W, 3), FLOOR + 60, np.uint8), _pred(BLOCKED))
    msg = str(e.value)
    assert "does not recover" in msg
    assert "diff_thr" in msg and "plate_alpha" in msg, (
        "both constants, since the product governs"
    )
    assert "confirmed empty" in msg, "the fix is a vouched plate, and the message should say so"


def test_a_seed_time_occupant_who_leaves_is_reported_even_though_it_is_not_refused():
    """The localised half of the self-sealing failure, which the global refusal must not
    fire on and which the panel cannot show.

    A person in frame 1 is baked into the plate; when they leave, that region disagrees
    with a plate that still holds them, is non-static forever, and the plate cannot
    recover. Three quarters of the frame still smooths normally, so `static_share` stays
    high and nothing trips -- the panel looks *better* than it should, not worse.

    `sealed_share` is the number that shows it, and it is why the measure counts
    *consecutive shut frames per pixel* rather than "never static": at frame 1 the
    occupant **is** the plate, so those pixels read static and any never-static measure
    reports 0.0 for exactly this case. Checked, and it did.
    """
    stab = FixedCameraStabiliser(window=3, num_classes=3, seal_patience=10)
    _settle(stab, _frame(person_rows=5), _pred(GO), 5)  # occupant present from frame 1
    for _ in range(30):  # they leave; the vacated rows can never agree with the plate
        stab(_frame(), _pred(GO))

    assert stab.sealed_share() == pytest.approx(0.25, abs=0.01), "5 of 20 rows, stuck shut"
    assert stab.diagnostics()["static_share"] == pytest.approx(0.75, abs=0.01), (
        "the other three quarters smooth normally, which is why nothing looks wrong"
    )


def test_a_window_below_one_is_refused():
    """`window=0` would mean an empty ring and an argmax over nothing."""
    with pytest.raises(ValueError):
        FixedCameraStabiliser(window=0)


def test_the_vote_is_confined_to_the_declared_label_space():
    """`num_classes` is the size of the label space being voted on, and a caller passing
    a smaller number than the predictions actually use gets a vote that cannot see the
    missing ids.

    Pinned because the failure is silent and the call site is easy to get wrong:
    `cli/scene.py` passes `len(TRAV_COLORS)`, so the palette and the label space are the
    same fact in two places. If they ever disagree, this is the behaviour that follows.
    """
    stab = FixedCameraStabiliser(window=3, num_classes=2)  # declares {0, 1}, gets 2s
    frame = _frame()
    out = _settle(stab, frame, _pred(GO), 4)
    assert np.all(out != GO), (
        "a class outside num_classes cannot win a vote -- if this ever passes with GO, "
        "the guard below is obsolete and the call site no longer needs to match"
    )

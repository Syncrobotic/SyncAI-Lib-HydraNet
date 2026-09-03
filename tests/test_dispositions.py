"""The disposition log's failures would all be silent, and they are all data loss.

A torn append that welds two rows, a reader that quietly drops a corrupt line, a
verdict filed against an alert that does not exist -- none of these raise anywhere
downstream; they surface months later as training rows that never were. Hence tests.

pytest tests/test_dispositions.py -v
"""

import json
import threading
from datetime import datetime, timezone

import pytest

from syncai_hydranet.analytics.events import SecurityEvent
from syncai_hydranet.serving.dispositions import (
    SCHEMA_VERSION,
    AlertRecord,
    CorruptLogError,
    append_row,
    config_hash,
    current_dispositions,
    day_path,
    file_hash,
    iter_records,
    model_identity,
    record_alert,
    record_disposition,
)

NOW = datetime(2026, 8, 19, 3, 30, tzinfo=timezone.utc)

MODEL = {
    "checkpoint": "runs/hydranet-retail-xl/best.pt",
    "git": {"available": True, "commit": "ee1bdc1", "dirty": False},
    "config_hash": "sha256:deadbeef",
}


def event(**overrides) -> SecurityEvent:
    fields = {
        "type": "loitering",
        "camera": "cam04",
        "frame_start": 100,
        "frame_end": 160,
        "fps": 5.0,
        "track_ids": (7,),
        "zone": "entrance",
        "value": 12.2,
        "threshold": 10.0,
        "basis": "seconds a single track stayed inside the polygon",
    }
    fields.update(overrides)
    return SecurityEvent(**fields)


def log_one(root, **kw) -> AlertRecord:
    kw.setdefault("model", MODEL)
    kw.setdefault("clip", "assets/archive_20260816-113012_20260816-113518.mp4")
    kw.setdefault("now", NOW)
    return record_alert(root, event(), **kw)


# ------------------------------------------------------------------- schema round-trip


def test_alert_round_trips_and_keeps_the_event_row_verbatim(tmp_path):
    ev = event()
    rec = log_one(tmp_path)
    (back,) = list(iter_records(tmp_path))
    assert back == rec
    assert back.event == ev.as_row(), "the event row must land and return untouched"
    assert back.schema_version == SCHEMA_VERSION
    assert back.frame_ref["frame_start"] == 100
    assert back.frame_ref["clip"].endswith(".mp4")


def test_rows_are_filed_under_the_utc_day():
    path = day_path("lake", NOW)
    assert path.name == "2026-08-19.jsonl"
    with pytest.raises(ValueError, match="timezone"):
        day_path("lake", datetime(2026, 8, 19, 3, 30))


def test_calib_path_is_hashed_and_none_means_uncalibrated(tmp_path):
    calib = tmp_path / "cam04.calib.json"
    calib.write_text('{"vfov_deg": 62.0}')
    rec = log_one(tmp_path / "lake", calib=calib)
    assert rec.calib_version == file_hash(calib)
    assert rec.calib_version.startswith("sha256:")
    assert log_one(tmp_path / "lake2").calib_version is None


def test_config_hash_is_stable_under_key_order():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_alert_without_pullable_footage_is_refused(tmp_path):
    # No clip and no wall-clock start: nobody could ever review this against pixels.
    with pytest.raises(ValueError, match="footage"):
        record_alert(tmp_path, event(), model=MODEL, now=NOW)
    # A stream timestamp on the event is an acceptable substitute for a clip path.
    stamped = event(clip_start=datetime(2026, 8, 16, 11, 30, tzinfo=timezone.utc))
    record_alert(tmp_path, stamped, model=MODEL, now=NOW)


def test_alert_with_an_unnamed_instrument_is_refused(tmp_path):
    with pytest.raises(ValueError, match="basis"):
        record_alert(tmp_path, event(basis=""), model=MODEL, clip="c.mp4", now=NOW)


# ------------------------------------------------------------------------ dispositions


def test_a_fresh_alert_is_unreviewed_by_default(tmp_path):
    rec = log_one(tmp_path)
    status = current_dispositions(iter_records(tmp_path))
    assert status[rec.alert_id].status == "unreviewed"
    assert status[rec.alert_id].by is None


def test_a_verdict_updates_status_and_the_last_one_wins(tmp_path):
    rec = log_one(tmp_path)
    record_disposition(tmp_path, rec.alert_id, "confirmed", by="operator-a", at=NOW)
    record_disposition(
        tmp_path, rec.alert_id, "rejected", by="operator-b", reason="hanging packet", at=NOW
    )
    verdict = current_dispositions(iter_records(tmp_path))[rec.alert_id]
    assert verdict.status == "rejected"
    assert verdict.by == "operator-b"
    assert verdict.reason == "hanging packet"
    # Append-only: the overruled verdict is still a row, not an edit.
    kinds = [r.kind for r in iter_records(tmp_path)]
    assert kinds == ["alert", "disposition", "disposition"]


def test_a_disposition_for_an_unknown_alert_is_an_error(tmp_path):
    log_one(tmp_path)
    with pytest.raises(KeyError, match="no-such-alert"):
        record_disposition(tmp_path, "no-such-alert", "confirmed", by="operator-a", at=NOW)


def test_a_disposition_needs_a_known_status_and_a_reviewer(tmp_path):
    rec = log_one(tmp_path)
    with pytest.raises(ValueError, match="maybe"):
        record_disposition(tmp_path, rec.alert_id, "maybe", by="operator-a", at=NOW)
    with pytest.raises(ValueError, match="reviewer"):
        record_disposition(tmp_path, rec.alert_id, "confirmed", by="", at=NOW)


# ------------------------------------------------------------------ corruption + append


def test_the_reader_raises_on_corruption_unless_asked_to_count(tmp_path):
    rec = log_one(tmp_path)
    path = day_path(tmp_path, NOW)
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{this is not json\n")
    log_one(tmp_path)  # a good row after the bad one must survive

    with pytest.raises(CorruptLogError, match="corrupt"):
        list(iter_records(tmp_path))

    bad = []
    good = list(iter_records(tmp_path, corrupt=bad))
    assert len(good) == 2 and good[0] == rec
    assert len(bad) == 1
    assert bad[0].line_no == 2 and bad[0].path.endswith("2026-08-19.jsonl")


def test_a_row_from_a_newer_schema_is_counted_not_half_read(tmp_path):
    rec = log_one(tmp_path)
    row = rec.as_row() | {"schema_version": SCHEMA_VERSION + 1}
    append_row(day_path(tmp_path, NOW), row)
    bad = []
    assert len(list(iter_records(tmp_path, corrupt=bad))) == 1
    assert len(bad) == 1 and "newer" in bad[0].reason


def test_append_after_a_torn_tail_keeps_the_new_row_whole(tmp_path):
    log_one(tmp_path)
    path = day_path(tmp_path, NOW)
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"kind": "alert", "alert')  # a writer killed mid-line: no newline
    rec = log_one(tmp_path)  # must not weld itself onto the torn line

    bad = []
    good = list(iter_records(tmp_path, corrupt=bad))
    assert rec in good and len(good) == 2
    assert len(bad) == 1, "the torn line stays its own, countable, corruption"


def test_concurrent_appends_interleave_whole_lines(tmp_path):
    path = tmp_path / "2026-08-19.jsonl"
    n_threads, n_rows = 8, 25

    def work(tid):
        for i in range(n_rows):
            append_row(path, {"schema_version": 1, "kind": "x", "tid": tid, "i": i})

    threads = [threading.Thread(target=work, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = [ln for ln in path.read_text().splitlines() if ln]
    rows = [json.loads(ln) for ln in lines]  # every line parses: no byte interleaving
    assert len(rows) == n_threads * n_rows
    for tid in range(n_threads):
        assert sorted(r["i"] for r in rows if r["tid"] == tid) == list(range(n_rows))


# ------------------------------------------------------------------------------ filters


def test_filters_narrow_without_lying(tmp_path):
    a = log_one(tmp_path)
    b = record_alert(tmp_path, event(camera="cam11"), model=MODEL, clip="c.mp4", now=NOW)
    record_disposition(tmp_path, a.alert_id, "confirmed", by="op", at=NOW)
    assert [r.alert_id for r in iter_records(tmp_path, camera="cam11")] == [b.alert_id]
    assert [r.kind for r in iter_records(tmp_path, alert_id=a.alert_id)] == [
        "alert",
        "disposition",
    ]
    assert list(iter_records(tmp_path / "no-such-dir")) == []


def test_model_identity_builds_the_dict_the_records_expect():
    """The constructor for AlertRecord.model, wired to the suite like its siblings.

    It had zero callers anywhere -- the tests hand-built the dict -- so nothing pinned
    its shape to the one `record_alert`'s docstring promises.
    """
    ident = model_identity("runs/x/best.pt", config={"a": 1})
    assert set(ident) == set(MODEL)
    assert ident["checkpoint"] == "runs/x/best.pt"
    assert ident["config_hash"] == config_hash({"a": 1})
    assert isinstance(ident["git"], dict) and "available" in ident["git"]

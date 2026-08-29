"""The operator-feedback disposition log: the data engine's first brick.

Every alert an operator sees is recorded here at the moment it is raised, and every
verdict an operator passes on one is appended after it. The point is not the alarm UI --
it is that confirmations and rejections are training data, and training data with no
provenance is the mistake this repository keeps paying for (`RETAIL_DATA.md` opens with
the bill). So a row carries everything needed to reproduce the alert months later:
the event row verbatim (`basis`/`value`/`threshold` -- the instrument, the measurement,
the setting, never a score), the frame it can be pulled from, the checkpoint and commit
that raised it, and the hash of the calibration the geometry ran under. Schema before
product: this file was decided on 2026-08-19
(`git show b7457c2:docs/journal/2026-08-19-security-retail-teachers-and-methodology.md`
section 4 item 4).

---------------------------------------------------------------------------
THE ONE-SIDEDNESS, WHICH IS A PROPERTY OF THE INSTRUMENT AND NOT A BUG

**This store labels the confident-positive frontier and nothing else. It is a precision
signal only.** An operator can confirm or reject only what alerted, and what alerts is
what the current model and thresholds were already most sure about. A recall failure --
the fall nobody was alerted to, the intrusion under the detection threshold -- never
produces a row here, so the absence of rejected alerts is not evidence of a low miss
rate, and a "99% confirmed" week can coexist with a model that misses half of
everything. **Never read this store as a miss-rate instrument.** The recall instrument
is the statistical-anomaly mining layer, deliberately kept separate; a survey of this
log answers "of what we showed operators, how much was real", and no other question.

---------------------------------------------------------------------------
STORAGE, AND WHY IT IS A DIRECTORY OF JSONL AND NOT A DATABASE

One file per UTC day under a root directory, `<root>/YYYY-MM-DD.jsonl`, one JSON object
per line. UTC because the site corpus's own filenames are UTC and the stores are UTC+8
-- `events.clip_start_from_name` documents what mixing those cost -- so every timestamp
in a row carries its offset and the *filing* date is the one zone nobody has to guess.
JSONL because the consumer is a training-data pipeline and a person with grep, in that
order, and a database dependency would make the lake's first brick the only artefact in
the project that cannot be read without a server.

The log is **append-only**. A disposition does not mutate the alert row it judges; it is
its own row, joined by `alert_id`, so a re-review is a second row and the history of
minds changing is itself data. `current_dispositions` does the fold; last verdict wins.

Reads are corruption-tolerant and never silently lossy: a line that does not parse is
either raised on (the default) or skipped **and counted** into a list the caller
supplied. A reader that quietly drops rows would turn a torn write into missing
training data with no error, which is the failure mode this project ranks worst.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..analytics.events import SecurityEvent

SCHEMA_VERSION = 1

# The full vocabulary. "unreviewed" is recordable on purpose: an operator retracting a
# verdict is a legitimate act and an append-only log records it as one more row rather
# than by editing history.
DISPOSITIONS = ("unreviewed", "confirmed", "rejected")


class CorruptLogError(RuntimeError):
    """A log line that could not be read, raised only when the caller did not opt to count."""


@dataclass(frozen=True)
class CorruptLine:
    """One unreadable line: where it is and why it failed, so nothing is dropped silently."""

    path: str
    line_no: int
    reason: str


@dataclass(frozen=True)
class Disposition:
    """An operator's verdict on one alert: what, who, when, and optionally why."""

    status: str
    by: str | None = None
    at: str | None = None  # ISO-8601 with offset
    reason: str | None = None


UNREVIEWED = Disposition("unreviewed")


@dataclass(frozen=True)
class AlertRecord:
    """One alert as it was raised, before any human has judged it.

    ``event`` is the event row **verbatim** (`SecurityEvent.as_row()` output):
    `basis`/`value`/`threshold`/`type` untouched, because a disposition is a label on
    what the operator was actually shown, and a log that normalises the row is a log of
    something else. ``frame_ref`` is the pullable footage reference -- a clip path with
    the row's frame offsets, or a stream wall-clock time -- and an alert with neither is
    refused at write time: a verdict nobody can re-check against pixels is not a label.
    """

    alert_id: str
    camera: str
    logged_at: str  # ISO-8601 with offset, when this row was filed
    frame_ref: dict[str, Any]  # clip / frame_start / frame_end / stream_time
    event: dict[str, Any]  # the event row verbatim
    model: dict[str, Any]  # checkpoint / git / config_hash, see model_identity()
    calib_version: str | None  # hash of the camera's calib.json, None = uncalibrated
    schema_version: int = SCHEMA_VERSION
    kind: str = "alert"

    def as_row(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "alert_id": self.alert_id,
            "camera": self.camera,
            "logged_at": self.logged_at,
            "frame_ref": dict(self.frame_ref),
            "event": dict(self.event),
            "model": dict(self.model),
            "calib_version": self.calib_version,
        }


@dataclass(frozen=True)
class DispositionRecord:
    """One operator verdict, appended after (never into) the alert row it judges."""

    alert_id: str
    status: str
    by: str
    at: str  # ISO-8601 with offset
    reason: str | None = None
    schema_version: int = SCHEMA_VERSION
    kind: str = "disposition"

    def as_row(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "alert_id": self.alert_id,
            "status": self.status,
            "by": self.by,
            "at": self.at,
            "reason": self.reason,
        }


Record = AlertRecord | DispositionRecord


# ----------------------------------------------------------------- identity helpers


def _sha256(data: bytes) -> str:
    # Same shape as data/fingerprint.py's digest, so the two provenance records read
    # alike: "sha256:" plus the first 32 hex characters.
    return f"sha256:{hashlib.sha256(data).hexdigest()[:32]}"


def file_hash(path: str | Path) -> str:
    """Content hash of a small file (a calib.json, a config). Raises if it is missing.

    Contents rather than path-and-size, unlike `fingerprint_dir`: that trade was about
    not hashing gigabytes per run, and a calibration file is a kilobyte whose *content*
    is exactly what "which calibration was live" means.
    """
    return _sha256(Path(path).read_bytes())


def config_hash(cfg: Mapping[str, Any] | str | Path) -> str:
    """Stable hash of a resolved config: a mapping, or a path to the file itself."""
    if isinstance(cfg, (str, Path)):
        return file_hash(cfg)
    canon = json.dumps(dict(cfg), sort_keys=True, ensure_ascii=False, default=str)
    return _sha256(canon.encode("utf-8"))


def model_identity(
    checkpoint: str | Path,
    config: Mapping[str, Any] | str | Path | None = None,
    repo: Path | None = None,
) -> dict[str, Any]:
    """checkpoint path + git state + config hash, the same identity meta.json records.

    Reuses `utils.runmeta.git_state` rather than re-deriving it, so a disposition row
    and a training run name their code the same way. The import is deferred because
    `runmeta` imports torch at module level and this module runs at the serving edge,
    where a log writer that needs torch on the box would be a dependency nobody asked
    for -- pass a prebuilt dict to `record_alert` on such a box instead.
    """
    from ..utils.runmeta import git_state

    return {
        "checkpoint": str(checkpoint),
        "git": git_state(repo),
        "config_hash": None if config is None else config_hash(config),
    }


# ------------------------------------------------------------------ storage plumbing


def day_path(root: str | Path, when: datetime) -> Path:
    """`<root>/YYYY-MM-DD.jsonl`, filed under the UTC date of ``when``."""
    if when.tzinfo is None:
        raise ValueError(
            f"{when!r} has no timezone; a naive time cannot be filed under a UTC date "
            "without guessing, and events.clip_start_from_name records what guessing "
            "cost the last time"
        )
    return Path(root) / f"{when.astimezone(timezone.utc):%Y-%m-%d}.jsonl"


def append_row(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append one JSON line, atomically enough to survive concurrent writers and kills.

    The whole serialised line goes down in **one os.write to an O_APPEND descriptor**,
    which local filesystems apply as a single append, so two processes logging at once
    interleave whole lines rather than bytes. (Like `runmeta._lock_holder`'s pid test,
    this is exact for the deployment -- one box, local disk -- and would need a real
    lock on NFS.) fsync before returning, because a disposition acknowledged to an
    operator and then lost to a power cut is a label someone believes exists.

    A previous writer killed mid-line leaves a tail with no newline; appending straight
    after it would weld two rows into one corrupt line, losing *both*. So a torn tail
    gets a newline first: the partial line stays its own, countable, corruption and the
    new row lands whole. When two writers race that repair, the file gains a blank
    line, which the reader skips as the deliberate residue of this rule.
    """
    line = json.dumps(dict(row), ensure_ascii=False) + "\n"
    data = line.encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_RDWR rather than O_WRONLY because the torn-tail probe below preads the last
    # byte, and pread on a write-only descriptor is EBADF. O_APPEND still pins every
    # write to the end.
    fd = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        size = os.fstat(fd).st_size
        if size > 0 and os.pread(fd, 1, size - 1) != b"\n":
            data = b"\n" + data
        written = os.write(fd, data)
        if written != len(data):
            raise OSError(f"short write to {path}: {written} of {len(data)} bytes")
        os.fsync(fd)
    finally:
        os.close(fd)


def parse_row(row: Mapping[str, Any]) -> Record:
    """One JSON object back into a typed record. Raises ValueError on anything else."""
    if not isinstance(row, Mapping):
        raise ValueError(f"a log line must hold a JSON object, got {type(row).__name__}")
    version = row.get("schema_version")
    if not isinstance(version, int) or version < 1:
        raise ValueError(f"missing or invalid schema_version: {version!r}")
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {version} is newer than this reader ({SCHEMA_VERSION}); "
            "refusing to half-read it"
        )
    kind = row.get("kind")
    try:
        if kind == "alert":
            return AlertRecord(
                alert_id=str(row["alert_id"]),
                camera=str(row["camera"]),
                logged_at=str(row["logged_at"]),
                frame_ref=dict(row["frame_ref"]),
                event=dict(row["event"]),
                model=dict(row["model"]),
                calib_version=row.get("calib_version"),
                schema_version=version,
            )
        if kind == "disposition":
            status = str(row["status"])
            if status not in DISPOSITIONS:
                raise ValueError(f"unknown disposition status {status!r}")
            return DispositionRecord(
                alert_id=str(row["alert_id"]),
                status=status,
                by=str(row["by"]),
                at=str(row["at"]),
                reason=row.get("reason"),
                schema_version=version,
            )
    except KeyError as missing:
        raise ValueError(f"{kind} row is missing field {missing}") from missing
    raise ValueError(f"unknown record kind {kind!r}")


# ----------------------------------------------------------------------------- API


def _require(row: Mapping[str, Any], keys: tuple[str, ...], what: str) -> None:
    missing = [k for k in keys if k not in row]
    if missing:
        raise ValueError(f"{what} is missing {', '.join(missing)}; refusing to log it")


def record_alert(
    root: str | Path,
    event: SecurityEvent | Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    calib: str | Path | None = None,
    clip: str | Path | None = None,
    alert_id: str | None = None,
    now: datetime | None = None,
) -> AlertRecord:
    """File one alert, unreviewed, into the day's log. Returns the record as written.

    ``event`` is a `SecurityEvent` or an already-flat row from `as_row()`; either way
    the row lands verbatim. ``model`` is `model_identity()`'s dict (or an equivalent
    prebuilt one -- all three keys are required, because a disposition with no model
    identity cannot be filtered out of a training set when that model is superseded).
    ``calib`` is the camera's calib.json path (hashed here) or a precomputed hash
    string; None records honestly that the camera was uncalibrated at alert time.

    Refused outright: an event row with no ``basis`` (a number whose production nobody
    can name -- the standing rule), and an alert with neither a ``clip`` nor a stream
    wall-clock time, because footage nobody can pull cannot be reviewed and the
    resulting "verdict" would label the operator's imagination.
    """
    row: dict[str, Any] = dict(event) if isinstance(event, Mapping) else event.as_row()
    _require(row, ("type", "camera", "basis", "value", "threshold"), "event row")
    if not row["basis"]:
        raise ValueError(
            "event row has an empty basis: nothing names the instrument that measured "
            "it, and an operator cannot judge a number whose production nobody can name"
        )
    stream_time = row.get("started_at")
    if clip is None and stream_time is None:
        raise ValueError(
            "alert carries neither a clip path nor a stream wall-clock time "
            "(event started_at is None); footage nobody can pull cannot be reviewed. "
            "Pass clip=..., or stamp the event via events.with_clip_start first."
        )
    _require(model, ("checkpoint", "git", "config_hash"), "model identity")

    when = now if now is not None else datetime.now(timezone.utc)
    if when.tzinfo is None:
        raise ValueError("now must be timezone-aware; this log never files naive times")
    calib_version = calib if calib is None or isinstance(calib, str) else file_hash(calib)
    record = AlertRecord(
        alert_id=alert_id if alert_id is not None else uuid.uuid4().hex,
        camera=str(row["camera"]),
        logged_at=when.isoformat(),
        frame_ref={
            "clip": None if clip is None else str(clip),
            "frame_start": row.get("frame_start"),
            "frame_end": row.get("frame_end"),
            "stream_time": stream_time,
        },
        event=row,
        model=dict(model),
        calib_version=calib_version,
    )
    append_row(day_path(root, when), record.as_row())
    return record


def record_disposition(
    root: str | Path,
    alert_id: str,
    status: str,
    *,
    by: str,
    reason: str | None = None,
    at: datetime | None = None,
) -> DispositionRecord:
    """Append one operator verdict on a previously recorded alert.

    An unknown ``alert_id`` is an error, not a row: a verdict that attaches to nothing
    would sit in the lake looking like a label until a join silently dropped it, and
    the caller holding a wrong id is a bug worth hearing about today.
    """
    if status not in DISPOSITIONS:
        raise ValueError(f"unknown disposition {status!r}; known: {', '.join(DISPOSITIONS)}")
    if not by:
        raise ValueError(
            "a disposition needs a reviewer (by=...); an anonymous verdict cannot be "
            "audited and this log exists to be audited"
        )
    corrupt: list[CorruptLine] = []
    known = {r.alert_id for r in iter_records(root, kinds=("alert",), corrupt=corrupt)}
    if alert_id not in known:
        skipped = (
            f" ({len(corrupt)} corrupt lines were skipped while looking)" if corrupt else ""
        )
        raise KeyError(
            f"alert_id {alert_id!r} is not in the log under {root}{skipped}; a "
            "disposition must attach to a recorded alert"
        )
    when = at if at is not None else datetime.now(timezone.utc)
    if when.tzinfo is None:
        raise ValueError("at must be timezone-aware; this log never files naive times")
    record = DispositionRecord(
        alert_id=alert_id, status=status, by=by, at=when.isoformat(), reason=reason
    )
    append_row(day_path(root, when), record.as_row())
    return record


def iter_records(
    root: str | Path,
    *,
    camera: str | None = None,
    alert_id: str | None = None,
    kinds: tuple[str, ...] | None = None,
    corrupt: list[CorruptLine] | None = None,
) -> Iterator[Record]:
    """Yield typed records from every day file under ``root``, oldest day first.

    Corruption is never silent: a line that does not parse raises `CorruptLogError` by
    default, or -- when the caller passes ``corrupt=[]`` -- is skipped and appended to
    that list with its file, line number and reason. Blank lines alone are passed over
    without comment, because `append_row`'s torn-tail repair legitimately leaves them.

    The ``camera`` filter matches alert rows only; disposition rows carry no camera, so
    setting it excludes them -- join with `current_dispositions` over an unfiltered
    read when you need verdicts per camera.
    """
    root = Path(root)
    if not root.is_dir():
        return
    for path in sorted(root.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                text = line.strip()
                if not text:
                    continue
                try:
                    record = parse_row(json.loads(text))
                except (ValueError, TypeError) as err:  # json errors are ValueErrors
                    if corrupt is None:
                        raise CorruptLogError(
                            f"{path.name}:{line_no}: {err}. Pass corrupt=[] to skip "
                            "and count unreadable lines instead of stopping on them."
                        ) from err
                    corrupt.append(CorruptLine(str(path), line_no, str(err)))
                    continue
                if kinds is not None and record.kind not in kinds:
                    continue
                if alert_id is not None and record.alert_id != alert_id:
                    continue
                if camera is not None and getattr(record, "camera", None) != camera:
                    continue
                yield record


def current_dispositions(records: Iterable[Record]) -> dict[str, Disposition]:
    """Fold an ordered read into each alert's current verdict. Unreviewed by default.

    Later rows win, which in file order means the most recent verdict -- so a
    confirm-then-retract reads as unreviewed, with the history still in the log.
    Dispositions whose alert fell outside the caller's filters are dropped here rather
    than invented as alerts.
    """
    seen: set[str] = set()
    verdicts: dict[str, Disposition] = {}
    for record in records:
        if isinstance(record, AlertRecord):
            seen.add(record.alert_id)
            verdicts.setdefault(record.alert_id, UNREVIEWED)
        else:
            verdicts[record.alert_id] = Disposition(
                status=record.status, by=record.by, at=record.at, reason=record.reason
            )
    return {aid: verdicts[aid] for aid in seen}

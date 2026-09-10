#!/usr/bin/env python3
"""The grading surface: an operator reads the day's alerts and files a verdict on each.

    python deploy/retail-security/review_server.py \\
        --root runs/serve_pilot01/dispositions --by "store manager" --port 8765

docs/PLAN.md section 4.5 calls the operator's verdicts the human test set -- free,
in-distribution, accumulating, and the only route to the success number in section 1 --
and section 9.2 records that none had been taken. This is the first surface a person can
take one on. It is deliberately small: the standard library's HTTP server, one page per
day, a form per alert, and every verdict written through `dispositions.record_disposition`
so the store keeps the same append-only shape whether a row came from here or from a
script. Nothing here is the alarm UI; it is the labelling tool the data engine starts
from.

---------------------------------------------------------------------------
WHAT THE REVIEWER SEES, AND WHY IT IS DRAWN FROM THE CALIBRATION

An alert row carries no image and no box (`serving/dispositions.py` -- `frame_ref` is a
pointer, deliberately). The frame is pulled from the clip the row points into, at the
crossing frame, and what is drawn on it comes from `camera.json` and the event alone:

* the zone's polygon, projected floor -> calibrated pixels -> the frame's pixels -> through
  the lens, so the reviewer sees the region the rule fired on;
* the shopper, as a marker at the floor position the live event recorded at the crossing
  (`extra.x_m / z_m`), with a box of a person's height above it from
  `geometry.pixel_row_at_height`; the head and shoulders of that box are **blurred before
  the image is encoded**, by the same `utils/face_blur.py` arithmetic the published
  figures use. A reviewer grades a posture and a position, not a face -- section 4.6.

The blur is by calibration, not by a detector, and that is a known limit: a second person
in the frame is not blurred by it. This server binds to localhost and serves an operator
inside the store; it is not a publishing path. Publishing goes through the audit gate in
front of `assets/`, which this does not touch.

The pilot loops its clips, so a frame index past the clip's end wraps -- `frame_time`
takes the modulus of the clip's duration, and says so on the page.

---------------------------------------------------------------------------
A ROW IS DRAWN AGAINST THE CALIBRATION IT WAS FILED UNDER, OR NOT AT ALL

Every alert row carries `calib_version`, the hash of the camera.json the geometry ran
under. On 2026-09-10 the first fleet re-read found a loitering event naming `fixture_04`
whose floor points sat in today's `fixture_03` every frame: another session had rewritten
the fleet's camera files that afternoon and the fixtures were renumbered, and the row's
hash was the only thing that said so. So this page compares the hash on the row with the
file it would draw from, and when they differ it draws **no zone and no marker**, says
why, and still lets the reviewer grade the raw frame -- a polygon from a different
calibration is not "roughly right", it is a different claim about the floor.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import io
import json
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from syncai_hydranet.geometry.camera_json import CameraFile  # noqa: E402
from syncai_hydranet.geometry.ground import (  # noqa: E402
    distort_points,
    ground_to_pixel,
    pixel_row_at_height,
)
from syncai_hydranet.serving.dispositions import (  # noqa: E402
    AlertRecord,
    current_dispositions,
    file_hash,
    iter_records,
    record_disposition,
)
from syncai_hydranet.utils.face_blur import blur_region  # noqa: E402

PERSON_HEIGHT_M = 1.8  # the box drawn above a floor point; a blur that is too tall is free
PERSON_HALF_WIDTH_M = 0.35
VERDICTS = ("confirmed", "rejected")


# ------------------------------------------------------------------ pure parts


def clip_duration_s(clip: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(clip),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(json.loads(out.stdout)["format"]["duration"])


def event_fps(event: dict) -> float:
    """The row carries frames and seconds, not fps; the ratio is exact by construction."""
    frames = int(event["frame_end"]) - int(event["frame_start"]) + 1
    return frames / float(event["seconds"])


def frame_time(event: dict, duration_s: float | None) -> tuple[float, bool]:
    """Seconds into the clip of the crossing frame, and whether it wrapped (a looped clip)."""
    t = int(event["frame_end"]) / event_fps(event)
    if duration_s is not None and duration_s > 0 and t >= duration_s:
        return t % duration_s, True
    return t, False


def to_frame_pixels(pts_m: np.ndarray, cam_file: CameraFile, frame_wh: tuple[int, int]):
    """Floor metres -> the frame's own (distorted) pixels, via the calibrated frame."""
    u, v, depth = ground_to_pixel(pts_m[:, 0], pts_m[:, 1], cam_file.camera, cam_file.plane)
    pts = np.stack([u, v], axis=1)
    if cam_file.lens is not None and abs(cam_file.lens.k1) > 1e-12:
        lens = cam_file.lens
        pts = distort_points(pts, lens.k1, lens.centre_px, lens.radius_px)
    w, h = cam_file.image_size_px
    scale = np.array([frame_wh[0] / w, frame_wh[1] / h])
    return pts * scale, depth


def person_box(event: dict, cam_file: CameraFile, frame_wh: tuple[int, int]):
    """A person-height box over the floor position the event recorded, in frame pixels.

    None when the event has no position (an occupancy row) or the geometry refuses it.
    """
    # `SecurityEvent.as_row` flattens `extra` into the row, so a filed row carries x_m
    # and z_m at top level; an unflattened event still carries them under `extra`.
    extra = event.get("extra") or {}
    x, z = event.get("x_m", extra.get("x_m")), event.get("z_m", extra.get("z_m"))
    if x is None or z is None or not np.isfinite([x, z]).all():
        return None
    feet, depth = to_frame_pixels(
        np.array([[x - PERSON_HALF_WIDTH_M, z], [x + PERSON_HALF_WIDTH_M, z]]),
        cam_file,
        frame_wh,
    )
    if not np.isfinite(feet).all() or (depth <= 0).any():
        return None
    head_v = pixel_row_at_height(x, z, PERSON_HEIGHT_M, cam_file.camera, cam_file.plane)
    if not np.isfinite(head_v):
        return None
    head_v *= frame_wh[1] / cam_file.image_size_px[1]
    x0, x1 = float(feet[:, 0].min()), float(feet[:, 0].max())
    y1 = float(feet[:, 1].mean())
    # A head above the frame's top row is clipped to it: the blur covers what is visible.
    return x0, max(0.0, min(head_v, y1 - 1.0)), x1, y1


def zone_outline(event: dict, cam_file: CameraFile, frame_wh: tuple[int, int]):
    name = event.get("zone")
    for z in cam_file.zones:
        if z.name == name and z.kind != "entrance_line":
            pts, depth = to_frame_pixels(np.asarray(z.points_m, float), cam_file, frame_wh)
            if np.isfinite(pts).all() and (depth > 0).all():
                return [tuple(map(float, p)) for p in pts]
    return None


def draw_review_frame(frame: Image.Image, event: dict, cam_file: CameraFile) -> Image.Image:
    """Blur the person, then draw the zone and the marker. Blur first: the drawing is on top."""
    img = frame.convert("RGB")
    box = person_box(event, cam_file, img.size)
    if box is not None:
        blur_region(img, *box)
    d = ImageDraw.Draw(img)
    outline = zone_outline(event, cam_file, img.size)
    if outline:
        d.polygon(outline, outline=(255, 200, 0), width=3)
    if box is not None:
        x0, y0, x1, y1 = box
        d.rectangle([x0, y0, x1, y1], outline=(230, 25, 75), width=3)
        d.ellipse([(x0 + x1) / 2 - 6, y1 - 6, (x0 + x1) / 2 + 6, y1 + 6], fill=(230, 25, 75))
    return img


def extract_frame(clip: Path, t_s: float, width: int = 960) -> Image.Image:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t_s:.3f}", "-i", str(clip), "-frames:v", "1",
         "-vf", f"scale={width}:-2", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True,
        check=True,
    )  # fmt: skip
    return Image.open(io.BytesIO(out.stdout))


# ------------------------------------------------------------------ the store view


class Store:
    def __init__(self, root: Path, commission: Path, by: str) -> None:
        self.root, self.commission, self.by = root, commission, by
        self._cams: dict[str, CameraFile | None] = {}
        self._durations: dict[str, float | None] = {}

    def camera_file(self, camera: str) -> CameraFile | None:
        if camera not in self._cams:
            p = self.commission / f"{camera}.camera.json"
            self._cams[camera] = CameraFile.load(p) if p.is_file() else None
        return self._cams[camera]

    def calibration_matches(self, rec: AlertRecord) -> bool | None:
        """True when the file on disk is the one the row was filed under; None if no file."""
        p = self.commission / f"{rec.camera}.camera.json"
        if not p.is_file() or rec.calib_version is None:
            return None
        return file_hash(p) == rec.calib_version

    def duration(self, clip: Path) -> float | None:
        key = str(clip)
        if key not in self._durations:
            try:
                self._durations[key] = clip_duration_s(clip)
            except (subprocess.CalledProcessError, FileNotFoundError, KeyError, ValueError):
                self._durations[key] = None
        return self._durations[key]

    def alerts(self) -> list[tuple[AlertRecord, str]]:
        """Every alert with its current verdict, newest first."""
        records = list(iter_records(self.root))
        verdicts = current_dispositions(records)
        out = []
        for r in records:
            if isinstance(r, AlertRecord):
                v = verdicts.get(r.alert_id)
                out.append((r, "unreviewed" if v is None else v.status))
        out.sort(key=lambda t: t[0].logged_at, reverse=True)
        return out

    def alert(self, alert_id: str) -> AlertRecord | None:
        for r in iter_records(self.root, alert_id=alert_id, kinds=("alert",)):
            if isinstance(r, AlertRecord):
                return r
        return None

    def review_image(self, rec: AlertRecord) -> bytes | None:
        clip_ref = rec.frame_ref.get("clip")
        cam_file = self.camera_file(rec.camera)
        if not clip_ref:
            return None
        clip = ROOT / clip_ref
        if not clip.is_file():
            return None
        t, _wrapped = frame_time(rec.event, self.duration(clip))
        try:
            frame = extract_frame(clip, t)
        except subprocess.CalledProcessError:
            return None
        draw = cam_file is not None and self.calibration_matches(rec) is True
        img = draw_review_frame(frame, rec.event, cam_file) if draw else frame.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        return buf.getvalue()

    def file_verdict(self, alert_id: str, status: str, reason: str | None) -> None:
        if status not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {status!r}")
        record_disposition(self.root, alert_id, status, by=self.by, reason=reason or None)


# ------------------------------------------------------------------ the page


ROW = """<div class=a><div><img src="/frame/{aid}.jpg" alt="frame"></div><div>
<b>{type}</b> &middot; {camera} &middot; zone {zone}<br>
{value} vs {threshold} &middot; <small>{basis}</small><br>
<small>{started} &middot; frames {f0}-{f1} &middot; {clip}{wrapped}</small><br>
<small>alert {aid} &middot; calib {calib} &middot; model {model}</small>
<p>verdict: <b>{verdict}</b></p>
<form method=post action="/verdict"><input type=hidden name=alert_id value="{aid}">
<input name=reason placeholder="reason (optional)" size=32>
<button name=status value=confirmed>confirm</button>
<button name=status value=rejected>reject</button>
<button name=status value=rejected
 onclick="this.form.reason.value=this.form.reason.value||'staff'">reject: staff</button>
</form></div></div>"""

STYLE = """<style>body{font:14px system-ui;margin:1.5em;max-width:1100px}
.a{display:grid;grid-template-columns:480px 1fr;gap:1em;border-top:1px solid #ccc;padding:1em 0}
img{width:480px;background:#eee}form{display:inline}button{margin-right:.5em}
small{color:#666}</style>"""


def render_index(store: Store, show: str = "unreviewed") -> str:
    every = store.alerts()
    rows = every if show == "all" else [(r, v) for r, v in every if v == show]
    counts: dict[str, int] = {}
    for _, v in every:
        counts[v] = counts.get(v, 0) + 1
    e = html.escape
    links = " &middot; ".join(
        f'<a href="/?show={e(k)}">{e(k)} {n}</a>' for k, n in sorted(counts.items())
    )
    parts = [
        "<!doctype html><meta charset=utf-8><title>alert review</title>",
        STYLE,
        f"<h1>alert review &middot; {e(str(store.root))}</h1>",
        f'<p>{links} &middot; <a href="/?show=all">all</a> &middot; '
        f"verdicts filed as <b>{e(store.by)}</b></p>",
    ]
    if not rows:
        parts.append(f"<p>nothing {e(show)}.</p>")
    for rec, verdict in rows:
        ev = rec.event
        started = ev.get("started_at") or rec.frame_ref.get("stream_time") or rec.logged_at
        clip = rec.frame_ref.get("clip") or "(stream)"
        wrapped = ""
        if rec.frame_ref.get("clip"):
            _, w = frame_time(ev, store.duration(ROOT / rec.frame_ref["clip"]))
            wrapped = " <small>(looped clip: frame index wrapped)</small>" if w else ""
        match = store.calibration_matches(rec)
        if match is False:
            wrapped += (
                " <b>calibration changed since this alert was filed:</b> "
                "<small>zone and marker not drawn; the row's zone name may not be "
                "today's</small>"
            )
        elif match is None:
            wrapped += " <small>(no camera file: raw frame only)</small>"
        parts.append(
            ROW.format(
                aid=e(rec.alert_id),
                type=e(str(ev["type"])),
                camera=e(rec.camera),
                zone=e(str(ev.get("zone"))),
                value=e(str(ev.get("value"))),
                threshold=e(str(ev.get("threshold"))),
                basis=e(str(ev.get("basis"))),
                started=e(str(started)),
                f0=ev.get("frame_start"),
                f1=ev.get("frame_end"),
                clip=e(str(clip)),
                wrapped=wrapped,
                calib=e(str(rec.calib_version))[:19],
                model=e(str(rec.model.get("checkpoint"))),
                verdict=e(verdict),
            )
        )
    return "\n".join(parts)


class Handler(BaseHTTPRequestHandler):
    store: Store  # set on the class by `serve`

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        if url.path == "/":
            show = parse_qs(url.query).get("show", ["unreviewed"])[0]
            self._send(200, render_index(self.store, show).encode(), "text/html; charset=utf-8")
            return
        if url.path.startswith("/frame/") and url.path.endswith(".jpg"):
            rec = self.store.alert(url.path[len("/frame/") : -len(".jpg")])
            data = self.store.review_image(rec) if rec else None
            if data is None:
                self._send(404, b"no frame", "text/plain")
                return
            self._send(200, data, "image/jpeg")
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/verdict":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", "0"))
        form = parse_qs(self.rfile.read(n).decode())
        try:
            self.store.file_verdict(
                form["alert_id"][0], form["status"][0], form.get("reason", [""])[0].strip()
            )
        except (KeyError, ValueError) as exc:
            self._send(400, str(exc).encode(), "text/plain")
            return
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def log_message(self, fmt, *args):  # quieter than the default, which logs every image
        if "/frame/" not in (args[0] if args else ""):
            super().log_message(fmt, *args)


def serve(store: Store, host: str, port: int) -> None:
    Handler.store = store
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"review sheet on http://{host}:{port}/  root {store.root}  by {store.by!r}")
    with contextlib.suppress(KeyboardInterrupt):
        httpd.serve_forever()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", type=Path, required=True, help="the disposition store")
    ap.add_argument("--commission", type=Path, default=ROOT / "runs/commission01")
    ap.add_argument("--by", required=True, help="who is grading; every verdict names them")
    ap.add_argument("--host", default="127.0.0.1", help="localhost on purpose; see the header")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    if not args.root.is_dir():
        raise SystemExit(f"{args.root} is not a directory; nothing to review")
    serve(Store(args.root, args.commission, args.by), args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

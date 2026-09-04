"""The zone-drawing tool PLAN step 2 has been missing: click the floor, get metres.

    uv run python tools/commissioning/zone_draw.py --camera Taichung-cam01
    # open runs/zone_draw01/Taichung-cam01.html in a browser, draw, press Copy JSON
    uv run python tools/commissioning/zone_draw.py --camera Taichung-cam01 --apply drawn.json

---------------------------------------------------------------------------
WHY A HUMAN DRAWS THIS, AFTER TWO AUTOMATIC ATTEMPTS

`footprints_from_masks.py` records both, with what each measured. The input is not the
problem -- the 2026-08-25 masks segment this shop's furniture cleanly. The problem is
that **one camera cannot see behind a counter**. Every automatic footprint therefore
assumes its far edge, and an assumed edge on the most important fixture in the room is
not something a confirm pass can repair: on Taichung-cam01 the main service counter
produced no footprint at all under one construction and a strip of open floor under the
other.

A person looking at the frame knows the counter is there and knows roughly how deep it
is. That is the missing measurement, it costs a minute per camera, and nothing else in
this pipeline can supply it.

---------------------------------------------------------------------------
TWO WAYS TO DRAW, BECAUSE TWO THINGS ARE BEING SAID

* **region** -- click a closed polygon on visible floor. Every vertex projects through
  `camera.json` independently, so the shape is metres the moment it is drawn. Right for
  an aisle, a queue area, a forbidden square: things whose whole extent is on the floor
  and in view.
* **base + depth** -- click along the line where a fixture meets the floor, then type how
  deep it is. The tool offsets that line by the stated depth, away from the camera. Right
  for counters and tables, where the near edge is visible and the far edge never is.
  The depth is the human's measurement and it is written into the zone's provenance, so a
  later reader sees that the far edge was stated rather than seen.

The page is a **local file with the frame embedded**, not a published page: it holds a
real frame of a real shop, and that is not something to put on a URL to save a click.

---------------------------------------------------------------------------
WHAT COMES BACK

The browser's Copy JSON gives pixel coordinates in the frame's own resolution plus a
name, a kind and (for a base) a depth. `--apply` undistorts, projects, and writes zones
into `camera.json` -- refusing a kind outside `ZONE_KINDS` and a polygon whose points do
not all land on the floor, because a vertex above the horizon comes back NaN and a zone
with a NaN corner tests False for every point in it, forever, silently.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import json
from pathlib import Path

import numpy as np

from syncai_hydranet.geometry.camera_json import ZONE_KINDS, CameraFile, Zone

# The repo root, derived rather than written out: every one of these 26 tools had it
# as an absolute path, so a second checkout ran against the first one's `runs/` and
# any machine but this one failed at import with a path and no reason. Two levels up
# from `tools/<group>/<tool>.py`, and `tests/test_no_absolute_sys_path.py` keeps it so.
ROOT = Path(__file__).resolve().parents[2]
COMMISSIONED = ROOT / "runs/commission01"
OUT = ROOT / "runs/zone_draw01"
DRAWABLE_KINDS = sorted(ZONE_KINDS - {"walkable"})

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>__CAMERA__ &mdash; zone draw</title>
<style>
 body{margin:0;background:#111;color:#eee;font:14px/1.5 system-ui,sans-serif}
 header{padding:8px 12px;background:#000;position:sticky;top:0;z-index:2}
 #wrap{position:relative;display:inline-block}
 canvas{display:block;cursor:crosshair}
 aside{position:fixed;right:0;top:0;width:330px;height:100vh;overflow:auto;
       background:#181818;padding:12px;box-sizing:border-box;border-left:1px solid #333}
 main{margin-right:330px}
 button{background:#2b6;color:#000;border:0;padding:6px 10px;border-radius:4px;
        font-weight:600;cursor:pointer;margin:2px 0}
 button.g{background:#444;color:#eee}
 input,select{width:100%;box-sizing:border-box;background:#222;color:#eee;
              border:1px solid #444;border-radius:4px;padding:5px;margin:2px 0 8px}
 li{margin:4px 0;border-bottom:1px solid #2a2a2a;padding-bottom:4px}
 .k{color:#8bd}
 code{background:#222;padding:1px 4px;border-radius:3px}
</style>
<header><b>__CAMERA__</b> &mdash; click to add points &middot;
 <code>Enter</code> finish &middot; <code>Backspace</code> undo point &middot;
 <code>Esc</code> cancel &middot; existing zones are drawn in blue</header>
<main><div id="wrap"><canvas id="c"></canvas></div></main>
<aside>
 <label>name</label><input id="name" placeholder="front_counter">
 <label>kind</label><select id="kind">__KINDS__</select>
 <label>mode</label><select id="mode">
   <option value="region">region &mdash; closed polygon on visible floor</option>
   <option value="base">base + depth &mdash; the line a fixture meets the floor</option>
 </select>
 <label>depth (m, base mode only)</label><input id="depth" value="0.8">
 <button id="done">finish shape (Enter)</button>
 <button class="g" id="undo">undo point</button>
 <button class="g" id="clear">clear current</button>
 <hr><b>drawn</b><ul id="list"></ul>
 <button id="copy">Copy JSON</button>
 <button class="g" id="pop">remove last</button>
 <p style="color:#888">Paste the JSON back to the session; it is applied with
 <code>zone_draw.py --apply</code>.</p>
</aside>
<script>
const IMG = "__DATAURI__", W = __W__, H = __H__, SCALE = __SCALE__;
const EXISTING = __EXISTING__;
const c = document.getElementById('c'), ctx = c.getContext('2d');
const img = new Image(); let pts = [], shapes = [];
img.onload = () => { c.width = W*SCALE; c.height = H*SCALE; draw(); };
img.src = IMG;
const COLS = ['#ff6040','#5adc82','#ffcd46','#78aaff','#eb78eb','#6ee6e6','#ff9659'];
function draw(){
  ctx.drawImage(img,0,0,c.width,c.height);
  ctx.lineWidth = 3;
  ctx.strokeStyle = '#5ac8ff';
  for (const z of EXISTING){ ring(z.points, true); }
  shapes.forEach((s,i) => { ctx.strokeStyle = COLS[i%COLS.length];
    ring(s.points, s.mode === 'region'); label(s, i, COLS[i%COLS.length]); });
  ctx.strokeStyle = '#fff';
  ring(pts, false);
  ctx.fillStyle = '#fff';
  for (const p of pts) ctx.fillRect(p[0]*SCALE-3, p[1]*SCALE-3, 6, 6);
}
function ring(p, close){
  if (!p.length) return;
  ctx.beginPath(); ctx.moveTo(p[0][0]*SCALE, p[0][1]*SCALE);
  for (const q of p.slice(1)) ctx.lineTo(q[0]*SCALE, q[1]*SCALE);
  if (close) ctx.closePath();
  ctx.stroke();
}
function label(s, i, col){
  const m = s.points.reduce((a,b)=>[a[0]+b[0],a[1]+b[1]],[0,0]).map(v=>v/s.points.length);
  ctx.fillStyle = col; ctx.font = 'bold 18px sans-serif';
  ctx.fillText(`${i}. ${s.name}`, m[0]*SCALE+6, m[1]*SCALE);
}
c.addEventListener('click', e => {
  const r = c.getBoundingClientRect();
  pts.push([(e.clientX-r.left)/SCALE, (e.clientY-r.top)/SCALE]); draw();
});
function finish(){
  const mode = document.getElementById('mode').value;
  if (pts.length < (mode === 'region' ? 3 : 2)) return;
  shapes.push({name: document.getElementById('name').value || `zone_${shapes.length+1}`,
               kind: document.getElementById('kind').value, mode,
               depth_m: parseFloat(document.getElementById('depth').value),
               points: pts});
  pts = []; render(); draw();
}
function render(){
  document.getElementById('list').innerHTML = shapes.map((s,i) =>
    `<li>${i}. <b>${s.name}</b> <span class="k">${s.kind}</span><br>` +
    `${s.mode}${s.mode==='base' ? ' &middot; '+s.depth_m+' m' : ''} &middot; ` +
    `${s.points.length} pts</li>`).join('');
}
document.getElementById('done').onclick = finish;
document.getElementById('undo').onclick = () => { pts.pop(); draw(); };
document.getElementById('clear').onclick = () => { pts = []; draw(); };
document.getElementById('pop').onclick = () => { shapes.pop(); render(); draw(); };
document.getElementById('copy').onclick = () => {
  const out = JSON.stringify({camera: "__CAMERA__", frame_px: [W, H], shapes}, null, 1);
  navigator.clipboard.writeText(out).catch(()=>{});
  const w = window.open(); w.document.write('<pre>'+out.replace(/</g,'&lt;')+'</pre>');
};
addEventListener('keydown', e => {
  if (e.key === 'Enter') finish();
  else if (e.key === 'Backspace') { e.preventDefault(); pts.pop(); draw(); }
  else if (e.key === 'Escape') { pts = []; draw(); }
});
</script>
"""


def build(camera: str, scale: float) -> Path:
    cam_file = CameraFile.load(COMMISSIONED / f"{camera}.camera.json")
    if not cam_file.plate_file:
        raise SystemExit(f"{camera}: camera.json names no plate")
    plate = ROOT / cam_file.plate_file
    data = base64.b64encode(plate.read_bytes()).decode()
    w, h = cam_file.image_size_px
    existing = []
    for z in cam_file.zones:
        pts = _to_px(np.asarray(z.points_m, dtype=float), cam_file)
        existing.append({"name": z.name, "points": [[float(a), float(b)] for a, b in pts]})
    page = (
        PAGE.replace("__CAMERA__", camera)
        .replace("__DATAURI__", f"data:image/png;base64,{data}")
        .replace("__W__", str(w))
        .replace("__H__", str(h))
        .replace("__SCALE__", str(scale))
        .replace("__EXISTING__", json.dumps(existing))
        .replace(
            "__KINDS__",
            "".join(f'<option value="{k}">{k}</option>' for k in DRAWABLE_KINDS),
        )
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{camera}.html"
    out.write_text(page)
    return out


def _to_px(points_m: np.ndarray, cam_file: CameraFile) -> np.ndarray:
    """Floor metres -> pixels on the raw plate the page shows, lens included.

    The page reports clicks in raw pixels and `to_metres` undistorts them; drawing the
    existing zones without putting the lens back would show them a few pixels off the
    floor they describe, and a person redrawing to "fix" that would be correcting the
    overlay rather than the zone.
    """
    from syncai_hydranet.geometry.ground import distort_points, ground_to_pixel

    u, v, _ = ground_to_pixel(points_m[:, 0], points_m[:, 1], cam_file.camera, cam_file.plane)
    px = np.stack([u, v], axis=1)
    lens = cam_file.lens
    if lens is not None:
        px = distort_points(px, lens.k1, lens.centre_px, lens.radius_px)
    return px


def extrude(base_m: np.ndarray, depth: float) -> np.ndarray:
    """A base line offset `depth` metres to the side away from the camera, closed.

    The offset is along the line's own normal rather than along the view ray: where a
    counter's face runs toward the viewer the ray is parallel to it, and offsetting
    radially slides the far edge along the line instead of behind it. The sign is chosen
    once for the whole line, from its midpoint, so a curved base cannot fold into a bowtie
    the way a per-point choice does.
    """
    d = base_m[-1] - base_m[0]
    n = np.array([-d[1], d[0]], dtype=float)
    norm = float(np.hypot(*n))
    if norm < 1e-9:
        raise SystemExit("base line has zero length")
    n /= norm
    mid = base_m[len(base_m) // 2]
    if float(np.dot(n, mid)) < 0:  # the camera sits at the origin, looking toward +z
        n = -n
    return np.concatenate([base_m, (base_m + depth * n)[::-1]])


def apply(camera: str, drawn: Path) -> None:
    payload = json.loads(drawn.read_text())
    if payload.get("camera") != camera:
        raise SystemExit(f"{drawn} was drawn on {payload.get('camera')!r}, not {camera!r}")
    cf_path = COMMISSIONED / f"{camera}.camera.json"
    cam_file = CameraFile.load(cf_path)
    zones = []
    for s in payload["shapes"]:
        if s["kind"] not in ZONE_KINDS:
            raise SystemExit(f"{s['name']}: kind {s['kind']!r} not in {sorted(ZONE_KINDS)}")
        pts = cam_file.ground_points(
            np.asarray(s["points"], dtype=float), above_horizon="raise", what=s["name"]
        )
        if s["mode"] == "base":
            pts = extrude(pts, float(s["depth_m"]))
        elif len(pts) < 3:
            raise SystemExit(f"{s['name']}: a region needs 3 points, got {len(pts)}")
        zones.append(
            Zone(
                s["name"],
                s["kind"],
                tuple((round(float(a), 2), round(float(b), 2)) for a, b in pts),
            )
        )
    keep = tuple(z for z in cam_file.zones if z.kind == "walkable")
    cam_file = dataclasses.replace(cam_file, zones=keep + tuple(zones))
    cam_file.validate()
    cam_file.save(cf_path)
    for z in zones:
        print(f"  {z.name:<18} {z.kind:<14} {len(z.points_m)} pts")
    print(f"{camera}: {len(zones)} drawn zones written ({len(keep)} kept from the mask pass)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True)
    ap.add_argument("--scale", type=float, default=2.0, help="page zoom over the frame")
    ap.add_argument("--apply", type=Path, metavar="DRAWN.JSON")
    args = ap.parse_args()
    if args.apply:
        apply(args.camera, args.apply)
    else:
        print(f"open file://{build(args.camera, args.scale)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

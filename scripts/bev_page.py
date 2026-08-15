#!/usr/bin/env python3
"""The 3D scene page served alongside the live view.

Drawn with plain canvas 2D and about eighty lines of projection maths rather than a 3D
library. That is not minimalism for its own sake: the page is served *from the robot*,
which sits on a network with no route to a CDN, and a viewer that needs the internet to
render a robot's local state is a viewer that fails exactly when it is needed. Nothing
here loads.

What it draws, and why each choice is the honest one:

* **Floor cells** at their measured distance, coloured by traversability. Cells the depth
  sensor never saw are simply absent -- a hole in the floor, not a guess. On a polished
  lobby floor that hole was 10.6% of the walkable pixels, and it is the most important
  thing on the screen.
* **Objects as wireframe cuboids** sized from the depth inside their box. Wireframe, not
  a solid asset: a solid chair model asserts a shape that was never measured, and at 4 m
  on a D435 the extent is already a guess. The wireframe shows the *evidence*.
* **A heading line only when the footprint has one.** A round footprint gets no arrow,
  because a chair's point cloud cannot say which way it faces.
* **A range ring at the configured distance gate**, so "within 5 m" is visible rather
  than implied.
"""

# ruff: noqa: E501 -- this module is one HTML/JS document; wrapping it would hurt it
PAGE = b"""<!doctype html><meta charset=utf-8><title>HydraNet scene</title>
<style>
 body{background:#0a0d12;color:#c9d3e0;font:13px/1.5 system-ui,sans-serif;margin:0}
 canvas{display:block;width:100vw;height:78vh}
 .bar{display:flex;gap:16px;align-items:center;padding:8px 14px;flex-wrap:wrap}
 .k{color:#5d6b7f} b{color:#e8eef7;font-weight:600}
 a{color:#8cf;text-decoration:none;border:1px solid #2a3646;border-radius:4px;padding:2px 8px}
 a:hover{background:#16202c}
 .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
</style>
<canvas id=c></canvas>
<div class=bar id=stats></div>
<div class=bar style="color:#5d6b7f">
  <span><i class=sw style="background:#28dc5a"></i>floor, sensor agrees</span>
  <span><i class=sw style="background:#fac828"></i>caution</span>
  <span><i class=sw style="background:#e0483c"></i>blocked</span>
  <span>gaps in the floor are where the depth sensor returned nothing &mdash; glass,
        mirror, or a polished surface. Not free space.</span>
  <a href="/">2D view</a>
</div>
<script>
const c = document.getElementById('c'), g = c.getContext('2d');
let scene = null, spin = 0.0, drag = null;
let VW = 0, VH = 0;                                // CSS pixels; the backing store is DPR-scaled

// A camera looking down at the scene from behind the robot. Everything is in metres in
// the robot's frame: x right, z forward, height up.
const cam = {height: 3.2, back: 3.0, pitch: 0.42, f: 520};

function project(p) {
  // Rotate about the vertical axis for the orbit, then into the viewing camera.
  const cs = Math.cos(spin), sn = Math.sin(spin);
  const x = p[0] * cs - p[2] * sn, z = p[0] * sn + p[2] * cs;
  const y = p[1];
  const zc = z + cam.back, yc = y - cam.height;
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const zr = zc * cp - yc * sp, yr = zc * sp + yc * cp;
  if (zr < 0.15) return null;                    // behind the eye
  return [VW / 2 + cam.f * x / zr, VH / 2 - cam.f * yr / zr, zr];
}

function poly(pts, fill, stroke) {
  const ps = pts.map(project);
  if (ps.some(p => p === null)) return;
  g.beginPath();
  ps.forEach((p, i) => i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]));
  g.closePath();
  if (fill) { g.fillStyle = fill; g.fill(); }
  if (stroke) { g.strokeStyle = stroke; g.stroke(); }
}

const CELL_COLOURS = {1: 'rgba(40,220,90,.55)', 2: 'rgba(250,200,40,.55)',
                      3: 'rgba(224,72,60,.45)'};

function drawGrid(grid) {
  const {nx, nz, cell_m: cell, x_min, cells} = grid;
  for (let iz = nz - 1; iz >= 0; iz--) {          // far to near, so near overdraws
    for (let ix = 0; ix < nx; ix++) {
      const v = cells[iz * nx + ix];
      if (!v) continue;                            // empty: never seen, draw nothing
      const x = x_min + ix * cell, z = iz * cell;
      poly([[x, 0, z], [x + cell, 0, z], [x + cell, 0, z + cell], [x, 0, z + cell]],
           CELL_COLOURS[v], 'rgba(0,0,0,.18)');
    }
  }
}

function drawRing(r) {                             // the distance gate, on the floor
  g.strokeStyle = 'rgba(140,200,255,.5)'; g.setLineDash([6, 6]); g.beginPath();
  let started = false;
  for (let a = -1.2; a <= 1.2; a += 0.04) {
    const p = project([r * Math.sin(a), 0, r * Math.cos(a)]);
    if (!p) { started = false; continue; }
    started ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]); started = true;
  }
  g.stroke(); g.setLineDash([]);
  const lab = project([0, 0, r]);
  if (lab) { g.fillStyle = 'rgba(140,200,255,.8)'; g.fillText(r.toFixed(0) + ' m', lab[0] + 4, lab[1] - 4); }
}

function drawObject(o) {
  const hw = Math.max(o.width_m, 0.15) / 2, hd = Math.max(o.depth_m, 0.15) / 2, h = Math.max(o.height_m, 0.2);
  const cy = Math.cos(o.yaw_confidence > 0.35 ? o.yaw_rad : 0);
  const sy = Math.sin(o.yaw_confidence > 0.35 ? o.yaw_rad : 0);
  const corner = (dx, dz, y) => [o.x_m + dx * cy - dz * sy, -y, o.z_m + dx * sy + dz * cy];
  const base = [corner(-hw, -hd, 0), corner(hw, -hd, 0), corner(hw, hd, 0), corner(-hw, hd, 0)];
  const top  = base.map(p => [p[0], -h, p[2]]);
  const shade = o.range_m > 4.5 ? .35 : .95;       // beyond a D435's honest range
  const line = `rgba(120,210,255,${shade})`;
  poly(base, 'rgba(120,210,255,.10)', line);
  poly(top, null, line);
  for (let i = 0; i < 4; i++) {
    const a = project(base[i]), b = project(top[i]);
    if (a && b) { g.strokeStyle = line; g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); }
  }
  if (o.yaw_confidence > 0.35) {                   // only when the footprint has an axis
    const a = project(corner(0, 0, h * 0.5)), b = project(corner(hw * 1.6, 0, h * 0.5));
    if (a && b) { g.strokeStyle = 'rgba(255,220,120,.9)'; g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); }
  }
  const lab = project([o.x_m, -h - 0.12, o.z_m]);
  if (lab) {
    g.fillStyle = `rgba(232,238,247,${shade})`; g.font = '12px system-ui';
    g.fillText(`${o.name} ${o.range_m.toFixed(1)}m`, lab[0] - 18, lab[1]);
  }
}

function draw() {
  // Setting canvas.width clears it and resets the transform, so it happens once, first.
  const dpr = devicePixelRatio || 1;
  VW = c.clientWidth; VH = c.clientHeight;
  c.width = VW * dpr; c.height = VH * dpr;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = '#0a0d12'; g.fillRect(0, 0, VW, VH);
  if (!scene) { g.fillStyle = '#5d6b7f'; g.fillText('waiting for a frame...', 20, 30); return; }
  drawGrid(scene.grid);
  drawRing(scene.range_m || 5);
  (scene.objects || []).sort((a, b) => b.z_m - a.z_m).forEach(drawObject);
}

c.addEventListener('pointerdown', e => drag = e.clientX);
addEventListener('pointerup', () => drag = null);
addEventListener('pointermove', e => { if (drag !== null) { spin += (e.clientX - drag) * 0.006; drag = e.clientX; draw(); } });

async function tick() {
  try {
    const r = await fetch('/scene.json');
    scene = await r.json();
    const b = scene.grid.blind_fraction;
    document.getElementById('stats').innerHTML =
      `<span><span class=k>objects</span><b>${(scene.objects || []).length}</b></span>` +
      `<span><span class=k>walkable with no depth</span><b>${(b * 100).toFixed(1)}%</b></span>` +
      `<span><span class=k>range gate</span><b>${(scene.range_m || 5).toFixed(0)} m</b></span>` +
      `<span class=k>drag to orbit</span>`;
    draw();
  } catch (e) { /* the stream restarts; keep the last frame */ }
  setTimeout(tick, 200);
}
addEventListener('resize', draw);
tick();
</script>
"""

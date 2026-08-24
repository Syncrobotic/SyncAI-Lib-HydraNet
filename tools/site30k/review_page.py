#!/usr/bin/env python3
"""Build the acceptance page for the campaign: a per-camera sample of finished frames.

Sampling is per CAMERA and spread over DATES, because that is where this campaign's
risk is. Six of the nine cameras have been looked at exactly once, on one frame, before
the run started; a page that shows twelve consecutive frames of the same camera-date
proves nothing that the first frame did not already prove.

    python tools/site30k/review_page.py --root datasets/site30k_v1 --per-camera 12

Writes review_page.html next to the dataset. Publish it with the Artifact tool.
"""

import argparse
import base64
import io
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_COLORS = {
    "floor": "#3CB44B",
    "wall": "#0082C8",
    "column": "#FFE119",
    "display_table": "#F58230",
    "shelf": "#F032E6",
    "person": "#E6194B",
}
IDS = {"floor": 1, "wall": 2, "column": 3, "display_table": 4, "shelf": 5, "person": 6}


def encode(path: Path, width: int, quality: int) -> str:
    im = Image.open(path).convert("RGB")
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("datasets/site30k_v1"))
    ap.add_argument("--per-camera", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.root / "review_page.html"

    previews = sorted((args.root / "preview").glob("*.jpg"))
    by_cam: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for p in previews:
        m = re.match(r"(.+?)__(\d{8})-\d{6}__\d{4}\.jpg", p.name)
        if m:
            by_cam[m.group(1)][m.group(2)].append(p)

    cameras = []
    for cam in sorted(by_cam):
        dates = sorted(by_cam[cam])
        # one frame per date, walking the dates, until the quota is filled: the spread
        # over days is the point, not the count.
        picks: list[Path] = []
        i = 0
        while len(picks) < args.per_camera and any(len(by_cam[cam][d]) > i for d in dates):
            for d in dates:
                if len(picks) >= args.per_camera:
                    break
                if len(by_cam[cam][d]) > i:
                    picks.append(by_cam[cam][d][i])
            i += 1
        frames = []
        for p in picks:
            name = p.stem
            mask_path = args.root / "masks" / f"{name}.png"
            shares, labelled = {}, None
            if mask_path.exists():
                m = np.asarray(Image.open(mask_path))
                shares = {k: round(100 * float((m == v).mean()), 2) for k, v in IDS.items()}
                labelled = round(100 * float((m != 255).mean()), 1)
            frames.append(
                {
                    "name": name,
                    "date": name.split("__")[1][:8],
                    "labelled": labelled,
                    "shares": shares,
                    "img": encode(p, 1100, 74),
                }
            )
        cameras.append(
            {
                "camera": cam,
                "dates": len(dates),
                "total": sum(len(v) for v in by_cam[cam].values()),
                "frames": frames,
            }
        )

    fails = []
    fpath = args.root / "failures.jsonl"
    if fpath.exists():
        fails = [json.loads(line) for line in fpath.read_text().splitlines() if line.strip()]
    n_masks = len(list((args.root / "masks").glob("*.png")))

    legend = "".join(
        f'<span class="lg"><i style="background:{c}"></i>{k}</span>'
        for k, c in CLASS_COLORS.items()
    )
    body = []
    for cam in cameras:
        cards = []
        for f in cam["frames"]:
            bar = "".join(
                f'<i style="width:{f["shares"].get(k, 0)}%;background:{c}" title="{k} '
                f'{f["shares"].get(k, 0)}%"></i>'
                for k, c in CLASS_COLORS.items()
            )
            unl = max(0.0, 100 - sum(f["shares"].values()))
            cards.append(
                f'<article class="card"><img src="data:image/jpeg;base64,{f["img"]}" '
                f'alt="{f["name"]}" loading="lazy">'
                f'<div class="meta"><div class="row"><span class="pct mono">'
                f"{f['labelled']}%<em>labelled</em></span>"
                f'<span class="date mono">{f["date"]}</span></div>'
                f'<div class="stack">{bar}<i class="unl" style="width:{unl}%"></i></div>'
                f'<div class="fname mono">{f["name"]}</div></div></article>'
            )
        body.append(
            f'<section><div class="cam"><h2>{cam["camera"]}</h2>'
            f'<span class="n">{cam["total"]} preview frames over {cam["dates"]} dates '
            f"&middot; showing {len(cam['frames'])}</span></div>"
            f'<div class="grid">{"".join(cards)}</div></section>'
        )

    fail_rows = "".join(
        f'<tr><td class="mono">{f.get("camera", "?")}</td><td class="mono">'
        f'{f.get("date", "?")}</td><td class="mono">{f.get("rc", "?")}</td>'
        f'<td class="mono small">{(f.get("tail", "") or "")[-160:]}</td></tr>'
        for f in fails[:40]
    )

    fail_section = (
        (
            "<section><div class='cam'><h2>Failed units</h2></div><table>"
            "<tr><th>camera</th><th>date</th><th>rc</th><th>last words</th></tr>"
            + fail_rows
            + "</table></section>"
        )
        if fails
        else ""
    )
    html = f"""<title>Site30k v1 Review</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400&display=swap">
<style>
:root {{ --ground:#0F1216; --panel:#161A21; --line:#282F3A; --ink:#E8ECF3;
        --muted:#96A0B0; --dim:#6C7686; --accent:#F58230; --ok:#3CB44B; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink);
        font-family:"Source Sans 3",system-ui,sans-serif; font-size:15px; line-height:1.55; }}
.mono {{ font-family:"IBM Plex Mono",ui-monospace,monospace;
        font-variant-numeric:tabular-nums; }}
header {{ border-bottom:1px solid var(--line); background:var(--panel);
          padding:22px clamp(16px,4vw,40px); }}
h1 {{ font-family:Archivo,sans-serif; font-size:clamp(22px,3vw,30px); margin:0;
      letter-spacing:-0.02em; }}
.sub {{ color:var(--muted); margin:6px 0 16px; max-width:70ch; }}
.stats {{ display:flex; gap:28px; flex-wrap:wrap; }}
.stat b {{ display:block; font-family:Archivo,sans-serif; font-size:22px; }}
.stat span {{ font-size:11px; letter-spacing:.09em;
        text-transform:uppercase; color:var(--dim); }}
.bar {{ display:flex; gap:20px; flex-wrap:wrap; margin-top:16px; }}
.lg {{ display:inline-flex; align-items:center;
        gap:7px; font-size:12.5px; color:var(--muted); }}
.lg i {{ width:11px; height:11px; border-radius:2px; }}
main {{ max-width:1500px; margin:0 auto; padding:26px clamp(16px,4vw,40px) 70px; }}
section {{ margin-bottom:40px; }}
.cam {{ display:flex; align-items:baseline; gap:14px; border-bottom:1px solid var(--line);
        padding-bottom:8px; margin-bottom:18px; }}
.cam h2 {{ font-family:Archivo,sans-serif; font-size:17px; margin:0; }}
.cam .n {{ font-size:12px; color:var(--dim); text-transform:uppercase; letter-spacing:.08em; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:16px; }}
.card {{ background:var(--panel); border:1px solid var(--line);
        border-radius:5px; overflow:hidden; }}
.card img {{ width:100%; display:block; aspect-ratio:16/9; object-fit:cover; }}
.meta {{ padding:10px 12px 12px; display:flex; flex-direction:column; gap:8px; }}
.row {{ display:flex; justify-content:space-between; align-items:baseline; }}
.pct {{ font-family:Archivo,sans-serif; font-size:15px; }}
.pct em {{ font-style:normal; color:var(--dim); font-size:11px; text-transform:uppercase;
           letter-spacing:.07em; margin-left:5px; }}
.date {{ color:var(--dim); font-size:12px; }}
.stack {{ display:flex; height:7px; border-radius:2px; overflow:hidden; background:#1C222B; }}
.stack i {{ display:block; height:100%; }}
.stack i.unl {{ background:repeating-linear-gradient(45deg,#232a34,
        #232a34 3px,#1a2028 3px,#1a2028 6px); }}
.fname {{ font-size:11.5px; color:var(--muted); word-break:break-all; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
td, th {{ border-bottom:1px solid var(--line); padding:6px 10px; text-align:left; }}
.small {{ font-size:11px; color:var(--muted); }}
footer {{ border-top:1px solid var(--line); color:var(--dim); font-size:12.5px;
          padding:18px clamp(16px,4vw,40px); max-width:80ch; }}
</style>
<header>
  <h1>Site30k v1 Review</h1>
  <p class="sub">Teacher annotations from recipe v6.2, sampled per camera and spread over
  dates. Every frame here is a pre-label, not ground truth.</p>
  <div class="stats">
    <div class="stat"><b class="mono">{n_masks}</b><span>frames labelled</span></div>
    <div class="stat"><b class="mono">{len(cameras)}</b><span>cameras</span></div>
    <div class="stat"><b class="mono">{len(fails)}</b><span>failed units</span></div>
  </div>
  <div class="bar">{legend}<span class="lg">hatched = unlabelled</span></div>
</header>
<main>{"".join(body)}
{fail_section}
</main>
<footer>Dataset {args.root}. Masks are 7-class PNGs (0 void, 1 floor, 2 wall, 3 column,
4 display_table, 5 shelf, 6 person, 255 unlabelled) plus product ids 7-10; images are the
frames they label. Previews exist for one frame in twenty.</footer>
"""
    out.write_text(html)
    print(
        f"{len(cameras)} cameras, {sum(len(c['frames']) for c in cameras)} sampled "
        f"frames, {len(fails)} failures -> {out} ({out.stat().st_size / 1e6:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

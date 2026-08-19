#!/usr/bin/env python3
"""Three inference-side output stabilisers, measured with flicker_baseline's own ruler.

  nice -n 10 .venv/bin/python scripts/stable_infer.py --suite \
    --config configs/hydranet_retail_security_b03_cw_xl.yaml \
    --checkpoint runs/hydranet_retail_security_b03_cw_xl/best.pt \
    --input datasets/studioa_clips/Kaohsiung-cam04/archive_20260816-112757_20260816-113301.mp4 \
    --static-mask datasets/studioa_static/Kaohsiung-cam04/static_20260816-112757.png \
    --out runs/stable01

Three mechanisms, each behind its own switch, so an ablation is just a combination of them:

1. **Static compositing** (``--static-composite``). Run the model once, offline, over that
   camera's plate for that time slot to get a *background label board*. At run time, take
   the per-frame difference against the plate and paste the board's label onto every pixel
   below a threshold. The threshold follows `static_plates.py`'s noise-floor reasoning:
   the slot's ``noise_floor_used`` from index.json, times ``--static-mult`` (default 8,
   which is `static_plates.DYNAMIC_MULT`). Measured on cam04 the two populations separate
   widely -- pixels sitting on the plate have a median diff of 3-4 grey levels, pixels a
   person occupies 50-150 -- and the curve is flat between thresholds of 12 and 32, so 25
   sits in the middle of the gap.
2. **Logits EMA over the moving region** (``--ema``). Exponentially smooth the probability
   map per pixel, then argmax; ``--ema-alpha`` is the weight on the new frame. Pixels the
   static compositing takes over end up with the plate's label, so the EMA only decides
   anything where it does not. (The EMA *state* is updated for the whole frame regardless,
   so a pixel entering or leaving the taken-over region does not carry stale state.)
3. **Track-rendered boxes** (``--tracks``). Detections go through the online half of
   `offline_tracks.OfflineForward` -- imported rather than copied -- which is a two-stage
   hysteresis association at 0.35 to be born and 0.20 to survive; a box's class is its
   track's majority vote, and a frame that missed the object is filled from the Kalman
   motion model for up to ``--track-max-age`` frames.

**The measurement is not rewritten.** `flicker_baseline.build_model` is monkeypatched so
that `flicker_baseline.main()` runs unchanged over the stabilised output: same loop, same
flip/bounce/fragmentation/survival arithmetic, same default thresholds. A second copy of
that arithmetic would be a second chance to get it wrong, so there is not one here.

**Immunity to `temporal.py`'s self-sealing failure is by design, not by test.** The plate
and the background label board are *offline, trusted artefacts and are never updated
online*. The worst case for a wrong gate decision is that the pixel falls back to the live
prediction -- which is the baseline behaviour -- rather than freezing an error into the
background. `temporal.py`'s failure chain (update the plate online -> the wrong place is
never static -> it is never repaired) has no first link here.

The costs, stated rather than left to be discovered:

* Where the background board is wrong, it is **stably** wrong. Regions the board still
  reads as `person` are the plate's dirty areas -- a plate is only as empty as its own
  clip, and someone standing at the counter for a few minutes is in the median (8.6% of
  this cam04 plate, measured). Those are marked **untrusted, never taken over** by
  default, so they fall back to the live prediction, which is baseline behaviour. Forcing
  a not-person label onto them instead pastes `fixture` over a real person, which the
  smoke test reproduces. It also means flips inside the static mask do not go to zero:
  dirty pixels still take the live (or EMA) path.
* EMA delays a real change: argmax flips once ``(1 - alpha)^n < 0.5``, which at
  ``alpha=0.35`` is 2 frames, or 0.4 s.
* Track rendering: ``min-hits 2`` makes a new object appear one frame late, and a box
  lingers for up to ``max-age`` frames after the object has gone.

When a training run shares the GPU: batch is always 1, and start the whole process under
`nice -n 10`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import flicker_baseline as fb
from offline_tracks import OfflineForward

from syncai_hydranet.data.label_maps_retail_security import get_det_vocab
from syncai_hydranet.data.video import finish_encoder, probe
from syncai_hydranet.preprocessing import IMAGENET_MEAN, IMAGENET_STD
from syncai_hydranet.utils.visualize import (
    letterbox,
    overlay,
    preprocess,
    terrain_palette,
)


def _save_heatmap_pil(arr, path, title, vmax=None):  # noqa: ARG001  no font, no title
    """A PIL stand-in for fb.save_heatmap: the venv has no matplotlib (it is not in
    pyproject). The measurement arithmetic is untouched -- this only draws the same array
    as a PNG on an approximation of inferno."""
    a = np.asarray(arr, np.float64)
    hi = float(vmax) if vmax else float(a.max() or 1.0)
    t = np.clip(a / max(hi, 1e-9), 0.0, 1.0)
    anchors = np.array(
        [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], np.float64
    )
    pos = t * (len(anchors) - 1)
    lo = np.floor(pos).astype(np.int64)
    hi_i = np.minimum(lo + 1, len(anchors) - 1)
    frac = (pos - lo)[..., None]
    Image.fromarray((anchors[lo] * (1 - frac) + anchors[hi_i] * frac).astype(np.uint8)).save(
        path
    )


try:
    import matplotlib  # noqa: F401
except ImportError:
    fb.save_heatmap = _save_heatmap_pil


# The same 8 as static_plates.DYNAMIC_MULT: threshold = noise floor x 8. Not imported,
# deliberately. That constant measures per-pixel deviation along a 0.5 fps time axis; this
# measures one frame's spatial difference against the plate. They share the reasoning
# ("some multiple of the noise floor") and not the measurement behind it, so restating the
# number is the honest form and importing it would claim a shared derivation.
STATIC_MULT_DEFAULT = 8.0
# Box blur (one side) before the difference. The plate is upsampled from 960x540 and the
# frame downsampled from 1920x1080, so the resampling sharpness difference concentrates on
# high-frequency edges; a 5 px mean filter pushes it back under the noise floor (measured:
# p99 difference under 5%).
DIFF_BLUR = 5


def _area_filter(mask: np.ndarray, min_area: float) -> np.ndarray:
    """Clear True components smaller than min_area. Fragmentation is decided by area, not
    guessed at through a kernel size."""
    from scipy import ndimage

    cc, n = ndimage.label(mask)
    if not n:
        return mask
    areas = ndimage.sum_labels(np.ones_like(cc, dtype=np.int64), cc, np.arange(1, n + 1))
    bad = np.flatnonzero(areas < min_area) + 1
    if not len(bad):
        return mask
    return mask & ~np.isin(cc, bad)


def _fill_small_components(lab: np.ndarray, min_frac: float) -> np.ndarray:
    """Fill components below min_frac of the area with the nearest large component's label.
    Offline, and done once, to the background board only."""
    from scipy import ndimage

    cut = min_frac * lab.size
    keep = np.ones_like(lab, dtype=bool)
    for c in np.unique(lab):
        mask = lab == c
        cc, n = ndimage.label(mask)
        if not n:
            continue
        areas = ndimage.sum_labels(np.ones_like(cc, dtype=np.int64), cc, np.arange(1, n + 1))
        small_ids = np.flatnonzero(areas < cut) + 1
        if len(small_ids):
            keep &= ~np.isin(cc, small_ids)
    if keep.all() or not keep.any():
        return lab
    _, (iy, ix) = ndimage.distance_transform_edt(~keep, return_indices=True)
    return lab[iy, ix]


# ---------------------------------------------------------------------------
# ------------- the stabilising wrapper: to flicker_baseline.main this *is* a model
# ---------------------------------------------------------------------------


class StabilisedModel:
    """Wraps the real model; `predict()` returns the stabilised output on HydraNet's
    interface.

    `flicker_baseline.main` touches exactly four things: `.to(device)`, `.eval()`,
    `.load_state_dict()` and `.predict(x, score_thr=, nms_thr=)`. The first three are
    delegated; the fourth is where the three mechanisms live.

    **Stateful** (EMA, tracker, frame index), so one instance takes one clip, in order.
    """

    def __init__(self, inner, opts, cfg, renderer=None):
        self.inner = inner
        self.opts = opts
        self.size = tuple(cfg["data"]["input_size"])  # (H, W)
        self.class_names = list(cfg["data"]["terrain_classes"])
        self.person_idx = (
            self.class_names.index("person") if "person" in self.class_names else None
        )
        self.renderer = renderer
        self.device = torch.device("cpu")
        self.frame_idx = 0
        self._ema: torch.Tensor | None = None
        self._plate_labels: torch.Tensor | None = None
        self._plate_onehot: torch.Tensor | None = None
        self._plate_trusted_np: np.ndarray | None = None
        self._plate_blur: torch.Tensor | None = None
        self._calm_streak: np.ndarray | None = None
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self.static_thr = opts.static_thr  # None -> derived from index.json on the first frame
        self.stats: dict[str, list] = {"takeover_share": [], "ghost_person_px": []}
        self.fwd = (
            OfflineForward(
                opts.track_birth,
                opts.track_keep,
                opts.track_iou,
                opts.track_iou_low,
                opts.track_max_age,
                opts.track_min_hits,
                opts.vel_scale,
            )
            if opts.tracks
            else None
        )

    # -- the three delegations flicker_baseline.main needs --------------------
    def to(self, device):
        self.inner.to(device)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        return self

    def eval(self):
        self.inner.eval()
        return self

    def load_state_dict(self, sd):
        return self.inner.load_state_dict(sd)

    # -- helpers --------------------------------------------------------------
    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised tensor -> [1, 3, H, W] float, 0..255 grey, in canvas coordinates."""
        assert self._mean is not None and self._std is not None
        return (x * self._std + self._mean).clamp(0, 1) * 255.0

    @staticmethod
    def _blur(img: torch.Tensor) -> torch.Tensor:
        k = DIFF_BLUR
        return torch.nn.functional.avg_pool2d(img, k, stride=1, padding=k // 2)

    def _resolve_static_thr(self) -> float:
        """Noise floor x --static-mult. The floor comes from static_plates' index.json, which
        is an offline artefact."""
        mask = Path(self.opts.static_mask)
        slot = mask.stem.split("_", 1)[1]
        cam = mask.parent.name
        index = mask.parent.parent / "index.json"
        if not index.exists():
            raise SystemExit(
                f"{index} does not exist, so the noise floor cannot be looked up; "
                f"pass --static-thr explicitly, in grey levels"
            )
        slot_info = json.loads(index.read_text())["cameras"][cam]["slots"][slot]
        floor = slot_info.get("noise_floor_used", slot_info.get("noise_floor_used_median"))
        if floor is None:
            raise SystemExit(f"index.json has no noise_floor_used for {cam}/{slot}")
        return float(floor) * self.opts.static_mult

    def _ensure_plate(self) -> None:
        """The background label board and the plate canvas it is compared against.

        Computed once and **never updated online** -- see the module docstring for why that
        is the whole immunity argument.

        Two offline tidy-ups, both done once under the premise that the board is a trusted
        artefact and then left alone:

        * **Small components are filled in.** A fragment below --plate-min-frac on the board
          is the model's noise on one plate; pasted into every frame it becomes a permanent
          fragment, so it takes the nearest large component's label instead.
        * **Regions the board reads as `person` are marked untrusted** and compositing never
          touches them (measured: 8.6% of this cam04 plate reads as person -- someone stood
          at the counter long enough to survive the median). Pasting a forced not-person
          label there is pasting `fixture` over a real person; falling back to the live
          prediction, which is baseline behaviour, is the honest treatment of a dirty plate.
        """
        if self._plate_labels is not None:
            return
        if self.static_thr is None:
            self.static_thr = self._resolve_static_thr()
        plate_img = Image.open(self.opts.plate).convert("RGB")
        px, _lb, _region = preprocess(plate_img, self.size)
        px = px.to(self.device)
        with torch.no_grad():
            logits = self.inner.forward(px)["terrain"][0]  # [C,H,W]
        lab = logits.argmax(0).cpu().numpy().astype(np.int64)
        lab = _fill_small_components(lab, self.opts.plate_min_frac)
        trusted = np.ones_like(lab, dtype=bool)
        if self.opts.plate_no_person and self.person_idx is not None:
            trusted = lab != self.person_idx
        self._plate_labels = torch.from_numpy(lab).to(self.device)
        self._plate_onehot = (
            torch.nn.functional.one_hot(self._plate_labels, len(self.class_names))
            .permute(2, 0, 1)
            .float()
        )  # [C,H,W] -- the "observation" when EMA is on too
        self._plate_trusted_np = trusted
        self._plate_blur = self._blur(self._denorm(px))  # [1,3,H,W]
        print(
            f"background label board ready: takeover threshold {self.static_thr:.1f} grey "
            f"levels, untrusted (person) area on the board {1 - trusted.mean():.1%}"
        )

    # -- the three mechanisms -------------------------------------------------
    def predict(self, x: torch.Tensor, score_thr: float, nms_thr: float = 0.6) -> dict:
        x = x.to(self.device)
        with torch.no_grad():
            out = self.inner.forward(x)
        probs = out["terrain"].softmax(dim=1)  # [1,C,H,W]
        live_labels = probs.argmax(dim=1)  # [1,H,W]

        # 1. The static-composite takeover mask. The label is applied further down, and
        #    how it is applied depends on whether EMA is on as well.
        calm_t: torch.Tensor | None = None
        if self.opts.static:
            self._ensure_plate()
            assert self._plate_blur is not None and self._plate_labels is not None
            assert self._plate_trusted_np is not None
            diff = (self._blur(self._denorm(x)) - self._plate_blur).abs().mean(dim=1)
            dyn = (diff[0] >= self.static_thr).cpu().numpy()  # [H,W]
            area = float(dyn.size)
            if self.opts.dynamic_min_frac > 0:
                # An isolated diff island far smaller than a person is compression noise,
                # not something entering the scene. Left in, the takeover region becomes
                # patchwork and the component count explodes -- 37 -> 129/91 on the smoke
                # clip. Deciding by area brings the stabilised count back to about live+10,
                # which morphological opening and closing could not.
                dyn = _area_filter(dyn, self.opts.dynamic_min_frac * area)
            if self.opts.static_dilate > 0:
                # Grow the moving region outward by r pixels: the rim around a person goes to
                # the live prediction, so a trailing edge is not pasted with background.
                from scipy import ndimage

                dyn = ndimage.binary_dilation(dyn, iterations=self.opts.static_dilate)
            calm = ~dyn & self._plate_trusted_np
            if self.opts.calm_min_frac > 0:
                # Small calm islands inside the moving region, same reasoning: pasting one
                # only adds a fragment.
                calm = _area_filter(calm, self.opts.calm_min_frac * area)
            if self.opts.static_entry > 1:
                # Entry hysteresis: k consecutive calm frames before takeover, and exit
                # immediately. Measured over the full 900 frames without this gate: the calm
                # test chatters frame to frame near the threshold, board and live labels
                # alternate, and the static-region flip rate is eaten back to where it
                # started by the leading edge (1.53% against a 1.50% baseline). The
                # asymmetry is on the safe side -- taking over late costs a few more frames
                # of live prediction, while releasing late would paste stale background.
                if self._calm_streak is None:
                    self._calm_streak = np.zeros(calm.shape, dtype=np.int32)
                self._calm_streak = (self._calm_streak + 1) * calm
                calm = self._calm_streak >= self.opts.static_entry
            calm_t = torch.from_numpy(calm).to(self.device)
            self.stats["takeover_share"].append(float(calm.mean()))

        # 2. Logits EMA over the moving region, and what it means combined with (1).
        if self.opts.ema:
            obs = probs
            if calm_t is not None:
                # Both on: a taken-over pixel feeds the board's one-hot into the EMA as an
                # *observation* rather than being pasted. Measured over 900 frames, pasting
                # is worse: the moving region sweeps frame to frame at the edge of a crowd,
                # board and live labels alternate, and the static-region flip rate comes out
                # above EMA alone (0.93% against 0.67%). Letting the EMA's lag absorb the
                # sweep means a leading edge has to out-vote the accumulated probability
                # before it can turn a page.
                assert self._plate_onehot is not None
                obs = torch.where(calm_t[None, None], self._plate_onehot[None], probs)
            self._ema = (
                obs
                if self._ema is None
                else self.opts.ema_alpha * obs + (1.0 - self.opts.ema_alpha) * self._ema
            )
            labels = self._ema.argmax(dim=1)
        else:
            labels = live_labels.clone()
            if calm_t is not None:
                # Static compositing alone: the paste as specified -- a pixel whose difference
                # is under the threshold takes the board's label directly.
                assert self._plate_labels is not None
                labels = torch.where(calm_t[None], self._plate_labels[None], labels)

        if self.person_idx is not None:
            ghost = int(
                ((labels == self.person_idx) & (live_labels != self.person_idx)).sum().item()
            )
            self.stats["ghost_person_px"].append(ghost)

        # 3. Track-rendered boxes.
        det_head = self.inner.det_head
        assert det_head is not None, "checkpoint has no detection head"
        meta: list[dict] = []
        if self.fwd is not None:
            raw = det_head.decode(
                out["det_cls"],
                out["det_reg"],
                out["det_ctr"],
                # Down to the keep threshold; birth lives in the associator.
                score_thr=self.opts.track_keep,
                nms_thr=nms_thr,
                img_size=self.size,
            )[0]
            boxes = raw["boxes"].cpu().numpy().astype(np.float64)
            scores = raw["scores"].cpu().numpy().astype(np.float64)
            labs = raw["labels"].cpu().numpy().astype(np.int64)
            self.fwd.update(boxes, scores, self.frame_idx)
            self._vote(boxes, scores, labs)
            det = self._tracked_boxes(meta)
        else:
            det = det_head.decode(
                out["det_cls"],
                out["det_reg"],
                out["det_ctr"],
                score_thr=score_thr,
                nms_thr=nms_thr,
                img_size=self.size,
            )[0]

        result = {"terrain": labels, "detection": [det]}
        if self.renderer is not None:
            self.renderer(self, x, labels[0], live_labels[0], det, meta)
        self.frame_idx += 1
        return result

    def _vote(self, boxes: np.ndarray, scores: np.ndarray, labs: np.ndarray) -> None:
        """Record this frame's observed class back onto the fragment that consumed it.

        Matched by value equality, since `update` copies the box through unchanged.
        """
        if not len(boxes):
            return
        assert self.fwd is not None, "_vote is only reached when --tracks built the forward"
        for t in self.fwd.tracks:
            if not t.frames or t.frames[-1] != self.frame_idx:
                continue
            j = int(np.abs(boxes - np.asarray(t.boxes[-1])).sum(axis=1).argmin())
            t.__dict__.setdefault("label_votes", []).append(int(labs[j]))
            t.__dict__["last_score"] = float(scores[j])

    def _tracked_boxes(self, meta: list[dict]) -> dict:
        """Confirmed live tracks -> boxes. age == 0 uses the observed box; otherwise the
        Kalman prediction fills in, up to max-age."""
        h, w = self.size
        bs, ls, ss = [], [], []
        assert self.fwd is not None, (
            "_tracked_boxes is only reached when --tracks built the forward"
        )
        for t in self.fwd.tracks:
            if not t.confirmed:
                continue
            votes = t.__dict__.get("label_votes")
            if not votes:
                continue
            box = np.asarray(t.boxes[-1] if t.age == 0 else t.kalman.box, np.float64)
            box[0::2] = box[0::2].clip(0, w)
            box[1::2] = box[1::2].clip(0, h)
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            bs.append(box)
            counts = np.bincount(votes)
            # numpy 2.5 stubs reject a plain argmax() here; the code is right (same
            # stub family as dwell.py's float(arr.min())), so the ignore is scoped.
            ls.append(int(counts.argmax()))  # ty: ignore[no-matching-overload]
            ss.append(float(t.__dict__.get("last_score", 0.0)))
            meta.append({"id": t.frag_id, "coasting": t.age > 0})
        if bs:
            return {
                "boxes": torch.from_numpy(np.stack(bs)),
                "labels": torch.from_numpy(np.asarray(ls, np.int64)),
                "scores": torch.from_numpy(np.asarray(ss, np.float64)),
            }
        return {
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "scores": torch.zeros((0,)),
        }


# ---------------------------------------------------------------------------
# ---- rendering, attached only on the all-on pass: the video and the measurement share
# ---- one forward pass, so what is measured is what is seen
# ---------------------------------------------------------------------------


class Renderer:
    GHOST_DUMP_PX = 1500  # grab a frame when stabilisation adds more person pixels than this
    GHOST_DUMP_CAP = 24

    def __init__(self, out_mp4, frames_dir, region, palette, names, fps, dump_every):
        self.out_mp4 = str(out_mp4)
        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.region = region  # (x0, y0, w, h)
        self.palette = palette
        self.names = names
        self.fps = fps
        self.dump_every = dump_every
        self.writer = None
        self.ghost_dumps = 0

    def __call__(self, sm, x, labels, live_labels, det, meta):
        x0, y0, cw, ch = self.region
        canvas = sm._denorm(x)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        content = Image.fromarray(canvas[y0 : y0 + ch, x0 : x0 + cw])
        lab = labels.cpu().numpy()[y0 : y0 + ch, x0 : x0 + cw]
        vis = overlay(content, lab, self.palette)
        self._draw_boxes(vis, det, meta)

        if self.writer is None:
            ow, oh = vis.width - vis.width % 2, vis.height - vis.height % 2
            self.enc_size = (ow, oh)
            self.writer = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{ow}x{oh}", "-r", f"{self.fps}", "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", self.out_mp4],
                stdin=subprocess.PIPE,
            )  # fmt: skip
        img = vis if vis.size == self.enc_size else vis.resize(self.enc_size)
        assert self.writer.stdin is not None
        self.writer.stdin.write(np.asarray(img, np.uint8).tobytes())

        # Grabs: one on a fixed interval plus every ghost candidate (a frame where
        # stabilisation added person pixels), as a triptych for frame-by-frame comparison
        ghost = sm.stats["ghost_person_px"][-1] if sm.stats["ghost_person_px"] else 0
        periodic = self.dump_every and sm.frame_idx % self.dump_every == 0
        ghosty = ghost >= self.GHOST_DUMP_PX and self.ghost_dumps < self.GHOST_DUMP_CAP
        if periodic or ghosty:
            live = overlay(
                content, live_labels.cpu().numpy()[y0 : y0 + ch, x0 : x0 + cw], self.palette
            )
            trip = Image.new("RGB", (cw * 3, ch))
            for i, p in enumerate((content, live, vis)):
                trip.paste(p, (cw * i, 0))
            tag = "ghost" if ghosty and not periodic else "every"
            trip.resize((cw * 3 // 2, ch // 2)).save(
                self.frames_dir / f"{tag}_{sm.frame_idx:04d}.png"
            )
            if ghosty:
                self.ghost_dumps += 1

    def _draw_boxes(self, vis, det, meta):
        x0, y0, cw, ch = self.region
        boxes = det.get("boxes")
        if boxes is None or not len(boxes):
            return
        draw = ImageDraw.Draw(vis)
        metas = meta if len(meta) == len(boxes) else [{}] * len(boxes)
        for box, score, label, m in zip(
            boxes.cpu().numpy(), det["scores"].cpu().numpy(),
            det["labels"].cpu().numpy(), metas, strict=True,
        ):  # fmt: skip
            b = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
            if b[2] <= 0 or b[3] <= 0 or b[0] >= cw or b[1] >= ch:
                continue
            coast = m.get("coasting", False)
            color = (180, 180, 180) if coast else (255, 255, 255)
            draw.rectangle(b, outline=color, width=2)
            tid = m.get("id")
            text = f"{self.names(int(label))}"
            if tid is not None:
                text += f" #{tid}"
            text += " ~" if coast else f" {score:.2f}"
            draw.text((b[0] + 2, b[1] + 2), text, fill=color)

    def close(self):
        code = finish_encoder(self.writer)
        if code not in (None, 0):
            raise RuntimeError(f"ffmpeg exited {code} while encoding {self.out_mp4}")


# ---------------------------------------------------------------------------
# ---- running a variant: monkeypatch flicker_baseline.build_model, instrument untouched
# ---------------------------------------------------------------------------


def run_variant(args, name: str, *, static: bool, ema: bool, tracks: bool, renderer=None):
    out_dir = args.out / name
    opts = argparse.Namespace(
        static=static,
        ema=ema,
        tracks=tracks,
        plate=args.plate,
        static_mask=args.static_mask,
        static_thr=args.static_thr,
        static_mult=args.static_mult,
        static_dilate=args.static_dilate,
        dynamic_min_frac=args.dynamic_min_frac,
        calm_min_frac=args.calm_min_frac,
        static_entry=args.static_entry,
        plate_min_frac=args.plate_min_frac,
        plate_no_person=not args.plate_person,
        ema_alpha=args.ema_alpha,
        track_birth=args.track_birth,
        track_keep=args.track_keep,
        track_iou=args.track_iou,
        track_iou_low=args.track_iou_low,
        track_max_age=args.track_max_age,
        track_min_hits=args.track_min_hits,
        vel_scale=args.vel_scale,
    )
    holder: dict = {}
    orig = fb.build_model

    def patched(cfg):
        holder["sm"] = StabilisedModel(orig(cfg), opts, cfg, renderer)
        return holder["sm"]

    fb.build_model = patched
    try:
        print(f"\n=== variant {name}: static={static} ema={ema} tracks={tracks} ===")
        metrics = fb.main(
            [
                "--config",
                args.config,
                "--checkpoint",
                args.checkpoint,
                "--input",
                args.input,
                "--static-mask",
                args.static_mask,
                "--out",
                str(out_dir),
                "--fps",
                str(args.fps),
                "--max-frames",
                str(args.max_frames),
                "--score-thr",
                str(args.score_thr),
                "--nms-thr",
                str(args.nms_thr),
            ]
        )
    finally:
        fb.build_model = orig
    sm = holder["sm"]
    stats = {
        "static_thr": sm.static_thr,
        "takeover_share_mean": (
            float(np.mean(sm.stats["takeover_share"])) if sm.stats["takeover_share"] else None
        ),
        "ghost_person_px": fb.dist(sm.stats["ghost_person_px"]),
        "mechanisms": {"static": static, "ema": ema, "tracks": tracks},
        "params": {
            k: getattr(opts, k)
            for k in (
                "static_mult",
                "static_dilate",
                "dynamic_min_frac",
                "calm_min_frac",
                "static_entry",
                "plate_min_frac",
                "plate_no_person",
                "ema_alpha",
                "track_birth",
                "track_keep",
                "track_max_age",
                "track_min_hits",
            )
        },
    }
    (out_dir / "stabiliser_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1) + "\n"
    )
    return metrics, stats


def summarise(m: dict) -> dict:
    """metrics.json -> one row of the comparison table. Field selection only; no new
    arithmetic happens here."""
    det = m["detection"]
    frag = m["fragmentation"]
    fx = m["boundary_bands"].get("fixture")
    person = m["per_class_flip"].get("person", {})
    return {
        "flip_rate_in_static": m["flip"]["in_static"]["flip_rate"],
        "flip_rate_out_static": m["flip"]["out_static"]["flip_rate"],
        "bounce_share_in_static": m["flip"]["in_static"]["bounce_share_of_flips"],
        "bounce_share_overall": m["flip"]["overall"]["bounce_share_of_flips"],
        "person_flip_in_static": (person.get("in_static") or {}).get("flip_rate"),
        "person_pixel_frames_in_static": (person.get("in_static") or {}).get("pixel_frames"),
        "fixture_band_flip_rate": fx["band_flip_rate"] if fx else None,
        "fixture_band_over_interior": (
            fx["band_flip_rate"] / fx["interior_flip_rate"]
            if fx and fx["interior_flip_rate"]
            else None
        ),
        "components_mean": frag["total_components_per_frame"]["mean"],
        "components_p90": frag["total_components_per_frame"]["p90"],
        "small_share_mean": frag["small_share_per_frame"]["mean"],
        "boxes_per_frame_mean": det["boxes_per_frame"]["overall"]["mean"],
        "boxes_per_frame_var": det["boxes_per_frame"]["overall"]["var"],
        "tracks": det["tracks"]["overall"]["tracks"],
        "track_len_p50": det["tracks"]["overall"]["length"]["p50"],
        "track_len_mean": det["tracks"]["overall"]["length"]["mean"],
        "one_frame_share": det["tracks"]["overall"]["one_frame_share"],
        "ge5_share": det["tracks"]["overall"]["ge5_share"],
    }


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="stable_infer",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--input", required=True, help="a fixed-camera clip; the same one the baseline used"
    )
    ap.add_argument(
        "--static-mask", required=True, help="static_plates.py's mask for the same time slot"
    )
    ap.add_argument(
        "--plate",
        default=None,
        help="the plate for the same slot; by default derived from the mask filename, "
        "static_ -> plate_",
    )
    ap.add_argument("--out", type=Path, required=True, help="runs/stable01")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--max-frames", type=int, default=900)
    ap.add_argument(
        "--score-thr",
        type=float,
        default=0.30,
        help="detection threshold on the non-track path",
    )
    ap.add_argument("--nms-thr", type=float, default=0.6)
    # Mechanism switches, for the single-run mode. --suite arranges its own ablation.
    ap.add_argument("--static-composite", action="store_true", dest="static_on")
    ap.add_argument("--ema", action="store_true", dest="ema_on")
    ap.add_argument("--tracks", action="store_true", dest="tracks_on")
    ap.add_argument(
        "--suite",
        action="store_true",
        help="run the full set: cited baseline, each mechanism alone, then all on",
    )
    ap.add_argument(
        "--baseline-metrics",
        default="runs/flicker_baseline01/metrics.json",
        help="an existing baseline metrics.json from the same instrument and clip, cited "
        "rather than re-run",
    )
    # Mechanism 1: static compositing
    ap.add_argument(
        "--static-thr",
        type=float,
        default=None,
        help="grey levels; defaults to the noise floor x --static-mult",
    )
    ap.add_argument("--static-mult", type=float, default=STATIC_MULT_DEFAULT)
    ap.add_argument(
        "--static-dilate",
        type=int,
        default=4,
        help="pixels to grow the moving region by, so a trailing edge is not pasted",
    )
    ap.add_argument(
        "--dynamic-min-frac",
        type=float,
        default=0.001,
        help="a moving island below this area fraction is noise and is re-read as calm; "
        "0 disables",
    )
    ap.add_argument(
        "--calm-min-frac",
        type=float,
        default=0.003,
        help="a calm island below this area fraction is not taken over; 0 disables",
    )
    ap.add_argument(
        "--static-entry",
        type=int,
        default=5,
        help="consecutive calm frames required before takeover -- entry hysteresis "
        "against leading-edge chatter. Exit is always immediate; <=1 disables",
    )
    ap.add_argument(
        "--plate-min-frac",
        type=float,
        default=0.001,
        help="fill board components below this area fraction",
    )
    ap.add_argument(
        "--plate-person",
        action="store_true",
        help="trust the board's person region and paste it. By default that region is "
        "read as the plate's dirty area and is never taken over",
    )
    # Mechanism 2: logits EMA
    ap.add_argument(
        "--ema-alpha",
        type=float,
        default=0.35,
        help="weight on the new frame; smaller is smoother and slower",
    )
    # Mechanism 3: track-rendered boxes
    ap.add_argument(
        "--track-birth",
        type=float,
        default=0.35,
        help="birth threshold, the upper edge of the hysteresis",
    )
    ap.add_argument(
        "--track-keep",
        type=float,
        default=0.20,
        help="keep threshold, the lower edge of the hysteresis",
    )
    ap.add_argument("--track-iou", type=float, default=0.3)
    ap.add_argument("--track-iou-low", type=float, default=0.4)
    ap.add_argument(
        "--track-max-age",
        type=int,
        default=5,
        help="how many missed frames the Kalman model fills in for",
    )
    ap.add_argument(
        "--track-min-hits",
        type=int,
        default=2,
        help="hits before a track is confirmed; this is what kills one-frame boxes",
    )
    ap.add_argument(
        "--vel-scale", type=float, default=None, help="defaults to 25 / sampling fps"
    )
    ap.add_argument(
        "--dump-every",
        type=int,
        default=100,
        help="save one comparison triptych every N frames",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.plate is None:
        m = Path(args.static_mask)
        args.plate = str(m.with_name(m.name.replace("static_", "plate_", 1)))
    if not Path(args.plate).exists():
        raise SystemExit(f"plate does not exist: {args.plate}")
    if args.vel_scale is None:
        args.vel_scale = 25.0 / args.fps

    cfg = fb.load_config(args.config, [])
    size = cfg["data"]["input_size"]
    palette = terrain_palette(cfg["data"].get("terrain_classes"))
    vocabs = {d["det_vocab"] for d in cfg["data"].get("datasets", []) if d.get("det_vocab")}
    if len(vocabs) == 1:
        det_names = list(get_det_vocab(next(iter(vocabs))).classes)
    else:
        det_names = list(
            ((cfg.get("model", {}).get("heads", {}) or {}).get("detection") or {}).get(
                "classes", []
            )
        )
    name_of = lambda i: det_names[i] if i < len(det_names) else str(i)  # noqa: E731

    src_w, src_h, _ = probe(args.input)
    _, region = letterbox(Image.new("RGB", (src_w, src_h)), size)

    if not args.suite:
        renderer = None
        if args.static_on or args.ema_on or args.tracks_on:
            renderer = Renderer(
                args.out / "stable_cam04.mp4", args.out / "frames", region, palette,
                name_of, args.fps, args.dump_every,
            )  # fmt: skip
        try:
            metrics, stats = run_variant(
                args, "single",
                static=args.static_on, ema=args.ema_on, tracks=args.tracks_on,
                renderer=renderer,
            )  # fmt: skip
        finally:
            if renderer is not None:
                renderer.close()
        print(json.dumps(summarise(metrics), ensure_ascii=False, indent=1))
        return 0

    # ---- the full set: cited baseline, each mechanism alone, all on (which also renders) ---
    baseline = json.loads(Path(args.baseline_metrics).read_text())
    rows: dict[str, dict] = {"baseline": summarise(baseline)}
    stab: dict[str, dict] = {}

    for name, mech in (
        ("static_only", {"static": True, "ema": False, "tracks": False}),
        ("ema_only", {"static": False, "ema": True, "tracks": False}),
        ("tracks_only", {"static": False, "ema": False, "tracks": True}),
    ):
        metrics, stats = run_variant(args, name, **mech)
        rows[name] = summarise(metrics)
        stab[name] = stats

    renderer = Renderer(
        args.out / "stable_cam04.mp4", args.out / "full" / "frames", region, palette,
        name_of, args.fps, args.dump_every,
    )  # fmt: skip
    try:
        metrics, stats = run_variant(
            args, "full", static=True, ema=True, tracks=True, renderer=renderer
        )
    finally:
        renderer.close()
    rows["full"] = summarise(metrics)
    stab["full"] = stats

    table = {
        "measured": (
            "output stabilisation: baseline against each mechanism alone and all on, "
            "same instrument, same clip, same thresholds"
        ),
        "instrument": (
            "scripts/flicker_baseline.py, reusing its main() via a build_model monkeypatch"
        ),
        "baseline_source": str(args.baseline_metrics),
        "input": args.input,
        "static_mask": args.static_mask,
        "plate": args.plate,
        "frames": args.max_frames,
        "sample_fps": args.fps,
        "rows": rows,
        "stabiliser": stab,
    }
    out_json = args.out / "metrics.json"
    out_json.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n")
    print(f"\ncomparison table written to {out_json}")

    cols = list(rows)
    keys = list(next(iter(rows.values())))
    print(f"\n{'metric':32s}" + "".join(f"{c:>14s}" for c in cols))
    for k in keys:
        line = f"{k:32s}"
        for c in cols:
            v = rows[c].get(k)
            line += f"{v:14.4f}" if isinstance(v, float) else f"{v!s:>14s}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

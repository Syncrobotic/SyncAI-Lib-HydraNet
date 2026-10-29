"""Four-panel preview from two checkpoints: original, terrain, detection, pose.

Why this is not `hydranet-infer-video --layout quad`: that renders one checkpoint,
and terrain here comes from a different run to detection and pose. The 22-class
terrain head was trained alone (`retail22_smoke`) because its taxonomy has five
annotated frames, while detection and pose share `detpose_coco01`. Two models, one
grid.

Terrain is inferred on the FIRST FRAME ONLY and held for the clip. That is the
caller's choice and it is the honest one for this checkpoint: 4 training images
cannot support a per-frame claim, and a static panel says so by not moving.

Layout, clockwise from top left: source frame, terrain, pose, detection.

The pose panel has two modes. `--pose-panel heatmap` draws the head's own output -- 17
stride-8 heatmaps, shown as their per-pixel max -- and `skeleton` draws what `predict`
makes of it after box-conditioned argmax. The heatmap is the head; the skeleton is the
head plus the detection boxes it decodes inside, so a bad skeleton over a clean heatmap
means the boxes failed, not the joints.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from syncai_hydranet.config import load_config
from syncai_hydranet.models import build_model
from syncai_hydranet.models.heads.detection import SCORE_THR_VIEW
from syncai_hydranet.utils.checkpoint import load_checkpoint, select_weights
from syncai_hydranet.utils.visualize import crop_box, overlay, preprocess, terrain_palette

SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (0, 1), (0, 2), (1, 3), (2, 4),
)  # fmt: skip
LIMB, JOINT, BOX = (70, 230, 255), (255, 215, 60), (120, 220, 120)
KP_MIN_CONF = 0.2

# A dark-to-hot ramp, so that "no joint here" stays out of the way instead of tinting the
# whole frame the low end of a rainbow.
HEAT_STOPS = np.array(
    [(10, 10, 45), (45, 15, 120), (150, 35, 130), (232, 92, 62), (250, 205, 55),
     (255, 255, 225)], dtype=np.float32,
)  # fmt: skip


# One hue per COCO-17 joint, walking the wheel so that neighbours on a limb differ.
KP_COLORS = np.array(
    [
        [int(255 * c) for c in __import__("colorsys").hsv_to_rgb((k * 7 % 17) / 17, 0.85, 1.0)]
        for k in range(17)
    ],
    dtype=np.float32,
)


def heat_rgb(p: np.ndarray) -> np.ndarray:
    """Map probabilities in [0,1] through HEAT_STOPS -> an HxWx3 float array."""
    x = np.clip(p, 0.0, 1.0) * (len(HEAT_STOPS) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(HEAT_STOPS) - 2)
    f = (x - i)[..., None]
    return HEAT_STOPS[i] * (1.0 - f) + HEAT_STOPS[i + 1] * f


def predict_with_raw(model, images, score_thr: float, nms_thr: float = 0.6):
    """`HydraNet.predict` plus the pre-decode head outputs, from one forward pass.

    `predict` replaces the pose head's heatmaps in `result` with box-decoded coordinates,
    so a panel that wants to draw the heatmap has to hold on to the raw dict. Running
    `forward` a second time would cost a second pass and, with any dropout-like layer,
    would not even be the same tensor.
    """
    out = model.forward(images)
    result: dict = {}
    for head in model.heads():
        head.decode_into(
            result, out, score_thr=score_thr, nms_thr=nms_thr, img_size=images.shape[-2:]
        )
    return result, out


def load(cfg_path: str, ckpt_path: str, device: str, weights: str = "ema"):
    """Load a run for preview. `ema` by default because that is the weight set every
    evaluation in this repo scores, so a preview drawn from `model` would not be the
    thing the reported numbers describe. Falls back when a checkpoint has no EMA."""
    cfg = load_config(cfg_path, [])
    # Older runs declare a head type this architecture dropped; keep only what the
    # panels read so a legacy terrain run can still be previewed.
    for k in [
        k for k, h in cfg["model"]["heads"].items() if h.get("type") == "semantic_fpn_depth"
    ]:
        cfg["model"]["heads"][k]["type"] = "semantic_fpn"
    model = build_model(cfg).to(device).eval()
    ck = load_checkpoint(ckpt_path)
    sd = select_weights(ck, weights if weights in ck else "model")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith(("backbone.", "neck."))]
    if missing:
        raise SystemExit(
            f"{ckpt_path}: {len(missing)} head tensors missing, e.g. {missing[:3]}"
        )
    size = tuple(cfg["data"]["input_size"])
    return cfg, model, size


def frames(path: str, w: int, h: int, fps: float, seconds: float):
    """Decode `seconds` of video as RGB frames via the system ffmpeg."""
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-t",
        str(seconds),
        "-i",
        path,
        "-vf",
        f"fps={fps}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = w * h * 3
    while chunk := proc.stdout.read(n):
        if len(chunk) < n:
            break
        yield Image.fromarray(np.frombuffer(chunk, np.uint8).reshape(h, w, 3))
    proc.stdout.close()
    proc.wait()


def label(img: Image.Image, text: str) -> Image.Image:
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 260, 26], fill=(0, 0, 0))
    d.text((8, 6), text, fill=(255, 255, 255))
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--detpose-config", required=True)
    ap.add_argument("--detpose-ckpt", required=True)
    ap.add_argument("--terrain-config", required=True)
    ap.add_argument("--terrain-ckpt", required=True)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--score-thr", type=float, default=SCORE_THR_VIEW)
    ap.add_argument(
        "--pose-panel", choices=("skeleton", "heatmap", "channels"), default="skeleton"
    )
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    _dp_cfg, dp_model, dp_size = load(a.detpose_config, a.detpose_ckpt, a.device)
    tr_cfg, tr_model, tr_size = load(a.terrain_config, a.terrain_ckpt, a.device)
    tr_names = tr_cfg["data"].get("terrain_classes", [])
    tr_palette = terrain_palette(tr_names, len(tr_names))

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            a.input,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    src_w, src_h = (int(x) for x in probe.split("x"))

    out_dir = Path(a.output).with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    terrain_panel = None
    n = 0

    for i, frame in enumerate(frames(a.input, src_w, src_h, a.fps, a.seconds)):
        # ---- terrain: first frame only, then reused ----------------------------
        if terrain_panel is None:
            t_in, t_lb, t_region = preprocess(frame, tr_size)
            with torch.no_grad():
                t_res = tr_model.predict(t_in.to(a.device))
            tx, ty, tw, th = t_region
            t_content = t_lb.crop((tx, ty, tx + tw, ty + th))
            t_map = crop_box(t_res["terrain"][0].cpu().numpy(), t_region)
            terrain_panel = label(
                overlay(t_content, t_map, tr_palette).resize(frame.size),
                "terrain (frame 1 only)",
            )

        # ---- detection + pose: every frame -------------------------------------
        d_in, d_lb, d_region = preprocess(frame, dp_size)
        with torch.no_grad():
            res, raw = predict_with_raw(dp_model, d_in.to(a.device), a.score_thr)

        # predict() returns detection as a per-image list and pose as a parallel list
        # already decoded against those very boxes -- the pose head borrows them (see
        # heads/registry.py), so decoding a second time here would risk a different box
        # set in the two panels.
        det = res["detection"][0]
        boxes = det["boxes"].detach().cpu()
        scores = det["scores"].detach().cpu().numpy()
        kps_all = res["pose"][0].detach().cpu().numpy()

        # Both panels draw on the letterboxed content in input pixels, then the panel is
        # resized back. Cropping `lb` rather than resizing the source is what keeps the
        # overlay aligned: `preprocess` rounds the letterbox, and recomputing the content
        # size independently was off by 5 rows.
        dx, dy, cw, ch = d_region
        content = d_lb.crop((dx, dy, dx + cw, dy + ch))

        det_panel = content.copy()
        dd = ImageDraw.Draw(det_panel)
        for b, s in zip(boxes.numpy(), scores, strict=True):
            x0, y0, x1, y1 = (b - np.array([dx, dy] * 2)).tolist()
            dd.rectangle([x0, y0, x1, y1], outline=BOX, width=2)
            dd.text((x0 + 3, max(0, y0 - 12)), f"{s:.2f}", fill=BOX)
        det_panel = label(det_panel.resize(frame.size), f"detection  {len(boxes)} boxes")

        if a.pose_panel == "channels":
            # Same tensor as `heatmap`, but coloured by WHICH joint owns each pixel rather
            # than by how strong the winner is. It is the panel that shows the head emits
            # 17 maps and not one: a max-over-channels view collapses a wrist and an elbow
            # into the same orange blob, and then the heatmap is easy to mistake for a set
            # of points.
            hm = torch.sigmoid(raw["pose"][0])
            hm = torch.nn.functional.interpolate(
                hm[None], size=tuple(dp_size), mode="bilinear", align_corners=False
            )[0]
            conf, which = hm.max(0)
            prob = crop_box(conf.cpu().numpy(), d_region)
            idx = crop_box(which.cpu().numpy(), d_region)
            alpha = np.clip(prob * 1.6, 0.0, 0.9)[..., None]
            blended = (
                np.asarray(content, np.float32) * (1 - alpha) + KP_COLORS[idx] * alpha
            ).astype(np.uint8)
            pose_panel = label(
                Image.fromarray(blended).resize(frame.size), "pose heatmap, coloured by joint"
            )
        elif a.pose_panel == "heatmap":
            # The head's own output, at stride 8. Max over the 17 channels rather than a
            # sum: the panel answers "is a joint here", and seventeen faint shoulders must
            # not out-glow one crisp wrist. Bilinear back to input size is for looking at
            # only -- decode_boxes reads the stride-8 grid directly.
            hm = torch.sigmoid(raw["pose"][0]).amax(0)[None, None]
            hm = torch.nn.functional.interpolate(
                hm, size=tuple(dp_size), mode="bilinear", align_corners=False
            )
            prob = crop_box(hm[0, 0].cpu().numpy(), d_region)
            alpha = np.clip(prob * 1.6, 0.0, 0.9)[..., None]
            blended = (
                np.asarray(content, np.float32) * (1 - alpha) + heat_rgb(prob) * alpha
            ).astype(np.uint8)
            pose_panel = label(
                Image.fromarray(blended).resize(frame.size),
                f"pose heatmap  peak {float(prob.max()):.2f}",
            )
        else:
            pose_panel = content.copy()
            pd = ImageDraw.Draw(pose_panel)
            n_person = 0
            if len(boxes):
                for b, kp in zip(boxes.numpy(), kps_all, strict=True):
                    kp = kp.copy()
                    kp[:, 0] -= dx
                    kp[:, 1] -= dy
                    ok = kp[:, 2] >= KP_MIN_CONF
                    if not ok.any():
                        continue
                    n_person += 1
                    bh = b[3] - b[1]
                    w = max(1, round(bh / 90))
                    r = max(1.2, bh / 70)
                    for p, q in SKELETON:
                        if ok[p] and ok[q]:
                            pd.line([tuple(kp[p, :2]), tuple(kp[q, :2])], fill=LIMB, width=w)
                    for k in range(17):
                        if ok[k]:
                            x, y = kp[k, :2]
                            pd.ellipse([x - r, y - r, x + r, y + r], fill=JOINT)
            pose_panel = label(pose_panel.resize(frame.size), f"pose  {n_person} persons")

        src_panel = label(frame.copy(), "source")

        grid = Image.new("RGB", (src_w * 2, src_h * 2))
        grid.paste(src_panel, (0, 0))
        grid.paste(terrain_panel, (src_w, 0))
        grid.paste(det_panel, (0, src_h))
        grid.paste(pose_panel, (src_w, src_h))
        grid.resize((src_w, src_h)).save(out_dir / f"{i:05d}.jpg", quality=88)
        n = i + 1

    if n == 0:
        print("no frames decoded", file=sys.stderr)
        return 1

    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-framerate",
            str(a.fps),
            "-i",
            str(out_dir / "%05d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            a.output,
        ],
        check=True,
    )
    print(f"{n} frames -> {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

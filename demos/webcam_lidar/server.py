"""Webcam -> MoGe-2 depth -> Lidar-style point cloud, broadcast over WebSocket.

One frame in, one cloud out: nothing accumulates across frames except the depth EMA.

    uv run server.py                  # voxel mode (4 cm)
    uv run server.py --scanlines      # simulated 32-line Lidar
    uv run server.py --synthetic      # no webcam / no model: analytic room, for testing

Wire format (little-endian), one binary WebSocket message per frame:

    0   4s   magic b"LPC1"
    4   u32  frame sequence number
    8   u32  N points
    12  u32  mode (0 = voxel, 1 = scanlines)
    16  f64  server send time, unix seconds (viewer shows latency when on the same clock)
    24  f16[N] x, f16[N] y, f16[N] z   (planar)

Coordinates are sensor-centred, right-handed, Y up, looking down -Z (the Three.js
convention), in metres. OpenCV's camera frame (Y down, Z forward) is converted here so
the viewer never has to know about it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

MAGIC = b"LPC1"
HEADER = struct.Struct("<4sIIId")
MODE_VOXEL, MODE_SCAN = 0, 1

_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
# A unit-slope ramp gives a 3x3 Sobel response of 8, so dividing by 8 turns the response
# back into "depth change per pixel".
_SOBEL_GAIN = 8.0


# --------------------------------------------------------------------------------------
# Frame sources
# --------------------------------------------------------------------------------------
class Camera:
    """Grabs frames on its own thread and keeps only the newest one.

    Reading the camera inline would make inference wait on exposure and, worse, consume
    frames from the driver's queue that are already stale; the viewer would then lag the
    world by the queue depth.
    """

    def __init__(self, source: int | str, width: int, height: int) -> None:
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source {source!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.is_file = isinstance(source, str)
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._cond = threading.Condition()
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        period = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30.0) if self.is_file else 0.0
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:  # loop a video file
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue
            with self._cond:
                self._frame, self._seq = frame, self._seq + 1
                self._cond.notify_all()
            if period:
                time.sleep(period)

    def latest(self, after_seq: int, timeout: float = 2.0) -> tuple[int, np.ndarray]:
        """Block until a frame newer than `after_seq` exists, then return it."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > after_seq, timeout):
                raise TimeoutError("camera produced no frame")
            assert self._frame is not None
            return self._seq, self._frame

    def close(self) -> None:
        self._stop = True
        self.cap.release()


# --------------------------------------------------------------------------------------
# Depth estimators. Both return (depth HxW float32 on device, inf/nan = invalid,
# intrinsics (fx, fy, cx, cy) normalised by image width/height).
# --------------------------------------------------------------------------------------
class MoGeDepth:
    def __init__(self, repo: str, device: torch.device, fp16: bool, num_tokens: int | None):
        from moge.model.v2 import MoGeModel

        self.device = device
        self.fp16 = fp16 and device.type == "cuda"
        self.num_tokens = num_tokens
        self.model = MoGeModel.from_pretrained(repo).to(device).eval()

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(rgb).to(self.device, non_blocking=True)
        img = img.permute(2, 0, 1).float().div_(255.0)
        out = self.model.infer(img, num_tokens=self.num_tokens, use_fp16=self.fp16)
        k = out["intrinsics"]
        depth = out["depth"].float()
        depth = torch.where(out["mask"].bool(), depth, torch.full_like(depth, math.nan))
        return depth, torch.stack([k[0, 0], k[1, 1], k[0, 2], k[1, 2]]).float()


class SyntheticDepth:
    """A box room with a cube in it, ray-cast analytically.

    It imitates the two defects that matter for the acceptance criteria: per-pixel depth
    noise (wavy walls) and blurred depth across discontinuities (flying pixels). A
    5x5 box blur of a step is exactly what a regression network does at an object edge.
    """

    def __init__(
        self, device: torch.device, noise: float = 0.01, box_x: float | None = None
    ) -> None:
        self.device = device
        self.noise = noise
        self.box_x = box_x  # None: the cube slides back and forth
        self.t0 = time.perf_counter()
        self.k = torch.tensor([0.95, 0.95 * 518 / 291, 0.5, 0.5], device=device)

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = rgb.shape[:2]
        fx, fy, cx, cy = self.k.tolist()
        v, u = torch.meshgrid(
            (torch.arange(h, device=self.device) + 0.5) / h,
            (torch.arange(w, device=self.device) + 0.5) / w,
            indexing="ij",
        )
        dx, dy = (u - cx) / fx, (v - cy) / fy  # ray with dz = 1, OpenCV frame
        inf = torch.full_like(dx, math.inf)
        # Room: walls at x=-2 / x=2.5, floor y=1.2 (Y down), ceiling y=-1.5, back z=5.
        z = torch.minimum(inf, torch.full_like(dx, 5.0))
        for plane, coord in ((-2.0, dx), (2.5, dx), (1.2, dy), (-1.5, dy)):
            t = plane / coord
            z = torch.where((t > 0) & (t < z), t, z)
        # A 0.8 m cube sliding left and right, front face at z = 2.5.
        bx = self.box_x
        if bx is None:
            bx = 0.6 * math.sin(0.5 * (time.perf_counter() - self.t0))
        hit = ((dx * 2.5 - bx).abs() < 0.4) & ((dy * 2.5 - 0.8).abs() < 0.4)
        z = torch.where(hit & (z > 2.5), torch.full_like(z, 2.5), z)
        z = z * (1 + self.noise * torch.randn_like(z))
        z = F.avg_pool2d(z[None, None], 5, stride=1, padding=2, count_include_pad=False)[0, 0]
        return z, self.k.clone()


# --------------------------------------------------------------------------------------
# Depth -> point cloud
# --------------------------------------------------------------------------------------
@dataclass
class CloudParams:
    alpha: float = 0.3
    edge_thresh: float = 0.05
    edge_dilate: int = 1
    voxel: float = 0.04
    max_depth: float = 10.0
    scanlines: bool = False
    n_lines: int = 32
    elev_min: float = -25.0
    elev_max: float = 15.0
    az_step: float = 0.25


class CloudBuilder:
    def __init__(self, p: CloudParams) -> None:
        self.p = p
        self.depth: torch.Tensor | None = None
        self.k: torch.Tensor | None = None
        self._grid: tuple[torch.Tensor, torch.Tensor] | None = None

    # -- temporal ---------------------------------------------------------------------
    def smooth(self, depth: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """EMA with weight `alpha` on the new frame.

        A pixel invalid now is invalid in the output: holding the old value would leave
        a ghost of whatever just moved away. The intrinsics are smoothed too, because
        MoGe re-estimates the focal length every frame and a jittering focal length
        makes the whole cloud breathe even when every depth is perfectly stable.
        """
        a = self.p.alpha
        depth = torch.where(torch.isfinite(depth) & (depth > 0), depth, math.nan)
        if self.depth is None or self.depth.shape != depth.shape:
            self.depth, self.k = depth, k
        else:
            assert self.k is not None
            prev = self.depth
            blended = a * depth + (1 - a) * prev
            self.depth = torch.where(torch.isnan(prev), depth, blended)
            self.k = a * k + (1 - a) * self.k
        return self.depth, self.k

    # -- spatial ----------------------------------------------------------------------
    def edges(self, depth: torch.Tensor) -> torch.Tensor:
        """True where a pixel sits on (or next to) a depth discontinuity, or is invalid.

        The gradient is normalised by depth itself: |grad d| / d is the relative depth
        change per pixel. That makes the threshold scale-free -- the same value means the
        same thing for a cup at 0.5 m and a wall at 5 m -- and stable from frame to
        frame, which a per-frame min-max normalisation is not (one large step elsewhere
        in the image would move the threshold everywhere).

        In these units the threshold is 0.05, not 0.15. A real surface stays low: a
        plane at 85 deg incidence with ~0.002 rad per pixel changes ~0.02 per pixel. But
        a network does not draw an occlusion as a step, it smears it over a few pixels,
        and a 2:1 depth step smeared over 5 px is only 0.10-0.14 per pixel -- under
        0.15, so exactly the flying points this filter exists for would pass. 0.05 sits
        between the two with margin on both sides (see test_pointcloud.py).
        """
        bad = ~torch.isfinite(depth)
        d = torch.nan_to_num(depth, nan=0.0, posinf=0.0)[None, None]
        kx = _SOBEL_X.to(d)[None, None]
        dp = F.pad(d, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(dp, kx)
        gy = F.conv2d(dp, kx.transpose(-1, -2))
        rel = torch.sqrt(gx * gx + gy * gy)[0, 0] / (_SOBEL_GAIN * d[0, 0].clamp_min(1e-6))
        edge = (rel > self.p.edge_thresh) | bad
        # A Sobel response is 3 px wide but a network's smear across an occlusion is
        # often wider; dilating catches the stragglers that make "flying" points.
        r = self.p.edge_dilate
        if r > 0:
            edge = F.max_pool2d(edge[None, None].float(), 2 * r + 1, 1, r)[0, 0] > 0
        return edge | bad

    def _pixel_grid(self, h: int, w: int, device: torch.device):
        if self._grid is None or self._grid[0].shape != (h, w):
            v, u = torch.meshgrid(
                (torch.arange(h, device=device) + 0.5) / h,
                (torch.arange(w, device=device) + 0.5) / w,
                indexing="ij",
            )
            self._grid = (u, v)
        return self._grid

    def unproject(self, depth: torch.Tensor, k: torch.Tensor, keep: torch.Tensor):
        u, v = self._pixel_grid(*depth.shape, depth.device)
        fx, fy, cx, cy = k
        z = depth[keep]
        x = (u[keep] - cx) / fx * z
        y = (v[keep] - cy) / fy * z
        return torch.stack([x, -y, -z], dim=1)  # OpenCV -> Y up, -Z forward

    def voxel_downsample(self, pts: torch.Tensor) -> torch.Tensor:
        """Centroid per occupied voxel. Averaging (rather than keeping one sample) is
        itself a denoiser: a wall's points inside one 4 cm cell collapse to their mean."""
        if pts.shape[0] == 0:
            return pts
        q = torch.floor(pts / self.p.voxel).long()
        q -= q.min(dim=0).values
        dims = q.max(dim=0).values + 1
        key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
        _, inv, counts = torch.unique(key, return_inverse=True, return_counts=True)
        sums = torch.zeros(counts.shape[0], 3, device=pts.device, dtype=pts.dtype)
        sums.index_add_(0, inv, pts)
        return sums / counts[:, None].to(pts.dtype)

    def scanlines(self, depth: torch.Tensor, k: torch.Tensor, edge: torch.Tensor):
        """Sample depth along the rays of a virtual 32-line spinning Lidar at the camera.

        Each ray (elevation, azimuth) is projected into the image and the depth is read
        bilinearly. Bilinear sampling straddling a discontinuity would invent a point in
        mid-air, so the edge mask is sampled the same way and any ray whose footprint
        touches an edge or invalid pixel is dropped. Rays outside the camera's field of
        view (the top and bottom rings, with a typical webcam) simply return nothing.
        """
        p = self.p
        dev = depth.device
        el = torch.deg2rad(torch.linspace(p.elev_min, p.elev_max, p.n_lines, device=dev))
        az = torch.deg2rad(torch.arange(-90.0 + p.az_step / 2, 90.0, p.az_step, device=dev))
        el, az = torch.meshgrid(el, az, indexing="ij")
        # Ray directions in the OpenCV frame (Y down, Z forward).
        dx = torch.cos(el) * torch.sin(az)
        dy = -torch.sin(el)
        dz = torch.cos(el) * torch.cos(az)
        fx, fy, cx, cy = k
        u = fx * dx / dz + cx
        v = fy * dy / dz + cy
        inside = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        u, v, dx, dy, dz = u[inside], v[inside], dx[inside], dy[inside], dz[inside]
        grid = torch.stack([2 * u - 1, 2 * v - 1], dim=-1)[None, None]  # 1x1xNx2
        planes = torch.stack([torch.nan_to_num(depth, nan=0.0, posinf=0.0), edge.float()])
        s = F.grid_sample(
            planes[None], grid, mode="bilinear", padding_mode="border", align_corners=False
        )[0, :, 0]
        zs, bad = s[0], s[1]
        ok = (bad == 0) & (zs > 0) & (zs < p.max_depth)
        t = zs[ok] / dz[ok]  # depth is z-depth; scale the unit ray to reach it
        return torch.stack([dx[ok] * t, -dy[ok] * t, -dz[ok] * t], dim=1)

    def __call__(self, depth: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        depth, k = self.smooth(depth, k)
        edge = self.edges(depth)
        if self.p.scanlines:
            return self.scanlines(depth, k, edge)
        keep = ~edge & (depth < self.p.max_depth)
        return self.voxel_downsample(self.unproject(depth, k, keep))


def pack(pts: np.ndarray, seq: int, mode: int) -> bytes:
    """float32 Nx3 -> header + planar float16 x, y, z."""
    planar = np.ascontiguousarray(pts.T, dtype=np.float16)
    return HEADER.pack(MAGIC, seq, pts.shape[0], mode, time.time()) + planar.tobytes()


# --------------------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------------------
class Timer:
    """Rolling means of named stages, printed once a second."""

    def __init__(self, window: int = 30) -> None:
        self.hist: dict[str, deque[float]] = {}
        self.window = window
        self.last_print = time.perf_counter()
        self.frames: deque[float] = deque(maxlen=window)

    def add(self, **ms: float) -> None:
        for name, v in ms.items():
            self.hist.setdefault(name, deque(maxlen=self.window)).append(v)

    def tick(self, extra: str) -> None:
        now = time.perf_counter()
        self.frames.append(now)
        if now - self.last_print < 1.0 or len(self.frames) < 2:
            return
        self.last_print = now
        fps = (len(self.frames) - 1) / (self.frames[-1] - self.frames[0])
        parts = [f"{n} {sum(h) / len(h):6.1f}ms" for n, h in self.hist.items()]
        print(" | ".join([*parts, f"{fps:5.1f} FPS", extra]), flush=True)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


async def run(args: argparse.Namespace) -> None:
    from websockets.asyncio.server import broadcast, serve

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    else:
        print("warning: running on CPU, expect well under 8 FPS", flush=True)

    in_w = args.width
    in_h = round(args.cam_height * in_w / args.cam_width)
    if args.synthetic:
        cam = None
        estimator = SyntheticDepth(device)
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        cam = Camera(source, args.cam_width, args.cam_height)
        estimator = MoGeDepth(args.model, device, not args.fp32, args.num_tokens)

    builder = CloudBuilder(
        CloudParams(
            alpha=args.alpha,
            edge_thresh=args.edge_thresh,
            edge_dilate=args.edge_dilate,
            voxel=args.voxel,
            max_depth=args.max_depth,
            scanlines=args.scanlines,
        )
    )
    mode = MODE_SCAN if args.scanlines else MODE_VOXEL

    def step(last_seq: int) -> tuple[int, np.ndarray, dict[str, float]]:
        t0 = time.perf_counter()
        if cam is None:
            seq, frame = last_seq + 1, np.zeros((in_h, in_w, 3), np.uint8)
        else:
            seq, frame = cam.latest(last_seq)
            frame = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t1 = time.perf_counter()
        depth, k = estimator(frame)
        _sync(device)
        t2 = time.perf_counter()
        with torch.inference_mode():
            pts = builder(depth, k).float().cpu().numpy()
        t3 = time.perf_counter()
        ms = {"cap": (t1 - t0) * 1e3, "infer": (t2 - t1) * 1e3, "post": (t3 - t2) * 1e3}
        return seq, pts, ms

    clients: set = set()

    async def handler(ws) -> None:
        clients.add(ws)
        print(f"client connected ({len(clients)})", flush=True)
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)

    timer = Timer()
    period = 1.0 / args.fps
    # Compression off: float16 coordinates barely deflate and the CPU cost is real.
    async with serve(handler, args.host, args.port, compression=None, max_size=None):
        print(
            f"ws://{args.host}:{args.port}  mode={'scanlines' if args.scanlines else 'voxel'}"
        )
        seq = 0
        try:
            while True:
                t_start = time.perf_counter()
                seq, pts, ms = await asyncio.to_thread(step, seq)
                t4 = time.perf_counter()
                payload = pack(pts, seq, mode)
                broadcast(clients, payload)
                # broadcast() only queues onto each socket; this is pack + enqueue time.
                # Wire time shows up in the viewer's latency readout instead.
                ms["send"] = (time.perf_counter() - t4) * 1e3
                timer.add(**ms)
                timer.tick(
                    f"{pts.shape[0]:6d} pts  {len(payload) / 1024:6.0f} KiB  "
                    f"{len(clients)} client(s)"
                )
                await asyncio.sleep(max(0.0, period - (time.perf_counter() - t_start)))
        finally:
            if cam is not None:
                cam.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", default="0", help="camera index or video file path")
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--width", type=int, default=518, help="inference input width")
    ap.add_argument("--model", default="Ruicheng/moge-2-vits-normal")
    ap.add_argument(
        "--num-tokens",
        type=int,
        default=None,
        help="MoGe token budget; lower is faster (model default if unset)",
    )
    ap.add_argument("--fp32", action="store_true", help="disable fp16 inference")
    ap.add_argument("--device", default=None)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--fps", type=float, default=10.0, help="target output rate (cap)")
    ap.add_argument("--alpha", type=float, default=0.3, help="EMA weight of the new frame")
    ap.add_argument(
        "--edge-thresh",
        type=float,
        default=0.05,
        help="relative depth change per pixel above which a pixel is an edge",
    )
    ap.add_argument("--edge-dilate", type=int, default=1, help="edge mask dilation, px")
    ap.add_argument("--voxel", type=float, default=0.04, help="voxel size, m")
    ap.add_argument("--max-depth", type=float, default=10.0, help="drop points beyond, m")
    ap.add_argument(
        "--scanlines",
        action="store_true",
        help="simulate a 32-line Lidar instead of voxel downsampling",
    )
    ap.add_argument(
        "--synthetic", action="store_true", help="analytic test scene instead of webcam + model"
    )
    return ap.parse_args(argv)


def main() -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()

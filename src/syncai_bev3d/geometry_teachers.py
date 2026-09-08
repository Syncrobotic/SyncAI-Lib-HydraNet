"""Any-view geometry models as commissioning witnesses: one plate in, a lens and a depth out.

Two models, asked the same two questions the fleet cannot answer from inside (PLAN §7a.32):

* **What is the lens?** 22 of 23 selling-floor cameras carry
  `vfov_source: fleet_hardware_assumed` (PLAN §7.19); Taichung-cam01's tile grid is the
  only measured vfov there is, and every metre downstream is divided by the focal length
  that angle implies. A model that predicts intrinsics from one image is an independent
  estimate of that number -- MapAnything gave 38.26 deg against 70.4 and was refused for
  it (`tools/commissioning/map_anything_eval.py`). This module asks Depth Anything 3 and
  VGGT the same question on the same undistorted plate.
* **Where is the floor, and how tall is the furniture?** Depth-Anything V2 Metric-Indoor
  has been the world model's depth since commissioning began, and its measured weakness
  is textureless white surfaces -- wall and column heights are drawn at a stated constant
  because of it, and the Tao-Hsin pair's white fixtures are the teacher blindness the
  commissioning masks record. `tools/commissioning/geometry_bench.py` scores any depth
  source against the floor the commissioned camera already knows is at height zero; this
  module gives it two more sources to score.

**What this is not.** Neither model is a tape measure. Single-image focal estimation is
ill-posed -- focal length and scene scale are entangled in one view and a network
separates them only through learned priors about how big things are -- so a vfov from
here is a third estimate to set beside the tile grid and the assumption, never a
replacement for either. And VGGT's depth is *relative*: a scale has to be fitted before
the bench can read it in metres, and whoever fits one says so in the source's label.

Two conventions this module holds, both learned the expensive way elsewhere in the tree:

* **The revision is an argument, not a constant.** Every other teacher pins a
  `*_REVISION` commit id beside its repo id, and `tests/test_teacher_revisions_are_pinned.py`
  refuses a load without one. These two were wired up on a machine that could not reach
  the Hub, so the commit ids could not be read -- and a pin copied from memory is a pin
  that names nothing. So `revision` is required at every call site, :func:`pinned`
  refuses anything but a full 40-character commit id, and the first run on the
  commissioning box records the id it used in its own output. Whoever promotes either
  model to a constant takes the id from that run, not from upstream `main`.
* **Intrinsics are read at the size the model saw.** Both models resize the plate to
  their own working resolution (DA3: longer side 504; VGGT: width 518, height to a
  multiple of 14) and return `K` for *that* image. A vfov is invariant to uniform scaling
  and not to a changed aspect, so :func:`aspect_drift` is recorded on every reading, and
  the depth is resized back to the plate's own lattice before anyone unprojects it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

#: `depth-anything/DA3NESTED-GIANT-LARGE` is the Giant geometry with the Metric-Large
#: scale head nested inside it, and its README states its depth is already in metres.
#: That claim is exactly what the bench's floor half tests.
DA3_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE"
#: VGGT-1B predicts camera intrinsics, extrinsics, depth and point maps from any number
#: of views; single-view is a case it "was never trained for" (its README). Its depth is
#: up to scale.
VGGT_MODEL = "facebook/VGGT-1B"

#: The working resolutions the two models' own loaders use. Restated here so
#: :func:`vggt_preprocess` can reproduce `load_and_preprocess_images` on an array rather
#: than a file path, and so a test can hold the drift figure without a model.
DA3_PROCESS_RES = 504
VGGT_WIDTH = 518
VGGT_PATCH = 14

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def pinned(revision: str) -> str:
    """Refuse a revision that is not a full commit id -- `"main"` moves, a tag can be moved.

    The same rule `tests/test_teacher_revisions_are_pinned.py` applies to the constants
    the other teachers carry, applied at run time to an argument because this module has
    no constant to check.
    """
    if not isinstance(revision, str) or not FULL_SHA.match(revision):
        raise ValueError(
            f"revision {revision!r} is not a 40-character commit id. Read it from "
            "`huggingface_hub.HfApi().model_info(repo_id).sha` on a machine that can reach "
            "the Hub, and record it with the numbers it produced."
        )
    return revision


@dataclass(frozen=True)
class GeometryReading:
    """What one model said about one plate. Every field is at the size stated beside it."""

    source: str
    model: str
    revision: str
    #: (H, W) float32 on the *input* plate's lattice. NaN where the model gave nothing.
    depth: np.ndarray
    #: 3x3, in pixels of the *processed* image, which is what the model saw.
    intrinsics: np.ndarray
    processed_hw: tuple[int, int]
    input_hw: tuple[int, int]
    #: `metric_claimed` -- the model says metres and the bench decides; `relative` -- the
    #: scale is undefined and a caller that fits one must say so.
    scale: str
    vfov_deg: float
    #: processed aspect / input aspect - 1. Zero means the vfov was read on the same
    #: picture; a few tenths of a percent is the patch rounding both loaders do.
    aspect_drift: float


def vfov_from_intrinsics(k_matrix: np.ndarray, height_px: int) -> float:
    """The vertical field of view `K` implies for an image `height_px` tall, in degrees.

    The inverse of `Camera.from_vfov`: `fy = (H / 2) / tan(vfov / 2)`.
    """
    fy = float(np.asarray(k_matrix)[1, 1])
    if not (fy > 0):
        raise ValueError(f"fy = {fy}; an intrinsics matrix with no positive focal has no vfov")
    return math.degrees(2.0 * math.atan((height_px / 2.0) / fy))


def aspect_drift(input_hw: tuple[int, int], processed_hw: tuple[int, int]) -> float:
    """How far the model's working image departed from the plate's shape.

    A vfov read off a cropped or padded image is a vfov of a different picture. Reported
    as a ratio minus one so `0.0` reads as "same shape" and `0.009` as "0.9% -- the
    multiple-of-14 rounding", which is what both loaders produce on a 16:9 plate.
    """
    ih, iw = input_hw
    ph, pw = processed_hw
    return (pw / ph) / (iw / ih) - 1.0


def resize_depth(depth: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """A model's depth map, bilinearly onto another lattice, holes kept as NaN.

    Non-positive and non-finite values are holes: they are set aside before the resize
    so they do not bleed into their neighbours as small depths, and re-imposed after.
    """
    d = np.asarray(depth, dtype=np.float32).squeeze()
    if d.ndim != 2:
        raise ValueError(f"a depth map is 2-D; got shape {d.shape}")
    valid = np.isfinite(d) & (d > 0)
    h, w = hw
    if d.shape == (h, w):
        return np.where(valid, d, np.nan).astype(np.float32)
    # Normalised interpolation: resample the zero-filled depth and the validity mask with
    # the same kernel and divide, so a pixel half-covered by a hole reads the depth of its
    # valid half rather than half of it. Pixels mostly hole stay holes.
    filled = np.where(valid, d, 0.0).astype(np.float32)
    out = np.asarray(
        Image.fromarray(filled, mode="F").resize((w, h), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    mask = np.asarray(
        Image.fromarray(valid.astype(np.float32), mode="F").resize(
            (w, h), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(mask > 0.5, out / mask, np.nan)
    return out.astype(np.float32)


def reading_from_arrays(
    *,
    source: str,
    model: str,
    revision: str,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    processed_hw: tuple[int, int],
    input_hw: tuple[int, int],
    scale: str,
) -> GeometryReading:
    """Assemble a reading from what a model returned. Pure, so it is what the tests hold.

    The two model runners below do nothing but call their model and hand the arrays here,
    which is what keeps the conversion -- the part that can be wrong quietly -- testable
    without a checkpoint.
    """
    k = np.asarray(intrinsics, dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"intrinsics are 3x3; got shape {k.shape}")
    if scale not in ("metric_claimed", "relative"):
        raise ValueError(f"scale must be 'metric_claimed' or 'relative', not {scale!r}")
    return GeometryReading(
        source=source,
        model=model,
        revision=revision,
        depth=resize_depth(depth, input_hw),
        intrinsics=k,
        processed_hw=(int(processed_hw[0]), int(processed_hw[1])),
        input_hw=(int(input_hw[0]), int(input_hw[1])),
        scale=scale,
        vfov_deg=vfov_from_intrinsics(k, int(processed_hw[0])),
        aspect_drift=aspect_drift(input_hw, processed_hw),
    )


def _device(device: str | None) -> str:
    """`cuda` if there is one, else `cpu`. Not `syncai_hydranet.utils.device.pick_device`:
    `tests/test_package_boundaries.py` names the four hydranet edges this package may use
    and the serving utilities are not among them, and a once-per-camera teacher does not
    need the kernel-architecture refusal that exists for the training box."""
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Depth Anything 3


def da3_reading(rgb: np.ndarray, revision: str, device: str | None = None) -> GeometryReading:
    """Depth Anything 3 on one undistorted plate: metres claimed, and a lens.

    `depth_anything_3` is not a dependency of this repository -- like MapAnything it wants
    its own environment (`pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3`
    plus `xformers`), so it is imported here and nowhere else. `inference` takes an array
    directly; `process_res` is left at the loader's own default so the reading describes
    the model as its authors run it.
    """
    from depth_anything_3.api import DepthAnything3

    dev = _device(device)
    model = DepthAnything3.from_pretrained(DA3_MODEL, revision=pinned(revision)).to(dev).eval()
    pred = model.inference([np.ascontiguousarray(rgb)], process_res=DA3_PROCESS_RES)
    processed = np.asarray(pred.processed_images)
    return reading_from_arrays(
        source="da3",
        model=DA3_MODEL,
        revision=revision,
        depth=np.asarray(pred.depth)[0],
        intrinsics=np.asarray(pred.intrinsics)[0],
        processed_hw=(int(processed.shape[1]), int(processed.shape[2])),
        input_hw=(int(rgb.shape[0]), int(rgb.shape[1])),
        scale="metric_claimed",
    )


# ---------------------------------------------------------------------------
# VGGT


def vggt_processed_hw(input_hw: tuple[int, int]) -> tuple[int, int]:
    """The shape `load_and_preprocess_images(mode="crop")` gives a plate of this shape.

    Width to 518, height scaled with it and rounded to a multiple of 14, then cropped to
    518 if taller than that. A 1080x1920 plate becomes 294x518 -- 0.9% narrower than it
    was, the height having been rounded up to the patch.
    """
    h, w = input_hw
    new_w = VGGT_WIDTH
    new_h = int(round(h * (new_w / w) / VGGT_PATCH) * VGGT_PATCH)
    return (min(new_h, VGGT_WIDTH), new_w)


def vggt_preprocess(rgb: np.ndarray) -> np.ndarray:
    """`load_and_preprocess_images` on an array: (3, H, W) float32 in [0, 1].

    Reproduced rather than called because the original takes file paths only, and
    writing the plate to a temporary PNG to read it straight back is a round trip that
    exists for no reason. Held to the loader's rules by :func:`vggt_processed_hw`.
    """
    h, w = rgb.shape[:2]
    new_h, new_w = vggt_processed_hw((h, w))
    scaled_h = int(round(h * (new_w / w) / VGGT_PATCH) * VGGT_PATCH)
    img = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize(
        (new_w, scaled_h), Image.Resampling.BICUBIC
    )
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if scaled_h > new_h:  # centre crop, as the loader does for a tall image
        top = (scaled_h - new_h) // 2
        arr = arr[top : top + new_h]
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def vggt_reading(rgb: np.ndarray, revision: str, device: str | None = None) -> GeometryReading:
    """VGGT-1B on one undistorted plate: a lens, and a depth that is *relative*.

    Single-view is the case its README says it was never trained for, and the reading
    says `scale="relative"` so nothing downstream reads its depth as metres by accident.
    `vggt` is imported here only (`pip install git+https://github.com/facebookresearch/vggt`).
    """
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    dev = _device(device)
    model = VGGT.from_pretrained(VGGT_MODEL, revision=pinned(revision)).to(dev).eval()
    images = torch.from_numpy(vggt_preprocess(rgb)).to(dev)[None]  # (S=1, 3, H, W)
    use_amp = dev.startswith("cuda")
    if use_amp:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=dtype):
                pred = model(images)
        else:
            pred = model(images)
    _ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    depth = pred["depth"].float().cpu().numpy()[0, 0, ..., 0]
    return reading_from_arrays(
        source="vggt",
        model=VGGT_MODEL,
        revision=revision,
        depth=depth,
        intrinsics=intr.float().cpu().numpy()[0, 0],
        processed_hw=(int(images.shape[-2]), int(images.shape[-1])),
        input_hw=(int(rgb.shape[0]), int(rgb.shape[1])),
        scale="relative",
    )


READERS = {"da3": da3_reading, "vggt": vggt_reading}

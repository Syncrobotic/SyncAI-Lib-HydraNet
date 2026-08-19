"""NYU Depth V2 as a depth-supervising dataset, with its own preprocessing and a reason.

47,584 train / 654 official-test indoor frames, RGB plus dense metric Kinect depth, stored
one `.h5` per frame. It is here to train the depth head (`models/heads/depth.py`) on real
metric ground truth, because the public teacher we measured does not transfer its metres:
zero-shot on this same split it over-predicts by a flat 15% while keeping its geometry
(`scripts/eprep_teacher_nyuv2.py`).

---------------------------------------------------------------------------
WHY THIS DOES NOT USE THE SHARED TRANSFORM PIPELINE

`transforms.ToTensor` ends with

    s["masks"] = {k: torch.from_numpy(m.astype(np.int64)) for k, m in s["masks"].items()}

which is correct for every mask that existed when it was written -- they are class ids.
A depth map routed through `masks` would be **silently truncated to integer metres**: 2.73 m
becomes 2, 0.34 m becomes 0, and 0 is also this dataset's no-return sentinel, so the near
field would not merely quantise, it would be deleted and then masked out as invalid. The
loss would fall, the run would look healthy, and the head would learn a staircase.

Extending the shared pipeline to carry float targets is the right long-term fix and a wider
change than a first depth run should smuggle in -- `transforms.py` is shared with two other
active lines. So this dataset does the four steps it needs itself, using the repo's own
normalisation constants so an image is preprocessed identically either way.

**Depth resizes with nearest, never bilinear.** Interpolating across a no-return boundary
invents a depth halfway between a real surface and the 0.0 that means "no measurement",
placing a phantom surface at the edge of every specular patch. Nearest keeps every value a
value the sensor actually returned.

**Augmentation is horizontal flip only.** Not conservatism for its own sake: a random
resized crop changes the effective focal length, and metric depth is tied to it -- crop a
frame to 70% and the true depth of every pixel is unchanged while the network is shown a
scene that, geometrically, is 1.4x closer. Scale augmentation for a metric head needs the
intrinsics adjusted alongside it, which this dataset has no way to express yet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..preprocessing import IMAGENET_MEAN, IMAGENET_STD


class NyuDepthDataset(Dataset[dict[str, Any]]):
    """One `.h5` per frame, each holding `rgb` [3,H,W] uint8 and `depth` [H,W] float32 m."""

    def __init__(
        self,
        root: str | Path,
        folder: str,
        input_size,
        train: bool = False,
        supervises=("depth",),
        head_name: str = "depth",
        max_depth: float = 10.0,
    ):
        self.files = sorted((Path(root) / folder).rglob("*.h5"))
        if not self.files:
            raise FileNotFoundError(f"no .h5 files under {Path(root) / folder}")
        self.size = (int(input_size[0]), int(input_size[1]))  # (H, W)
        self.train = bool(train)
        self.supervises = list(supervises)
        # The head's config key, not a fixed string: the target dict is keyed by head name
        # and `compute_losses` looks each head up by its own name, so a config that calls
        # this head `ground_depth` has to reach a target under that name.
        self.head_name = head_name
        self.max_depth = float(max_depth)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Imported here, not at module scope: h5py is not a declared dependency of this
        # package, and a top-level import would break `import syncai_hydranet.data` for
        # every run that has nothing to do with NYUv2.
        import h5py

        with h5py.File(self.files[index], "r") as h:
            rgb = np.asarray(h["rgb"][:]).transpose(1, 2, 0)  # [H, W, 3] uint8
            depth = np.asarray(h["depth"][:], dtype=np.float32)  # [H, W] metres

        h_out, w_out = self.size
        img = Image.fromarray(rgb).resize((w_out, h_out), Image.Resampling.BILINEAR)
        # `np.array`, not `asarray`: PIL hands back a read-only buffer and the validity
        # masking below writes in place.
        dep = np.array(
            Image.fromarray(depth).resize((w_out, h_out), Image.Resampling.NEAREST),
            dtype=np.float32,
        )

        if self.train and bool(torch.rand(()) < 0.5):
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            dep = np.ascontiguousarray(dep[:, ::-1])

        arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        # Out-of-range returns are zeroed rather than clipped. Clipping to `max_depth`
        # would turn every too-far pixel into a dense wall of surfaces at exactly the
        # ceiling, and the loss would dutifully train the head to predict it.
        dep[(dep <= 0) | (dep > self.max_depth) | ~np.isfinite(dep)] = 0.0
        return {
            "image": torch.from_numpy(arr.transpose(2, 0, 1)).contiguous(),
            "targets": {self.head_name: torch.from_numpy(dep)},
            "supervises": self.supervises,
        }


class RenderedDepthDataset(Dataset[dict[str, Any]]):
    """Rendered RGB/depth pairs from `scripts/hm3d_render.py`.

    Separate from `NyuDepthDataset` despite serving the same head, because the two differ
    in the one place that matters: NYUv2's depth is a filled Kinect product with no
    invalid pixels, while a raycast returns exactly 0 where a ray escaped the mesh through
    a scan hole, and HM3D scans have holes at every window and doorway. Merging them would
    mean one loader whose validity handling is right for one source and silently generous
    for the other.

    Depth is stored as 16-bit millimetre PNG rather than float32 `.npy`: lossless to 1 mm
    across the range this needs, a third of the size, and readable by anything.
    """

    def __init__(
        self,
        root: str | Path,
        folder: str,
        input_size,
        train: bool = False,
        supervises=("depth",),
        head_name: str = "depth",
        max_depth: float = 10.0,
    ):
        # rglob, so a split may be one directory or a tree of per-scene shards. Rendering
        # parallelises one process per scene, and letting each write its own subdirectory
        # avoids renumbering frames on merge -- which is the step that would quietly pair
        # an RGB from one scene with a depth from another.
        self.files = sorted((Path(root) / folder).rglob("*_rgb.png"))
        if not self.files:
            raise FileNotFoundError(f"no *_rgb.png under {Path(root) / folder}")
        self.size = (int(input_size[0]), int(input_size[1]))
        self.train = bool(train)
        self.supervises = list(supervises)
        self.head_name = head_name
        self.max_depth = float(max_depth)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rgb_path = self.files[index]
        dep_path = rgb_path.with_name(rgb_path.name.replace("_rgb.png", "_depth.png"))
        h_out, w_out = self.size
        img = (
            Image.open(rgb_path)
            .convert("RGB")
            .resize((w_out, h_out), Image.Resampling.BILINEAR)
        )
        dep = np.array(
            Image.open(dep_path).resize((w_out, h_out), Image.Resampling.NEAREST),
            dtype=np.float32,
        )
        dep /= 1000.0  # millimetres as stored -> metres as trained

        if self.train and bool(torch.rand(()) < 0.5):
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            dep = np.ascontiguousarray(dep[:, ::-1])

        arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        dep[(dep <= 0) | (dep > self.max_depth) | ~np.isfinite(dep)] = 0.0
        return {
            "image": torch.from_numpy(arr.transpose(2, 0, 1)).contiguous(),
            "targets": {self.head_name: torch.from_numpy(dep)},
            "supervises": self.supervises,
        }

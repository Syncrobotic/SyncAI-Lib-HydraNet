"""Palettes, overlays, letterboxing and TensorBoard comparison grids.

Shared by the image and video inference CLIs and by the trainer's validation logging.

Why letterbox: ``data.transforms.Resize`` stretches straight to the target size. RUGD
is roughly 1.25:1 and the default 512x640 input is also 1.25:1, so training on it is
lossless. A phone clip is 16:9 (1.78:1) or portrait 9:16 (0.56:1); stretching that to
1.25:1 squeezes the frame horizontally by up to a factor of two. Inference always
letterboxes; training does so when ``data.letterbox`` is set.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Traversability: blocked / caution / go. The one taxonomy that does not change between
# deployments, so it needs no selection.
TRAV_COLORS = np.array([[220, 40, 40], [250, 200, 40], [40, 200, 80]], dtype=np.uint8)

# Terrain colours are per taxonomy, because the taxonomies are not the same list of
# things. There used to be a single TERRAIN_COLORS holding the off-road one, and every
# indoor and retail run was rendered with it: `wall` came out sky blue, `glass` came out
# water blue, `person` came out the orange belonging to `stairs_log`. Nothing failed --
# an index is an index -- so a val_pred grid looked plausible while naming the wrong
# things, in the one place meant to catch a model calling a floor a wall.
#
# Index order matches ``data.terrain_classes`` in the matching config.

# configs/hydranet_regnet800mf.yaml
TERRAIN_COLORS_OFFROAD = np.array(
    [
        [0, 0, 0],  # void
        [139, 90, 43],  # dirt
        [60, 180, 75],  # grass
        [0, 100, 0],  # tree_bush
        [128, 128, 128],  # pavement_concrete
        [210, 180, 140],  # gravel_mulch
        [178, 34, 34],  # building_wall
        [135, 206, 235],  # sky
        [30, 100, 220],  # water
        [255, 215, 0],  # vehicle_object
        [112, 128, 144],  # rock
        [255, 140, 0],  # stairs_log
    ],
    dtype=np.uint8,
)

# configs/hydranet_indoor.yaml. Taken from scripts/live_view_orin.py, which had been
# carrying the only indoor palette in the repo -- under a comment claiming it matched
# the training-time grids, which is how the divergence stayed invisible.
TERRAIN_COLORS_INDOOR = np.array(
    [
        [0, 0, 0],  # void
        [139, 90, 43],  # floor_hard
        [60, 180, 75],  # floor_soft
        [0, 100, 0],  # floor_metal
        [128, 128, 128],  # wet_slippery
        [255, 130, 0],  # stairs
        [190, 100, 200],  # threshold_ramp
        [130, 130, 130],  # wall
        [70, 200, 235],  # glass
        [245, 220, 60],  # door
        [230, 80, 80],  # obstacle_furniture
        [255, 100, 180],  # person
    ],
    dtype=np.uint8,
)

# configs/hydranet_retail.yaml: the indoor 12 with display_fixture appended. Without a
# 13th entry the old code clipped it to index 11, so every display fixture in every
# rendered frame was drawn in the colour of `person`.
TERRAIN_COLORS_RETAIL = np.concatenate(
    [TERRAIN_COLORS_INDOOR, np.array([[120, 80, 200]], dtype=np.uint8)]  # display_fixture
)

# configs/hydranet_retail_objects.yaml: the object taxonomy, seven classes.
#
# Deliberately keyed to the retail palette rather than freshly picked, so that the two
# retail taxonomies can be read side by side in a review: floor keeps the brown it has
# in every indoor and retail grid, wall keeps its grey, person keeps its pink, and
# `fixture` takes display_fixture's violet because that is the class it absorbs.
#
# Only the two new classes need a colour of their own. `column` gets a deeper, bluer
# grey than `wall` -- near it, because that is where it came from, and separable from
# it, because telling those two apart is the entire reason this taxonomy exists.
# `product` gets amber, the one hue nothing else in any palette here uses.
TERRAIN_COLORS_RETAIL_OBJECTS = np.array(
    [
        [0, 0, 0],  # void
        [139, 90, 43],  # floor      -- as floor_hard everywhere else
        [130, 130, 130],  # wall     -- as wall everywhere else
        [70, 90, 150],  # column     -- new: wall's neighbour, not wall's twin
        [120, 80, 200],  # fixture   -- as display_fixture, the class it absorbs
        [255, 170, 30],  # product   -- new
        [255, 100, 180],  # person   -- as person everywhere else
    ],
    dtype=np.uint8,
)

# Most specific first, and the order is load-bearing: retail is indoor plus one class,
# so it contains every indoor marker and would match `floor_hard` if that came first.
# Selection is by name rather than by length because indoor and off-road are both
# twelve long -- length is precisely the test that would keep agreeing while the
# colours meant different things.
#
# The object taxonomy is matched on `product`, which no other taxonomy contains. Its
# own class names are otherwise a subset of words the others use (`wall`, `person`),
# so a less distinctive marker would be a live collision rather than a theoretical one.
_TERRAIN_PALETTES = (
    ("product", TERRAIN_COLORS_RETAIL_OBJECTS),
    ("display_fixture", TERRAIN_COLORS_RETAIL),
    ("floor_hard", TERRAIN_COLORS_INDOOR),
    ("dirt", TERRAIN_COLORS_OFFROAD),
)


def _generated_palette(n: int) -> np.ndarray:
    """Distinct colours for a taxonomy nobody has named.

    Deterministic, and index 0 stays black to match the `void` convention everywhere
    else. These carry no meaning by design: they are for a config that declares no
    ``terrain_classes``, where there are no class names for a colour to contradict.
    """
    import colorsys

    colors = [(0, 0, 0)]
    for i in range(max(n - 1, 0)):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.75, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(colors or [(0, 0, 0)], dtype=np.uint8)


def terrain_palette(classes, n_classes: int | None = None) -> np.ndarray:
    """Pick the palette belonging to a config's ``data.terrain_classes``.

    A *named* taxonomy must have a palette built for it, and one that has outgrown its
    palette is an error here rather than a wrong colour later -- that pairing is the
    whole point. A config that declares no class names is a different case and not an
    error: with nothing named, no colour can be wrong, so ``n_classes`` gets a
    generated set. Refusing here instead would only push callers toward passing some
    other taxonomy's palette, which is what this function exists to stop.
    """
    names = list(classes or [])
    if not names:
        if n_classes is None:
            raise ValueError("terrain_palette needs either class names or n_classes")
        return _generated_palette(n_classes)
    for marker, palette in _TERRAIN_PALETTES:
        if marker in names:
            if len(palette) < len(names):
                raise ValueError(
                    f"{len(names)} terrain classes but the matched palette has "
                    f"{len(palette)} colours; add the missing entries in "
                    f"utils/visualize.py"
                )
            return palette
    raise ValueError(
        f"no terrain palette for classes {names}: add one to utils/visualize.py "
        f"rather than letting an unrelated palette be indexed by these labels"
    )


def overlay(
    base: Image.Image, mask: np.ndarray, palette: np.ndarray, alpha: float = 0.45
) -> Image.Image:
    """Blend a class-index map over an image using ``palette``.

    A class the palette does not cover is an error, not something to clamp. Clamping is
    what drew retail's `display_fixture` in `person`'s colour in every frame the project
    has rendered, and a wrong colour is indistinguishable from a wrong prediction.
    """
    top = int(np.max(mask)) if mask.size else 0
    if top >= len(palette):
        raise ValueError(
            f"mask contains class {top} but the palette has {len(palette)} colours; "
            f"pass the palette matching this head's taxonomy (see terrain_palette)"
        )
    color = palette[mask]
    out = (
        np.asarray(base, dtype=np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
    ).astype(np.uint8)
    return Image.fromarray(out)


def letterbox(img: Image.Image, size, fill=(114, 114, 114)):
    """Scale to fit inside ``(H, W)`` preserving aspect ratio, then centre-pad.

    Returns ``(image, (x0, y0, w, h))``; the region locates the real content inside the
    canvas so predictions can be cropped back to the source aspect ratio.
    """
    h, w = size
    ow, oh = img.size
    s = min(w / ow, h / oh)
    nw, nh = max(round(ow * s), 1), max(round(oh * s), 1)
    canvas = Image.new("RGB", (w, h), fill)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.Resampling.BILINEAR), (x0, y0))
    return canvas, (x0, y0, nw, nh)


def preprocess(img: Image.Image, size, use_letterbox: bool = True):
    """Turn one frame into ``(tensor, canvas, region)`` the way every inference path does.

    ``region`` locates the real content inside the canvas, so predictions can be cropped
    back and the output keeps the source aspect ratio instead of carrying grey bars.

    This existed four times: byte-identical in bev_video, live_view_ros and
    probe_ros_realsense, and once more in infer_image with the letterbox switch. Four
    copies of a normalisation is four chances to feed the model something it was not
    trained on, and the failure mode is a model that merely looks worse in one tool.
    """
    import torch

    from ..data.transforms import IMAGENET_MEAN, IMAGENET_STD

    h, w = size
    img = img.convert("RGB")
    if use_letterbox:
        img, region = letterbox(img, size)
    else:
        img, region = img.resize((w, h), Image.Resampling.BILINEAR), (0, 0, w, h)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1))[None], img, region


def crop_box(arr: np.ndarray, region) -> np.ndarray:
    """Crop an ``HxW`` array produced on a letterboxed canvas back to its content."""
    x0, y0, w, h = region
    return arr[y0 : y0 + h, x0 : x0 + w]


# ---------------------------------------------------------------------------
# Training-time monitoring
# ---------------------------------------------------------------------------


def denormalize(t) -> np.ndarray:
    """Convert a normalized ``[3, H, W]`` tensor back to an ``[H, W, 3]`` uint8 array."""
    from ..data.transforms import IMAGENET_MEAN, IMAGENET_STD

    arr = t.detach().cpu().numpy().transpose(1, 2, 0)
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def prediction_grid(images, preds, gts, palette, max_n: int = 4, gap: int = 4) -> np.ndarray:
    """Build an "input | prediction | label" comparison grid, one row per sample.

    Returns ``[H, W, 3]`` uint8, ready for ``SummaryWriter.add_image(..., dataformats="HWC")``.
    Loss curves only tell you the loss is falling; this shows whether the model is
    calling an entire floor a wall, and how much of the frame is ignore padding.
    """
    rows = []
    n = min(max_n, len(images))
    for i in range(n):
        base = Image.fromarray(denormalize(images[i]))
        p = np.asarray(preds[i].detach().cpu().numpy(), dtype=np.int64)
        g = np.asarray(gts[i].detach().cpu().numpy(), dtype=np.int64)
        pred_img = np.asarray(overlay(base, p, palette))
        # Paint label ignore (255) black so unsupervised regions are obvious.
        gt_img = np.asarray(overlay(base, np.where(g == 255, 0, g), palette))
        gt_img = np.where((g == 255)[..., None], 0, gt_img).astype(np.uint8)
        sep = np.full((gt_img.shape[0], gap, 3), 255, np.uint8)
        rows.append(np.concatenate([np.asarray(base), sep, pred_img, sep, gt_img], axis=1))
    if not rows:
        return np.zeros((1, 1, 3), np.uint8)
    w = max(r.shape[1] for r in rows)
    hsep = np.full((gap, w, 3), 255, np.uint8)
    out = []
    for r in rows:
        if r.shape[1] < w:
            r = np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255)
        out += [r, hsep]
    return np.concatenate(out[:-1], axis=0)

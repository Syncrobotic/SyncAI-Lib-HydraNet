"""The shapes of the two per-frame scene payloads this package emits.

There are two of them because there are two ways to know where the floor is. `bev.scene`
fits a ground plane and projects a segmentation mask onto it, which is what a single RGB
camera can do. `depth_scene.build_scene` back-projects a depth image, which is what a
D435-class sensor can do. Both answer "what is the floor and what is standing on it", in
metres, for the same renderer.

They are **not** interchangeable, and that is the whole reason this module exists. The
two payloads share their top-level key *names* and almost none of their contents:

    plane  grid -> x_min x_max z_min z_max cell_m shape
    depth  grid -> x_min cell_m cells nx nz blind_fraction

A renderer that reads `scene["grid"]["shape"]` works against a plane payload and raises
KeyError against a depth one; `scene["objects"][0]["label"]` does the same. Until now the
only thing standing between those two was the `source` string and whoever remembered to
check it -- both functions were annotated `-> dict`, so nothing could tell a consumer it
had wired up the wrong half of the package. (`bev.scene` was in fact annotated `-> dict`
while returning a `(dict, ndarray)` tuple, and every caller unpacked two values; the
annotation had been wrong for as long as it had existed, because nothing read it.)

`source` is a `Literal`, so `Scene` is a discriminated union: a checker that sees

    if payload["source"] == "plane":
        rows, cols = payload["grid"]["shape"]

knows `shape` is there, and knows it is *not* there on the other branch. That is the
check the comment in `build_scene` has been asking a human to perform by hand.

These describe payloads that get serialised to JSON and handed to renderers, so they are
TypedDicts rather than dataclasses: the wire format is a plain dict either way, and a
dataclass would add a conversion step at every boundary for no gain in what can be
checked.
"""

from __future__ import annotations

from typing import Literal, TypedDict

# --------------------------------------------------------------------- objects


class PlaneObject(TypedDict):
    """One detection placed on a fitted ground plane.

    `width_m`/`height_m` are None when the box's extent could not be measured -- a
    filler number would be indistinguishable from a real one, so the renderer is made
    to say what it substitutes. See the note at the construction site in `bev.scene`.
    """

    x_m: float
    z_m: float
    range_m: float
    width_m: float | None
    height_m: float | None
    label: int
    name: str
    score: float


class DepthObjectMetrics(TypedDict):
    """What `depth_scene.object_from_box` can say from the depth inside one box.

    Split from `DepthObject` because it is genuinely a separate stage: this half comes
    from the geometry, and the detector supplies `name`/`score` afterwards. Modelling it
    as one type would make `object_from_box` look like it returns two fields it has no
    way to know.
    """

    x_m: float
    z_m: float
    range_m: float
    depth_m: float
    width_m: float
    height_m: float
    yaw_rad: float
    yaw_confidence: float
    n_points: int


class DepthObject(DepthObjectMetrics):
    """A `DepthObjectMetrics` once the detection it came from has named it."""

    name: str
    score: float


# ----------------------------------------------------------------------- grids


class PlaneGrid(TypedDict):
    """Bounds and resolution of the BEV raster. `shape` is [rows, cols]."""

    x_min: float
    x_max: float
    z_min: float
    z_max: float
    cell_m: float
    shape: list[int]


class DepthGrid(TypedDict):
    """Occupancy actually observed, rather than bounds that were configured.

    `blind_fraction` has no counterpart on the plane side: depth reports how much of the
    grid it could not see, which is the number that stops "unobserved" being drawn as
    "clear".
    """

    nx: int
    nz: int
    cell_m: float
    x_min: float
    cells: list
    blind_fraction: float


# ---------------------------------------------------------------------- scenes


class PlaneScene(TypedDict):
    """`bev.scene`'s payload: the ground-plane half of this package."""

    source: Literal["plane"]
    camera: dict[str, float]
    plane: dict[str, float]
    grid: PlaneGrid
    class_share: dict[str, float]
    objects: list[PlaneObject]


class DepthScene(TypedDict):
    """`depth_scene.build_scene`'s payload: the depth half."""

    source: Literal["depth"]
    grid: DepthGrid
    objects: list[DepthObject]


# Discriminated on `source`. Annotate a consumer with this rather than with one half,
# and narrowing on `payload["source"]` is what buys access to the fields that differ.
Scene = PlaneScene | DepthScene

__all__ = [
    "DepthGrid",
    "DepthObject",
    "DepthObjectMetrics",
    "DepthScene",
    "PlaneGrid",
    "PlaneObject",
    "PlaneScene",
    "Scene",
]

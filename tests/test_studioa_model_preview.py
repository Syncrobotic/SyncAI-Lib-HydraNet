"""New-model 3D previews must use observed geometry and preserve floor projection."""

import numpy as np
import pytest

from syncai_bev3d.studioa_preview import visible_mesh


def arrays():
    z, x = np.mgrid[0:4, 0:4].astype(float) * 0.1
    return {
        "gx": x,
        "gz": z + 1,
        "lx": x + 3,
        "lz": z + 4,
        "height": np.ones((4, 4)),
        "geom_ok": np.ones((4, 4), bool),
    }


def test_visible_floor_uses_ground_coordinates_and_objects_use_depth():
    a = arrays()
    labels = np.zeros((4, 4), np.uint8)
    confidence = np.ones((4, 4))
    vertices, faces, _, _ = visible_mesh(a, labels, confidence, stride=1)
    assert len(faces) == 18 and np.all(vertices[:, 1] == 0)
    assert vertices[:, 0].max() < 0.4 and vertices[:, 2].max() < 1.4
    labels[:] = 7
    vertices, faces, _, _ = visible_mesh(a, labels, confidence, stride=1)
    assert np.all(vertices[:, 1] == 1) and vertices[:, 0].min() == 3


def test_mesh_refuses_unsupported_or_unconfident_surfaces():
    a = arrays()
    labels = np.zeros((4, 4), np.uint8)
    with pytest.raises(ValueError, match="no valid"):
        visible_mesh(a, labels, np.zeros((4, 4)), stride=1)
    a["gx"] *= 100
    with pytest.raises(ValueError, match="no valid"):
        visible_mesh(a, labels, np.ones((4, 4)), stride=1)
    labels[:] = 255
    with pytest.raises(ValueError, match="unknown semantic"):
        visible_mesh(arrays(), labels, np.ones((4, 4)), stride=1)

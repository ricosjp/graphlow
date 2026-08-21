"""Unit tests for utils.points (as_xyz_points)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graphlow.core.backend.base import Backend, TensorLike
from graphlow.utils.dimension import get_dimension
from graphlow.utils.points import as_xyz_points


def test_as_xyz_points_returns_n3_unchanged(backend: Backend[TensorLike]):
    """Shape ``(n_points, 3)`` is returned as-is."""
    points = backend.as_tensor(np.arange(12, dtype=np.float64).reshape(4, 3))
    actual = as_xyz_points(points)
    assert actual is points
    np.testing.assert_array_equal(
        backend.to_numpy(actual), backend.to_numpy(points)
    )


def test_as_xyz_points_squeezes_feature_axis(backend: Backend[TensorLike]):
    """Shape ``(n_points, 3, 1)`` is reduced to ``(n_points, 3)``."""
    xyz = np.arange(12, dtype=np.float64).reshape(4, 3)
    points = backend.as_tensor(xyz[:, :, None], dimension={"L": 1})
    actual = as_xyz_points(points)
    assert tuple(actual.shape) == (4, 3)
    np.testing.assert_array_equal(backend.to_numpy(actual), xyz)
    assert get_dimension(actual) == get_dimension(points)


@pytest.mark.parametrize(
    "shape",
    [(4, 2), (4, 3, 2), (4,), (4, 3, 1, 1), (3, 1)],
)
def test_as_xyz_points_raises_for_invalid_shape(
    backend: Backend[TensorLike],
    shape: tuple[int, ...],
):
    """Shapes other than ``(N, 3)`` or ``(N, 3, 1)`` raise ValueError."""
    points = backend.as_tensor(np.zeros(shape, dtype=np.float64))
    with pytest.raises(ValueError, match="points must have shape"):
        as_xyz_points(points)


def test_as_xyz_points_preserves_autograd(backend: Backend[TensorLike]):
    """Squeezing ``(N, 3, 1)`` keeps the computation graph."""
    points = backend.as_tensor(
        np.arange(12, dtype=np.float64).reshape(4, 3, 1), dimension={"L": 1}
    )
    backend.to_torch(points).requires_grad_(True)
    xyz = as_xyz_points(points)
    torch.sum(backend.to_torch(xyz)).backward()
    grad = backend.to_torch(points).grad
    assert grad is not None
    assert tuple(grad.shape) == (4, 3, 1)

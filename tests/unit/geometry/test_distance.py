"""Tests for geometry.distance (Hausdorff and Chamfer)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import pyvista as pv
import torch
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from graphlow.core.mesh import TensorMesh
from graphlow.io.pyvista import from_pyvista


def _make_line_segment_mesh_from_points(
    pts: np.ndarray,
) -> TensorMesh[torch.Tensor]:
    """Build a TensorMesh with two points and one line cell."""
    lines = np.array([2, 0, 1], dtype=np.int32)
    poly = pv.PolyData(pts.astype(np.float64), lines=lines)
    return from_pyvista(poly, "torch", torch.float64)


def _make_triangle_mesh_from_points(
    pts: np.ndarray,
) -> TensorMesh[torch.Tensor]:
    """Build a TensorMesh of a triangle."""
    cells = np.array([3, 0, 1, 2])
    ctypes = np.array([5])
    grid = pv.UnstructuredGrid(cells, ctypes, pts.astype(np.float64))
    return from_pyvista(grid, "torch", torch.float64)


# =============================================================================
# Hausdorff distance (multi-point vs multi-point)
# =============================================================================


def test_hausdorff_same_points():
    """Hausdorff distance between identical point sets is 0."""
    icosphere = from_pyvista(
        pv.Icosphere(radius=1.0, nsub=3), "torch", torch.float64
    )
    target = icosphere.points
    hd = icosphere.geometry.hausdorff_distance(target)
    assert hd.shape == (1,)
    # assert hd.dimension == phlower_dimension_tensor({"L": 1})
    assert hd.numpy().item() == 0.0


def test_hausdorff_multi_vs_multi():
    """Hausdorff distance: N points vs M points.

    P = {(0,0,0), (3,0,0), (0,3,0)}
    Q = {(1,0,0), (2,0,0), (0,1,0)}
    - d(p,Q): (0,0,0)->1, (3,0,0)->1, (0,3,0)->2  => 2
    - d(Q,p): (1,0,0)->1, (2,0,0)->1, (0,1,0)->1  => 1
    So Hausdorff = max(2, 1) = 2.
    """
    pts_p = np.array(
        [[0.0, 0, 0], [3.0, 0, 0], [0.0, 3, 0]],
        dtype=np.float64,
    )
    pts_q = np.array(
        [[1.0, 0, 0], [2.0, 0, 0], [0.0, 1, 0]],
        dtype=np.float64,
    )
    mesh = _make_triangle_mesh_from_points(pts_p)
    target = mesh.backend.as_tensor(pts_q)

    hd = mesh.geometry.hausdorff_distance(target)
    assert hd.shape == (1,)
    # assert hd.dimension == phlower_dimension_tensor({"L": 1})
    assert hd.numpy().item() == 2.0


def _reference_soft_hausdorff(dists: np.ndarray, tau: float) -> float:
    """Reference Hausdorff: soft-min"""
    n_p, n_q = dists.shape
    d_p2q = tau * (math.log(n_q) - logsumexp(-dists / tau, axis=1))
    d_q2p = tau * (math.log(n_p) - logsumexp(-dists / tau, axis=0))
    h1 = tau * (logsumexp(d_p2q / tau, axis=None) - math.log(n_p))
    h2 = tau * (logsumexp(d_q2p / tau, axis=None) - math.log(n_q))
    return np.maximum(h1, h2)


@pytest.mark.parametrize("tau", [0.01, 0.5, 1.0])
def test_hausdorff_softmin(tau: float):
    """2x2 L2 distances vs explicit soft Hausdorff reference."""
    pts_p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    pts_q = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    dists = cdist(pts_p, pts_q)

    expected = _reference_soft_hausdorff(dists, tau)

    mesh = _make_line_segment_mesh_from_points(pts_p)
    target = mesh.backend.as_tensor(pts_q)
    hd = mesh.geometry.hausdorff_distance(target, softmin_temperature=tau)

    actual = hd.detach().cpu().numpy()
    np.testing.assert_almost_equal(actual, expected)


# =============================================================================
# Chamfer distance (multi-point vs multi-point)
# =============================================================================
def test_chamfer_same_points():
    """Chamfer distance between identical point sets is 0."""
    icosphere = from_pyvista(
        pv.Icosphere(radius=1.0, nsub=3), "torch", torch.float64
    )
    target = icosphere.points
    cd = icosphere.geometry.chamfer_distance(target)
    assert cd.shape == (1,)
    assert cd.numpy().item() == 0.0


def test_chamfer_multi_vs_multi():
    """Chamfer distance: N points vs M points.

    P = {(0,0,0), (3,0,0), (0,3,0)}
    Q = {(1,0,0), (2,0,0), (0,1,0)}
    CD = (1/|P|) sum_p min_q |p-q|^2 + (1/|Q|) sum_q min_p |p-q|^2.
    - d(p,Q): (0,0,0)->1, (3,0,0)->1, (0,3,0)->2; total = 4  => mean = 2
    - d(Q,p): (1,0,0)->1, (2,0,0)->1, (0,1,0)->1; total = 3  => mean = 1
    So Chamfer = 2 + 1 = 3.
    """
    pts_p = np.array(
        [[0.0, 0, 0], [3.0, 0, 0], [0.0, 3, 0]],
        dtype=np.float64,
    )
    pts_q = np.array(
        [[1.0, 0, 0], [2.0, 0, 0], [0.0, 1, 0]],
        dtype=np.float64,
    )
    mesh = _make_triangle_mesh_from_points(pts_p)
    target = mesh.backend.as_tensor(pts_q)

    cd = mesh.geometry.chamfer_distance(target)
    assert cd.shape == (1,)
    # assert cd.dimension == phlower_dimension_tensor({"L": 1})
    assert cd.numpy().item() == 3.0


def _reference_soft_chamfer(dists: np.ndarray, tau: float) -> float:
    """Reference Chamfer: soft-min"""
    n_p, n_q = dists.shape
    d_p2q = tau * (math.log(n_q) - logsumexp(-dists / tau, axis=1))
    d_q2p = tau * (math.log(n_p) - logsumexp(-dists / tau, axis=0))
    return float(np.mean(d_p2q) + np.mean(d_q2p))


@pytest.mark.parametrize("tau", [0.01, 0.5, 1.0])
def test_chamfer_softmin(tau: float):
    """2x2 squared distances vs explicit soft-min Chamfer reference."""
    # P = {(0,0,0), (1,0,0)}, Q = {(0,0,0), (2,0,0)}
    # dists_sq = [[0, 4], [1, 1]] — row1 soft-min equals hard min (tie).
    pts_p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    pts_q = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
    dists = cdist(pts_p, pts_q) ** 2

    expected = _reference_soft_chamfer(dists, tau)

    mesh = _make_line_segment_mesh_from_points(pts_p)
    target = mesh.backend.as_tensor(pts_q)
    cd = mesh.geometry.chamfer_distance(target, softmin_temperature=tau)

    actual = cd.detach().cpu().numpy()
    np.testing.assert_almost_equal(actual, expected)

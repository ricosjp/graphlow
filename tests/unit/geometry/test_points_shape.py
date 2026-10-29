"""Geometry ops accept phlower-style points of shape ``(n_points, 3, 1)``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from graphlow.core.backend.base import TensorLike
from graphlow.core.mesh import TensorMesh


def _with_feature_axis[T: TensorLike](mesh: TensorMesh[T]) -> TensorMesh[T]:
    """Store points as ``(n_points, 3, 1)`` without copying values."""
    if mesh.points.ndim == 2:
        mesh.points = mesh.points[:, :, None]
    return mesh


def test_cell_volumes_match_for_points_n31(tet_mesh: TensorMesh[TensorLike]):
    """Cell volumes are unchanged when points gain a feature axis."""
    desired = tet_mesh.backend.to_numpy(tet_mesh.geometry.cell_volumes())
    actual = _with_feature_axis(tet_mesh).geometry.cell_volumes()
    np.testing.assert_allclose(tet_mesh.backend.to_numpy(actual), desired)


def test_face_area_vectors_match_for_points_n31(
    tet_mesh: TensorMesh[TensorLike],
):
    """Face area vectors are unchanged when points gain a feature axis."""
    surface = tet_mesh.extract_surface()
    desired = surface.backend.to_numpy(surface.geometry.face_area_vectors())
    actual = _with_feature_axis(surface).geometry.face_area_vectors()
    np.testing.assert_allclose(surface.backend.to_numpy(actual), desired)


def test_isoAM_match_for_points_n31(tet_mesh: TensorMesh[TensorLike]):
    """isoAM is unchanged when points gain a feature axis."""
    desired, desired_minv = tet_mesh.geometry.isoAM()
    backend = tet_mesh.backend
    desired_am = backend.to_torch(desired).to_dense().detach().cpu().numpy()
    desired_m = backend.to_numpy(desired_minv)

    actual, actual_minv = _with_feature_axis(tet_mesh).geometry.isoAM()
    actual_am = backend.to_torch(actual).to_dense().detach().cpu().numpy()
    np.testing.assert_allclose(actual_am, desired_am)
    np.testing.assert_allclose(backend.to_numpy(actual_minv), desired_m)


def test_cell_local_fem_rigidity_match_for_points_n31(
    tet_mesh: TensorMesh[TensorLike],
):
    """FEM rigidity is unchanged when points gain a feature axis."""
    desired = tet_mesh.geometry.cell_local_fem_rigidity(rank=0)
    actual = _with_feature_axis(tet_mesh).geometry.cell_local_fem_rigidity(
        rank=0
    )
    backend = tet_mesh.backend
    for cell_type, block in desired.items():
        np.testing.assert_allclose(
            backend.to_numpy(actual[cell_type]),
            backend.to_numpy(block),
        )


def test_chamfer_distance_match_for_points_n31(
    tet_mesh: TensorMesh[TensorLike],
):
    """Chamfer distance accepts ``(N, 3, 1)`` for mesh and target points."""
    if tet_mesh.backend.name == "phlower":
        pytest.skip("torch.cdist is not implemented for PhlowerTensor")
    backend = tet_mesh.backend
    xyz = backend.to_numpy(tet_mesh.xyz)
    target = backend.as_tensor(xyz + 0.1, dimension={"L": 1})
    desired = tet_mesh.geometry.chamfer_distance(target)
    mesh = _with_feature_axis(tet_mesh)
    actual = mesh.geometry.chamfer_distance(target[:, :, None])
    np.testing.assert_allclose(
        backend.to_numpy(actual), backend.to_numpy(desired)
    )


def test_extract_surface_gradient_with_points_n31(
    tet_mesh: TensorMesh[TensorLike],
):
    """Autograd through surface areas reaches ``(N, 3, 1)`` volume points."""
    mesh = _with_feature_axis(tet_mesh)
    mesh.requires_grad(True)
    surface = mesh.extract_surface()
    assert tuple(surface.points.shape) == (surface.n_points, 3, 1)
    loss = torch.sum(surface.geometry.face_areas())
    loss.backward()
    grad = mesh.backend.to_torch(mesh.points).grad
    assert grad is not None
    assert tuple(grad.shape) == tuple(mesh.points.shape)

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pyvista as pv
import torch

from graphlow.core.backend.base import TensorLike
from graphlow.core.blocks import (
    CellBlock,
    FaceBlock,
    FixedCellBlock,
    FixedFaceBlock,
    JaggedCellBlock,
    JaggedFaceBlock,
)
from graphlow.geometry.methods import analytic
from graphlow.utils.dimension import get_dimension
from graphlow.utils.safe_ops import safe_normalize
from graphlow.utils.topology_helper import TopologyDim, get_cell_dimension

if TYPE_CHECKING:
    from graphlow.core.mesh import TensorMesh


Dispatcher = Callable[
    [pv.CellType, FaceBlock | CellBlock, "TensorMesh"], TensorLike
]

# Fixed-length area-vector functions: (points, conn, backend) -> (N, 3)
_AREA_VEC_FN = {
    pv.CellType.TRIANGLE: analytic.triangle_area_vectors,
    pv.CellType.QUAD: analytic.quad_area_vectors,
    pv.CellType.PIXEL: analytic.pixel_area_vectors,
}

# Fixed-length centroid functions: (points, conn, backend) -> (N, 3)
_CENTROIDS_FN = {
    pv.CellType.TRIANGLE: analytic.triangle_centroids,
    pv.CellType.QUAD: analytic.quad_centroids,
    pv.CellType.PIXEL: analytic.pixel_centroids,
}

_SUPPORTED_FACE_TYPES = (
    pv.CellType.TRIANGLE,
    pv.CellType.QUAD,
    pv.CellType.PIXEL,
    pv.CellType.POLYGON,
)


def face_area_vectors[T: TensorLike](
    mesh: TensorMesh[T],
) -> T:
    """
    Face area vectors. Shape ``(n_faces, 3)`` for volume meshes or
    ``(n_cells, 3)`` for surface meshes.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh with differentiable points.

    Returns
    -------
    TensorLike
        For 2D meshes: shape ``(n_cells, 3)``
            indices match PyVista cell IDs.
        For 3D meshes: shape ``(n_faces, 3)``
            indices match FaceBlock global_indices.
    """
    fn = _dispatch_area_vectors
    if mesh.topology.mesh_dim() == TopologyDim.SURFACE:
        return _compute_2d_surface(mesh, fn)
    return _compute_3d_face(mesh, fn)


def face_areas[T: TensorLike](
    mesh: TensorMesh[T],
) -> T:
    """
    Face areas. Shape ``(n_faces, 1)`` for volume meshes or
    ``(n_cells, 1)`` for surface meshes.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh with differentiable points.

    Returns
    -------
    TensorLike
        For 2D meshes: shape ``(n_cells, 1)``
            indices match PyVista cell IDs.
        For 3D meshes: shape ``(n_faces, 1)``
            indices match FaceBlock global_indices.
    """
    return torch.linalg.vector_norm(
        face_area_vectors(mesh), dim=-1, keepdim=True
    )


def face_normals[T: TensorLike](
    mesh: TensorMesh[T],
    eps: float = 1e-12,
) -> T:
    """
    Face unit normals. Shape ``(n_faces, 3)`` for volume meshes or
    ``(n_cells, 3)`` for surface meshes.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh with differentiable points.
    eps : float, optional
        Epsilon for normalization. Default is 1e-12.

    Returns
    -------
    TensorLike
        For 2D meshes: shape ``(n_cells, 3)``
            indices match PyVista cell IDs.
        For 3D meshes: shape ``(n_faces, 3)``
            indices match FaceBlock global_indices.
    """
    area_vecs = face_area_vectors(mesh)
    return safe_normalize(area_vecs, dim=1, eps=eps)


def face_centroids[T: TensorLike](
    mesh: TensorMesh[T],
) -> T:
    """
    Face centroids. Shape ``(n_faces, 3)`` for volume meshes or
    ``(n_cells, 3)`` for surface meshes.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh with differentiable points.

    Returns
    -------
    TensorLike
        For 2D meshes: shape ``(n_cells, 3)``
            indices match PyVista cell IDs.
        For 3D meshes: shape ``(n_faces, 3)``
            indices match FaceBlock global_indices.
    """
    fn = _dispatch_face_centroids
    if mesh.topology.mesh_dim() == TopologyDim.SURFACE:
        return _compute_2d_surface(mesh, fn)
    return _compute_3d_face(mesh, fn)


# =========================================================================
# General computation flow (router layer)
# =========================================================================
def _compute_2d_surface[T: TensorLike](
    mesh: TensorMesh[T], dispatcher: Dispatcher
) -> T:
    vecs, indices = [], []
    for ct_val in mesh.topology.unique_cell_types():
        cell_type = pv.CellType(ct_val)
        if get_cell_dimension(cell_type) != TopologyDim.SURFACE:
            continue
        block = mesh.topology.cell_block(cell_type)
        if block is None:
            continue
        vecs.append(dispatcher(cell_type, block, mesh))
        indices.append(mesh.backend.as_index_tensor(block.global_indices))

    if not vecs:
        raise ValueError("No supported surface cells found.")
    stacked = torch.cat(vecs, dim=0)
    idx = torch.cat(indices, dim=0)
    out = mesh.backend.zeros(
        (mesh.n_cells,) + stacked.shape[1:],
        dimension=get_dimension(stacked),
    )
    return out.index_add_(0, idx, stacked)


def _compute_3d_face[T: TensorLike](
    mesh: TensorMesh[T], dispatcher: Dispatcher
) -> T:
    vecs, indices = [], []
    n_faces = 0
    for cell_type in _SUPPORTED_FACE_TYPES:
        block = mesh.topology.face_block(cell_type)
        if block is None:
            continue
        n_faces += block.n_faces
        vecs.append(dispatcher(cell_type, block, mesh))
        indices.append(mesh.backend.as_index_tensor(block.global_indices))

    if not vecs:
        raise ValueError("No supported faces found.")

    idx = mesh.backend.as_index_tensor(np.concatenate(indices, axis=0))
    stacked = torch.cat(vecs, dim=0)
    out = mesh.backend.zeros(
        (n_faces,) + stacked.shape[1:],
        dimension=get_dimension(stacked),
    )
    return out.index_add_(0, idx, stacked)


# =========================================================================
# Dispatchers
# =========================================================================
def _dispatch_area_vectors[T: TensorLike](
    cell_type: pv.CellType, block: FaceBlock | CellBlock, mesh: TensorMesh[T]
) -> T:
    """Dispatch the algorithm based on the block type and cell type."""
    if isinstance(block, (FixedFaceBlock, FixedCellBlock)):
        fn = _AREA_VEC_FN.get(cell_type)
        if fn is None:
            raise ValueError(f"Unsupported fixed face/cell type: {cell_type}")
        conn = mesh.backend.as_index_tensor(block.conn)
        return fn(mesh.points, conn, mesh.backend)
    if isinstance(block, (JaggedFaceBlock, JaggedCellBlock)):
        if cell_type != pv.CellType.POLYGON:
            raise ValueError(f"Unsupported jagged face/cell type: {cell_type}")
        conn = mesh.backend.as_index_tensor(block.conn)
        offsets = mesh.backend.as_index_tensor(block.offsets)
        return analytic.polygon_area_vectors(
            mesh.points, conn, offsets, mesh.backend
        )
    raise ValueError(f"Unknown block type: {type(block)}")


def _dispatch_face_centroids[T: TensorLike](
    cell_type: pv.CellType, block: FaceBlock | CellBlock, mesh: TensorMesh[T]
) -> T:
    """Dispatch the algorithm based on the block type and cell type."""
    if isinstance(block, (FixedFaceBlock, FixedCellBlock)):
        fn = _CENTROIDS_FN.get(cell_type)
        if fn is None:
            raise ValueError(f"Unsupported fixed face/cell type: {cell_type}")
        conn = mesh.backend.as_index_tensor(block.conn)
        return fn(mesh.points, conn, mesh.backend)
    if isinstance(block, (JaggedFaceBlock, JaggedCellBlock)):
        if cell_type != pv.CellType.POLYGON:
            raise ValueError(f"Unsupported jagged face/cell type: {cell_type}")
        conn = mesh.backend.as_index_tensor(block.conn)
        offsets = mesh.backend.as_index_tensor(block.offsets)
        return analytic.polygon_centroids(
            mesh.points, conn, offsets, mesh.backend
        )
    raise ValueError(f"Unknown block type: {type(block)}")

from __future__ import annotations

from typing import TYPE_CHECKING

import pyvista as pv
import torch

from graphlow.core.backend.base import TensorLike
from graphlow.core.blocks import FixedCellBlock
from graphlow.geometry.methods import analytic
from graphlow.utils.topology_helper import (
    TopologyDim,
    is_surface_mesh_watertight,
)

if TYPE_CHECKING:
    from graphlow.core.mesh import TensorMesh

# Fixed-length volume functions: (points, conn, backend) -> (N, 1)
_VOLUME_FN = {
    pv.CellType.TETRA: analytic.tetra_volume,
    pv.CellType.PYRAMID: analytic.pyramid_volume,
    pv.CellType.WEDGE: analytic.wedge_volume,
    pv.CellType.VOXEL: analytic.voxel_volume,
    pv.CellType.HEXAHEDRON: analytic.hexahedron_volume,
}


def surface_volume[T: TensorLike](mesh: TensorMesh[T]) -> T:
    """
    Compute enclosed volume of a watertight surface mesh.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh containing differentiable points.

    Returns
    -------
    TensorLike
        Enclosed volume as a tensor of shape ``(1,)``.

    Raises
    ------
    ValueError
        If the mesh is not a watertight surface mesh.
    """
    if (
        mesh.topology.mesh_dim() != TopologyDim.SURFACE
        or not is_surface_mesh_watertight(mesh.pvmesh)
    ):
        raise ValueError(
            "surface_volume is only supported for watertight surface meshes."
        )
    return _enclosed_volume_surface_mesh(mesh)


def cell_volumes[T: TensorLike](mesh: TensorMesh[T]) -> T:
    """
    Compute per-cell volumes for a volume mesh.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh containing differentiable points.

    Returns
    -------
    TensorLike
        Cell volumes with shape ``(n_cells, 1)``.

    Raises
    ------
    ValueError
        If the mesh is not a volume mesh.
    """
    if mesh.topology.mesh_dim() != TopologyDim.VOLUME:
        raise ValueError("cell_volumes is only supported for volume meshes.")

    if pv.CellType.POLYHEDRON in mesh.topology.unique_cell_types():
        return _compute_volume_using_divergence_theorem(mesh)

    # Fixed-length cells
    volumes = mesh.backend.zeros(
        (mesh.n_cells, 1),
        dimension={"L": 3},
    )
    for cell_type in _VOLUME_FN.keys():
        block = mesh.topology.cell_block(cell_type)
        if block is None or not isinstance(block, FixedCellBlock):
            continue
        conn = mesh.backend.as_index_tensor(block.conn)
        sub_volumes = _VOLUME_FN[cell_type](mesh.points, conn, mesh.backend)
        gidx = mesh.backend.as_index_tensor(block.global_indices)
        volumes.index_add_(0, gidx, sub_volumes)
    return volumes


def cell_centroids[T: TensorLike](mesh: TensorMesh[T]) -> T:
    """
    Compute per-cell centroids for a volume mesh.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh containing differentiable points.

    Returns
    -------
    TensorLike
        Cell centroids with shape ``(n_cells, 3)``.
    """
    Vc = cell_volumes(mesh)
    gf = mesh.geometry.face_centroids()
    Sf = mesh.geometry.face_area_vectors()
    moment_f = torch.sum(Sf * gf, dim=-1, keepdim=True) * gf
    moment_c = mesh.topology.map_face_to_cell(moment_f, "div", "segment")
    return moment_c / (4.0 * Vc)


# =========================================================================
# Enclosed volume for watertight surface mesh (divergence theorem)
# =========================================================================
def _enclosed_volume_surface_mesh[T: TensorLike](mesh: TensorMesh[T]) -> T:
    """
    Enclosed volume of a closed surface mesh.

    Notes
    -----
    Uses the divergence theorem:

    .. math::

        V = \\frac{1}{3} \\int_{\\partial \\Omega} \\mathbf{r}
        \\cdot \\mathbf{n}\\, dS
    """
    Sf = mesh.geometry.face_area_vectors()  # (F, 3)
    gf = mesh.geometry.face_centroids()  # (F, 3)
    return torch.sum(Sf * gf)[None] / 3.0


# =========================================================================
# Volume using divergence theorem
# =========================================================================
def _compute_volume_using_divergence_theorem[T: TensorLike](
    mesh: TensorMesh[T],
) -> T:
    """
    Compute volumes using divergence theorem (signed cone sum).

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh containing differentiable points.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, 1)``.
    """
    # HACK: Improve the performance of this function.
    points_on_faces = mesh.topology.map_point_to_face(
        mesh.points, "mean", "segment"
    )
    face_area_vectors = mesh.geometry.face_area_vectors()  # (Nf, 3)
    cone_volumes = (
        torch.sum(face_area_vectors * points_on_faces, dim=-1, keepdim=True)
        / 3.0
    )  # (Nf, 1)
    return mesh.topology.map_face_to_cell(cone_volumes, "div", "segment")

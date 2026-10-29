from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import pyvista as pv

from graphlow.core.backend.base import TensorLike
from graphlow.geometry import fem
from graphlow.geometry.distance import (
    chamfer_distance as _chamfer_distance,
)
from graphlow.geometry.distance import (
    hausdorff_distance as _hausdorff_distance,
)
from graphlow.geometry.operator import (
    isoAM as _isoAM,
)
from graphlow.geometry.operator import (
    isoAM_with_neumann as _isoAM_with_neumann,
)
from graphlow.geometry.surface import (
    face_area_vectors as _face_area_vectors,
)
from graphlow.geometry.surface import (
    face_areas as _face_areas,
)
from graphlow.geometry.surface import (
    face_centroids as _face_centroids,
)
from graphlow.geometry.surface import (
    face_normals as _face_normals,
)
from graphlow.geometry.volume import (
    cell_centroids as _cell_centroids,
)
from graphlow.geometry.volume import (
    cell_volumes as _cell_volumes,
)
from graphlow.geometry.volume import (
    surface_volume as _surface_volume,
)

if TYPE_CHECKING:
    from graphlow.core.mesh import TensorMesh


class MeshGeometry[T: TensorLike]:
    """
    Differentiable geometry evaluated on a mesh.

    This is a thin wrapper around functions in ``graphlow.geometry.*`` that
    operate on a ``TensorMesh`` and return backend tensors. Autograd flows
    through ``mesh.points`` (and any differentiable data used by the operator).

    Notes
    -----
    Returned tensor shapes are always ``(n_elements, feature_dim)``. For
    example, cell volumes are ``(n_cells, 1)``.
    """

    def __init__(self, mesh: TensorMesh[T]) -> None:
        self._mesh = mesh

    def face_area_vectors(self) -> T:
        """
        Compute face area vectors.

        Returns
        -------
        T
            Backend tensor of shape ``(n_faces, 3)`` for volume meshes or
            ``(n_cells, 3)`` for surface meshes (each cell is a face).
        """
        return _face_area_vectors(self._mesh)

    def face_areas(self) -> T:
        """
        Compute face areas.

        Returns
        -------
        T
            Backend tensor of shape ``(n_faces, 1)`` for volume meshes or
            ``(n_cells, 1)`` for surface meshes.
        """
        return _face_areas(self._mesh)

    def face_normals(self, eps: float = 1e-12) -> T:
        """
        Compute unit face normals.

        Parameters
        ----------
        eps : float, default=1e-12
            Numerical epsilon used for normalization.

        Returns
        -------
        T
            Backend tensor of shape ``(n_faces, 3)`` for volume meshes or
            ``(n_cells, 3)`` for surface meshes (each cell is a face).
        """
        return _face_normals(self._mesh, eps=eps)

    def face_centroids(self) -> T:
        """
        Compute face centroids.

        Returns
        -------
        T
            Backend tensor of shape ``(n_faces, 3)`` for volume meshes or
            ``(n_cells, 3)`` for surface meshes (each cell is a face).
        """
        return _face_centroids(self._mesh)

    def surface_volume(self) -> T:
        """
        Compute the enclosed volume of a surface mesh.

        Returns
        -------
        T
            Backend tensor of shape ``(1,)``.

        Notes
        -----
        This is defined for closed, consistently oriented surface meshes.
        """
        return _surface_volume(self._mesh)

    def cell_volumes(self) -> T:
        """
        Compute per-cell volumes for a volume mesh.

        Returns
        -------
        T
            Backend tensor of shape ``(n_cells, 1)``.
        """
        return _cell_volumes(self._mesh)

    def cell_centroids(self) -> T:
        """
        Compute per-cell centroids for a volume mesh.

        Returns
        -------
        T
            Backend tensor of shape ``(n_cells, 3)``.
        """
        return _cell_centroids(self._mesh)

    def hausdorff_distance(
        self,
        target_points: T,
        *,
        softmin_temperature: float | None = None,
    ) -> T:
        """
        Compute Hausdorff distance to a target point set.

        Parameters
        ----------
        target_points : T
            Target points of shape ``(n_target, 3)`` or ``(n_target, 3, 1)``.
        softmin_temperature : float or None, optional
            If positive, use soft min/max so gradient flows to all vertices.

        Returns
        -------
        T
            Backend tensor of shape ``(1,)``.
        """
        return _hausdorff_distance(
            self._mesh,
            target_points,
            softmin_temperature=softmin_temperature,
        )

    def chamfer_distance(
        self,
        target_points: T,
        *,
        softmin_temperature: float | None = None,
    ) -> T:
        """
        Compute Chamfer distance to a target point set.

        Parameters
        ----------
        target_points : T
            Target points of shape ``(n_target, 3)`` or ``(n_target, 3, 1)``.
        softmin_temperature : float or None, optional
            If positive, use soft min so gradient flows to all vertices.

        Returns
        -------
        T
            Backend tensor of shape ``(1,)``.
        """
        return _chamfer_distance(
            self._mesh,
            target_points,
            softmin_temperature=softmin_temperature,
        )

    def isoAM(
        self,
        with_moment_matrix: bool = True,
        consider_volume: bool = False,
        normal_interp_mode: Literal["mean", "conservative"] = "conservative",
        eps: float = 1e-12,
    ) -> tuple[T, T | None]:
        """
        Compute isotropic anisotropic metric (IsoAM) operator.

        Parameters
        ----------
        with_moment_matrix : bool, default=True
            If True, also returns the inverse moment matrix.
        consider_volume : bool, default=False
            If True, includes volume-related terms (for volume meshes).
        normal_interp_mode : {"mean", "conservative"}, default="conservative"
            Mode used to interpolate normals.
        eps : float, default=1e-12
            Numerical epsilon used for normalization.

        Returns
        -------
        isoam : T
            IsoAM operator of shape ``(dims, n_points, n_points)``.
        moment_inv : T or None
            Inverse moment matrix of shape ``(n_points, dims, dims)`` if
            requested; otherwise None.
        """
        return _isoAM(
            self._mesh,
            with_moment_matrix=with_moment_matrix,
            consider_volume=consider_volume,
            normal_interp_mode=normal_interp_mode,
            eps=eps,
        )

    def isoAM_with_neumann(
        self,
        with_moment_matrix: bool = True,
        consider_volume: bool = False,
        normal_weight: float = 10.0,
        normal_interp_mode: Literal["mean", "conservative"] = "conservative",
        eps: float = 1e-12,
    ) -> tuple[T, T, T | None]:
        """
        Compute IsoAM with a Neumann boundary model.

        Parameters
        ----------
        with_moment_matrix : bool, default=True
            If True, also returns the inverse moment matrix.
        consider_volume : bool, default=False
            If True, includes volume-related terms (for volume meshes).
        normal_weight : float, default=10.0
            Weight for the Neumann boundary normal term.
        normal_interp_mode : {"mean", "conservative"}, default="conservative"
            Mode used to interpolate normals.
        eps : float, default=1e-12
            Numerical epsilon used for normalization.

        Returns
        -------
        isoam : T
            IsoAM operator of shape ``(dims, n_points, n_points)``.
        weighted_normals : T
            Weighted normals for Neumann boundary condition, shape
            ``(n_points, dims)``.
        moment_inv : T or None
            Inverse moment matrix of shape ``(n_points, dims, dims)`` if
            requested; otherwise None.
        """
        return _isoAM_with_neumann(
            self._mesh,
            with_moment_matrix=with_moment_matrix,
            consider_volume=consider_volume,
            normal_weight=normal_weight,
            normal_interp_mode=normal_interp_mode,
            eps=eps,
        )

    def global_fem_matrix_tet(self, cell_local_matrix_tet: T) -> T:
        """Deprecated wrapper for :meth:`global_fem_matrix`.

        .. deprecated::
            Use :meth:`global_fem_matrix` instead.
        """
        return fem.global_fem_matrix_tet(self._mesh, cell_local_matrix_tet)

    def apply_dirichlet_to_global_fem_matrix(
        self, global_matrix: T, global_b: T, sparse_dirichlet: T
    ) -> tuple[T, T]:
        """
        Apply the given Dirichlet boundary condition to the linear problem.

        Parameters
        ----------
        global_matrix: T
            Sparse global matrix.
        global_b: T
            Dense global tensor.
        sparse_dirichlet: T
            Sparse tensor describing the Dirichlet boundary condition.

        Returns
        -------
        global_matrix_with_dirichlet: T
            Sparse global matrix after taking into account the Dirichlet bc.
        global_b_with_dirichlet: T
            Global b after taking into account the Dirichlet bc.
        """
        return fem.apply_dirichlet_to_global_fem_matrix(
            self._mesh, global_matrix, global_b, sparse_dirichlet
        )

    def apply_cell_local_fem_matrix_tet(
        self,
        cell_local_matrix_tet: T,
        u: T,
        vector_rank: int | None = None,
        matrix_rank: int | None = None,
    ) -> T:
        """Deprecated wrapper for :meth:`apply_cell_local_fem_matrix`.

        .. deprecated::
            Use :meth:`apply_cell_local_fem_matrix` instead.
        """
        return fem.apply_cell_local_fem_matrix_tet(
            self._mesh,
            cell_local_matrix_tet=cell_local_matrix_tet,
            u=u,
            vector_rank=vector_rank,
            matrix_rank=matrix_rank,
        )

    def cell_local_fem_rigidity_tet(
        self, cell_material_coeff: T | None = None, rank: int = 0
    ) -> T:
        """Deprecated wrapper for :meth:`cell_local_fem_rigidity`.

        .. deprecated::
            Use :meth:`cell_local_fem_rigidity` instead.
        """
        return fem.cell_local_fem_rigidity_tet(
            self._mesh, cell_material_coeff=cell_material_coeff, rank=rank
        )

    def cell_local_fem_mass_tet(
        self,
        cell_density: T | None = None,
    ) -> T:
        """Deprecated wrapper for :meth:`cell_local_fem_mass`.

        .. deprecated::
            Use :meth:`cell_local_fem_mass` instead.
        """
        return fem.cell_local_fem_mass_tet(
            self._mesh, cell_density=cell_density
        )

    def cell_local_fem_rigidity(
        self,
        cell_material_coeff: T | Mapping[pv.CellType, T] | None = None,
        rank: int = 0,
        *,
        quadrature_order: int | None = None,
    ) -> dict[pv.CellType, T]:
        """Generate quadrature-based cell-local rigidity matrices.

        Parameters
        ----------
        cell_material_coeff : T, Mapping[pyvista.CellType, T], or None, optional
            Cell-constant material coefficient in the rigidity integrand.
            ``None`` uses the identity for ``rank``. A tensor with leading
            size 1 is one coefficient for every cell. A tensor with leading
            size ``n_cells`` follows mesh cell order and is gathered onto
            each block with ``global_indices``. A mapping holds one
            block-ordered tensor per cell type. Layouts are ``(c, f)``,
            ``(c, i, j, f)``, and ``(c, i, j, k, l, f)`` for ranks 0, 2,
            and 4.
        rank : {0, 2, 4}, default=0
            Tensor rank of ``cell_material_coeff``.
        quadrature_order : int or None, optional
            Quadrature order. ``None`` selects the safe cell-type default.

        Returns
        -------
        dict[pyvista.CellType, T]
            One block-local rigidity matrix per cell type. Row ``i`` matches
            ``cell_block(cell_type).global_indices[i]``.
        """
        return fem.cell_local_fem_rigidity(
            self._mesh,
            cell_material_coeff=cell_material_coeff,
            rank=rank,
            quadrature_order=quadrature_order,
        )

    def cell_local_fem_mass(
        self,
        cell_density: T | Mapping[pv.CellType, T] | None = None,
        *,
        quadrature_order: int | None = None,
    ) -> dict[pv.CellType, T]:
        """Generate quadrature-based consistent cell-local mass matrices.

        Parameters
        ----------
        cell_density : T, Mapping[pyvista.CellType, T], or None, optional
            Cell-constant density with layout ``(c, f)``. ``None`` uses unit
            density. A tensor with leading size 1 is shared by every cell.
            A tensor with leading size ``n_cells`` is mesh cell order. A
            mapping is one block-ordered tensor per cell type.
        quadrature_order : int or None, optional
            Quadrature order. ``None`` selects the safe cell-type default.

        Returns
        -------
        dict[pyvista.CellType, T]
            One consistent block-local mass matrix per cell type. Row ``i``
            matches ``cell_block(cell_type).global_indices[i]``.
        """
        return fem.cell_local_fem_mass(
            self._mesh,
            cell_density=cell_density,
            quadrature_order=quadrature_order,
        )

    def apply_cell_local_fem_matrix(
        self,
        cell_local_matrix: T | Mapping[pv.CellType, T],
        u: T,
        vector_rank: int | None = None,
        matrix_rank: int | None = None,
    ) -> T:
        """Apply block-local FEM matrices and sum the products onto points.

        Parameters
        ----------
        cell_local_matrix : T or Mapping[pyvista.CellType, T]
            One block-local matrix, or one matrix per cell type.
        u : T
            Point-wise physical variable to be multiplied by the matrix.
        vector_rank : int or None, optional
            Rank of ``u``. If omitted, a Phlower tensor uses ``u.rank()``
            and any other tensor uses ``u.ndim - 2``.
        matrix_rank : int or None, optional
            Rank of the local matrix between the node axes and the feature
            axis. If omitted, a Phlower tensor uses ``rank() - 2`` and any
            other tensor uses ``ndim - 4``.

        Returns
        -------
        T
            Sum of the assembled matvec results.
        """
        return fem.apply_cell_local_fem_matrix(
            self._mesh,
            cell_local_matrix=cell_local_matrix,
            u=u,
            vector_rank=vector_rank,
            matrix_rank=matrix_rank,
        )

    def global_fem_matrix(
        self, cell_local_matrix: T | Mapping[pv.CellType, T]
    ) -> T:
        """Assemble block-local FEM matrices into one global sparse matrix.

        Parameters
        ----------
        cell_local_matrix : T or Mapping[pyvista.CellType, T]
            One block-local matrix, or one matrix per cell type.

        Returns
        -------
        T
            Sum of the assembled blocks as one COO matrix.
        """
        return fem.global_fem_matrix(self._mesh, cell_local_matrix)

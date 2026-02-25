from __future__ import annotations

import numpy as np
import phlower_tensor as pt
import torch
from scipy import sparse as sp

from graphlow.base.mesh_interface import IReadOnlyGraphlowMesh
from graphlow.util.enums import FeatureName, SparseMatrixName
from graphlow.util.sparse_tensor import threashold_sparse_tensor


class GraphProcessor:
    """A class for graph processing."""

    def __init__(self) -> None:
        pass

    def compute_cell_point_incidence(
        self, mesh: IReadOnlyGraphlowMesh, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_cells, n_points)-shaped sparse incidence matrix.
        The result is cached in the mesh object.

        Parameters
        ----------
        mesh: GraphlowMesh
        refresh_cache: bool, optional [False]
            If True, recompute the incidence matrix.
            Otherwise, return the cached result if available.

        Returns
        -------
        pt.PhlowerTensor
            (n_cells, n_points)-shaped sparse coo tensor.
        """
        if (
            not refresh_cache
            and SparseMatrixName.CELL_POINT_INCIDENCE in mesh.dict_sparse_tensor
        ):
            return mesh.dict_sparse_tensor[
                SparseMatrixName.CELL_POINT_INCIDENCE
            ]

        indices = mesh.pvmesh.cell_connectivity
        indptr = mesh.pvmesh.offset
        data = np.ones(len(indices), dtype=bool)
        size = (mesh.n_cells, mesh.n_points)
        scipy_cp_inc = sp.csr_array((data, indices, indptr), shape=size)
        torch_cp_inc = pt.phlower_array(scipy_cp_inc).to_tensor().to(mesh.dtype)
        cp_inc = pt.phlower_tensor(torch_cp_inc.coalesce(), dimension={}).to(
            device=mesh.device
        )

        mesh.dict_sparse_tensor.update(
            {SparseMatrixName.CELL_POINT_INCIDENCE: cp_inc},
            overwrite=True,
        )
        return cp_inc

    def compute_cell_adjacency(
        self, mesh: IReadOnlyGraphlowMesh, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_cells, n_cells)-shaped sparse adjacency matrix including
        self-loops. The result is cached in the mesh object.

        Parameters
        ----------
        mesh: GraphlowMesh
        refresh_cache: bool, optional [False]
            If True, recompute the adjacency matrix.
            Otherwise, return the cached result if available.

        Returns
        -------
        pt.PhlowerTensor
            (n_cells, n_cells)-shaped sparse coo tensor.
        """
        if (
            not refresh_cache
            and SparseMatrixName.CELL_ADJACENCY in mesh.dict_sparse_tensor
        ):
            return mesh.dict_sparse_tensor[SparseMatrixName.CELL_ADJACENCY]

        cp_inc = mesh.compute_cell_point_incidence()
        cell_adjacency = threashold_sparse_tensor(
            cp_inc @ cp_inc.transpose(0, 1)
        )

        mesh.dict_sparse_tensor.update(
            {SparseMatrixName.CELL_ADJACENCY: cell_adjacency},
            overwrite=True,
        )
        return cell_adjacency

    def compute_point_adjacency(
        self, mesh: IReadOnlyGraphlowMesh, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_points, n_points)-shaped sparse adjacency matrix
        including self-loops. The result is cached in the mesh object.

        Parameters
        ----------
        mesh: GraphlowMesh
        refresh_cache: bool, optional [False]
            If True, recompute the adjacency matrix.
            Otherwise, return the cached result if available.

        Returns
        -------
        pt.PhlowerTensor
            (n_points, n_points)-shaped sparse coo tensor.
        """
        if (
            not refresh_cache
            and SparseMatrixName.POINT_ADJACENCY in mesh.dict_sparse_tensor
        ):
            return mesh.dict_sparse_tensor[SparseMatrixName.POINT_ADJACENCY]

        cp_inc = mesh.compute_cell_point_incidence()
        point_adjacency = threashold_sparse_tensor(
            cp_inc.transpose(0, 1) @ cp_inc
        )

        mesh.dict_sparse_tensor.update(
            {SparseMatrixName.POINT_ADJACENCY: point_adjacency},
            overwrite=True,
        )
        return point_adjacency

    def compute_point_degree(
        self, mesh: IReadOnlyGraphlowMesh, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_points, n_points)-shaped degree matrix.
        The result is cached in the mesh object.

        Parameters
        ----------
        mesh: GraphlowMesh
        refresh_cache: bool, optional [False]
            If True, recompute the degree matrix.
            Otherwise, return the cached result if available.

        Returns
        -------
        pt.PhlowerTensor
            (n_points, n_points)-shaped sparse coo tensor.
        """
        if (
            not refresh_cache
            and SparseMatrixName.POINT_DEGREE in mesh.dict_sparse_tensor
        ):
            return mesh.dict_sparse_tensor[SparseMatrixName.POINT_DEGREE]

        point_adjacency = self.compute_point_adjacency(mesh)
        point_degree = self._compute_degree(point_adjacency)

        mesh.dict_sparse_tensor.update(
            {SparseMatrixName.POINT_DEGREE: point_degree}, overwrite=True
        )
        return point_degree

    def compute_cell_degree(
        self, mesh: IReadOnlyGraphlowMesh, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_cells, n_cells)-shaped degree matrix.
        The result is cached in the mesh object.

        Parameters
        ----------
        mesh: GraphlowMesh
        refresh_cache: bool, optional [False]
            If True, recompute the degree matrix.
            Otherwise, return the cached result if available.

        Returns
        -------
        pt.PhlowerTensor
            (n_cells, n_cells)-shaped sparse coo tensor.
        """
        if (
            not refresh_cache
            and SparseMatrixName.CELL_DEGREE in mesh.dict_sparse_tensor
        ):
            return mesh.dict_sparse_tensor[SparseMatrixName.CELL_DEGREE]

        cell_adjacency = self.compute_cell_adjacency(mesh)
        cell_degree = self._compute_degree(cell_adjacency)

        mesh.dict_sparse_tensor.update(
            {SparseMatrixName.CELL_DEGREE: cell_degree}, overwrite=True
        )
        return cell_degree

    def compute_normalized_point_adjacency(
        self, mesh: IReadOnlyGraphlowMesh, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_points, n_points)-shaped normalized adjacency matrix.
        The result is cached in the mesh object.

        Parameters
        ----------
        mesh: GraphlowMesh
        refresh_cache: bool, optional [False]
            If True, recompute the normalized adjacency matrix.
            Otherwise, return the cached result if available.

        Returns
        -------
        pt.PhlowerTensor
            (n_points, n_points)-shaped sparse coo tensor.
        """
        if (
            not refresh_cache
            and SparseMatrixName.NORMALIZED_POINT_ADJ in mesh.dict_sparse_tensor
        ):
            return mesh.dict_sparse_tensor[
                SparseMatrixName.NORMALIZED_POINT_ADJ
            ]

        point_adj = self.compute_point_adjacency(mesh)
        normalized_point_adj = self._compute_normalized_adjacency(point_adj)

        mesh.dict_sparse_tensor.update(
            {SparseMatrixName.NORMALIZED_POINT_ADJ: normalized_point_adj},
            overwrite=True,
        )
        return normalized_point_adj

    def compute_normalized_cell_adjacency(
        self, mesh: IReadOnlyGraphlowMesh, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_cells, n_cells)-shaped normalized adjacency matrix.
        The result is cached in the mesh object.

        Parameters
        ----------
        mesh: GraphlowMesh
        refresh_cache: bool, optional [False]
            If True, recompute the normalized adjacency matrix.
            Otherwise, return the cached result if available.

        Returns
        -------
        pt.PhlowerTensor
            (n_cells, n_cells)-shaped sparse coo tensor.
        """
        if (
            not refresh_cache
            and SparseMatrixName.NORMALIZED_CELL_ADJ in mesh.dict_sparse_tensor
        ):
            return mesh.dict_sparse_tensor[SparseMatrixName.NORMALIZED_CELL_ADJ]

        cell_adj = self.compute_cell_adjacency(mesh)
        normalized_cell_adj = self._compute_normalized_adjacency(cell_adj)

        mesh.dict_sparse_tensor.update(
            {SparseMatrixName.NORMALIZED_CELL_ADJ: normalized_cell_adj},
            overwrite=True,
        )
        return normalized_cell_adj

    def compute_point_relative_incidence(
        self, mesh: IReadOnlyGraphlowMesh, other_mesh: IReadOnlyGraphlowMesh
    ) -> pt.PhlowerTensor:
        """Compute (n_points_other, n_points_self)-shaped sparse incidence
        matrix based on points.

        Parameters
        ----------
        mesh: GraphlowMesh
        other_mesh: graphlow.GraphlowMesh
            The other mesh object to be compared against.

        Returns
        -------
        pt.PhlowerTensor
            (n_points_other, n_points_self)-shaped sparse coo tensor.
        """
        if other_mesh.n_points > mesh.n_points:
            return other_mesh.compute_point_relative_incidence(mesh).transpose(
                0, 1
            )

        if FeatureName.ORIGINAL_INDEX not in other_mesh.pvmesh.point_data:
            raise ValueError(
                f"{FeatureName.ORIGINAL_INDEX} not found in "
                f"{other_mesh.pvmesh.point_data.keys()}.\n"
                "Run mesh operation with add_original_index=True option."
            )

        col = torch.from_numpy(
            other_mesh.pvmesh.point_data[FeatureName.ORIGINAL_INDEX]
        )
        row = torch.arange(len(col))
        values = torch.ones(len(col), dtype=mesh.dtype)
        indices = torch.stack([row, col], dim=0)
        size = (other_mesh.n_points, mesh.n_points)
        torch_tensor = torch.sparse_coo_tensor(
            indices, values, size=size, device=mesh.device
        )
        point_relative_inc = pt.phlower_tensor(
            torch_tensor.coalesce(), dimension={}
        ).to(device=mesh.device)
        return point_relative_inc

    def compute_cell_relative_incidence(
        self,
        mesh: IReadOnlyGraphlowMesh,
        other_mesh: IReadOnlyGraphlowMesh,
        minimum_n_sharing: int | None = None,
    ) -> pt.PhlowerTensor:
        """Compute (n_cells_other, n_cells_self)-shaped sparse incidence
        matrix based on cells.

        Parameters
        ----------
        mesh: GraphlowMesh
        other_mesh: graphlow.GraphlowMesh
            The other mesh object to be compared against.
        minimum_n_sharing: int | None
            Minimum number of sharing points to define connectivity. If not
            set, it will be the number of points for each cell.

        Returns
        -------
        pt.PhlowerTensor
            (n_cells_other, n_cells_self)-shaped sparse coo tensor.
        """
        if other_mesh.n_points > mesh.n_points:
            return other_mesh.compute_cell_relative_incidence(
                mesh, minimum_n_sharing=minimum_n_sharing
            ).transpose(0, 1)

        # calculate point relative incidence (n_points_other, n_points_self)
        point_relative_inc = self.compute_point_relative_incidence(
            mesh, other_mesh
        )

        # calculate cp_inc of other mesh (n_cells_other, n_points_other)
        other_cp_inc = other_mesh.compute_cell_point_incidence()

        # calculate mapped cp_inc from other mesh to self mesh
        # (n_cells_other, n_points_self)
        mapped_cp_inc = threashold_sparse_tensor(
            other_cp_inc @ point_relative_inc
        )

        # calculate self cell point incidence (n_self_points, n_self_cells)
        self_pc_inc = mesh.compute_cell_point_incidence().transpose(0, 1)

        # calculate how many points are shared between other and self cells
        # (n_other_cells, n_self_cells)
        cc = mapped_cp_inc @ self_pc_inc

        # Threshold per (other_cell, self_cell): require shared points >= this.
        if minimum_n_sharing is None:
            # (self-mesh) points per other-mesh cell (full containment).
            other_cell_n_vertex = torch.bincount(
                mapped_cp_inc.indices()[0], minlength=mapped_cp_inc.shape[0]
            ).to(mesh.dtype)
            threshold = other_cell_n_vertex[cc.indices()[0]]
        else:
            threshold = minimum_n_sharing

        mask = cc.values() >= threshold
        relative_incidence = torch.sparse_coo_tensor(
            cc.indices()[:, mask],
            torch.ones(mask.sum(), dtype=mesh.dtype),
            cc.shape,
        )

        return pt.phlower_tensor(
            relative_incidence.coalesce(), dimension={}
        ).to(device=mesh.device)

    def _compute_degree(self, adj: pt.PhlowerTensor) -> pt.PhlowerTensor:
        """Compute degree matrix from adjacency matrix.

        Parameters
        ----------
        adjacency: pt.PhlowerTensor
            Adjacency matrix.

        Returns
        -------
        pt.PhlowerTensor
            sparse coo tensor.
        """
        row = adj.indices()[0]
        val = adj.values()
        shape = adj.shape
        dtype = adj.dtype
        device = adj.device
        n = shape[0]
        degrees = torch.bincount(row, weights=val, minlength=n).to(dtype)
        idx = torch.arange(n, device=device)
        tensor = torch.sparse_coo_tensor(
            torch.stack([idx, idx]), degrees, shape
        )
        return pt.phlower_tensor(tensor.coalesce(), dimension={}).to(
            device=device
        )

    def _compute_normalized_adjacency(
        self, adj: pt.PhlowerTensor
    ) -> pt.PhlowerTensor:
        """Compute normalized adjacency matrix from adjacency matrix.

        Parameters
        ----------
        adjacency: pt.PhlowerTensor
            Adjacency matrix.

        Returns
        -------
        pt.PhlowerTensor
            Normalized adjacency matrix.
        """
        row = adj.indices()[0]
        val = adj.values()
        shape = adj.shape
        dtype = adj.dtype
        device = adj.device
        n = shape[0]

        degrees = torch.bincount(row, weights=val, minlength=n).to(dtype)
        D_inv_sqrt_values = 1.0 / torch.sqrt(degrees)
        idx = torch.arange(n, device=device)

        D_inv_sqrt = pt.phlower_tensor(
            torch.sparse_coo_tensor(
                torch.stack([idx, idx]),
                D_inv_sqrt_values,
                shape,
            ).coalesce(),
            dimension={},
        ).to(device=device)
        tensor = D_inv_sqrt @ adj @ D_inv_sqrt
        return tensor.coalesce()

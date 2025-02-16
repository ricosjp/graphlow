from __future__ import annotations

import numpy as np
import torch
from scipy import sparse as sp

from graphlow.base.mesh_interface import IReadOnlyGraphlowMesh
from graphlow.util import array_handler
from graphlow.util.enums import FeatureName, SparseMatrixName


class GraphProcessor:
    """A class for graph processing."""

    def __init__(self) -> None:
        pass

    def compute_cell_point_incidence(
        self, mesh: IReadOnlyGraphlowMesh, cache: bool = True
    ) -> torch.Tensor:
        """Compute (n_cells, n_points)-shaped sparse incidence matrix.
        The method is cached.

        Parameters
        ----------
        mesh: GraphlowMesh
        cache: bool
            If True, the result is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_cells, n_points)-shaped sparse csr tensor.
        """
        if SparseMatrixName.CELL_POINT_INCIDENCE in mesh.dict_sparse_tensor:
            return mesh.dict_sparse_tensor[
                SparseMatrixName.CELL_POINT_INCIDENCE
            ]

        indices = torch.from_numpy(mesh.pvmesh.cell_connectivity.copy())
        indptr = torch.from_numpy(mesh.pvmesh.offset.copy())
        data = torch.ones(len(indices), dtype=mesh.dtype)
        size = (mesh.n_cells, mesh.n_points)
        cell_point_incidence = torch.sparse_csr_tensor(
            indptr, indices, data, size=size, device=mesh.device
        )

        if cache:
            mesh.dict_sparse_tensor.update(
                {SparseMatrixName.CELL_POINT_INCIDENCE: cell_point_incidence},
                overwrite=True,
            )
        return cell_point_incidence

    def compute_cell_adjacency(
        self, mesh: IReadOnlyGraphlowMesh, cache: bool = True
    ) -> torch.Tensor:
        """Compute (n_cells, n_cells)-shaped sparse adjacency matrix including
        self-loops. The method is cached.

        Parameters
        ----------
        mesh: GraphlowMesh
        cache: bool
            If True, the result is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_cells, n_cells)-shaped sparse csr tensor.
        """
        if SparseMatrixName.CELL_ADJACENCY in mesh.dict_sparse_tensor:
            return mesh.dict_sparse_tensor[SparseMatrixName.CELL_ADJACENCY]

        scipy_cp_inc = array_handler.convert_to_scipy_sparse_csr(
            mesh.compute_cell_point_incidence()
        ).astype(bool)
        cell_adjacency = array_handler.convert_to_torch_sparse_csr(
            (scipy_cp_inc @ scipy_cp_inc.T).astype(float),
            device=mesh.device,
            dtype=mesh.dtype,
        )

        if cache:
            mesh.dict_sparse_tensor.update(
                {SparseMatrixName.CELL_ADJACENCY: cell_adjacency},
                overwrite=True,
            )
        return cell_adjacency

    def compute_point_adjacency(
        self, mesh: IReadOnlyGraphlowMesh, cache: bool = True
    ) -> torch.Tensor:
        """Compute (n_points, n_points)-shaped sparse adjacency matrix
        including self-loops. The method is cached.

        Parameters
        ----------
        mesh: GraphlowMesh
        cache: bool
            If True, the result is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_points, n_points)-shaped sparse csr tensor.
        """
        if SparseMatrixName.POINT_ADJACENCY in mesh.dict_sparse_tensor:
            return mesh.dict_sparse_tensor[SparseMatrixName.POINT_ADJACENCY]

        scipy_cp_inc = array_handler.convert_to_scipy_sparse_csr(
            mesh.compute_cell_point_incidence()
        ).astype(bool)
        point_adjacency = array_handler.convert_to_torch_sparse_csr(
            (scipy_cp_inc.T @ scipy_cp_inc).astype(float),
            device=mesh.device,
            dtype=mesh.dtype,
        )

        if cache:
            mesh.dict_sparse_tensor.update(
                {SparseMatrixName.POINT_ADJACENCY: point_adjacency},
                overwrite=True,
            )
        return point_adjacency

    def compute_point_degree(
        self, mesh: IReadOnlyGraphlowMesh, cache: bool = True
    ) -> torch.Tensor:
        """Compute (n_points, n_points)-shaped degree matrix.

        Parameters
        ----------
        mesh: GraphlowMesh
        cache: bool
            If True, the result is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_points, n_points)-shaped sparse csr tensor.
        """
        if SparseMatrixName.POINT_DEGREE in mesh.dict_sparse_tensor:
            return mesh.dict_sparse_tensor[SparseMatrixName.POINT_DEGREE]

        point_adjacency = self.compute_point_adjacency(mesh)
        point_degree = self._compute_degree(point_adjacency)

        if cache:
            mesh.dict_sparse_tensor.update(
                {SparseMatrixName.POINT_DEGREE: point_degree}, overwrite=True
            )
        return point_degree

    def compute_cell_degree(
        self, mesh: IReadOnlyGraphlowMesh, cache: bool = True
    ) -> torch.Tensor:
        """Compute (n_cells, n_cells)-shaped degree matrix.

        Parameters
        ----------
        mesh: GraphlowMesh
        cache: bool
            If True, the result is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_cells, n_cells)-shaped sparse csr tensor.
        """
        if SparseMatrixName.CELL_DEGREE in mesh.dict_sparse_tensor:
            return mesh.dict_sparse_tensor[SparseMatrixName.CELL_DEGREE]

        cell_adjacency = self.compute_cell_adjacency(mesh)
        cell_degree = self._compute_degree(cell_adjacency)

        if cache:
            mesh.dict_sparse_tensor.update(
                {SparseMatrixName.CELL_DEGREE: cell_degree}, overwrite=True
            )
        return cell_degree

    def compute_normalized_point_adjacency(
        self, mesh: IReadOnlyGraphlowMesh, cache: bool = True
    ) -> torch.Tensor:
        """Compute (n_points, n_points)-shaped normalized adjacency matrix.

        Parameters
        ----------
        mesh: GraphlowMesh
        cache: bool
            If True, the result is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_points, n_points)-shaped sparse csr tensor.
        """
        if SparseMatrixName.NORMALIZED_POINT_ADJ in mesh.dict_sparse_tensor:
            return mesh.dict_sparse_tensor[
                SparseMatrixName.NORMALIZED_POINT_ADJ
            ]

        point_adj = self.compute_point_adjacency(mesh)
        normalized_point_adj = self._compute_normalized_adjacency(point_adj)

        if cache:
            mesh.dict_sparse_tensor.update(
                {SparseMatrixName.NORMALIZED_POINT_ADJ: normalized_point_adj},
                overwrite=True,
            )
        return normalized_point_adj

    def compute_normalized_cell_adjacency(
        self, mesh: IReadOnlyGraphlowMesh, cache: bool = True
    ) -> torch.Tensor:
        """Compute (n_cells, n_cells)-shaped normalized adjacency matrix.

        Parameters
        ----------
        mesh: GraphlowMesh
        cache: bool
            If True, the result is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_cells, n_cells)-shaped sparse csr tensor.
        """
        if SparseMatrixName.NORMALIZED_CELL_ADJ in mesh.dict_sparse_tensor:
            return mesh.dict_sparse_tensor[SparseMatrixName.NORMALIZED_CELL_ADJ]

        cell_adj = self.compute_cell_adjacency(mesh)
        normalized_cell_adj = self._compute_normalized_adjacency(cell_adj)

        if cache:
            mesh.dict_sparse_tensor.update(
                {SparseMatrixName.NORMALIZED_CELL_ADJ: normalized_cell_adj},
                overwrite=True,
            )
        return normalized_cell_adj

    def compute_point_relative_incidence(
        self, mesh: IReadOnlyGraphlowMesh, other_mesh: IReadOnlyGraphlowMesh
    ) -> torch.Tensor:
        """Compute (n_points_other, n_points_self)-shaped sparse incidence
        matrix based on points.

        Parameters
        ----------
        mesh: GraphlowMesh
        other_mesh: graphlow.GraphlowMesh
            The other mesh object to be compared against.

        Returns
        -------
        torch.Tensor[float]
            (n_points_other, n_points_self)-shaped sparse csr tensor.
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
        coo = torch.sparse_coo_tensor(
            indices, values, size=size, device=mesh.device
        )
        return coo.to_sparse_csr()

    def compute_cell_relative_incidence(
        self,
        mesh: IReadOnlyGraphlowMesh,
        other_mesh: IReadOnlyGraphlowMesh,
        minimum_n_sharing: int | None = None,
    ) -> torch.Tensor:
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
        torch.Tensor[float]
            (n_cells_other, n_cells_self)-shaped sparse csr tensor.
        """
        if other_mesh.n_points > mesh.n_points:
            return other_mesh.compute_cell_relative_incidence(
                mesh, minimum_n_sharing=minimum_n_sharing
            ).transpose(0, 1)

        other_self_point_incidence = array_handler.convert_to_scipy_sparse_csr(
            mesh.compute_point_relative_incidence(other_mesh)
        ).astype(bool)

        # (n_other_cells, n_self_points)
        other_incidence = (
            array_handler.convert_to_scipy_sparse_csr(
                other_mesh.compute_cell_point_incidence()
            ).astype(bool)
            @ other_self_point_incidence
        ).astype(int)

        # (n_self_points, n_self_cells)
        self_incidence = (
            array_handler.convert_to_scipy_sparse_csr(
                mesh.compute_cell_point_incidence()
            )
            .astype(int)
            .T
        )

        # (n_other_cells, n_self_cells)
        dot: sp.csr_array = other_incidence @ self_incidence

        if minimum_n_sharing is None:
            other_cell_n_vertex = np.array(other_incidence.sum(axis=1))
            coo_dot = dot.tocoo()
            row = coo_dot.row
            filter_ = coo_dot.data >= other_cell_n_vertex[row]
            relative_incidence = sp.csr_array(
                (
                    np.ones(np.sum(filter_), dtype=bool),
                    (coo_dot.row[filter_], coo_dot.col[filter_]),
                ),
                shape=dot.shape,
            )
        else:
            relative_incidence = dot >= minimum_n_sharing

        return array_handler.convert_to_torch_sparse_csr(
            relative_incidence.astype(float),
            device=mesh.device,
            dtype=mesh.dtype,
        )

    def _compute_degree(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Compute degree matrix from adjacency matrix.

        Parameters
        ----------
        adjacency: torch.Tensor
            Adjacency matrix.

        Returns
        -------
        torch.Tensor[float]
            sparse csr tensor.
        """
        degrees = adjacency.sum(dim=1, keepdim=True).to_dense().reshape(-1)
        return torch.sparse.spdiags(
            degrees,
            offsets=torch.tensor([0]),
            shape=adjacency.shape,
        ).to_sparse_csr()

    def _compute_normalized_adjacency(
        self, adjacency: torch.Tensor
    ) -> torch.Tensor:
        """Compute normalized adjacency matrix from adjacency matrix.

        Parameters
        ----------
        adjacency: torch.Tensor
            Adjacency matrix.

        Returns
        -------
        torch.Tensor[float]
            Normalized adjacency matrix.
        """
        degrees = adjacency.sum(dim=1, keepdim=True).to_dense().reshape(-1)
        D_inv_sqrt_values = 1.0 / torch.sqrt(degrees)
        indices = torch.stack(
            [
                torch.arange(adjacency.shape[0]),
                torch.arange(adjacency.shape[1]),
            ]
        )
        D_inv_sqrt = torch.sparse_coo_tensor(
            indices,
            D_inv_sqrt_values,
            size=adjacency.shape,
            device=adjacency.device,
        ).to_sparse_csr()
        return D_inv_sqrt @ adjacency @ D_inv_sqrt

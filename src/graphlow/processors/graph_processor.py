from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from graphlow.util import array_handler
from graphlow.util.enums import SparseMatrixName

if TYPE_CHECKING:
    from graphlow.base import GraphlowMesh


class GraphProcessorMixin:
    """A mix-in class for graph processing."""

    def compute_cell_point_incidence(
            self, *, use_cache: bool = True) -> torch.Tensor:
        """Compute (n_cells, n_points)-shaped sparse incidence matrix.
        The method is cached.

        Parameters
        ----------
        use_cache: bool
            If True, use cached data if exists. If False, re-compute and
            re-cache new data. The default is False.

        Returns
        -------
        torch.Tensor[float]
            (n_cells, n_points)-shapece sparse incidence matrix.
        """
        if use_cache and SparseMatrixName.CELL_POINT_INCIDENCE \
                in self.dict_sparse_tensor:
            return self.dict_sparse_tensor[
                SparseMatrixName.CELL_POINT_INCIDENCE]

        indices = self.mesh.cell_connectivity
        indptr = self.mesh.offset
        data = torch.ones(len(indices))
        size = (self.mesh.n_cells, self.mesh.n_points)
        cell_point_incidence = array_handler.send(
            torch.sparse_csr_tensor(
                indptr, indices, data, size=size),
            device=self.device, dtype=self.dtype)

        self.dict_sparse_tensor.update(
            {SparseMatrixName.CELL_POINT_INCIDENCE: cell_point_incidence},
            overwrite=True)
        return cell_point_incidence

    def compute_cell_adjacency(
            self, *, use_cache: bool = True) -> torch.Tensor:
        """Compute (n_cells, n_cells)-shaped sparse adjacency matrix including
        self-loops. The method is cached.

        Parameters
        ----------
        use_cache: bool
            If True, use cached data if exists. If False, re-compute and
            re-cache new data. The default is False.

        Returns
        -------
        torch.Tensor[float]
            (n_cells, n_cells)-shapece sparse adjacency matrix.
        """
        if use_cache and SparseMatrixName.CELL_ADJACENCY \
                in self.dict_sparse_tensor:
            return self.dict_sparse_tensor[SparseMatrixName.CELL_ADJACENCY]

        scipy_cp_inc = array_handler.convert_to_scipy_sparse_csr(
            self.compute_cell_point_incidence()).astype(bool)
        cell_adjacency = array_handler.convert_to_torch_sparse_csr(
            (scipy_cp_inc @ scipy_cp_inc.T).astype(float),
            device=self.device, dtype=self.dtype)

        self.dict_sparse_tensor.update(
            {SparseMatrixName.CELL_ADJACENCY: cell_adjacency}, overwrite=True)
        return cell_adjacency

    def compute_point_adjacency(
            self, *, use_cache: bool = True) -> torch.Tensor:
        """Compute (n_points, n_points)-shaped sparse adjacency matrix
        including self-loops. The method is cached.

        Parameters
        ----------
        use_cache: bool
            If True, use cached data if exists. If False, re-compute and
            re-cache new data. The default is False.

        Returns
        -------
        torch.Tensor[float]
            (n_points, n_points)-shapece sparse adjacency matrix.
        """
        if use_cache and SparseMatrixName.POINT_ADJACENCY \
                in self.dict_sparse_tensor:
            return self.dict_sparse_tensor[SparseMatrixName.POINT_ADJACENCY]

        scipy_cp_inc = array_handler.convert_to_scipy_sparse_csr(
            self.compute_cell_point_incidence()).astype(bool)
        point_adjacency = array_handler.convert_to_torch_sparse_csr(
            (scipy_cp_inc.T @ scipy_cp_inc).astype(float),
            device=self.device, dtype=self.dtype)

        self.dict_sparse_tensor.update(
            {SparseMatrixName.POINT_ADJACENCY: point_adjacency},
            overwrite=True)
        return point_adjacency

    def compute_point_relative_incidence(self, other_mesh: GraphlowMesh):
        """Compute (n_points_other, n_points_self)-shaped sparse incidence
        matrix based on points.

        Parameters
        ----------
        other_mesh: graphlow.GraphlowMesh
            Other mesh object to be

        Returns
        -------
        torch.Tensor[float]
            (n_points, n_points)-shapece sparse adjacency matrix.
        """
        col = torch.from_numpy(other_mesh.mesh.point_data['original_index'])
        row = torch.arange(len(col))
        value = torch.ones(len(col))
        indices = torch.stack([row, col], dim=0)
        size = (other_mesh.mesh.n_points, self.mesh.n_points)
        return array_handler.convert_to_torch_sparse_csr(
            torch.sparse_coo_tensor(
                indices, value, size=size),
            device=self.device, dtype=self.dtype)

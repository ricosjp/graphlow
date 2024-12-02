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
        self, mesh: IReadOnlyGraphlowMesh
    ) -> torch.Tensor:
        """Compute (n_cells, n_points)-shaped sparse incidence matrix.
        The method is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_cells, n_points)-shaped sparse csr tensor.
        """
        indices = mesh.pvmesh.cell_connectivity
        indptr = mesh.pvmesh.offset
        data = torch.ones(len(indices))
        size = (mesh.n_cells, mesh.n_points)
        cell_point_incidence = array_handler.send(
            torch.sparse_csr_tensor(indptr, indices, data, size=size),
            device=mesh.device,
            dtype=mesh.dtype,
        )

        mesh.dict_sparse_tensor.update(
            {SparseMatrixName.CELL_POINT_INCIDENCE: cell_point_incidence},
            overwrite=True,
        )
        return cell_point_incidence

    def compute_cell_adjacency(
        self, mesh: IReadOnlyGraphlowMesh
    ) -> torch.Tensor:
        """Compute (n_cells, n_cells)-shaped sparse adjacency matrix including
        self-loops. The method is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_cells, n_cells)-shaped sparse csr tensor.
        """
        scipy_cp_inc = array_handler.convert_to_scipy_sparse_csr(
            mesh.compute_cell_point_incidence()
        ).astype(bool)
        cell_adjacency = array_handler.convert_to_torch_sparse_csr(
            (scipy_cp_inc @ scipy_cp_inc.T).astype(float),
            device=mesh.device,
            dtype=mesh.dtype,
        )

        mesh.dict_sparse_tensor.update(
            {SparseMatrixName.CELL_ADJACENCY: cell_adjacency}, overwrite=True
        )
        return cell_adjacency

    def compute_point_adjacency(
        self, mesh: IReadOnlyGraphlowMesh
    ) -> torch.Tensor:
        """Compute (n_points, n_points)-shaped sparse adjacency matrix
        including self-loops. The method is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_points, n_points)-shaped sparse csr tensor.
        """
        scipy_cp_inc = array_handler.convert_to_scipy_sparse_csr(
            mesh.compute_cell_point_incidence()
        ).astype(bool)
        point_adjacency = array_handler.convert_to_torch_sparse_csr(
            (scipy_cp_inc.T @ scipy_cp_inc).astype(float),
            device=mesh.device,
            dtype=mesh.dtype,
        )

        mesh.dict_sparse_tensor.update(
            {SparseMatrixName.POINT_ADJACENCY: point_adjacency}, overwrite=True
        )
        return point_adjacency

    def compute_point_relative_incidence(
        self, mesh: IReadOnlyGraphlowMesh, other_mesh: IReadOnlyGraphlowMesh
    ) -> torch.Tensor:
        """Compute (n_points_other, n_points_self)-shaped sparse incidence
        matrix based on points.

        Parameters
        ----------
        other_mesh: graphlow.GraphlowMesh
            Other mesh object to be

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
        values = torch.ones(len(col))
        indices = torch.stack([row, col], dim=0)
        size = (other_mesh.n_points, mesh.n_points)
        return array_handler.convert_to_torch_sparse_csr(
            torch.sparse_coo_tensor(indices, values, size=size),
            device=mesh.device,
            dtype=mesh.dtype,
        )

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
        other_mesh: graphlow.GraphlowMesh
            Other mesh object to be
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
            mesh.compute_point_relative_incidence(other_mesh).to(bool)
        )
        other_incidence = (
            array_handler.convert_to_scipy_sparse_csr(
                other_mesh.compute_cell_point_incidence().to(bool)
            )
            @ other_self_point_incidence
        ).astype(int)  # (n_cells_other, n_points_self)
        self_incidence = array_handler.convert_to_scipy_sparse_csr(
            mesh.compute_cell_point_incidence().to(int)
        ).T  # (n_points_self, n_cells_self)

        # (n_other_cells, n_self_cells)
        dot = other_incidence.dot(self_incidence)

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

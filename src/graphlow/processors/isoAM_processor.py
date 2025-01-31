from __future__ import annotations

import torch

from graphlow.base.mesh_interface import IReadOnlyGraphlowMesh
from graphlow.util.logger import get_logger

logger = get_logger(__name__)


class IsoAMProcessor:
    """A class for isoAM calculation."""

    def __init__(self) -> None:
        pass

    def compute_isoAM(
        self,
        mesh: IReadOnlyGraphlowMesh,
        with_moment_matrix: bool = True,
        consider_volume: bool = False,
    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        """Compute (dims, n_points, n_points)-shaped isoAM.

        Parameters
        ----------
        with_moment_matrix: bool, optional [True]
            If True, scale the matrix with moment matrices, which are
            tensor products of relative position tensors.
        consider_volume: bool, optional [False]
            If True, consider effective volume of each vertex.

        Returns
        -------
        If `with_moment_matrix` is True, returns a tuple of 2 tensors,
            (isoAM, Minv).
        If `with_moment_matrix` is False, returns a tuple,
            (isoAM, None).

        isoAM: (dims, n_points, n_points)-shaped sparse coo tensor
        Minv: (n_points, dims, dims)-shaped tensor
        """
        points = mesh.points
        adj = mesh.compute_point_adjacency().to_sparse_coo()
        n_points, dim = points.shape

        # compute x_jk - x_ik
        diff_kij = self._compute_differences(points, adj)

        # compute: 1 / || x_j - x_i ||^2
        weight_by_squarenorm_ij = self._compute_inverse_square_norm(diff_kij)

        # consider effective volumes as weight: V_j / V_i
        if consider_volume:
            Wij = self._compute_weight_from_volume(mesh, adj)
            weight_by_squarenorm_ij *= Wij

        weighted_diff_kij = torch.stack(
            [diff_kij[k] * weight_by_squarenorm_ij for k in range(dim)]
        )

        # without moment matrix
        if not with_moment_matrix:
            isoAM = self._create_grad_operator_from(
                weighted_diff_kij
            ).coalesce()
            return isoAM, None

        # compute inversed moment matrix for each point
        diff_ijk = diff_kij.permute((1, 2, 0))
        weighted_diff_ikj = weighted_diff_kij.permute(1, 0, 2)
        inversed_moment_tensors = torch.stack(
            [
                torch.pinverse(weighted_diff_ikj[i].mm(diff_ijk[i]).to_dense())
                for i in range(n_points)
            ]
        )

        # compute isoAM using inversed moment matrix
        Dkij = torch.stack(
            [
                (
                    inversed_moment_tensors[i] @ weighted_diff_ikj[i]
                ).to_sparse_coo()
                for i in range(n_points)
            ]
        ).permute((1, 0, 2))
        isoAM = self._create_grad_operator_from(Dkij).coalesce()
        return isoAM, inversed_moment_tensors

    def compute_isoAM_with_neumann(
        self,
        mesh: IReadOnlyGraphlowMesh,
        normal_weight: float = 10.0,
        with_moment_matrix: bool = True,
        consider_volume: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, None | torch.Tensor]:
        """Compute (dims, n_points, n_points)-shaped
        Neumann boundary model IsoAM.

        Parameters
        ----------
        normal_weight: float, optional [10.0]
            Weight of the normal vector.
        with_moment_matrix: bool, optional [True]
            If True, scale the matrix with moment matrices, which are
            tensor products of relative position tensors.
        consider_volume: bool, optional [False]
            If True, consider effective volume of each vertex.

        Returns
        -------
        If `with_moment_matrix` is True, returns a tuple of 3 tensors,
            (NIsoAM, weighted_normals, Minv).
        If `with_moment_matrix` is False, returns a tuple,
            (NIsoAM, weighted_normals, None).

        NIsoAM: (dims, n_points, n_points)-shaped sparse coo tensor
        weighted_normals: (n_points, dims)-shaped tensor
        Minv: (n_points, dims, dims)-shaped tensor
        """
        points = mesh.points
        adj = mesh.compute_point_adjacency().to_sparse_coo()
        n_points, dim = points.shape

        # compute normals
        normals = self._compute_normals_on_surface_points(mesh)
        weighted_normals = normal_weight * normals

        # compute x_jk - x_ik
        diff_kij = self._compute_differences(points, adj)

        # compute: 1 / || x_j - x_i ||^2
        weight_by_squarenorm_ij = self._compute_inverse_square_norm(diff_kij)

        # consider effective volumes as weight: V_j / V_i
        if consider_volume:
            Wij = self._compute_weight_from_volume(mesh, adj)
            weight_by_squarenorm_ij *= Wij

        weighted_diff_kij = torch.stack(
            [diff_kij[k] * weight_by_squarenorm_ij for k in range(dim)]
        )

        # without moment matrix
        if not with_moment_matrix:
            isoAM = self._create_grad_operator_from(
                weighted_diff_kij
            ).coalesce()
            return isoAM, weighted_normals, None

        # compute inversed moment matrix for each point
        diff_ijk = diff_kij.permute((1, 2, 0))
        weighted_diff_ikj = weighted_diff_kij.permute(1, 0, 2)
        inversed_moment_tensors = torch.stack(
            [
                torch.pinverse(
                    weighted_diff_ikj[i].mm(diff_ijk[i]).to_dense()
                    + weighted_normals[i].outer(normals[i])
                )
                for i in range(n_points)
            ]
        )

        # compute isoAM using inversed moment matrix
        Dkij = torch.stack(
            [
                (
                    inversed_moment_tensors[i] @ weighted_diff_ikj[i]
                ).to_sparse_coo()
                for i in range(n_points)
            ]
        ).permute((1, 0, 2))
        isoAM = self._create_grad_operator_from(Dkij).coalesce()
        return isoAM, weighted_normals, inversed_moment_tensors

    def _compute_differences(
        self, points: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        """Compute: x_{jk} - x_{ik}, where x represents a point

        Parameters
        ----------
        points: (n_points, dim)-shaped torch tensor

        adj: (n_points, n_points)-shaped torch sparse coo tensor

        Returns:
        --------
            (dim, n_points, n_points)-shaped torch sparse coo tensor
        """
        # Extract indices from sparse adjacency matrix
        n_points, dim = points.shape
        adj_indices = adj.indices()

        # remove diag elements to avoid division by zero
        adj_indices = adj_indices[:, adj_indices[0, :] != adj_indices[1, :]]
        rows, cols = adj_indices

        # calculate differences between points
        diff_xj_xi = points[cols] - points[rows]
        dim_idx = torch.arange(dim).repeat_interleave(adj_indices.shape[1])
        diff_indices = torch.cat(
            (dim_idx.unsqueeze(0), adj_indices.repeat(1, dim))
        )
        diff_vals = diff_xj_xi.T.flatten()
        diff_kij = torch.sparse_coo_tensor(
            diff_indices, diff_vals, size=(dim, *adj.shape)
        ).coalesce()
        return diff_kij

    def _compute_inverse_square_norm(self, d_kij: torch.Tensor) -> torch.Tensor:
        """Compute: 1 / || sum_k d_{kij}^2 ||^2

        Parameters
        ----------
        diff_kij: (dim, n_points, n_points)-shaped torch sparse coo tensor

        Returns:
        --------
            (n_points, n_points)-shaped torch sparse coo tensor
        """
        squarenorm_ij = d_kij.pow(2).sum(dim=0).pow(-1)
        if torch.isinf(squarenorm_ij.values()).any():
            raise ZeroDivisionError("The input contains duplicate points.")
        return squarenorm_ij

    def _compute_weight_from_volume(
        self, mesh: IReadOnlyGraphlowMesh, adj: torch.Tensor
    ) -> torch.Tensor:
        """Compute: V_j / V_i

        Parameters
        ----------
        mesh: GraphlowMesh

        adj: (n_points, n_points)-shaped torch sparse coo tensor

        Returns:
        --------
            (n_points, n_points)-shaped torch sparse coo tensor
        """
        adj_indices = adj.indices()
        rows, cols = adj_indices

        cell_volumes = mesh.compute_volumes()
        effective_volumes = mesh.convert_elemental2nodal(
            cell_volumes, mode="effective"
        )
        W_vals = effective_volumes[cols] / effective_volumes[rows]
        Wij = torch.sparse_coo_tensor(adj_indices, W_vals, size=adj.shape)
        return Wij

    def _create_grad_operator_from(self, Akij: torch.Tensor) -> torch.Tensor:
        """Create a grad operator from a given sparse coo matrix
        where each row(i) has a value and diag(i=j) has no value (sparse)

        Parameters
        ----------
        Akij: (dim, n_points, n_points)-shaped torch sparse coo tensor
            To ensure correct calculation, each row(i) must contain a value and
            diag(i=j) has no value (sparse)

        Returns:
        --------
            (dim, n_points, n_points)-shaped torch sparse coo tensor
        """
        dim, n_points, _ = Akij.shape
        L_values = torch.sparse.sum(Akij, dim=2).to_dense()
        L_indices = torch.arange(n_points).expand(2, -1)
        grad_op = torch.stack(
            [
                Akij[k]
                - torch.sparse_coo_tensor(
                    L_indices, L_values[k], size=(n_points, n_points)
                )
                for k in range(dim)
            ]
        )
        return grad_op

    def _compute_normals_on_surface_points(
        self, mesh: IReadOnlyGraphlowMesh
    ) -> torch.Tensor:
        surf = mesh.extract_surface(pass_points=True)
        surf_vol_rel_inc = (
            mesh.compute_point_relative_incidence(surf).to_sparse_coo().T
        )
        normals_on_faces = surf.compute_normals()
        normals_on_points = surf.convert_elemental2nodal(
            normals_on_faces, "mean"
        )
        normals_on_points /= normals_on_points.norm(dim=1, keepdim=True)
        return surf_vol_rel_inc @ normals_on_points

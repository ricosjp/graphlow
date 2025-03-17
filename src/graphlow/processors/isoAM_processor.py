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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute (dims, n_points, n_points)-shaped isoAM.

        Parameters
        ----------
        mesh: GraphlowMesh
        with_moment_matrix: bool, optional [True]
            If True, scale the matrix with moment matrices, which are
            tensor products of relative position tensors.
        consider_volume: bool, optional [False]
            If True, consider effective volume of each vertex.

        Returns
        -------
        isoAM: torch.Tensor | None
            (dims, n_points, n_points)-shaped sparse coo tensor
        Minv: torch.Tensor | None
            if `with_moment_matrix` is True,
                return (n_points, dims, dims)-shaped tensor
            if `with_moment_matrix` is False,
                return None
        """
        points = mesh.points
        n_points, dim = points.shape
        adj = mesh.compute_point_adjacency().to_sparse_coo()
        i_indices, j_indices = adj.indices()  # (2, nnz)
        diag_mask = i_indices == j_indices

        # Compute differences x_j - x_i
        diff = points[j_indices] - points[i_indices]  # (nnz, dim)

        # Compute squared norms ||x_j - x_i||^2
        norm_sq = torch.norm(diff, dim=1) ** 2  # (nnz,)

        # Compute (x_j - x_i) / ||x_j - x_i||^2
        p = diff / norm_sq.unsqueeze(1)  # (nnz, dim)
        p[diag_mask] = 0.0
        if torch.isinf(p).any():
            raise ZeroDivisionError("Input mesh contains duplicate points")

        # Compute weights
        weights = torch.ones(
            i_indices.shape[0], device=points.device, dtype=points.dtype
        )  # (nnz,)

        if consider_volume:
            weights = self._compute_weights_nnz_from_volume(mesh)

        if not with_moment_matrix:
            nnz_tensor = p * weights.unsqueeze(1)  # (nnz, dim)
            isoAM = self._create_grad_operator_from(
                i_indices, j_indices, n_points, nnz_tensor
            )
            return isoAM, None

        moment_matrix = self._compute_moment_matrix(
            i_indices, j_indices, points, weights
        )

        # Precompute normals to avoid singular matrices
        normals = self._compute_normals_on_surface_points(mesh)
        normals_outer = normals.unsqueeze(2) * normals.unsqueeze(1)

        moment_rank = torch.linalg.matrix_rank(moment_matrix, hermitian=True)
        batch_mask = moment_rank < dim
        moment_matrix[batch_mask] += normals_outer[batch_mask]

        # Compute the inverse of M_i
        moment_inv = torch.linalg.inv(moment_matrix)  # (n_points, dim, dim)

        # Get M_i^{-1} for each edge (i,j)
        moment_inv_i = moment_inv[i_indices]  # (nnz, dim, dim)

        # Compute M_i^{-1} @ p for each edge
        temp = torch.bmm(moment_inv_i, p.unsqueeze(2)).squeeze(2)  # (nnz, dim)

        # Compute D_{k,ij} = w_ij * (M_i^{-1} @ p)_k
        nnz_tensor = weights.unsqueeze(1) * temp  # (nnz, dim)
        isoAM = self._create_grad_operator_from(
            i_indices, j_indices, n_points, nnz_tensor
        )
        return isoAM, moment_inv

    def compute_isoAM_with_neumann(
        self,
        mesh: IReadOnlyGraphlowMesh,
        normal_weight: float = 10.0,
        with_moment_matrix: bool = True,
        consider_volume: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
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
        NIsoAM: torch.Tensor
            (dims, n_points, n_points)-shaped sparse coo tensor
        weighted_normals: torch.Tensor
            (n_points, dims)-shaped tensor
        Minv: torch.Tensor | None
            if `with_moment_matrix` is True,
                return (n_points, dims, dims)-shaped tensor
            if `with_moment_matrix` is False,
                return None
        """
        points = mesh.points
        n_points = points.shape[0]
        adj = mesh.compute_point_adjacency().to_sparse_coo()
        i_indices, j_indices = adj.indices()  # (2, nnz)
        diag_mask = i_indices == j_indices

        # Compute normals
        normals = self._compute_normals_on_surface_points(mesh)
        weighted_normals = normal_weight * normals  # (n_points, dim)
        normals_outer = weighted_normals.unsqueeze(2) * normals.unsqueeze(
            1
        )  # (n_points, dim, dim)

        # Compute differences x_j - x_i
        diff = points[j_indices] - points[i_indices]  # (nnz, dim)

        # Compute squared norms ||x_j - x_i||^2
        norm_sq = torch.norm(diff, dim=1) ** 2  # (nnz,)

        # Compute (x_j - x_i) / ||x_j - x_i||^2
        p = diff / norm_sq.unsqueeze(1)  # (nnz, dim)
        p[diag_mask] = 0.0
        if torch.isinf(p).any():
            raise ZeroDivisionError("Input mesh contains duplicate points")

        # Compute weights
        weights = torch.ones(
            i_indices.shape[0], device=points.device, dtype=points.dtype
        )  # (nnz,)

        if consider_volume:
            weights = self._compute_weights_nnz_from_volume(mesh)

        if not with_moment_matrix:
            nnz_tensor = p * weights.unsqueeze(1)  # (nnz, dim)
            isoAM = self._create_grad_operator_from(
                i_indices, j_indices, n_points, nnz_tensor
            )
            return isoAM, weighted_normals, None

        moment_matrix = (
            self._compute_moment_matrix(i_indices, j_indices, points, weights)
            + normals_outer
        )  # (n_points, dim, dim)

        # Compute the inverse of M_i
        moment_inv = torch.linalg.inv(moment_matrix)  # (n_points, dim, dim)

        # Get M_i^{-1} for each edge (i,j)
        moment_inv_i = moment_inv[i_indices]  # (nnz, dim, dim)

        # Compute M_i^{-1} @ p for each edge
        temp = torch.bmm(moment_inv_i, p.unsqueeze(2)).squeeze(2)  # (nnz, dim)

        # Compute D_{k,ij} = w_ij * (M_i^{-1} @ p)_k
        nnz_tensor = weights.unsqueeze(1) * temp  # (nnz, dim)
        isoAM = self._create_grad_operator_from(
            i_indices, j_indices, n_points, nnz_tensor
        )
        return isoAM, weighted_normals, moment_inv

    def _compute_moment_matrix(
        self,
        i_indices: torch.Tensor,
        j_indices: torch.Tensor,
        points: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the moment matrix M_i for each point.

        Parameters
        ----------
        mesh : GraphlowMesh
            The mesh to compute the moment matrix for.

        Returns
        -------
        torch.Tensor
            (n_points, dim, dim)-shaped tensor sparse coo tensor
        """
        diag_mask = i_indices == j_indices

        # Compute differences
        diff = points[j_indices] - points[i_indices]  # (nnz, dim)

        # Compute norms
        norms = torch.norm(diff, dim=1)  # (nnz,)

        # Compute unit vectors
        u = diff / norms.unsqueeze(1)  # (nnz, dim)
        u[diag_mask] = 0.0
        if torch.isinf(u).any():
            raise ZeroDivisionError("Input mesh contains duplicate points")

        # Compute outer products: (nnz, dim, dim)
        u_outer = u.unsqueeze(2) * u.unsqueeze(1)  # (nnz, dim, dim)

        # Compute weighted outer products: weights * u_outer
        weighted_u_outer = u_outer * weights.unsqueeze(1).unsqueeze(
            2
        )  # (nnz, dim, dim)

        # Initialize M_i as (n_points, dim, dim)
        n_points, dim = points.shape
        M = torch.zeros(
            n_points, dim, dim, dtype=points.dtype, device=points.device
        )

        # Sum each row
        M.index_add_(0, i_indices, weighted_u_outer)
        return M

    def _compute_weights_nnz_from_volume(
        self, mesh: IReadOnlyGraphlowMesh
    ) -> torch.Tensor:
        """Compute: V_j / V_i

        Parameters
        ----------
        mesh: GraphlowMesh

        Returns
        -------
        (nnz,)-shaped tensor
        """
        adj = mesh.compute_point_adjacency().to_sparse_coo()
        i_indices, j_indices = adj.indices()
        cell_volumes = torch.abs(mesh.compute_volumes())
        effective_volumes = mesh.convert_elemental2nodal(
            cell_volumes, mode="effective"
        )
        weights = effective_volumes[j_indices] / effective_volumes[i_indices]
        return weights

    def _create_grad_operator_from(
        self,
        i_indices: torch.Tensor,
        j_indices: torch.Tensor,
        n_points: int,
        nnz_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Create a grad operator from a given tensor

        Parameters
        ----------
        nnz_tensor: (nnz, dim)-shaped torch tensor

        Returns
        -------
        (dim, n_points, n_points)-shaped torch sparse coo tensor
        """
        dim = nnz_tensor.shape[1]

        # Compute sum_D_per_i_k: (n_points, dim)
        sum_D_per_i_k = torch.zeros(
            n_points, dim, dtype=nnz_tensor.dtype, device=nnz_tensor.device
        )
        sum_D_per_i_k.index_add_(0, i_indices, nnz_tensor)

        # Identify self-loop edges (i == j)
        diag_mask = i_indices == j_indices

        # Adjust tilde_D by subtracting sum_D_per_i_k for self-loop edges
        grad_adj = nnz_tensor.clone()
        grad_adj[diag_mask] -= sum_D_per_i_k[i_indices[diag_mask]]

        # nnz, dim -> dim, n_points, n_points
        indices = torch.stack([i_indices, j_indices], dim=0)
        result = torch.stack(
            [
                torch.sparse_coo_tensor(
                    indices, grad_adj[:, k], (n_points, n_points)
                )
                for k in range(dim)
            ]
        )
        return result

    def _compute_normals_on_surface_points(
        self, mesh: IReadOnlyGraphlowMesh
    ) -> torch.Tensor:
        """Compute normals tensor with values only on the surface points.

        Parameters
        ----------
        mesh: GraphlowMesh

        Returns
        -------
        normals: (n_points, dim)-shaped tensor
        """
        surf = mesh.extract_surface(pass_point_data=True)
        surf_vol_rel_inc = (
            mesh.compute_point_relative_incidence(surf).to_sparse_coo().T
        )
        normals_on_faces = surf.compute_normals()
        normals_on_points = surf.convert_elemental2nodal(
            normals_on_faces, "mean"
        )
        normals_on_points /= normals_on_points.norm(dim=1, keepdim=True)
        return surf_vol_rel_inc @ normals_on_points

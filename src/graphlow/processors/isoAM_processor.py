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

        # precompute normals outer product to avoid recomputation
        normals = self._compute_normals_on_surface_points(mesh)
        normals_outer = normals.unsqueeze(2) * normals.unsqueeze(1)

        inversed_moment_tensors = []
        for i in range(n_points):
            mi = weighted_diff_ikj[i].mm(diff_ijk[i]).to_dense()
            if torch.linalg.matrix_rank(mi) < dim:
                mi += normals_outer[i]
            inversed_moment_tensors.append(torch.inverse(mi))
        inversed_moment_tensors = torch.stack(inversed_moment_tensors)

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

        # precompute normals outer product to avoid recomputation
        normals_outer = weighted_normals.unsqueeze(2) * normals.unsqueeze(1)
        inversed_moment_tensors = []
        for i in range(n_points):
            mi = weighted_diff_ikj[i].mm(diff_ijk[i]).to_dense()
            mi += normals_outer[i]
            inversed_moment_tensors.append(torch.inverse(mi))
        inversed_moment_tensors = torch.stack(inversed_moment_tensors)

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
        cell_volumes = mesh.compute_volumes()
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

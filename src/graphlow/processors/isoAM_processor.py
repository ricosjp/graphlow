from __future__ import annotations

from typing import Literal

import phlower_tensor as pt
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
        normal_interp_mode: Literal["mean", "conservative"] = "conservative",
    ) -> tuple[pt.PhlowerTensor, pt.PhlowerTensor | None]:
        """Compute (dims, n_points, n_points)-shaped isoAM.

        Parameters
        ----------
        mesh: GraphlowMesh
        with_moment_matrix: bool, optional [True]
            If True, scale the matrix with moment matrices, which are
            tensor products of relative position tensors.
        consider_volume: bool, optional [False]
            If True, consider effective volume of each vertex.
        normal_interp_mode: Literal["mean", "conservative"], \
            default: "conservative" \
            The way to interpolate normals. cf. convert_elemental2nodal.
            - "mean": For each node, \
                we consider all the elements that share this node \
                and compute the average of their values.
                This approach provides \
                a smoothed representation at each node.
            - "conservative": For each element,
                we consider all the nodes that share this element \
                and distribute the element value to them equally.
                The values are then summed at each node. \
                This approach ensures that the total quantity \
                (such as mass or volume) is conserved.

        Returns
        -------
        isoAM: pt.PhlowerTensor | None
            (dims, n_points, n_points)-shaped sparse coo tensor
        Minv: pt.PhlowerTensor | None
            if `with_moment_matrix` is True,
                return (n_points, dims, dims)-shaped tensor
            if `with_moment_matrix` is False,
                return None
        """
        points = mesh.points
        n_points, dim = points.shape
        adj = mesh.compute_point_adjacency()
        i_indices, j_indices = adj.indices()  # (2, nnz)
        diag_mask = i_indices == j_indices

        # Compute differences: x_j - x_i
        diff = points[j_indices] - points[i_indices]  # (nnz, dim)

        # Compute squared norms: ||x_j - x_i||^2
        squared_norm = torch.linalg.vector_norm(diff, dim=1) ** 2  # (nnz,)

        # Compute weights: w_ij
        weights = pt.phlower_tensor(
            torch.ones(
                i_indices.shape[0], device=points.device, dtype=points.dtype
            ),
            dimension={},
        ).to(device=points.device)  # (nnz,)

        if consider_volume:
            weights = self._compute_weights_nnz_from_volume(mesh)

        # Compute weighted inverse of squared norms w_ij / ||x_j - x_i||^2
        weighted_inv_squarenorm = weights / squared_norm  # (nnz,)
        weighted_inv_squarenorm[diag_mask] = 0.0
        if torch.isinf(weighted_inv_squarenorm.to_tensor()).any():
            raise ZeroDivisionError("Input mesh contains duplicate points")

        # Compute element tensor: w_ij (x_j - x_i) / ||x_j - x_i||^2
        element = diff * weighted_inv_squarenorm[:, None]  # (nnz, dim)

        if not with_moment_matrix:
            isoAM = self._create_grad_operator_from(
                i_indices, j_indices, n_points, element
            )
            return isoAM, None

        moment_matrix = self._compute_moment_matrix(
            i_indices, j_indices, points, weights
        )

        # Precompute normals to avoid singular matrices
        normals = self._compute_normals_on_surface_points(
            mesh, normal_interp_mode
        )
        n_otimes_n = (
            normals[:, None, :] * normals[:, :, None]
        )  # (n_points, dim, dim)

        moment_rank = torch.linalg.matrix_rank(
            moment_matrix.to_tensor(), hermitian=True
        )
        batch_mask = moment_rank < dim
        moment_matrix[batch_mask] += n_otimes_n[batch_mask]

        # Compute the inverse of M_i
        moment_inv_dim = moment_matrix.dimension * -1
        moment_inv = pt.phlower_tensor(
            torch.linalg.inv(moment_matrix.to_tensor()),
            dimension=moment_inv_dim,
        ).to(device=points.device)  # (n_points, dim, dim)

        # Get M_i^{-1} for each edge (i,j)
        moment_inv_i = moment_inv[i_indices]  # (nnz, dim, dim)

        # Compute element tensor:  M_i^{-1} w_ij (x_j - x_i) / ||x_j - x_i||^2
        elem_col = element[:, :, None]
        element_with_moment = pt.phlower_tensor(
            torch.bmm(moment_inv_i.to_tensor(), elem_col.to_tensor()).squeeze(
                2
            ),
            dimension=element.dimension,
        ).to(device=points.device)  # (nnz, dim)

        # Compute D_{k,ij}
        isoAM = self._create_grad_operator_from(
            i_indices, j_indices, n_points, element_with_moment
        )

        return isoAM, moment_inv

    def compute_isoAM_with_neumann(
        self,
        mesh: IReadOnlyGraphlowMesh,
        normal_weight: float = 10.0,
        with_moment_matrix: bool = True,
        consider_volume: bool = False,
        normal_interp_mode: Literal["mean", "conservative"] = "conservative",
    ) -> tuple[pt.PhlowerTensor, pt.PhlowerTensor, pt.PhlowerTensor | None]:
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
        normal_interp_mode: Literal["mean", "conservative"], \
            default: "conservative" \
            The way to interpolate normals. cf. convert_elemental2nodal.
            - "mean": For each node, \
                we consider all the elements that share this node \
                and compute the average of their values.
                This approach provides \
                a smoothed representation at each node.
            - "conservative": For each element,
                we consider all the nodes that share this element \
                and distribute the element value to them equally.
                The values are then summed at each node. \
                This approach ensures that the total quantity \
                (such as mass or volume) is conserved.

        Returns
        -------
        NIsoAM: pt.PhlowerTensor
            (dims, n_points, n_points)-shaped sparse coo tensor
        weighted_normals: pt.PhlowerTensor
            (n_points, dims)-shaped tensor
        Minv: pt.PhlowerTensor | None
            if `with_moment_matrix` is True,
                return (n_points, dims, dims)-shaped tensor
            if `with_moment_matrix` is False,
                return None
        """
        points = mesh.points
        n_points = points.shape[0]
        adj = mesh.compute_point_adjacency()
        i_indices, j_indices = adj.indices()  # (2, nnz)
        diag_mask = i_indices == j_indices

        # Compute normals
        normals = self._compute_normals_on_surface_points(
            mesh, normal_interp_mode
        )
        weighted_normals = normal_weight * normals  # (n_points, dim)
        n_otimes_n = (
            weighted_normals[:, None, :] * normals[:, :, None]
        )  # (n_points, dim, dim)

        # Compute differences: x_j - x_i
        diff = points[j_indices] - points[i_indices]  # (nnz, dim)

        # Compute squared norms: ||x_j - x_i||^2
        squared_norm = torch.linalg.vector_norm(diff, dim=1) ** 2  # (nnz,)

        # Compute weights: w_ij
        weights = pt.phlower_tensor(
            torch.ones(
                i_indices.shape[0], device=points.device, dtype=points.dtype
            ),
            dimension={},
        ).to(device=points.device)  # (nnz,)

        if consider_volume:
            weights = self._compute_weights_nnz_from_volume(mesh)

        # Compute weighted inverse of squared norms: w_ij / ||x_j - x_i||^2
        weighted_inv_squarenorm = weights / squared_norm  # (nnz,)
        weighted_inv_squarenorm[diag_mask] = 0.0
        if torch.isinf(weighted_inv_squarenorm.to_tensor()).any():
            raise ZeroDivisionError("Input mesh contains duplicate points")

        # Compute element tensor: w_ij (x_j - x_i) / ||x_j - x_i||^2
        element = diff * weighted_inv_squarenorm[:, None]  # (nnz, dim)

        if not with_moment_matrix:
            isoAM = self._create_grad_operator_from(
                i_indices, j_indices, n_points, element
            )
            return isoAM, weighted_normals, None

        moment_matrix = (
            self._compute_moment_matrix(i_indices, j_indices, points, weights)
            + n_otimes_n
        )  # (n_points, dim, dim)

        # Compute the inverse of M_i
        moment_inv_dim = moment_matrix.dimension * -1
        moment_inv = pt.phlower_tensor(
            torch.linalg.inv(moment_matrix.to_tensor()),
            dimension=moment_inv_dim,
        ).to(device=points.device)  # (n_points, dim, dim)

        # Get M_i^{-1} for each edge (i,j)
        moment_inv_i = moment_inv[i_indices]  # (nnz, dim, dim)

        # Compute element tensor
        elem_col = element[:, :, None]
        element_with_moment = pt.phlower_tensor(
            torch.bmm(moment_inv_i.to_tensor(), elem_col.to_tensor()).squeeze(
                2
            ),
            dimension=element.dimension,
        ).to(device=points.device)  # (nnz, dim)

        # Compute D_{k,ij}
        isoAM = self._create_grad_operator_from(
            i_indices, j_indices, n_points, element_with_moment
        )
        return isoAM, weighted_normals, moment_inv

    def _compute_moment_matrix(
        self,
        i_indices: torch.Tensor,
        j_indices: torch.Tensor,
        points: pt.PhlowerTensor,
        weights: pt.PhlowerTensor,
    ) -> pt.PhlowerTensor:
        """
        Compute the moment matrix M_i for each point.

        Parameters
        ----------
        i_indices: torch.Tensor
            The row indices of adjacency coo matrix. (nnz,)
        j_indices: torch.Tensor
            The column indices of adjacency coo matrix. (nnz,)
        points: pt.PhlowerTensor
            The points to compute the moment matrix for. (n_points, dim)
        weights: pt.PhlowerTensor
            The weights of the points. (nnz,)

        Returns
        -------
        pt.PhlowerTensor
            (n_points, dim, dim)-shaped tensor sparse coo tensor
        """
        n_points, dim = points.shape
        diag_mask = i_indices == j_indices

        # Compute differences: x_j - x_i
        diff = points[j_indices] - points[i_indices]  # (nnz, dim)

        # Compute squared norms: ||x_j - x_i||^2
        squared_norm = torch.linalg.vector_norm(diff, dim=1) ** 2  # (nnz,)

        # Compute weighted inverse of squared norms: w_ij / ||x_j - x_i||^2
        weighted_inv_squarenorm = weights / squared_norm  # (nnz,)
        weighted_inv_squarenorm[diag_mask] = 0.0
        if torch.isinf(weighted_inv_squarenorm.to_tensor()).any():
            raise ZeroDivisionError("Input mesh contains duplicate points")

        # Compute tensor products: (x_j - x_i) \otimes (x_j - x_i)
        d_otimes_d = diff[:, None, :] * diff[:, :, None]  # (nnz, dim, dim)

        # Compute weighted tensor products:
        # w_ij * (x_j - x_i) \otimes (x_j - x_i) / ||x_j - x_i||^2
        element = (
            d_otimes_d * weighted_inv_squarenorm[:, None, None]
        )  # (nnz, dim, dim)

        # Initialize moment matrix as (n_points, dim, dim)
        moment_matrix = torch.zeros(
            n_points, dim, dim, dtype=diff.dtype, device=diff.device
        )

        # Sum each row
        moment_matrix.index_add_(0, i_indices, element.to_tensor())
        return pt.phlower_tensor(moment_matrix, dimension=element.dimension)

    def _compute_weights_nnz_from_volume(
        self, mesh: IReadOnlyGraphlowMesh
    ) -> pt.PhlowerTensor:
        """Compute: V_j / V_i

        Parameters
        ----------
        mesh: GraphlowMesh

        Returns
        -------
        (nnz,)-shaped tensor
        """
        adj = mesh.compute_point_adjacency()
        i_indices, j_indices = adj.indices()
        cell_volumes = torch.abs(mesh.compute_volumes())
        effective_volumes = mesh.convert_elemental2nodal(
            cell_volumes, mode="conservative"
        )
        weights = effective_volumes[j_indices] / effective_volumes[i_indices]
        return weights

    def _create_grad_operator_from(
        self,
        i_indices: torch.Tensor,
        j_indices: torch.Tensor,
        n_points: int,
        element: pt.PhlowerTensor,
    ) -> pt.PhlowerTensor:
        """Create a grad operator from a given tensor

        Parameters
        ----------
        i_indices: torch.Tensor
            The row indices of adjacency coo matrix. (nnz,)
        j_indices: torch.Tensor
            The column indices of adjacency coo matrix. (nnz,)
        n_points: int
            The number of points.
        element: pt.PhlowerTensor
            The non-zero elements to create the grad operator from. (nnz, dim)

        Returns
        -------
        (dim, n_points, n_points)-shaped phlower sparse coo tensor
        """
        dim = element.shape[1]

        # Compute \sum_l D_{k,il}: (n_points, dim)
        sum_by_row = torch.zeros(
            n_points, dim, dtype=element.dtype, device=element.device
        )
        sum_by_row.index_add_(0, i_indices, element.to_tensor())
        sum_by_row = pt.phlower_tensor(sum_by_row, dimension=element.dimension)

        # Identify self-loop edges (i == j)
        diag_mask = i_indices == j_indices  # (nnz,)

        # substract diagonal elements: D_{k,ij} - \delta_{ij} \sum_l D_{k,il}
        grad_adj = (
            element
            - diag_mask.to(element.dtype)[:, None] * sum_by_row[i_indices]
        )

        # nnz, dim -> dim, n_points, n_points
        indices = torch.stack([i_indices, j_indices], dim=0)
        stacked = torch.stack(
            [
                torch.sparse_coo_tensor(
                    indices, grad_adj[:, k].to_tensor(), (n_points, n_points)
                )
                for k in range(dim)
            ]
        )
        return pt.phlower_tensor(stacked.coalesce(), dimension={}).to(
            device=element.device
        )

    def _compute_normals_on_surface_points(
        self,
        mesh: IReadOnlyGraphlowMesh,
        mode: Literal["mean", "conservative"] = "conservative",
        epsilon: float = 1.0e-3,
    ) -> pt.PhlowerTensor:
        """Compute normals tensor with values only on the surface points.

        Parameters
        ----------
        mesh: GraphlowMesh
            The mesh to compute the normals for.
        mode: Literal["mean", "conservative"], \
            default: "conservative" \
            The way to interpolate normals. cf. convert_elemental2nodal.
            - "mean": For each node, \
                we consider all the elements that share this node \
                and compute the average of their values.
                This approach provides \
                a smoothed representation at each node.
            - "conservative": For each element,
                we consider all the nodes that share this element \
                and distribute the element value to them equally.
                The values are then summed at each node. \
                This approach ensures that the total quantity \
                (such as mass or volume) is conserved.
        epsilon: float
            default: 1.0e-3
            Threshold to detect zero-normed normal vectors.

        Returns
        -------
        normals: (n_points, dim)-shaped tensor
        """
        surf = mesh.extract_surface(pass_point_data=True)
        surf_vol_rel_inc = mesh.compute_point_relative_incidence(
            surf
        ).transpose(0, 1)
        normals_on_faces = surf.compute_normals()
        normals_on_points = surf.convert_elemental2nodal(normals_on_faces, mode)
        norm = torch.linalg.vector_norm(normals_on_points, dim=1)  # (n_points,)

        filter_non_zero = norm.to_tensor() > epsilon
        filtered_normal = normals_on_points[
            filter_non_zero
        ]  # (n_non_zero, dim)
        normals_on_points[filter_non_zero] = (
            filtered_normal / norm[filter_non_zero, None]
        )
        normals_on_points[~filter_non_zero] = 0.0
        return surf_vol_rel_inc @ normals_on_points

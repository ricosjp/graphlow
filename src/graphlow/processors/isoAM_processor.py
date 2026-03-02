from __future__ import annotations

from typing import Literal

import phlower_tensor as pt
import torch

from graphlow.base.mesh_interface import IReadOnlyGraphlowMesh
from graphlow.util.logger import get_logger
from graphlow.util.phlower_helper import (
    phlower_ones,
    phlower_zeros,
)

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
        eps: float | None = None,
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
        eps: float | None, optional [None]
            The epsilon value to avoid zero division.
            If None, use the default epsilon value for the dtype.

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

        # Compute weights
        weights = phlower_ones(
            (n_points,), dimension={}, dtype=points.dtype, device=points.device
        )  # (n_points,)

        if consider_volume:
            volumes = torch.abs(mesh.compute_volumes())
            weights = mesh.convert_elemental2nodal(volumes, mode="conservative")

        # Precompute normals to avoid singular matrices
        normals = _compute_normals_on_surface_points(
            mesh, normal_interp_mode, eps
        )
        n_otimes_n = (
            normals[:, :, None] * normals[:, None, :]
        )  # (n_points, dim, dim)

        # Compute moment matrix
        if with_moment_matrix:
            moment_matrix = self._compute_moment_matrix(
                adj, points, weights, eps
            )
            moment_rank = torch.linalg.matrix_rank(
                moment_matrix.to_tensor(), hermitian=True
            )
            batch_mask = moment_rank < dim
            moment_matrix[batch_mask] += n_otimes_n[batch_mask]
        else:
            moment_matrix = None

        # Compute D_{k,ij}
        rawAM, moment_inv = _compute_rawAM_and_moment_inv(
            adj, points, weights, moment_matrix, eps
        )

        # Compute \tilde{D}_{k,ij}
        isoAM = _create_grad_operator_from(rawAM)
        return isoAM, moment_inv

    def compute_isoAM_with_neumann(
        self,
        mesh: IReadOnlyGraphlowMesh,
        normal_weight: float = 10.0,
        with_moment_matrix: bool = True,
        consider_volume: bool = False,
        normal_interp_mode: Literal["mean", "conservative"] = "conservative",
        eps: float | None = None,
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
        eps: float | None, optional [None]
            The epsilon value to avoid zero division.
            If None, use the default epsilon value for the dtype.

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
        n_points, _ = points.shape
        adj = mesh.compute_point_adjacency()

        # Compute weights
        weights = phlower_ones(
            (n_points,), dimension={}, dtype=points.dtype, device=points.device
        )  # (n_points,)

        if consider_volume:
            volumes = torch.abs(mesh.compute_volumes())
            weights = mesh.convert_elemental2nodal(volumes, mode="conservative")

        # Compute normals
        normals = _compute_normals_on_surface_points(
            mesh, normal_interp_mode, eps
        )
        weighted_normals = normal_weight * normals  # (n_points, dim)
        n_otimes_n = (
            weighted_normals[:, :, None] * normals[:, None, :]
        )  # (n_points, dim, dim)

        # Compute moment matrix
        if with_moment_matrix:
            moment_matrix = (
                self._compute_moment_matrix(adj, points, weights, eps)
                + n_otimes_n
            )
        else:
            moment_matrix = None

        # Compute D_{k,ij}
        rawAM, moment_inv = _compute_rawAM_and_moment_inv(
            adj, points, weights, moment_matrix, eps
        )

        # Compute \tilde{D}_{k,ij}
        NisoAM = _create_grad_operator_from(rawAM)
        return NisoAM, weighted_normals, moment_inv

    def _compute_moment_matrix(
        self,
        adj: pt.PhlowerTensor,
        points: pt.PhlowerTensor,
        weights: pt.PhlowerTensor,
        eps: float | None = None,
    ) -> pt.PhlowerTensor:
        r"""Compute the moment matrix M_i at each node i.

        M_i is the sum of (V_j/V_i) * u_ij \otimes u_ij over neighbors j,
        where u_ij is the unit vector from point i to j.

        Parameters
        ----------
        adj: pt.PhlowerTensor
            Point adjacency matrix. (N, N)
        points: pt.PhlowerTensor
            Node coordinates. (n_points, dim)
        weights: pt.PhlowerTensor
            The weights of the points. (n_points,)
        eps: float | None, optional [None]
            The epsilon value to avoid zero division.
            If None, use the default epsilon value for the dtype.

        Returns
        -------
        pt.PhlowerTensor
            Dense (n_points, dim, dim)-shaped moment matrix.
        """
        n_points, dim = points.shape
        i_indices, j_indices = adj.indices()  # (2, nnz)
        eps = eps or torch.finfo(points.dtype).eps

        # Compute differences: x_j - x_i
        diff = points[j_indices] - points[i_indices]  # (nnz, dim)

        # Compute distance (clamped to avoid zero division): ||x_j - x_i||
        distance = torch.clamp(
            torch.linalg.vector_norm(diff, dim=1), min=eps
        )  # (nnz,)

        # Compute unit vectors: (x_j - x_i) / ||x_j - x_i||
        u = diff / distance[:, None]  # (nnz, dim)

        # Compute weights: w_ij = V_j / V_i
        w = weights[j_indices] / torch.clamp(
            weights[i_indices], min=eps
        )  # (nnz,)

        # Compute tensor products
        outer = u[:, :, None] * u[:, None, :]  # (nnz, dim, dim)
        contrib = w[:, None, None] * outer  # (nnz, dim, dim)

        # Initialize moment matrix as (n_points, dim, dim) and sum by row
        moment_matrix = phlower_zeros(
            (n_points, dim, dim),
            dimension=contrib.dimension,
            dtype=contrib.dtype,
            device=contrib.device,
        )
        moment_matrix = moment_matrix.index_add(0, i_indices, contrib)
        return moment_matrix


def _compute_normals_on_surface_points(
    mesh: IReadOnlyGraphlowMesh,
    mode: Literal["mean", "conservative"] = "conservative",
    eps: float | None = None,
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
    eps: float | None, optional [None]
        The epsilon value to avoid zero division.
        If None, use the default epsilon value for the dtype.

    Returns
    -------
    pt.PhlowerTensor
        (n_points, dim)-shaped nodal normals.
    """
    eps = eps or torch.finfo(mesh.dtype).eps
    surf = mesh.extract_surface(pass_point_data=True)
    surf_vol_rel_inc = mesh.compute_point_relative_incidence(surf).transpose(
        0, 1
    )
    normals_on_faces = surf.compute_normals()
    normals_on_points = surf.convert_elemental2nodal(normals_on_faces, mode)
    norm: pt.PhlowerTensor = torch.linalg.vector_norm(
        normals_on_points, dim=1
    )  # (n_points,)

    filter_non_zero = norm.to_tensor() > eps
    filtered_normal = normals_on_points[filter_non_zero]  # (n_non_zero, dim)
    normals_on_points[filter_non_zero] = (
        filtered_normal / norm[filter_non_zero, None]
    )
    normals_on_points[~filter_non_zero] = 0.0
    return surf_vol_rel_inc @ normals_on_points


def _create_grad_operator_from(
    D: pt.PhlowerTensor,
) -> pt.PhlowerTensor:
    """Build the row-sum-zero gradient operator from the raw AM.

    Parameters
    ----------
    D: pt.PhlowerTensor
        The raw AM. (K, N, N)

    Returns
    -------
    pt.PhlowerTensor
        The grad operator. (K, N, N)
    """
    k_idx, i_idx, _ = D.indices()
    values = D.values()
    K, N, _ = D.shape

    # sum over l: sumD[k, i] = Σ_j D[k,i,j]
    sumD = torch.zeros((K, N), dtype=D.dtype, device=D.device)
    for kk in range(K):
        mask = k_idx == kk
        sumD[kk] = sumD[kk].index_add(0, i_idx[mask], values[mask])

    # add diagonal entries: D_tilde = D - diag(sumD)
    diag_k = torch.arange(K, device=D.device).repeat_interleave(N)
    diag_i = torch.arange(N, device=D.device).repeat(K)
    diag_j = diag_i
    diag_indices = torch.stack([diag_k, diag_i, diag_j], dim=0)
    diag_vals = -sumD.reshape(-1)
    new_idx = torch.cat([D.indices(), diag_indices], dim=1)
    new_vals = torch.cat([values, diag_vals], dim=0)
    return pt.phlower_tensor(
        torch.sparse_coo_tensor(
            new_idx, new_vals, size=(K, N, N), device=D.device, dtype=D.dtype
        ).coalesce(),
        dimension=D.dimension,
    ).to(device=D.device)


def _compute_rawAM_and_moment_inv(
    adj: pt.PhlowerTensor,
    x: pt.PhlowerTensor,
    V: pt.PhlowerTensor,
    moment_matrix: pt.PhlowerTensor | None = None,
    eps: float | None = None,
) -> tuple[pt.PhlowerTensor, pt.PhlowerTensor | None]:
    """Compute raw AM D_{k,ij} and optionally M_i^{-1}.

    Formula:
        D_{k,ij} = M_i^{-1} (x_jk-x_ik)
                    * (V_j/(||x_j-x_i||^2 V_i)) * A_ij
    If moment_matrix is None, M_i^{-1} is treated as identity.

    Parameters
    ----------
    adj: pt.PhlowerTensor
        Point adjacency matrix. (N, N)
    x: pt.PhlowerTensor
        Node coordinates. (N, K)
    V: pt.PhlowerTensor
        Nodal weights (e.g. ones or volumes when
        consider_volume=True). (N,)
    moment_matrix: pt.PhlowerTensor | None, optional [None]
        Per-node moment matrix (N, K, K).
        If None, raw AM is computed without it.
    eps: float | None, optional [None]
        The epsilon value to avoid zero division.
        If None, use the default epsilon value for the dtype.

    Returns
    -------
    raw_AM: pt.PhlowerTensor
        The raw AM. (K, N, N)
    moment_inv: pt.PhlowerTensor | None
        Inverse moment matrix (N, K, K) if moment_matrix was given, else None.
    """
    N, K = x.shape
    row, col = adj.indices()
    nnz = row.shape[0]
    eps = eps or torch.finfo(x.dtype).eps

    # Compute differences: x_jk - x_ik
    diff = x[col] - x[row]  # (nnz,k)

    # Compute squared norms: ||x_j - x_i||^2
    dist2 = torch.clamp(torch.sum(diff * diff, dim=1), min=eps)

    # Compute scalar: V_j / (V_i * ||x_j - x_i||^2)
    scalar = V[col] / (torch.clamp(V[row], min=eps) * dist2)  # (nnz,)

    rhs = diff[:, :, None]

    if moment_matrix is None:
        sol = rhs
        Minv = None
    else:
        M = moment_matrix[row]
        Minv = phlower_zeros(
            (N, K, K),
            dimension=moment_matrix.dimension * -1,
            dtype=moment_matrix.dtype,
            device=moment_matrix.device,
        )
        # solve M_i^{-1} @ (x_jk - x_ik)
        try:
            # Cholesky works only for SPD
            L = torch.linalg.cholesky(M)  # (nnz,k,k)
            sol = torch.cholesky_solve(rhs, L)  # (nnz,k,1)
            moment_inv = torch.cholesky_inverse(L)  # (nnz,k,k)
        except torch.linalg.LinAlgError:
            moment_inv = torch.linalg.inv(M)  # (nnz,k,k)
            sol = moment_inv @ rhs  # (nnz,k,1)

        # ---- pack (nnz,k,k) -> 3D sparse COO (n,k,k) ----
        Minv[row, :, :] = moment_inv

    D_vals = sol[:, :, 0] * scalar[:, None]  # (nnz,k)

    # ---- pack (nnz,k) -> 3D sparse COO (k,n,n) ----
    k_idx = torch.arange(K, device=x.device).repeat(nnz)  # (nnz*k,)
    i_idx = row.repeat_interleave(K)  # (nnz*k,)
    j_idx = col.repeat_interleave(K)  # (nnz*k,)
    values = D_vals.reshape((-1,))  # row-major: (nnz*k,)

    D = pt.phlower_tensor(
        torch.sparse_coo_tensor(
            torch.stack([k_idx, i_idx, j_idx], dim=0),
            values.to_tensor(),
            size=(K, N, N),
            dtype=D_vals.dtype,
        ).coalesce(),
        dimension=D_vals.dimension,
    ).to(device=D_vals.device)

    return D, Minv

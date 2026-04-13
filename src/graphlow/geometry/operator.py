from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch

from graphlow.core.backend.base import Backend, TensorLike
from graphlow.utils.dimension import get_dimension

if TYPE_CHECKING:
    from graphlow.core.mesh import TensorMesh


logger = logging.getLogger(__name__)


def isoAM[T: TensorLike](
    mesh: TensorMesh[T],
    with_moment_matrix: bool = True,
    consider_volume: bool = False,
    normal_interp_mode: Literal["mean", "conservative"] = "conservative",
    eps: float = 1e-12,
) -> tuple[T, T | None]:
    """
    Compute isoAM of shape ``(dims, n_points, n_points)``.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh with differentiable points.
    with_moment_matrix : bool, optional
        If True, use moment matrices (tensor products of relative
        position vectors) and return their inverse. Default is True.
    consider_volume : bool, optional
        If True, use cell volumes as vertex weights. Default is False.
    normal_interp_mode : {"mean", "conservative"}, optional
        How to map face normals to points. Default is "conservative".
    eps : float, optional
        Epsilon to avoid zero division. Default is 1e-12.

    Returns
    -------
    isoam : T
        IsoAM operator of shape ``(dims, n_points, n_points)`` (sparse COO).
    moment_inv : T or None
        Inverse of the moment matrix, shape ``(n_points, dims, dims)``,
        if ``with_moment_matrix`` is True; otherwise None.
    """
    backend = mesh.backend
    points = mesh.points
    n_points, dim = points.shape
    adj = mesh.topology.point_adjacency(layout="coo")

    # Compute weights
    weights = backend.ones((n_points, 1), dimension={})

    if consider_volume:
        volumes = torch.abs(mesh.geometry.cell_volumes())
        weights = mesh.topology.map_cell_to_point(
            volumes, normal_interp_mode, "segment"
        )
    weights = weights.reshape((-1,))

    # Compute moment matrix
    if with_moment_matrix:
        normals_on_surface_points = _compute_normals_on_surface_points(
            mesh, normal_interp_mode, eps
        )

        n_otimes_n = (
            normals_on_surface_points[:, :, None]
            * normals_on_surface_points[:, None, :]
        )  # (n_points, dim, dim)

        moment_matrix = _compute_moment_matrix(
            backend, adj, points, weights, eps
        )
        moment_rank = torch.linalg.matrix_rank(
            backend.to_torch(moment_matrix), hermitian=True
        )
        batch_mask = moment_rank < dim
        moment_matrix[batch_mask] += n_otimes_n[batch_mask]
    else:
        moment_matrix = None

    # Compute D_{k,ij}
    rawAM, moment_inv = _compute_rawAM_and_moment_inv(
        backend, adj, points, weights, moment_matrix, eps
    )

    # Compute \tilde{D}_{k,ij}
    isoAM = _create_grad_operator_from(backend, rawAM)
    return isoAM, moment_inv


def isoAM_with_neumann[T: TensorLike](
    mesh: TensorMesh[T],
    with_moment_matrix: bool = True,
    consider_volume: bool = False,
    normal_weight: float = 10.0,
    normal_interp_mode: Literal["mean", "conservative"] = "conservative",
    eps: float = 1e-12,
) -> tuple[T, T, T | None]:
    """
    Compute IsoAM with Neumann boundary model.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh with differentiable points.
    with_moment_matrix : bool, optional
        If True, use moment matrices and return their inverse. Default is True.
    consider_volume : bool, optional
        If True, use cell volumes as vertex weights. Default is False.
    normal_weight : float, optional
        Weight for the Neumann boundary normal term. Default is 10.0.
    normal_interp_mode : {"mean", "conservative"}, optional
        How to map face normals to points. Default is "conservative".
    eps : float, optional
        Epsilon to avoid zero division. Default is 1e-12.

    Returns
    -------
    isoam : T
        IsoAM operator of shape ``(dims, n_points, n_points)`` (sparse COO).
    weighted_normals : T
        Right-hand side for Neumann condition, shape ``(n_points, dims)``.
    moment_inv : T or None
        Inverse of the moment matrix, shape ``(n_points, dims, dims)``,
        if ``with_moment_matrix`` is True; otherwise None.
    """
    backend = mesh.backend
    points = mesh.points
    n_points, _ = points.shape
    adj = mesh.topology.point_adjacency(layout="coo")

    # Compute weights
    weights = backend.ones((n_points, 1), dimension={})

    if consider_volume:
        volumes = torch.abs(mesh.geometry.cell_volumes())
        weights = mesh.topology.map_cell_to_point(
            volumes, normal_interp_mode, "segment"
        )
    weights = weights.reshape((-1,))

    # Compute normals
    normals_on_surface_points = _compute_normals_on_surface_points(
        mesh, normal_interp_mode, eps
    )

    weighted_normals = (
        normal_weight * normals_on_surface_points
    )  # (n_points, dim)

    # Compute moment matrix
    if with_moment_matrix:
        n_otimes_n = (
            weighted_normals[:, :, None] * normals_on_surface_points[:, None, :]
        )  # (n_points, dim, dim)

        moment_matrix = (
            _compute_moment_matrix(backend, adj, points, weights, eps)
            + n_otimes_n
        )
    else:
        moment_matrix = None

    # Compute D_{k,ij}
    rawAM, moment_inv = _compute_rawAM_and_moment_inv(
        backend, adj, points, weights, moment_matrix, eps
    )

    # Compute \tilde{D}_{k,ij}
    NisoAM = _create_grad_operator_from(backend, rawAM)
    return NisoAM, weighted_normals, moment_inv


# =========================================================================
# Helper functions
# =========================================================================
def _compute_moment_matrix[T: TensorLike](
    backend: Backend[T],
    adj: T,
    points: T,
    weights: T,
    eps: float = 1e-12,
) -> T:
    r"""Compute the moment matrix M_i at each node i.

    M_i is the sum of (V_j/V_i) * u_ij \otimes u_ij over neighbors j,
    where u_ij is the unit vector from point i to j.

    Parameters
    ----------
    backend : Backend[T]
        Backend for tensor operations.
    adj : T
        Point adjacency matrix of shape ``(n_points, n_points)`` (sparse).
    points : T
        Node coordinates, shape ``(n_points, dim)``.
    weights : T
        Per-point weights, shape ``(n_points,)``.
    eps : float, optional
        Epsilon to avoid zero division. Default is 1e-12.

    Returns
    -------
    T
        Dense moment matrix of shape ``(n_points, dim, dim)``.
    """
    n_points, dim = points.shape
    i_indices, j_indices = adj.indices()  # (2, nnz)

    # Compute differences: x_j - x_i
    diff = points[j_indices] - points[i_indices]  # (nnz, dim)

    # Compute distance (clamped to avoid zero division): ||x_j - x_i||
    distance = torch.clamp(
        torch.linalg.vector_norm(diff, dim=1, keepdim=True), min=eps
    )  # (nnz, 1)

    # Compute unit vectors: (x_j - x_i) / ||x_j - x_i||
    u = diff / distance  # (nnz, dim)

    # Compute weights: w_ij = V_j / V_i
    w = weights[j_indices] / torch.clamp(weights[i_indices], min=eps)  # (nnz,)

    # Compute tensor products
    outer = u[:, :, None] * u[:, None, :]  # (nnz, dim, dim)
    contrib = w[:, None, None] * outer  # (nnz, dim, dim)

    # Initialize moment matrix as (n_points, dim, dim) and sum by row
    moment_matrix = backend.zeros(
        (n_points, dim, dim),
        dimension=get_dimension(contrib),
    )
    return moment_matrix.index_add_(0, i_indices, contrib)


def _compute_normals_on_surface_points[T: TensorLike](
    mesh: TensorMesh[T],
    normal_interp_mode: Literal["mean", "conservative"] = "conservative",
    eps: float = 1e-12,
) -> T:
    """
    Compute normals tensor with values only on the surface points.

    Parameters
    ----------
    mesh : TensorMesh[T]
        The mesh to compute the normals for.
    normal_interp_mode : {"mean", "conservative"}, optional
        How to map face normals to points (cf. map_cell_to_point).
        "mean": n_p = (1 / n_cells) * sum_{c \\ni p} n_c;
        "conservative": n_p = sum_{c \\ni p} n_c / n_cells.
        Default is "conservative".
    eps : float, optional
        Epsilon for normalization. Default is 1e-12.

    Returns
    -------
    T
        Nodal normals of shape ``(n_points, dim)`` (non-surface points are 0).
    """
    backend = mesh.backend
    surf = mesh.extract_surface()
    normals_on_faces = surf.geometry.face_normals(eps=eps)
    normals_on_points = surf.topology.map_cell_to_point(
        normals_on_faces, normal_interp_mode, "segment"
    )

    # normalize mapped normals on point
    norm = torch.linalg.vector_norm(normals_on_points, dim=1)
    non_zero_mask = backend.to_torch(norm) > eps
    normals_on_points[non_zero_mask] = (
        normals_on_points[non_zero_mask] / norm[non_zero_mask, None]
    )
    normals_on_points[~non_zero_mask] = 0.0
    return surf.scatter_add_to_parent_point_data(normals_on_points)


def _create_grad_operator_from[T: TensorLike](
    backend: Backend[T],
    D: T,
) -> T:
    """Build the row-sum-zero gradient operator from the raw AM.

    Parameters
    ----------
    backend : Backend[T]
        Backend for tensor operations.
    D : T
        Raw AM tensor of shape ``(dims, n_points, n_points)`` (sparse COO).

    Returns
    -------
    T
        Gradient operator of shape ``(dims, n_points, n_points)`` (sparse COO).
    """
    dimension = get_dimension(D)
    device = backend.device
    dtype = backend.dtype
    D = backend.to_torch(D)

    k_idx, i_idx, _ = D.indices()
    values = D.values()
    K, N, _ = D.shape

    # sum over l: sumD[k, i] = Σ_j D[k,i,j]
    sumD = torch.zeros((K, N), dtype=dtype, device=device)
    for kk in range(K):
        mask = k_idx == kk
        sumD[kk].index_add_(0, i_idx[mask], values[mask])

    # add diagonal entries: D_tilde = D - diag(sumD)
    diag_k = torch.arange(K, device=device).repeat_interleave(N)
    diag_i = torch.arange(N, device=device).repeat(K)
    diag_j = diag_i
    diag_indices = torch.stack([diag_k, diag_i, diag_j], dim=0)
    diag_vals = -sumD.reshape(-1)
    new_idx = torch.cat([D.indices(), diag_indices], dim=1)
    new_vals = torch.cat([values, diag_vals], dim=0)
    return backend.as_tensor(
        torch.sparse_coo_tensor(
            new_idx, new_vals, size=(K, N, N), dtype=dtype, device=device
        ).coalesce(),
        dimension=dimension,
    )


def _compute_rawAM_and_moment_inv[T: TensorLike](
    backend: Backend[T],
    adj: T,
    x: T,
    V: T,
    moment_matrix: T | None = None,
    eps: float = 1e-12,
) -> tuple[T, T | None]:
    """Compute raw AM D_{k,ij} and optionally M_i^{-1}.

    Formula:
        D_{k,ij} = M_i^{-1} (x_jk-x_ik)
                    * (V_j/(||x_j-x_i||^2 V_i)) * A_ij
    If moment_matrix is None, M_i^{-1} is treated as identity.

    Parameters
    ----------
    backend : Backend[T]
        Backend for tensor operations.
    adj : T
        Point adjacency matrix of shape ``(n_points, n_points)`` (sparse).
    x : T
        Node coordinates of shape ``(n_points, dims)``.
    V : T
        Nodal weights (e.g. ones or volumes when consider_volume=True),
        shape ``(n_points,)``.
    moment_matrix : T or None, optional
        Per-node moment matrix of shape ``(n_points, dims, dims)``.
        If None, raw AM is computed without it.
    eps : float, optional
        Epsilon to avoid zero division. Default is 1e-12.

    Returns
    -------
    raw_am : T
        Raw AM of shape ``(dims, n_points, n_points)`` (sparse COO).
    moment_inv : T or None
        Inverse moment matrix of shape ``(n_points, dims, dims)``
        if moment_matrix was given
        otherwise None.
    """
    device = backend.device
    dtype = backend.dtype
    N, K = x.shape
    row, col = adj.indices()
    nnz = row.shape[0]

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
        dimension = get_dimension(moment_matrix)
        if dimension is not None:
            dimension = dimension * -1
        Minv = backend.zeros(
            (N, K, K),
            dimension=dimension,
        )
        # solve M_i^{-1} @ (x_jk - x_ik)
        try:
            # Cholesky works only for SPD
            L = torch.linalg.cholesky(M)  # (nnz,k,k)
            sol = torch.cholesky_solve(rhs, L)  # (nnz,k,1)
            moment_inv = torch.cholesky_inverse(L)  # (nnz,k,k)
        except torch.linalg.LinAlgError:
            logger.warning(
                "Cholesky failed for moment matrix."
                "Falling back to dense inverse."
            )
            moment_inv = torch.linalg.inv(M)  # (nnz,k,k)
            sol = moment_inv @ rhs  # (nnz,k,1)

        # ---- pack (nnz,k,k) -> 3D sparse COO (n,k,k) ----
        Minv[row, :, :] = moment_inv

    D_vals = sol[:, :, 0] * scalar[:, None]  # (nnz,k)

    # ---- pack (nnz,k) -> 3D sparse COO (k,n,n) ----
    k_idx = torch.arange(K, device=device).repeat(nnz)  # (nnz*k,)
    i_idx = row.repeat_interleave(K)  # (nnz*k,)
    j_idx = col.repeat_interleave(K)  # (nnz*k,)
    values = D_vals.reshape((-1,))  # row-major: (nnz*k,)

    D = backend.as_tensor(
        torch.sparse_coo_tensor(
            torch.stack([k_idx, i_idx, j_idx], dim=0),
            backend.to_torch(values),
            size=(K, N, N),
            dtype=dtype,
            device=device,
        ).coalesce(),
        dimension=get_dimension(D_vals),
    )

    return D, Minv

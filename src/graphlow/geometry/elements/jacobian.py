"""Isoparametric Jacobian and physical-gradient operations."""

from __future__ import annotations

import torch

from graphlow.core.backend.base import Backend, TensorLike


def compute_jacobian[T: TensorLike](
    x_nodes: T, dN_dxi: T, backend: Backend[T]
) -> T:
    """Compute batched isoparametric Jacobians.

    Parameters
    ----------
    x_nodes : T
        Physical cell nodes with shape ``(n_cells, n_nodes, embedding_dim)``.
    dN_dxi : T
        Reference gradients with shape
        ``(n_points, n_nodes, reference_dimension)``.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Jacobians with shape
        ``(n_cells, n_points, embedding_dim, reference_dimension)``.
        The physical dimension is inferred from ``x_nodes`` and ``dN_dxi``.
    """
    if x_nodes.ndim != 3:
        raise ValueError(
            f"x_nodes must have rank 3, got {tuple(x_nodes.shape)}"
        )
    if dN_dxi.ndim == 2:
        dN_dxi = dN_dxi[None, ...]
    if dN_dxi.ndim != 3:
        raise ValueError(f"dN_dxi must have rank 3, got {tuple(dN_dxi.shape)}")
    if x_nodes.shape[1] != dN_dxi.shape[1]:
        raise ValueError("x_nodes and dN_dxi must have the same node count")
    # J = sum_a x_a ⊗ ∇_ξ N_a. Dimension follows the operands.
    return backend.einsum("cne,qnr->cqer", x_nodes, dN_dxi, dimension="auto")


def jacobian_measure[T: TensorLike](
    jacobian: T,
) -> tuple[T | None, T]:
    """Compute signed determinants and non-negative integration measures.

    Square Jacobians return their signed determinant and its absolute value.
    Embedded surface Jacobians return ``None`` for the signed determinant and
    ``sqrt(det(J.T @ J))`` as their measure.

    Parameters
    ----------
    jacobian : T
        Jacobians with shape
        ``(..., embedding_dimension, reference_dimension)``.

    Returns
    -------
    det_j : T or None
        Signed determinants for square Jacobians, or ``None`` for embedded
        surfaces.
    measure : T
        Non-negative integration measures with shape ``jacobian.shape[:-2]``.

    Raises
    ------
    NotImplementedError
        If the Jacobian dimensions are unsupported.
    """
    embedding_dimension = jacobian.shape[-2]
    reference_dimension = jacobian.shape[-1]
    if embedding_dimension == reference_dimension:
        if reference_dimension == 2:
            det_j = (
                jacobian[..., 0, 0] * jacobian[..., 1, 1]
                - jacobian[..., 0, 1] * jacobian[..., 1, 0]
            )
        elif reference_dimension == 3:
            det_j = (
                jacobian[..., 0, 0]
                * (
                    jacobian[..., 1, 1] * jacobian[..., 2, 2]
                    - jacobian[..., 1, 2] * jacobian[..., 2, 1]
                )
                - jacobian[..., 0, 1]
                * (
                    jacobian[..., 1, 0] * jacobian[..., 2, 2]
                    - jacobian[..., 1, 2] * jacobian[..., 2, 0]
                )
                + jacobian[..., 0, 2]
                * (
                    jacobian[..., 1, 0] * jacobian[..., 2, 1]
                    - jacobian[..., 1, 1] * jacobian[..., 2, 0]
                )
            )
        else:
            raise NotImplementedError(
                f"Unsupported Jacobian dimension: {reference_dimension}"
            )
        return det_j, torch.abs(det_j)
    if reference_dimension != 2 or embedding_dimension != 3:
        raise NotImplementedError(
            "Embedded Jacobians currently require shape (..., 3, 2)"
        )
    first = jacobian[..., :, 0]
    second = jacobian[..., :, 1]
    cross = torch.linalg.cross(first, second, dim=-1)
    measure = torch.linalg.vector_norm(cross, dim=-1)
    return None, measure


def physical_shape_function_gradients[T: TensorLike](
    jacobian: T, dN_dxi: T, backend: Backend[T]
) -> T:
    """Transform reference gradients to physical-space gradients.

    A Moore-Penrose inverse is used so the operation also covers embedded
    TRI3/QUAD4 surfaces and retains the existing degenerate-cell convention.

    Parameters
    ----------
    jacobian : T
        Jacobians with shape
        ``(n_cells, n_points, embedding_dim, reference_dimension)``.
    dN_dxi : T
        Reference gradients with shape
        ``(n_points, n_nodes, reference_dimension)``.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Physical gradients with shape
        ``(n_cells, n_points, n_nodes, embedding_dim)``.
        The physical dimension is inferred from ``jacobian`` and ``dN_dxi``.
    """
    # ∇_x N = ∇_ξ N @ J^{+}. Dimension follows the operands.
    inverse = torch.linalg.pinv(jacobian)
    return backend.einsum("qnr,cqre->cqne", dN_dxi, inverse, dimension="auto")

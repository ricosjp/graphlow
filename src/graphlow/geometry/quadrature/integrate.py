"""Batched integration on isoparametric elements."""

from __future__ import annotations

import pyvista as pv

from graphlow.core.backend.base import Backend, TensorLike
from graphlow.geometry.elements.jacobian import (
    compute_jacobian,
    jacobian_measure,
)
from graphlow.geometry.elements.shape_functions import (
    get_shape_function_gradients,
)
from graphlow.geometry.quadrature.rules import get_quadrature_rule


def integrate_element[T: TensorLike](
    x_nodes: T,
    cell_type: pv.CellType,
    integrand: T | None = None,
    required_order: int | None = None,
    *,
    backend: Backend[T],
) -> T:
    """Integrate sampled values over a batch of physical elements.

    Parameters
    ----------
    x_nodes : T
        Physical cell nodes with shape
        ``(n_cells, n_nodes, embedding_dim)``.
    cell_type : pyvista.CellType
        VTK cell type of the elements.
    integrand : T or None, optional
        Sampled values with shape ``(n_cells, n_quad, ...)``.
    required_order : int or None, optional
        Minimum polynomial order the rule must integrate exactly.
        ``None`` selects 2 for linear elements and 4 for quadratic
        elements. The returned rule may have a higher order.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Integrated values with shape ``(n_cells, ...)``.
    """
    rule = get_quadrature_rule(
        cell_type,
        required_order,
        backend=backend,
    )
    dN_dxi = get_shape_function_gradients(cell_type)(rule.xi)
    jacobian = compute_jacobian(x_nodes, dN_dxi, backend)
    _, abs_det_j = jacobian_measure(jacobian)

    # integrand's shape should be (n_cells, n_quad, ...)
    if integrand is None:
        integrand = backend.ones(abs_det_j.shape, dimension={})
    if integrand.ndim < 2:
        raise ValueError("integrand must have cell and quadrature dimensions")
    if tuple(integrand.shape[:2]) != tuple(abs_det_j.shape):
        raise ValueError(
            "integrand and measure must share cell/quadrature dimensions"
        )
    # A trailing ellipsis is rejected when the integrand has no extra axes.
    if integrand.ndim == 2:
        equation = "cq,q,cq->c"
    else:
        equation = "cq...,q,cq->c..."
    return backend.einsum(equation, integrand, rule.weights, abs_det_j)

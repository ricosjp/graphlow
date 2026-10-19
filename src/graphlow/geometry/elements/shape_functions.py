"""Analytic shape functions for supported reference elements."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pyvista as pv
import torch

from graphlow.core.backend.base import TensorLike
from graphlow.core.element_definitions import get_element

ShapeEvaluator = Callable[[TensorLike], TensorLike]


def validate_reference_coords(
    xi: TensorLike,
    cell_type: pv.CellType,
) -> None:
    """Validate that ``xi`` matches the element reference dimension.

    Parameters
    ----------
    xi : TensorLike
        Reference coordinates with trailing dimension equal to the
        element's reference dimension.
    cell_type : pyvista.CellType
        VTK cell type used to look up the expected dimension.

    Raises
    ------
    ValueError
        If ``xi`` is scalar or its last dimension is wrong.
    """
    reference_dimension = get_element(cell_type).reference_dimension
    if xi.ndim == 0 or xi.shape[-1] != reference_dimension:
        raise ValueError(
            "The last xi dimension must be "
            f"{reference_dimension}, got {tuple(xi.shape)}"
        )


def get_shape_functions(cell_type: pv.CellType) -> ShapeEvaluator:
    """Return the shape-function evaluator for a cell type.

    The returned callable does not validate ``xi``. Callers that accept
    external coordinates should use :func:`validate_reference_coords`.

    Parameters
    ----------
    cell_type : pyvista.CellType
        VTK cell type.

    Returns
    -------
    ShapeEvaluator
        Callable ``xi -> N`` with
        ``xi`` shape ``(..., reference_dimension)`` and
        ``N`` shape ``(..., n_nodes)``.

    Raises
    ------
    NotImplementedError
        If shape functions are not registered for ``cell_type``.
    """
    evaluators = _SHAPE_EVALUATORS.get(cell_type)
    if evaluators is None:
        raise NotImplementedError(
            f"Shape functions are not registered for {cell_type}"
        )
    shape_fn, _ = evaluators
    return shape_fn


def get_shape_function_gradients(cell_type: pv.CellType) -> ShapeEvaluator:
    """Return the shape-function gradient evaluator for a cell type.

    The returned callable does not validate ``xi``. Callers that accept
    external coordinates should use :func:`validate_reference_coords`.

    Parameters
    ----------
    cell_type : pyvista.CellType
        VTK cell type.

    Returns
    -------
    ShapeEvaluator
        Callable ``xi -> dN_dxi`` with
        ``xi`` shape ``(..., reference_dimension)`` and
        ``dN_dxi`` shape ``(..., n_nodes, reference_dimension)``.

    Raises
    ------
    NotImplementedError
        If shape-function gradients are not registered for ``cell_type``.
    """
    evaluators = _SHAPE_EVALUATORS.get(cell_type)
    if evaluators is None:
        raise NotImplementedError(
            f"Shape-function gradients are not registered for {cell_type}"
        )
    _, gradient_fn = evaluators
    return gradient_fn


# =============================================================================
# Linear elements
# =============================================================================


def _triangle_shape[T: TensorLike](xi: T) -> T:
    r, s = xi[..., 0], xi[..., 1]
    return torch.stack((1 - r - s, r, s), dim=-1)


def _triangle_gradient[T: TensorLike](xi: T) -> T:
    zero = xi[..., 0] * 0
    one = zero + 1
    return torch.stack(
        (
            torch.stack((-one, -one), dim=-1),
            torch.stack((one, zero), dim=-1),
            torch.stack((zero, one), dim=-1),
        ),
        dim=-2,
    )


def _quad_shape[T: TensorLike](xi: T) -> T:
    r, s = xi[..., 0], xi[..., 1]
    return 0.25 * torch.stack(
        (
            (1 - r) * (1 - s),
            (1 + r) * (1 - s),
            (1 + r) * (1 + s),
            (1 - r) * (1 + s),
        ),
        dim=-1,
    )


def _quad_gradient[T: TensorLike](xi: T) -> T:
    r, s = xi[..., 0], xi[..., 1]
    return 0.25 * torch.stack(
        (
            torch.stack((-(1 - s), -(1 - r)), dim=-1),
            torch.stack((1 - s, -(1 + r)), dim=-1),
            torch.stack((1 + s, 1 + r), dim=-1),
            torch.stack((-(1 + s), 1 - r), dim=-1),
        ),
        dim=-2,
    )


def _tetra_shape[T: TensorLike](xi: T) -> T:
    r, s, t = xi[..., 0], xi[..., 1], xi[..., 2]
    return torch.stack((1 - r - s - t, r, s, t), dim=-1)


def _tetra_gradient[T: TensorLike](xi: T) -> T:
    zero = xi[..., 0] * 0
    one = zero + 1
    return torch.stack(
        (
            torch.stack((-one, -one, -one), dim=-1),
            torch.stack((one, zero, zero), dim=-1),
            torch.stack((zero, one, zero), dim=-1),
            torch.stack((zero, zero, one), dim=-1),
        ),
        dim=-2,
    )


def _hexahedron_shape[T: TensorLike](xi: T) -> T:
    r, s, t = xi[..., 0], xi[..., 1], xi[..., 2]
    corners = get_element(pv.CellType.HEXAHEDRON).reference_coords
    return torch.stack(
        tuple(
            0.125 * (1 + r * r_sign) * (1 + s * s_sign) * (1 + t * t_sign)
            for r_sign, s_sign, t_sign in corners
        ),
        dim=-1,
    )


def _hexahedron_gradient[T: TensorLike](xi: T) -> T:
    r, s, t = xi[..., 0], xi[..., 1], xi[..., 2]
    corners = get_element(pv.CellType.HEXAHEDRON).reference_coords
    gradients = tuple(
        torch.stack(
            (
                0.125 * r_sign * (1 + s * s_sign) * (1 + t * t_sign),
                0.125 * s_sign * (1 + r * r_sign) * (1 + t * t_sign),
                0.125 * t_sign * (1 + r * r_sign) * (1 + s * s_sign),
            ),
            dim=-1,
        )
        for r_sign, s_sign, t_sign in corners
    )
    return torch.stack(gradients, dim=-2)


# =============================================================================
# Quadratic elements
# =============================================================================


def _quadratic_triangle_shape[T: TensorLike](xi: T) -> T:
    r, s = xi[..., 0], xi[..., 1]
    l0, l1, l2 = 1 - r - s, r, s
    return torch.stack(
        (
            l0 * (2 * l0 - 1),
            l1 * (2 * l1 - 1),
            l2 * (2 * l2 - 1),
            4 * l0 * l1,
            4 * l1 * l2,
            4 * l2 * l0,
        ),
        dim=-1,
    )


def _quadratic_triangle_gradient[T: TensorLike](xi: T) -> T:
    r, s = xi[..., 0], xi[..., 1]
    l0, l1, l2 = 1 - r - s, r, s
    zero = r * 0
    one = zero + 1
    dl0 = torch.stack((-one, -one), dim=-1)
    dl1 = torch.stack((one, zero), dim=-1)
    dl2 = torch.stack((zero, one), dim=-1)
    gradients = (
        (4 * l0 - 1)[..., None] * dl0,
        (4 * l1 - 1)[..., None] * dl1,
        (4 * l2 - 1)[..., None] * dl2,
        4 * (l0[..., None] * dl1 + l1[..., None] * dl0),
        4 * (l1[..., None] * dl2 + l2[..., None] * dl1),
        4 * (l2[..., None] * dl0 + l0[..., None] * dl2),
    )
    return torch.stack(gradients, dim=-2)


def _quadratic_quad_shape[T: TensorLike](xi: T) -> T:
    r, s = xi[..., 0], xi[..., 1]
    corners = get_element(pv.CellType.QUAD).reference_coords
    values = [
        0.25
        * (1 + r * r_sign)
        * (1 + s * s_sign)
        * (r * r_sign + s * s_sign - 1)
        for r_sign, s_sign in corners
    ]
    values.extend(
        (
            0.5 * (1 - r * r) * (1 - s),
            0.5 * (1 + r) * (1 - s * s),
            0.5 * (1 - r * r) * (1 + s),
            0.5 * (1 - r) * (1 - s * s),
        )
    )
    return torch.stack(tuple(values), dim=-1)


def _quadratic_quad_gradient[T: TensorLike](xi: T) -> T:
    r, s = xi[..., 0], xi[..., 1]
    corners = get_element(pv.CellType.QUAD).reference_coords
    gradients = []
    for r_sign, s_sign in corners:
        a = 1 + r * r_sign
        b = 1 + s * s_sign
        d = r * r_sign + s * s_sign - 1
        gradients.append(
            torch.stack(
                (
                    0.25 * r_sign * b * (d + a),
                    0.25 * s_sign * a * (d + b),
                ),
                dim=-1,
            )
        )
    gradients.extend(
        (
            torch.stack((-r * (1 - s), -0.5 * (1 - r * r)), dim=-1),
            torch.stack((0.5 * (1 - s * s), -(1 + r) * s), dim=-1),
            torch.stack((-r * (1 + s), 0.5 * (1 - r * r)), dim=-1),
            torch.stack((-0.5 * (1 - s * s), -(1 - r) * s), dim=-1),
        )
    )
    return torch.stack(tuple(gradients), dim=-2)


def _quadratic_tetra_shape[T: TensorLike](xi: T) -> T:
    r, s, t = xi[..., 0], xi[..., 1], xi[..., 2]
    l0, l1, l2, l3 = 1 - r - s - t, r, s, t
    return torch.stack(
        (
            l0 * (2 * l0 - 1),
            l1 * (2 * l1 - 1),
            l2 * (2 * l2 - 1),
            l3 * (2 * l3 - 1),
            4 * l0 * l1,
            4 * l1 * l2,
            4 * l2 * l0,
            4 * l0 * l3,
            4 * l1 * l3,
            4 * l2 * l3,
        ),
        dim=-1,
    )


def _quadratic_tetra_gradient[T: TensorLike](xi: T) -> T:
    r, s, t = xi[..., 0], xi[..., 1], xi[..., 2]
    l0, l1, l2, l3 = 1 - r - s - t, r, s, t
    zero = r * 0
    one = zero + 1
    dl = (
        torch.stack((-one, -one, -one), dim=-1),
        torch.stack((one, zero, zero), dim=-1),
        torch.stack((zero, one, zero), dim=-1),
        torch.stack((zero, zero, one), dim=-1),
    )
    lambdas = (l0, l1, l2, l3)
    gradients = [
        (4 * value - 1)[..., None] * derivative
        for value, derivative in zip(lambdas, dl, strict=True)
    ]
    for first, second in (
        (0, 1),
        (1, 2),
        (2, 0),
        (0, 3),
        (1, 3),
        (2, 3),
    ):
        gradients.append(
            4
            * (
                lambdas[first][..., None] * dl[second]
                + lambdas[second][..., None] * dl[first]
            )
        )
    return torch.stack(tuple(gradients), dim=-2)


def _quadratic_hexahedron_shape[T: TensorLike](xi: T) -> T:
    r, s, t = xi[..., 0], xi[..., 1], xi[..., 2]
    corners = get_element(pv.CellType.HEXAHEDRON).reference_coords
    values = [
        0.125
        * (1 + r * r_sign)
        * (1 + s * s_sign)
        * (1 + t * t_sign)
        * (r * r_sign + s * s_sign + t * t_sign - 2)
        for r_sign, s_sign, t_sign in corners
    ]
    r_edges = (
        (-1.0, -1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
        (1.0, 1.0),
    )
    edge_values = [
        0.25 * (1 - r * r) * (1 + s * ss) * (1 + t * tt) for ss, tt in r_edges
    ]
    # VTK interleaves r- and s-directed edges on each z plane.
    values.extend(
        (
            edge_values[0],
            0.25 * (1 + r) * (1 - s * s) * (1 - t),
            edge_values[1],
            0.25 * (1 - r) * (1 - s * s) * (1 - t),
            edge_values[2],
            0.25 * (1 + r) * (1 - s * s) * (1 + t),
            edge_values[3],
            0.25 * (1 - r) * (1 - s * s) * (1 + t),
        )
    )
    values.extend(
        0.25 * (1 + r * rr) * (1 + s * ss) * (1 - t * t)
        for rr, ss in (
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
        )
    )
    return torch.stack(tuple(values), dim=-1)


def _quadratic_hexahedron_gradient[T: TensorLike](xi: T) -> T:
    r, s, t = xi[..., 0], xi[..., 1], xi[..., 2]
    corners = get_element(pv.CellType.HEXAHEDRON).reference_coords
    gradients = []
    for r_sign, s_sign, t_sign in corners:
        a = 1 + r * r_sign
        b = 1 + s * s_sign
        c = 1 + t * t_sign
        d = r * r_sign + s * s_sign + t * t_sign - 2
        gradients.append(
            torch.stack(
                (
                    0.125 * r_sign * b * c * (d + a),
                    0.125 * s_sign * a * c * (d + b),
                    0.125 * t_sign * a * b * (d + c),
                ),
                dim=-1,
            )
        )

    def r_edge(ss: float, tt: float) -> T:
        return torch.stack(
            (
                -0.5 * r * (1 + s * ss) * (1 + t * tt),
                0.25 * (1 - r * r) * ss * (1 + t * tt),
                0.25 * (1 - r * r) * (1 + s * ss) * tt,
            ),
            dim=-1,
        )

    def s_edge(rr: float, tt: float) -> T:
        return torch.stack(
            (
                0.25 * rr * (1 - s * s) * (1 + t * tt),
                -0.5 * s * (1 + r * rr) * (1 + t * tt),
                0.25 * (1 + r * rr) * (1 - s * s) * tt,
            ),
            dim=-1,
        )

    gradients.extend(
        (
            r_edge(-1.0, -1.0),
            s_edge(1.0, -1.0),
            r_edge(1.0, -1.0),
            s_edge(-1.0, -1.0),
            r_edge(-1.0, 1.0),
            s_edge(1.0, 1.0),
            r_edge(1.0, 1.0),
            s_edge(-1.0, 1.0),
        )
    )
    for rr, ss in (
        (-1.0, -1.0),
        (1.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
    ):
        gradients.append(
            torch.stack(
                (
                    0.25 * rr * (1 + s * ss) * (1 - t * t),
                    0.25 * (1 + r * rr) * ss * (1 - t * t),
                    -0.5 * t * (1 + r * rr) * (1 + s * ss),
                ),
                dim=-1,
            )
        )
    return torch.stack(tuple(gradients), dim=-2)


_SHAPE_EVALUATORS: Mapping[
    pv.CellType, tuple[ShapeEvaluator, ShapeEvaluator]
] = {
    pv.CellType.TRIANGLE: (_triangle_shape, _triangle_gradient),
    pv.CellType.QUAD: (_quad_shape, _quad_gradient),
    pv.CellType.TETRA: (_tetra_shape, _tetra_gradient),
    pv.CellType.HEXAHEDRON: (_hexahedron_shape, _hexahedron_gradient),
    pv.CellType.QUADRATIC_TRIANGLE: (
        _quadratic_triangle_shape,
        _quadratic_triangle_gradient,
    ),
    pv.CellType.QUADRATIC_QUAD: (
        _quadratic_quad_shape,
        _quadratic_quad_gradient,
    ),
    pv.CellType.QUADRATIC_TETRA: (
        _quadratic_tetra_shape,
        _quadratic_tetra_gradient,
    ),
    pv.CellType.QUADRATIC_HEXAHEDRON: (
        _quadratic_hexahedron_shape,
        _quadratic_hexahedron_gradient,
    ),
}

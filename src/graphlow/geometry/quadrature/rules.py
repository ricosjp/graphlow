"""Standard quadrature rules for supported elements."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Literal

import numpy as np
import pyvista as pv
import vtk

from graphlow.core.backend.base import Backend, TensorLike
from graphlow.core.element_definitions import get_element
from graphlow.utils.topology_helper import TopologyDim


@dataclass(frozen=True)
class _QuadratureRule[T: TensorLike]:
    """Reference quadrature points and weights.

    Parameters
    ----------
    cell_type : pyvista.CellType
        VTK cell type associated with the rule.
    order : int
        Polynomial order this rule integrates exactly.
    xi : T
        Reference coordinates with shape
        ``(n_points, reference_dimension)``.
    weights : T
        Reference weights with shape ``(n_points,)``.
    """

    cell_type: pv.CellType
    order: int
    xi: T
    weights: T


def _triangle_rule(
    required_order: int,
) -> tuple[int, list[list[float]], list[float]]:
    """Return ``order`` and reference data for a triangle rule.

    Orders 3 and 4 share the order-four formula.
    """
    match required_order:
        case 1:
            return 1, [[1 / 3, 1 / 3]], [1 / 2]
        case 2:
            # Symmetric order-two rule on the reference triangle of area 1/2.
            # https://github.com/mfem/mfem/blob/v4.10/fem/intrules.cpp#L1466
            return (
                2,
                [[1 / 6, 1 / 6], [1 / 6, 2 / 3], [2 / 3, 1 / 6]],
                [1 / 6, 1 / 6, 1 / 6],
            )
        case 3 | 4:
            # Six-point order-four rule on the reference triangle.
            # https://github.com/mfem/mfem/blob/v4.10/fem/intrules.cpp#L1473
            a = 4.45948490915964890213e-01
            b = 1 - 2 * a
            c = 9.15762135097707430376e-02
            d = 1 - 2 * c
            return (
                4,
                [[a, a], [a, b], [b, a], [c, c], [c, d], [d, c]],
                [1.11690794839005735906e-01] * 3
                + [5.49758718276609353870e-02] * 3,
            )
        case _:
            raise NotImplementedError(
                f"Unsupported quadrature order: {required_order}"
            )


def _tetrahedron_rule(
    required_order: int,
) -> tuple[int, list[list[float]], list[float]]:
    """Return ``order`` and reference data for a tetrahedron rule.

    Order 3 has its own eight-point formula. Orders 4 and 5 share the
    order-five formula.
    """
    match required_order:
        case 1:
            return 1, [[1 / 4, 1 / 4, 1 / 4]], [1 / 6]
        case 2:
            # Symmetric order-two tetrahedron rule. The reference passes b.
            # https://github.com/mfem/mfem/blob/v4.10/fem/intrules.cpp#L1894
            b = 1.38196601125010531952e-01
            a = 1 - 3 * b
            return (
                2,
                [[b, b, b], [b, b, a], [b, a, b], [a, b, b]],
                [1 / 24] * 4,
            )
        case 3:
            # Eight-point order-three rule.
            # https://github.com/mfem/mfem/blob/v4.10/fem/intrules.cpp#L1901
            a = 3.28163302516381705232e-01
            b = 1 - 3 * a
            c = 1.08047249898428621151e-01
            d = 1 - 3 * c
            return (
                3,
                [
                    [a, a, a],
                    [a, a, b],
                    [a, b, a],
                    [b, a, a],
                    [c, c, c],
                    [c, c, d],
                    [c, d, c],
                    [d, c, c],
                ],
                [2.27029737561812265667e-02] * 4
                + [1.89636929104854412564e-02] * 4,
            )
        case 4 | 5:
            # Fourteen-point order-five rule. All weights are positive.
            # https://github.com/mfem/mfem/blob/v4.10/fem/intrules.cpp#L1909
            a = 3.10885919263300669613e-01
            b = 1 - 3 * a
            c = 9.27352503108912484819e-02
            d = 1 - 3 * c
            e = 4.54496295874350364485e-01
            f = 0.5 - e
            return (
                5,
                [
                    [a, a, a],
                    [a, a, b],
                    [a, b, a],
                    [b, a, a],
                    [c, c, c],
                    [c, c, d],
                    [c, d, c],
                    [d, c, c],
                    [e, e, f],
                    [e, f, e],
                    [f, e, e],
                    [f, f, e],
                    [f, e, f],
                    [e, f, f],
                ],
                [1.87813209530026427319e-02] * 4
                + [1.22488405193936587129e-02] * 4
                + [7.09100346284691120807e-03] * 6,
            )
        case _:
            raise NotImplementedError(
                f"Unsupported quadrature order: {required_order}"
            )


def _tensor_product_rule(
    dimension: TopologyDim, degree: int
) -> tuple[list[list[float]], list[float]]:
    """Return a Gauss-Legendre tensor-product rule of the given degree."""
    coordinates, one_dimensional_weights = np.polynomial.legendre.leggauss(
        degree
    )
    xi = [list(point) for point in product(coordinates, repeat=dimension)]
    weights = [
        prod(weight_tuple)
        for weight_tuple in product(one_dimensional_weights, repeat=dimension)
    ]
    return xi, weights


def _gauss_legendre_degree(required_order: int) -> int:
    """Return the smallest Gauss-Legendre degree for ``required_order``."""
    if required_order < 1:
        raise NotImplementedError(
            f"Unsupported quadrature order: {required_order}"
        )
    return (required_order + 2) // 2


def _recommended_quadrature_order(
    cell_type: pv.CellType,
) -> Literal[2, 4]:
    """Return the polynomial order a full rule must integrate exactly.

    Linear elements require order 2 and quadratic elements require
    order 4. :func:`get_quadrature_rule` may return a higher ``order``.

    Parameters
    ----------
    cell_type : pyvista.CellType
        VTK cell type.

    Returns
    -------
    int
        Polynomial order required for full integration.
    """
    if vtk.vtkCellTypeUtilities.IsLinear(int(cell_type)):
        return 2
    return 4


def get_quadrature_rule[T: TensorLike](
    cell_type: pv.CellType,
    required_order: int | None = None,
    *,
    backend: Backend[T],
) -> _QuadratureRule[T]:
    """Return a standard quadrature rule.

    ``required_order`` is the minimum polynomial order the rule must
    integrate exactly. ``None`` selects 2 for linear elements and 4 for
    quadratic elements. ``order`` may be higher when no formula has
    exactly the requested order. Triangles have formulas for orders
    1, 2, and 4, so a request of 3 returns 4. Tetrahedra have formulas
    for orders 1, 2, 3, and 5, so a request of 4 returns 5.
    Quadrilaterals and hexahedra use Gauss-Legendre rules. ``degree`` is
    the number of points in one direction, and the rule order is
    ``2 * degree - 1``, so a request of 2 returns 3 and a request of 4
    returns 5.

    Parameters
    ----------
    cell_type : pyvista.CellType
        VTK cell type to integrate.
    required_order : int or None, optional
        Minimum polynomial order the rule must integrate exactly.
        ``None`` selects the full rule for the element.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    QuadratureRule[T]
        Reference coordinates and weights.

    Raises
    ------
    NotImplementedError
        If the cell type has no rule, or no formula is exact for at
        least ``required_order``.
    """
    element = get_element(cell_type)
    reference_dimension = element.reference_dimension

    if required_order is None:
        required_order = _recommended_quadrature_order(cell_type)

    match cell_type:
        case pv.CellType.TRIANGLE | pv.CellType.QUADRATIC_TRIANGLE:
            order, xi, weights = _triangle_rule(required_order)
        case pv.CellType.TETRA | pv.CellType.QUADRATIC_TETRA:
            order, xi, weights = _tetrahedron_rule(required_order)
        case (
            pv.CellType.QUAD
            | pv.CellType.QUADRATIC_QUAD
            | pv.CellType.HEXAHEDRON
            | pv.CellType.QUADRATIC_HEXAHEDRON
        ):
            degree = _gauss_legendre_degree(required_order)
            order = 2 * degree - 1
            xi, weights = _tensor_product_rule(reference_dimension, degree)
        case _:
            raise NotImplementedError(
                f"No quadrature rule for VTK cell type: {cell_type}"
            )

    return _QuadratureRule(
        cell_type=cell_type,
        order=order,
        xi=backend.as_tensor(xi, dimension={}),
        weights=backend.as_tensor(weights, dimension={}),
    )

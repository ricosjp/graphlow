"""Unit tests for reference quadrature and isoparametric integration."""

from __future__ import annotations

import pytest
import pyvista as pv
import torch

from graphlow.core.backend.base import Backend
from graphlow.core.element_definitions import (
    get_element,
    reference_coords,
    supported_cell_types,
)
from graphlow.geometry.quadrature import (
    get_quadrature_rule,
    integrate_element,
)
from graphlow.geometry.quadrature.rules import _recommended_quadrature_order

_LINEAR_CELL_TYPES = (
    pv.CellType.TRIANGLE,
    pv.CellType.QUAD,
    pv.CellType.TETRA,
    pv.CellType.HEXAHEDRON,
)
_QUADRATIC_CELL_TYPES = (
    pv.CellType.QUADRATIC_TRIANGLE,
    pv.CellType.QUADRATIC_QUAD,
    pv.CellType.QUADRATIC_TETRA,
    pv.CellType.QUADRATIC_HEXAHEDRON,
)


def _reference_measure(cell_type: pv.CellType) -> float:
    """Return the reference-element area or volume."""
    match cell_type:
        case pv.CellType.TRIANGLE | pv.CellType.QUADRATIC_TRIANGLE:
            return 0.5
        case pv.CellType.QUAD | pv.CellType.QUADRATIC_QUAD:
            return 4.0  # 2^2
        case pv.CellType.TETRA | pv.CellType.QUADRATIC_TETRA:
            return 1.0 / 6.0
        case pv.CellType.HEXAHEDRON | pv.CellType.QUADRATIC_HEXAHEDRON:
            return 8.0  # 2^3
        case _:
            raise AssertionError(cell_type)


def _reference_coordinate_integral(cell_type: pv.CellType) -> float:
    """Return the integral of the first reference coordinate."""
    match cell_type:
        case pv.CellType.TRIANGLE | pv.CellType.QUADRATIC_TRIANGLE:
            return 1.0 / 6.0
        case pv.CellType.TETRA | pv.CellType.QUADRATIC_TETRA:
            return 1.0 / 24.0
        case (
            pv.CellType.QUAD
            | pv.CellType.QUADRATIC_QUAD
            | pv.CellType.HEXAHEDRON
            | pv.CellType.QUADRATIC_HEXAHEDRON
        ):
            return 0.0
        case _:
            raise AssertionError(cell_type)


class TestRecommendedOrder:
    """Default ``required_order`` by element linearity."""

    @pytest.mark.parametrize("cell_type", _LINEAR_CELL_TYPES)
    def test_linear_elements_recommend_order_two(self, cell_type: pv.CellType):
        """Linear elements request a rule exact for order 2."""
        assert _recommended_quadrature_order(cell_type) == 2

    @pytest.mark.parametrize("cell_type", _QUADRATIC_CELL_TYPES)
    def test_quadratic_elements_recommend_order_four(
        self, cell_type: pv.CellType
    ):
        """Quadratic elements request a rule exact for order 4."""
        assert _recommended_quadrature_order(cell_type) == 4


class TestGetQuadratureRule:
    """Selection and properties of reference quadrature rules."""

    @pytest.mark.parametrize(
        ("cell_type", "expected_order"),
        [
            (pv.CellType.TRIANGLE, 2),
            (pv.CellType.QUADRATIC_TRIANGLE, 4),
            (pv.CellType.TETRA, 2),
            (pv.CellType.QUADRATIC_TETRA, 5),
            (pv.CellType.QUAD, 3),
            (pv.CellType.QUADRATIC_QUAD, 5),
            (pv.CellType.HEXAHEDRON, 3),
            (pv.CellType.QUADRATIC_HEXAHEDRON, 5),
        ],
    )
    def test_default_rule_order(
        self,
        cell_type: pv.CellType,
        expected_order: int,
        backend: Backend,
    ):
        """Defaults cover the request; tensor products use Gauss order."""
        rule = get_quadrature_rule(cell_type, backend=backend)
        assert rule.cell_type == cell_type
        assert rule.order == expected_order

    @pytest.mark.parametrize("cell_type", supported_cell_types())
    def test_default_weights_sum_to_the_reference_measure(
        self, cell_type: pv.CellType, backend: Backend
    ):
        """Reference weights integrate the constant 1 exactly."""
        rule = get_quadrature_rule(cell_type, backend=backend)
        total = torch.sum(backend.to_torch(rule.weights))
        expected = torch.tensor(
            _reference_measure(cell_type),
            dtype=total.dtype,
            device=total.device,
        )
        torch.testing.assert_close(total, expected)

    def test_triangle_orders_three_and_four_share_a_formula(
        self, backend: Backend
    ):
        """Orders 3 and 4 select the same order-four triangle formula."""
        low = get_quadrature_rule(
            pv.CellType.TRIANGLE, required_order=3, backend=backend
        )
        high = get_quadrature_rule(
            pv.CellType.TRIANGLE, required_order=4, backend=backend
        )
        assert low.order == high.order == 4
        torch.testing.assert_close(
            backend.to_torch(low.xi), backend.to_torch(high.xi)
        )
        torch.testing.assert_close(
            backend.to_torch(low.weights), backend.to_torch(high.weights)
        )

    def test_tetrahedron_order_three_is_an_eight_point_rule(
        self, backend: Backend
    ):
        """Order 3 selects the eight-point tetrahedron formula."""
        rule = get_quadrature_rule(
            pv.CellType.TETRA, required_order=3, backend=backend
        )
        assert rule.order == 3
        assert tuple(rule.xi.shape) == (8, 3)

    def test_tetrahedron_orders_four_and_five_share_a_formula(
        self, backend: Backend
    ):
        """Orders 4 and 5 select the same order-five tetrahedron formula."""
        low = get_quadrature_rule(
            pv.CellType.TETRA, required_order=4, backend=backend
        )
        high = get_quadrature_rule(
            pv.CellType.TETRA, required_order=5, backend=backend
        )
        assert low.order == high.order == 5
        assert tuple(low.xi.shape) == (14, 3)
        torch.testing.assert_close(
            backend.to_torch(low.xi), backend.to_torch(high.xi)
        )
        torch.testing.assert_close(
            backend.to_torch(low.weights), backend.to_torch(high.weights)
        )

    def test_quad_orders_two_and_three_share_a_formula(self, backend: Backend):
        """Orders 2 and 3 select the same degree-two Gauss-Legendre rule."""
        low = get_quadrature_rule(
            pv.CellType.QUAD, required_order=2, backend=backend
        )
        high = get_quadrature_rule(
            pv.CellType.QUAD, required_order=3, backend=backend
        )
        assert low.order == high.order == 3
        assert tuple(low.xi.shape) == (4, 2)
        torch.testing.assert_close(
            backend.to_torch(low.xi), backend.to_torch(high.xi)
        )
        torch.testing.assert_close(
            backend.to_torch(low.weights), backend.to_torch(high.weights)
        )

    @pytest.mark.parametrize(
        ("cell_type", "required_order", "order", "n_points"),
        [
            (pv.CellType.QUAD, 1, 1, 1),
            (pv.CellType.QUAD, 4, 5, 9),
            (pv.CellType.QUAD, 6, 7, 16),
            (pv.CellType.HEXAHEDRON, 2, 3, 8),
            (pv.CellType.HEXAHEDRON, 4, 5, 27),
        ],
    )
    def test_gauss_legendre_uses_the_smallest_exact_rule(
        self,
        cell_type: pv.CellType,
        required_order: int,
        order: int,
        n_points: int,
        backend: Backend,
    ):
        """Tensor-product rules use the fewest points exact for the request.

        ``degree = (required_order + 2) // 2`` and ``order = 2 * degree - 1``.
        """
        rule = get_quadrature_rule(
            cell_type, required_order=required_order, backend=backend
        )
        reference_dimension = int(get_element(cell_type).reference_dimension)
        assert rule.order == order
        assert tuple(rule.xi.shape) == (n_points, reference_dimension)
        assert tuple(rule.weights.shape) == (n_points,)

    @pytest.mark.parametrize(
        ("linear", "quadratic"),
        [
            (pv.CellType.TRIANGLE, pv.CellType.QUADRATIC_TRIANGLE),
            (pv.CellType.QUAD, pv.CellType.QUADRATIC_QUAD),
            (pv.CellType.TETRA, pv.CellType.QUADRATIC_TETRA),
            (pv.CellType.HEXAHEDRON, pv.CellType.QUADRATIC_HEXAHEDRON),
        ],
    )
    def test_quadratic_elements_reuse_the_linear_reference_rule(
        self,
        linear: pv.CellType,
        quadratic: pv.CellType,
        backend: Backend,
    ):
        """Linear and quadratic cells of one shape share reference points."""
        left = get_quadrature_rule(linear, required_order=4, backend=backend)
        right = get_quadrature_rule(
            quadratic, required_order=4, backend=backend
        )
        assert left.order == right.order
        torch.testing.assert_close(
            backend.to_torch(left.xi), backend.to_torch(right.xi)
        )
        torch.testing.assert_close(
            backend.to_torch(left.weights), backend.to_torch(right.weights)
        )

    @pytest.mark.parametrize(
        ("cell_type", "required_order"),
        [
            (pv.CellType.TRIANGLE, 0),
            (pv.CellType.TRIANGLE, 5),
            (pv.CellType.TETRA, 6),
            (pv.CellType.QUAD, 0),
            (pv.CellType.HEXAHEDRON, -1),
        ],
    )
    def test_unsupported_order_raises(
        self,
        cell_type: pv.CellType,
        required_order: int,
        backend: Backend,
    ):
        """Orders without a formula raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Unsupported quadrature"):
            get_quadrature_rule(
                cell_type, required_order=required_order, backend=backend
            )

    def test_unsupported_cell_type_raises(self, backend: Backend):
        """Cells without a reference rule raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="No quadrature rule"):
            get_quadrature_rule(pv.CellType.VOXEL, backend=backend)


class TestIntegration:
    """Isoparametric integration on identity-mapped reference cells."""

    @pytest.mark.parametrize("cell_type", supported_cell_types())
    def test_unit_integrand_recovers_the_reference_measure(
        self, cell_type: pv.CellType, backend: Backend
    ):
        r"""Omitting the integrand returns the reference-cell area or volume.

        On the identity map,
        ``\int_{\hat{\Omega}} 1 \, d\xi = |\hat{\Omega}|``.
        """
        coords = reference_coords(cell_type, backend)
        reference_dimension = int(get_element(cell_type).reference_dimension)
        assert coords.shape[-1] == reference_dimension
        nodes_torch = backend.to_torch(coords[None]).detach().requires_grad_()
        measure = integrate_element(
            backend.as_tensor(nodes_torch), cell_type, backend=backend
        )
        expected = backend.as_tensor([_reference_measure(cell_type)])
        torch.testing.assert_close(
            backend.to_torch(measure),
            backend.to_torch(expected),
        )
        torch.sum(measure).backward()
        assert nodes_torch.grad is not None
        assert torch.isfinite(nodes_torch.grad).all()

    @pytest.mark.parametrize("cell_type", supported_cell_types())
    def test_uniform_scaling_multiplies_the_measure(
        self, cell_type: pv.CellType, backend: Backend
    ):
        """A uniformly scaled copy scales area by 4 and volume by 8."""
        coords = reference_coords(cell_type, backend)
        cells = torch.stack((coords, 2.0 * coords), dim=0)
        measure = integrate_element(cells, cell_type, backend=backend)
        reference = _reference_measure(cell_type)
        dimension = int(get_element(cell_type).reference_dimension)
        expected = backend.as_tensor([reference, reference * 2.0**dimension])
        torch.testing.assert_close(
            backend.to_torch(measure),
            backend.to_torch(expected),
        )

    @pytest.mark.parametrize("cell_type", supported_cell_types())
    def test_reference_coordinate_integrates_exactly(
        self, cell_type: pv.CellType, backend: Backend
    ):
        r"""The first reference coordinate integrates exactly.

        On the identity map, ``\int_{\hat{\Omega}} \xi_0 \, d\xi`` equals
        ``1/6`` on the triangle, ``1/24`` on the tetrahedron, and ``0`` on
        the tensor-product cells.
        """
        coords = reference_coords(cell_type, backend)
        rule = get_quadrature_rule(cell_type, backend=backend)
        xi = backend.to_torch(rule.xi)
        integrand = backend.as_tensor(xi[None, :, 0])
        integrated = integrate_element(
            coords[None], cell_type, integrand, backend=backend
        )
        expected = backend.as_tensor(
            [_reference_coordinate_integral(cell_type)]
        )
        torch.testing.assert_close(
            backend.to_torch(integrated),
            backend.to_torch(expected),
        )

    @pytest.mark.parametrize("cell_type", supported_cell_types())
    def test_constant_vector_integrand_scales_the_measure(
        self, cell_type: pv.CellType, backend: Backend
    ):
        r"""A constant vector integrand returns measure times that vector.

        ``\int_{\hat{\Omega}} \mathbf{v} \, d\xi = |\hat{\Omega}| \mathbf{v}``
        """
        coords = reference_coords(cell_type, backend)
        rule = get_quadrature_rule(cell_type, backend=backend)
        n_quad = rule.weights.shape[0]
        values = torch.tensor(
            [1.0, -2.0, 0.5], dtype=backend.dtype, device=backend.device
        )
        samples = values.view(1, 1, 3).expand(1, n_quad, 3).contiguous()
        integrated = integrate_element(
            coords[None],
            cell_type,
            backend.as_tensor(samples),
            backend=backend,
        )
        expected = values * _reference_measure(cell_type)
        torch.testing.assert_close(
            backend.to_torch(integrated),
            expected.view(1, 3),
        )


class TestIntegrandValidation:
    """Shape checks for ``integrate_element`` integrands."""

    @pytest.mark.parametrize(
        "integrand_shape",
        [
            (1, 1),  # wrong n_quad dimension
            (2, 3),  # wrong n_cells dimension
        ],
    )
    def test_rejects_wrong_leading_shape(
        self, integrand_shape: tuple[int, ...], backend: Backend
    ):
        """Leading axes must match ``(n_cells, n_quad)``."""
        cell_type = pv.CellType.TRIANGLE
        coords = reference_coords(cell_type, backend)[None]
        integrand = backend.ones(integrand_shape)
        with pytest.raises(
            ValueError, match="share cell/quadrature dimensions"
        ):
            integrate_element(coords, cell_type, integrand, backend=backend)

    @pytest.mark.parametrize(
        "integrand_shape",
        [
            (1,),
            (3,),
            (6,),
        ],
    )
    def test_rejects_rank_below_two(
        self, integrand_shape: tuple[int, ...], backend: Backend
    ):
        """The integrand must have cell and quadrature dimensions."""
        cell_type = pv.CellType.TRIANGLE
        nodes = reference_coords(cell_type, backend)[None]
        integrand = backend.ones(integrand_shape)
        with pytest.raises(ValueError, match="cell and quadrature dimensions"):
            integrate_element(nodes, cell_type, integrand, backend=backend)

    @pytest.mark.parametrize("cell_type", supported_cell_types())
    def test_accepts_correct_shape(
        self, cell_type: pv.CellType, backend: Backend
    ):
        """The integrand leading shape is ``(n_cells, n_quad, ...)``."""
        nodes = reference_coords(cell_type, backend)[None]
        rule = get_quadrature_rule(cell_type, backend=backend)
        integrand = backend.ones((1, rule.weights.shape[0], 2))
        integrate_element(nodes, cell_type, integrand, backend=backend)

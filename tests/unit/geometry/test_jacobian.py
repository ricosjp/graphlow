"""Unit tests for isoparametric Jacobians and physical gradients."""

from __future__ import annotations

import pytest
import pyvista as pv
import torch

from graphlow.core.backend.base import Backend, TensorLike
from graphlow.core.element_definitions import (
    get_element,
    reference_coords,
    supported_cell_types,
)
from graphlow.geometry.elements import (
    compute_jacobian,
    jacobian_measure,
    physical_shape_function_gradients,
)
from graphlow.geometry.elements.shape_functions import (
    get_shape_function_gradients,
)
from graphlow.utils.topology_helper import TopologyDim


@pytest.fixture
def sample_xi(cell_type: pv.CellType, backend: Backend) -> TensorLike:
    """A few interior-ish sample points in reference coordinates."""
    coords = reference_coords(cell_type, backend)
    centroid = torch.mean(coords, dim=0, keepdim=True)
    return torch.cat((centroid, (coords[:1] + centroid) / 2), dim=0)


def _float_tol(backend: Backend) -> tuple[float, float]:
    """Return (rtol, atol) matched to the backend float precision."""
    if backend.dtype == torch.float32:
        return 1e-5, 1e-6
    return 1e-10, 1e-12


def _affine_physical_nodes_and_gradient(
    cell_type: pv.CellType, backend: Backend
) -> tuple[TensorLike, TensorLike]:
    """Build physical nodes x(ξ) and a constant field gradient ∇u.

    The scalar field is the affine function ``u(x) = x · ∇u``. Surface
    elements are mapped into the ``z = 0`` plane; volume elements use a
    full-rank linear map in R^3.
    """
    xi = reference_coords(cell_type, backend)
    if get_element(cell_type).reference_dimension == TopologyDim.SURFACE:
        # x = 2 r, y = 3 s, z = 0
        r = xi[..., 0]
        s = xi[..., 1]
        zeros = r * 0
        x_nodes = torch.stack((2 * r, 3 * s, zeros), dim=-1)
        field_gradient = backend.as_tensor([1.5, -2.0, 0.0])
    else:
        # x = ξ @ A  with invertible A
        linear_map = backend.as_tensor(
            [
                [2.0, 0.2, 0.0],
                [0.0, 3.0, 0.1],
                [0.1, 0.0, 4.0],
            ]
        )
        x_nodes = xi @ linear_map
        field_gradient = backend.as_tensor([1.5, -2.0, 0.75])
    return x_nodes, field_gradient


@pytest.mark.parametrize("cell_type", supported_cell_types())
def test_jacobian_reconstructs_linear_field_gradient(
    cell_type: pv.CellType,
    backend: Backend,
    sample_xi: TensorLike,
):
    """∇_x N reconstructs gradients of an affine field on the mapped cell."""
    x_nodes, field_gradient = _affine_physical_nodes_and_gradient(
        cell_type, backend
    )
    x_torch = backend.to_torch(x_nodes[None]).detach().requires_grad_()
    x_nodes = backend.as_tensor(x_torch)

    dN_dxi = get_shape_function_gradients(cell_type)(sample_xi)
    jacobian = compute_jacobian(x_nodes, dN_dxi, backend)
    _, abs_det_j = jacobian_measure(jacobian)
    grad_x_N = physical_shape_function_gradients(jacobian, dN_dxi, backend)
    # Nodal values of u(x) = x · ∇u, then Σ_a u_a ∇_x N_a
    nodal_u = x_nodes[0] @ field_gradient
    reconstructed_grad = backend.einsum("cqad,a->cqd", grad_x_N, nodal_u)
    expected_grad = backend.to_torch(field_gradient).expand_as(
        backend.to_torch(reconstructed_grad)
    )
    rtol, atol = _float_tol(backend)
    torch.testing.assert_close(
        backend.to_torch(reconstructed_grad),
        expected_grad,
        rtol=rtol,
        atol=atol,
    )

    torch.sum(abs_det_j).backward()
    assert x_torch.grad is not None
    assert torch.isfinite(x_torch.grad).all()


@pytest.mark.parametrize("cell_type", supported_cell_types())
def test_identity_reference_mapping_has_unit_measure(
    cell_type: pv.CellType,
    backend: Backend,
    sample_xi: TensorLike,
):
    """Straight identity geometry yields unit integration measure."""
    coords = reference_coords(cell_type, backend)
    if coords.shape[-1] == 2:
        coords = torch.nn.functional.pad(coords, (0, 1))
    x_nodes_torch = backend.to_torch(coords[None]).detach().requires_grad_()
    x_nodes = backend.as_tensor(x_nodes_torch)

    dN_dxi = get_shape_function_gradients(cell_type)(sample_xi)
    _, abs_det_j = jacobian_measure(compute_jacobian(x_nodes, dN_dxi, backend))
    rtol, atol = _float_tol(backend)
    torch.testing.assert_close(
        backend.to_torch(abs_det_j),
        torch.ones_like(backend.to_torch(abs_det_j)),
        rtol=rtol,
        atol=atol,
    )
    torch.sum(abs_det_j).backward()
    assert x_nodes_torch.grad is not None
    assert torch.isfinite(x_nodes_torch.grad).all()


def test_inverted_tet_returns_signed_and_absolute_determinants(
    backend: Backend,
):
    """Orientation-reversed tet keeps signed det negative and |det| positive."""
    # VTK TET4 order: 0--1--2--3. Swap vertices 1 and 2 to flip orientation.
    coords = reference_coords(pv.CellType.TETRA, backend)
    inverted = torch.stack((coords[0], coords[2], coords[1], coords[3]), dim=0)
    cell_nodes = inverted[None]
    xi = backend.as_tensor([[0.25, 0.25, 0.25]])
    dN_dxi = get_shape_function_gradients(pv.CellType.TETRA)(xi)
    det_j, abs_det_j = jacobian_measure(
        compute_jacobian(cell_nodes, dN_dxi, backend)
    )
    assert det_j is not None
    assert torch.all(backend.to_torch(det_j) < 0)
    assert torch.all(backend.to_torch(abs_det_j) >= 0)
    assert bool(torch.any(backend.to_torch(det_j) <= 0).item())


def test_degenerate_surface_abs_det_j_has_finite_gradient(backend: Backend):
    """Collapsed triangle has zero measure with finite autograd."""
    nodes_torch = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        dtype=backend.dtype,
        device=backend.device,
        requires_grad=True,
    )
    nodes = backend.as_tensor(nodes_torch)
    xi = backend.as_tensor([[1 / 3, 1 / 3]])
    dN_dxi = get_shape_function_gradients(pv.CellType.TRIANGLE)(xi)
    _, abs_det_j = jacobian_measure(compute_jacobian(nodes, dN_dxi, backend))

    torch.testing.assert_close(
        backend.to_torch(abs_det_j),
        torch.zeros_like(backend.to_torch(abs_det_j)),
    )
    torch.sum(abs_det_j).backward()

    assert nodes_torch.grad is not None
    assert torch.isfinite(nodes_torch.grad).all()


# =============================================================================
# compute_jacobian
# =============================================================================
def test_compute_jacobian_accepts_rank2_gradients(backend: Backend):
    """A single reference point may be passed as rank-2 ``dN_dxi``."""
    x_nodes = backend.as_tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    dN_dxi = get_shape_function_gradients(pv.CellType.TRIANGLE)(
        backend.as_tensor([[1 / 3, 1 / 3]])
    )[0]
    assert dN_dxi.ndim == 2
    jacobian = compute_jacobian(x_nodes, dN_dxi, backend)
    assert tuple(jacobian.shape) == (1, 1, 3, 2)


def test_compute_jacobian_rejects_invalid_ranks(backend: Backend):
    """Rank checks reject malformed node and gradient tensors."""
    x_nodes = backend.as_tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]
    )
    dN_dxi = get_shape_function_gradients(pv.CellType.TRIANGLE)(
        backend.as_tensor([[1 / 3, 1 / 3]])
    )
    with pytest.raises(ValueError, match="x_nodes must have rank 3"):
        compute_jacobian(x_nodes[0], dN_dxi, backend)
    with pytest.raises(ValueError, match="dN_dxi must have rank 3"):
        compute_jacobian(x_nodes, dN_dxi[0, 0], backend)
    with pytest.raises(ValueError, match="same node count"):
        compute_jacobian(x_nodes[:, :2], dN_dxi, backend)


# =============================================================================
# jacobian_measure
# =============================================================================
def test_jacobian_measure_square_2d(backend: Backend):
    """Planar square Jacobians use the 2x2 determinant formula."""
    # Nodes live in R^2 so J has shape (..., 2, 2).
    x_nodes = backend.as_tensor([[[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]]])
    xi = backend.as_tensor([[1 / 3, 1 / 3]])
    dN_dxi = get_shape_function_gradients(pv.CellType.TRIANGLE)(xi)
    jacobian = compute_jacobian(x_nodes, dN_dxi, backend)
    det_j, abs_det_j = jacobian_measure(jacobian)
    assert det_j is not None
    assert tuple(jacobian.shape[-2:]) == (2, 2)
    rtol, atol = _float_tol(backend)
    torch.testing.assert_close(
        backend.to_torch(det_j),
        torch.full_like(backend.to_torch(det_j), 6.0),
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        backend.to_torch(abs_det_j),
        backend.to_torch(det_j),
        rtol=rtol,
        atol=atol,
    )


def test_jacobian_measure_rejects_unsupported_dimensions(backend: Backend):
    """Unsupported square and embedded shapes raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Unsupported Jacobian"):
        jacobian_measure(backend.as_tensor([[[1.0]]]))
    unsupported_embedded = backend.as_tensor(
        [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]]
    )
    with pytest.raises(NotImplementedError, match="Embedded Jacobians"):
        jacobian_measure(unsupported_embedded)


# =============================================================================
# helpers
# =============================================================================
def test_compute_jacobian_accepts_rank1_xi_via_gradients(backend: Backend):
    """Rank-1 ``xi`` yields rank-2 ``dN_dxi`` for ``compute_jacobian``."""
    coords = reference_coords(pv.CellType.TETRA, backend)
    x_nodes = coords[None]
    xi = backend.as_tensor([0.25, 0.25, 0.25])
    assert xi.ndim == 1
    dN_dxi = get_shape_function_gradients(pv.CellType.TETRA)(xi)
    assert dN_dxi.ndim == 2
    _, abs_det_j = jacobian_measure(compute_jacobian(x_nodes, dN_dxi, backend))
    assert tuple(abs_det_j.shape) == (1, 1)
    rtol, atol = _float_tol(backend)
    torch.testing.assert_close(
        backend.to_torch(abs_det_j),
        torch.ones_like(backend.to_torch(abs_det_j)),
        rtol=rtol,
        atol=atol,
    )


def test_abs_det_j_scales_with_uniform_cell_scaling(backend: Backend):
    """Uniformly scaled tet cells have integration measure 1 and 8."""
    coords = reference_coords(pv.CellType.TETRA, backend)
    x_nodes = torch.stack((coords, 2.0 * coords), dim=0)
    xi = backend.as_tensor([[0.25, 0.25, 0.25], [0.1, 0.1, 0.1]])
    dN_dxi = get_shape_function_gradients(pv.CellType.TETRA)(xi)
    _, abs_det_j = jacobian_measure(compute_jacobian(x_nodes, dN_dxi, backend))
    minima = torch.amin(abs_det_j, dim=1)
    assert tuple(minima.shape) == (2,)
    rtol, atol = _float_tol(backend)
    torch.testing.assert_close(
        backend.to_torch(minima),
        backend.to_torch(backend.as_tensor([1.0, 8.0])),
        rtol=rtol,
        atol=atol,
    )


def test_embedded_surface_has_no_signed_determinant(backend: Backend):
    """Embedded TRI3 Jacobians expose measure without signed det."""
    coords = reference_coords(pv.CellType.TRIANGLE, backend)
    padded = torch.nn.functional.pad(coords, (0, 1))
    xi = backend.as_tensor([[1 / 3, 1 / 3]])
    dN_dxi = get_shape_function_gradients(pv.CellType.TRIANGLE)(xi)
    det_j, abs_det_j = jacobian_measure(
        compute_jacobian(padded[None], dN_dxi, backend)
    )
    assert det_j is None
    assert abs_det_j is not None

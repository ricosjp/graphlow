"""Unit tests for isoparametric shape functions."""

from __future__ import annotations

import pytest
import pyvista as pv
import torch

from graphlow.core.backend.base import Backend
from graphlow.core.element_definitions import (
    reference_coords,
    supported_cell_types,
)
from graphlow.geometry.elements import (
    get_shape_function_gradients,
    get_shape_functions,
)


@pytest.mark.parametrize("cell_type", supported_cell_types())
def test_shape_functions_kronecker_and_partition(
    cell_type: pv.CellType, backend: Backend
):
    """Nodal Kronecker delta and partition of unity hold for all elements."""
    shape_fn = get_shape_functions(cell_type)
    coords = reference_coords(cell_type, backend)
    values = shape_fn(coords)
    eye = backend.as_tensor(torch.eye(values.shape[-1]).numpy())
    torch.testing.assert_close(backend.to_torch(values), backend.to_torch(eye))

    centroid = torch.mean(coords, dim=0, keepdim=True)
    xi = (coords + centroid) / 2
    ones = backend.ones((xi.shape[0],))
    partition = torch.sum(shape_fn(xi), dim=-1)
    torch.testing.assert_close(
        backend.to_torch(partition),
        backend.to_torch(ones),
    )


@pytest.mark.parametrize("cell_type", supported_cell_types())
def test_shape_function_gradients_sum_to_zero(
    cell_type: pv.CellType, backend: Backend
):
    """Gradients sum to zero (consistency with partition of unity)."""
    xi = reference_coords(cell_type, backend)
    gradients = get_shape_function_gradients(cell_type)(xi)
    summed = torch.sum(gradients, dim=-2)
    torch.testing.assert_close(
        backend.to_torch(summed),
        backend.to_torch(backend.zeros_like(summed)),
    )


@pytest.mark.parametrize("cell_type", supported_cell_types())
def test_shape_function_gradients_match_finite_difference(
    cell_type: pv.CellType, backend: Backend
):
    """Analytic ∇_ξ N matches central finite differences."""
    shape_fn = get_shape_functions(cell_type)
    gradient_fn = get_shape_function_gradients(cell_type)
    coords = reference_coords(cell_type, backend)
    xi = torch.mean(coords, dim=0, keepdim=True)
    analytic = gradient_fn(xi)[0]
    step = 1e-3 if backend.dtype == torch.float32 else 1e-6
    columns = []
    for axis in range(xi.shape[-1]):
        offset_torch = torch.zeros(
            tuple(xi.shape), dtype=backend.dtype, device=backend.device
        )
        offset_torch[:, axis] = step
        offset = backend.as_tensor(offset_torch)
        columns.append(
            (shape_fn(xi + offset) - shape_fn(xi - offset))[0] / (2 * step)
        )
    numerical = torch.stack(columns, dim=-1)
    torch.testing.assert_close(
        backend.to_torch(analytic),
        backend.to_torch(numerical),
        rtol=1e-3 if backend.dtype == torch.float32 else 1e-8,
        atol=1e-4 if backend.dtype == torch.float32 else 1e-9,
    )

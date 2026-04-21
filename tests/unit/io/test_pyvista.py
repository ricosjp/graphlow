"""Unit tests for PyVista I/O helpers."""

import numpy as np
import pytest
import pyvista as pv
import torch
from phlower_tensor import PhysicalDimensions

import graphlow


@pytest.fixture
def invalid_mesh() -> pv.UnstructuredGrid:
    """Invalid mesh with unused_points."""
    cells = np.array([4, 0, 1, 2, 3])
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    ctypes = np.array([4])
    mesh = pv.UnstructuredGrid(cells, ctypes, points)
    return mesh


def test_from_pyvista_validate_mesh(invalid_mesh: pv.UnstructuredGrid) -> None:
    """Ensure validate_mesh=True raises ValueError for invalid mesh."""
    with pytest.raises(ValueError):
        graphlow.from_pyvista(invalid_mesh, backend="torch", validate_mesh=True)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
def test_from_pyvista_invalid_dtype(
    mix_poly_grid: pv.UnstructuredGrid, dtype: torch.dtype
) -> None:
    """Ensure ValueError is raised for invalid dtype."""
    with pytest.raises(ValueError):
        graphlow.from_pyvista(mix_poly_grid, backend="torch", dtype=dtype)


@pytest.fixture
def valid_mesh() -> pv.UnstructuredGrid:
    grid = pv.ImageData(dimensions=(10, 10, 10), spacing=(1.0, 1.0, 1.0))
    grid.point_data["temperature"] = np.random.rand(grid.n_points)
    grid.cell_data["pressure"] = np.random.rand(grid.n_cells)
    grid.point_data["velocity"] = np.random.rand(grid.n_points, 3)
    return grid.cast_to_unstructured_grid()


@pytest.mark.parametrize(
    "dimension_collection",
    [
        None,
        {
            "temperature": {"T": 1},
            "pressure": {"M": 1, "L": -1, "T": -2},
            "velocity": {"L": 1, "T": -1},
        },
        {
            "temperature": {"T": 1},
            "velocity": {"L": 1, "T": -1},
        },
    ],
)
def test_from_pyvista_with_phlower(
    valid_mesh: pv.UnstructuredGrid,
    dimension_collection: dict[str, dict[str, float]] | None,
):
    mesh = graphlow.from_pyvista(
        valid_mesh, backend="phlower", dimension_collection=dimension_collection
    )

    assert mesh.points.has_dimension
    assert mesh.points.dimension.to_physics_dimension() == PhysicalDimensions(
        {"L": 1}
    )

    dimension_collection = dimension_collection or {}
    for name, data in mesh.point_data.items():
        if name in dimension_collection:
            assert data.dimension.to_physics_dimension() == PhysicalDimensions(
                dimension_collection[name]
            )

        else:
            assert data.dimension.is_dimensionless


@pytest.mark.parametrize(
    "dimension_collection",
    [
        None,
        {
            "temperature": {"T": 1},
            "pressure": {"M": 1, "L": -1, "T": -2},
            "velocity": {"L": 1, "T": -1},
        },
        {
            "temperature": {"T": 1},
            "velocity": {"L": 1, "T": -1},
        },
    ],
)
def test_from_pyvista_diable_dimensions(
    valid_mesh: pv.UnstructuredGrid,
    dimension_collection: dict[str, dict[str, float]] | None,
):
    mesh = graphlow.from_pyvista(
        valid_mesh,
        backend="phlower",
        dimension_collection=dimension_collection,
        disable_dimensions=True,
    )
    assert not mesh.points.has_dimension

    for data in mesh.point_data.values():
        assert not data.has_dimension

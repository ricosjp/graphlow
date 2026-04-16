"""Unit tests for PyVista I/O helpers."""

import numpy as np
import pytest
import pyvista as pv
import torch

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

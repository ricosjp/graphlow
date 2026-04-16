"""Tests for geometry.methods.analytic using primitive_cell meshes."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import pyvista as pv
import torch
from phlower_tensor import phlower_dimension_tensor
from pyvista.examples.cells import (
    Hexahedron,
    Polyhedron,
    Pyramid,
    Tetrahedron,
    Wedge,
)

import graphlow
from graphlow.io.pyvista import from_pyvista


# =============================================================================
# Surface volumes
# =============================================================================
@pytest.mark.parametrize(
    "filename",
    [
        # primitives
        pathlib.Path("tests/data/vtu/primitive_cell/tet.vtu"),  # triangle
        pathlib.Path("tests/data/vtu/primitive_cell/pyramid.vtu"),  # quad
        pathlib.Path("tests/data/vtu/primitive_cell/poly.vtu"),  # polygon
        pathlib.Path("tests/data/vts/cube/mesh.vts"),
        pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
        pathlib.Path("tests/data/vtu/cube/large.vtu"),
    ],
)
def test_surface_volume(filename: pathlib.Path):
    """Surface volume matches PyVista for watertight extracted surfaces."""
    grid: pv.DataSet = pv.read(filename)
    grid: pv.PolyData = grid.extract_surface(algorithm=None)
    surface_mesh = from_pyvista(grid, "phlower", dtype=torch.float64)
    surface_volume = surface_mesh.geometry.surface_volume()
    assert surface_volume.dimension == phlower_dimension_tensor({"L": 3})

    pv_surface_volume = np.array([grid.volume])
    np.testing.assert_almost_equal(surface_volume.numpy(), pv_surface_volume)


@pytest.mark.parametrize("grid", [pv.Disc(), pv.examples.cells.Hexahedron()])
def test_surface_volume_for_non_watertight_raises(
    grid: pv.UnstructuredGrid,
) -> None:
    """Non-watertight surface mesh: surface_volume raises ValueError."""
    grid = grid.cast_to_unstructured_grid()
    mesh = from_pyvista(grid, "phlower", dtype=torch.float64)
    with pytest.raises(
        ValueError, match="only supported for watertight surface meshes."
    ):
        mesh.geometry.surface_volume()


# =============================================================================
# Cell volumes
# =============================================================================
@pytest.mark.parametrize(
    "filename",
    [
        # primitives
        pathlib.Path("tests/data/vtu/primitive_cell/tet.vtu"),
        pathlib.Path("tests/data/vtu/primitive_cell/pyramid.vtu"),
        pathlib.Path("tests/data/vtu/primitive_cell/wedge.vtu"),
        pathlib.Path("tests/data/vtu/primitive_cell/hex.vtu"),
        pathlib.Path("tests/data/vtu/primitive_cell/poly.vtu"),
        pathlib.Path("tests/data/vts/cube/mesh.vts"),
        pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
        pathlib.Path("tests/data/vtu/cube/large.vtu"),
    ],
)
def test_cell_volumes(filename: pathlib.Path):
    """Cell volumes match PyVista's cell-size computation."""
    volmesh = graphlow.read(filename, "phlower", dtype=torch.float64)
    cell_volumes = volmesh.geometry.cell_volumes()
    assert cell_volumes.dimension == phlower_dimension_tensor({"L": 3})

    actual = cell_volumes.numpy()
    expected = (
        volmesh.pvmesh.compute_cell_sizes().cell_data["Volume"].reshape(-1, 1)
    )
    np.testing.assert_almost_equal(actual, expected)


def test_cell_volumes_for_non_volume_raises() -> None:
    """Non-volume mesh: cell_volumes raises ValueError."""
    grid = pv.Sphere().cast_to_unstructured_grid()
    mesh = from_pyvista(grid, "phlower", dtype=torch.float64)
    with pytest.raises(ValueError, match="only supported for volume meshes."):
        mesh.geometry.cell_volumes()


# =============================================================================
# Cell centroids
# =============================================================================


@pytest.mark.parametrize(
    "grid, expected",
    [
        (Tetrahedron(), np.array([[0, 0, 0]])),
        (Pyramid(), np.array([[0.0, 0.0, np.sqrt(2) / 8]])),
        (Wedge(), np.array([[1 / 2, 1 / 2, np.sqrt(3) / 6]])),
        (Hexahedron(), np.array([[1 / 2, 1 / 2, 1 / 2]])),
        (Polyhedron(), np.array([[3 / 8, 1 / 8, 1 / 4]])),
    ],
)
def test_cell_centroids(grid: pv.UnstructuredGrid, expected: np.ndarray):
    """Cell centroids match analytic references for primitive cells."""
    volmesh = from_pyvista(grid, "phlower", dtype=torch.float64)
    cell_centroids = volmesh.geometry.cell_centroids()
    assert cell_centroids.dimension == phlower_dimension_tensor({"L": 1})
    np.testing.assert_almost_equal(cell_centroids.numpy(), expected)

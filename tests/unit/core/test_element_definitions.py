"""Unit tests for static VTK element definitions."""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv
import vtk

from graphlow.core.backend.base import Backend
from graphlow.core.element_definitions import (
    get_element,
    get_fixed_face_patterns,
    reference_coords,
)

VTK_QUADRATIC_CELLS = (
    (
        pv.CellType.QUADRATIC_TRIANGLE,
        vtk.vtkQuadraticTriangle,
        2,
        False,
    ),
    (pv.CellType.QUADRATIC_QUAD, vtk.vtkQuadraticQuad, 2, True),
    (
        pv.CellType.QUADRATIC_TETRA,
        vtk.vtkQuadraticTetra,
        3,
        False,
    ),
    (
        pv.CellType.QUADRATIC_HEXAHEDRON,
        vtk.vtkQuadraticHexahedron,
        3,
        True,
    ),
)


@pytest.mark.parametrize(
    "cell_type, vtk_class, dimension, rescale", VTK_QUADRATIC_CELLS
)
def test_reference_coords_match_vtk_parametric_coordinates(
    cell_type: pv.CellType,
    vtk_class: type,
    dimension: int,
    rescale: bool,
    backend: Backend,
):
    """Reference coordinates match VTK parametric coordinates."""
    vtk_coordinates = np.asarray(vtk_class().GetParametricCoords()).reshape(
        (-1, 3)
    )[:, :dimension]
    # VTK uses [0, 1]^d for tensor-product parametric coordinates, while the
    # graphlow reference convention is [-1, 1]^d.
    if rescale:
        vtk_coordinates = 2 * vtk_coordinates - 1
    np.testing.assert_allclose(
        backend.to_numpy(reference_coords(cell_type, backend)),
        vtk_coordinates,
    )


@pytest.mark.parametrize(
    "cell_type, face_type, vtk_class",
    [
        (
            pv.CellType.QUADRATIC_TETRA,
            pv.CellType.QUADRATIC_TRIANGLE,
            vtk.vtkQuadraticTetra,
        ),
        (
            pv.CellType.QUADRATIC_HEXAHEDRON,
            pv.CellType.QUADRATIC_QUAD,
            vtk.vtkQuadraticHexahedron,
        ),
    ],
)
def test_face_patterns_match_vtk(
    cell_type: pv.CellType,
    face_type: pv.CellType,
    vtk_class: type,
):
    """Element face_patterns match VTK and feed get_fixed_face_patterns."""
    definition = get_element(cell_type)
    points = np.asarray(vtk_class().GetParametricCoords()).reshape((-1, 3))
    connectivity = np.arange(definition.n_nodes, dtype=np.int64)[None, :]
    grid = pv.UnstructuredGrid({cell_type: connectivity}, points)
    cell = grid.get_cell(0)
    vtk_faces = tuple(
        tuple(cell.get_face(face_id).point_ids)
        for face_id in range(cell.n_faces)
    )
    np.testing.assert_array_equal(
        definition.face_patterns[face_type], vtk_faces
    )
    np.testing.assert_array_equal(
        get_fixed_face_patterns(cell_type)[face_type],
        vtk_faces,
    )


@pytest.mark.parametrize(
    "cell_type",
    [
        pv.CellType.VOXEL,
        pv.CellType.WEDGE,
        pv.CellType.PYRAMID,
    ],
)
def test_topology_only_elements_have_face_patterns_without_reference_coords(
    cell_type: pv.CellType,
):
    """WEDGE / PYRAMID / VOXEL expose face patterns but no FEM coordinates."""
    definition = get_element(cell_type)
    assert definition.reference_coords is None
    assert get_fixed_face_patterns(cell_type) is definition.face_patterns
    assert definition.face_patterns


def test_get_element_unsupported_raises():
    """Unsupported cell types raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Unsupported VTK cell type"):
        get_element(pv.CellType.VERTEX)


def test_reference_coords_unavailable_for_topology_only_elements(
    backend: Backend,
):
    """Topology-only cells reject reference_coords materialization."""
    with pytest.raises(NotImplementedError, match="No reference coordinates"):
        reference_coords(pv.CellType.WEDGE, backend)

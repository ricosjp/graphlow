"""Unit tests for utils.topology_helper."""

from __future__ import annotations

import numpy as np
import pytest
import pyvista as pv

from graphlow.utils.topology_helper import (
    TopologyDim,
    gather_fixed_elements,
    gather_jagged_elements,
    get_cell_dimension,
    select_cells_by_type,
)


# =============================================================================
# Cell Type and Dimension
# =============================================================================
@pytest.mark.parametrize(
    "cell_type, expected",
    [
        (pv.CellType.TRIANGLE, TopologyDim.SURFACE),
        (pv.CellType.TETRA, TopologyDim.VOLUME),
        (pv.CellType.HEXAHEDRON, TopologyDim.VOLUME),
    ],
)
def test_get_cell_dimension(cell_type: pv.CellType, expected: TopologyDim):
    """get_cell_dimension returns correct TopologyDim for supported types."""
    assert get_cell_dimension(cell_type) == expected


def test_get_cell_dimension_unsupported_raises():
    """get_cell_dimension for unsupported cell type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown or unsupported"):
        get_cell_dimension(pv.CellType.VERTEX)


# =============================================================================
# Cell Selection
# =============================================================================
@pytest.mark.parametrize(
    "cell_types, cell_offsets, query_type, \
        expected_indices, expected_starts, expected_ends",
    [
        (
            np.array(
                [
                    pv.CellType.TETRA.value,
                    pv.CellType.TRIANGLE.value,
                    pv.CellType.TETRA.value,
                ],
            ),
            np.array([0, 4, 7, 11], dtype=np.int64),
            pv.CellType.TETRA,
            [0, 2],
            [0, 7],
            [4, 11],
        ),
        (
            np.array(
                [
                    pv.CellType.QUAD.value,
                    pv.CellType.POLYHEDRON.value,
                    pv.CellType.HEXAHEDRON.value,
                ],
            ),
            np.array([0, 4, 10, 18], dtype=np.int64),
            pv.CellType.POLYHEDRON,
            [1],
            [4],
            [10],
        ),
    ],
)
def test_select_cells_by_type(
    cell_types: np.ndarray,
    cell_offsets: np.ndarray,
    query_type: pv.CellType,
    expected_indices: np.ndarray,
    expected_starts: np.ndarray,
    expected_ends: np.ndarray,
):
    """select_cells_by_type returns global indices and offset ranges."""
    global_indices, start_offsets, end_offsets = select_cells_by_type(
        cell_types, cell_offsets, query_type
    )
    np.testing.assert_array_equal(global_indices, expected_indices)
    np.testing.assert_array_equal(start_offsets, expected_starts)
    np.testing.assert_array_equal(end_offsets, expected_ends)


def test_select_cells_by_type_empty_raises():
    """select_cells_by_type when no cell of type raises ValueError."""
    cell_types = np.array(
        [pv.CellType.TRIANGLE.value, pv.CellType.TRIANGLE.value], dtype=np.int64
    )
    cell_offsets = np.array([0, 3, 6], dtype=np.int64)
    with pytest.raises(ValueError, match="does not contain"):
        select_cells_by_type(cell_types, cell_offsets, pv.CellType.TETRA)


# =============================================================================
# Element Gathering
# =============================================================================
@pytest.fixture
def simple_poly_mixed() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    celltypes = np.array(
        [
            pv.CellType.TETRA.value,
            pv.CellType.POLYHEDRON.value,
            pv.CellType.HEXAHEDRON.value,
            pv.CellType.TETRA.value,
            pv.CellType.POLYHEDRON.value,
        ],
        dtype=np.int64,
    )
    # fmt: off
    conn = np.array([
            0, 1, 2, 3,                 # tet
            0, 1, 2, 4, 5,              # polyhedron
            1, 2, 4, 5, 6, 7, 8, 9,     # hex
            0, 2, 3, 10,                # tet
            7, 8, 9, 6, 11, 12, 13, 14, # polyhedron
        ], dtype=np.int64)
    # fmt: on
    offsets = np.array([0, 4, 9, 17, 21, 29], dtype=np.int64)
    return celltypes, conn, offsets


@pytest.mark.parametrize(
    "select_cell_type, expected",
    [
        (
            pv.CellType.TETRA,
            np.array([[0, 1, 2, 3], [0, 2, 3, 10]], dtype=np.int64),
        ),
        (
            pv.CellType.HEXAHEDRON,
            np.array([[1, 2, 4, 5, 6, 7, 8, 9]], dtype=np.int64),
        ),
    ],
)
def test_gather_fixed_elements(
    simple_poly_mixed: tuple[np.ndarray, np.ndarray, np.ndarray],
    select_cell_type: pv.CellType,
    expected: np.ndarray,
):
    """gather_fixed_elements produces (n_elements, k)"""
    celltypes, conn, offsets = simple_poly_mixed
    _, start_offsets, end_offsets = select_cells_by_type(
        celltypes, offsets, select_cell_type
    )
    out = gather_fixed_elements(conn, start_offsets, end_offsets)
    np.testing.assert_array_equal(out, expected)


def test_gather_fixed_elements_variable_length_raises(
    simple_poly_mixed: tuple[np.ndarray, np.ndarray, np.ndarray],
):
    """gather_fixed_elements raises if variable length."""
    celltypes, conn, offsets = simple_poly_mixed
    _, start_offsets, end_offsets = select_cells_by_type(
        celltypes, offsets, pv.CellType.POLYHEDRON
    )
    with pytest.raises(ValueError, match="requires fixed-length"):
        gather_fixed_elements(conn, start_offsets, end_offsets)


def test_gather_jagged_elements(
    simple_poly_mixed: tuple[np.ndarray, np.ndarray, np.ndarray],
):
    """gather_jagged_elements concatenates segments and builds new_offsets."""
    celltypes, conn, offsets = simple_poly_mixed
    _, start_offsets, end_offsets = select_cells_by_type(
        celltypes, offsets, pv.CellType.POLYHEDRON
    )
    gathered, new_offsets = gather_jagged_elements(
        conn, start_offsets, end_offsets
    )
    np.testing.assert_array_equal(
        gathered,
        np.array([0, 1, 2, 4, 5, 7, 8, 9, 6, 11, 12, 13, 14], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        new_offsets, np.array([0, 5, 13], dtype=np.int64)
    )

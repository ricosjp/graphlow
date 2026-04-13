"""Tests for derived adjacency matrices built from ``MeshTopology``."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import graphlow
from graphlow.graph.skeleton_builder import AdjacencyName
from graphlow.graph.skeleton_derived import DerivedMatrixName


# fmt: off
@pytest.mark.parametrize(
    "filename, adjacency_name, expected",
    [
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            AdjacencyName.CC,
            np.array(
                [
                    # 0  1
                    [2, 0],  #  0
                    [0, 2],  #  1
                ]
            ).astype(np.float64),
        ),
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            AdjacencyName.PP,
            np.array(
                [
                    # 0  1  2  3  4  5  6  7  8  9 10 11
                    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  0
                    [0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  1
                    [0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  2
                    [0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0],  #  3
                    [0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0],  #  4
                    [0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0],  #  5
                    [0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0],  #  6
                    [0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0],  #  7
                    [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],  #  8
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0],  #  9
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0],  # 10
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8],  # 11
                ]
            ).astype(np.float64),
        ),
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            AdjacencyName.CC,
            np.array(
                [
                    [3, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2],
                ]
            ).astype(np.float64),
        ),
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            AdjacencyName.PP,
            np.array(
                [
                    # 0   1   2   3   4   5   6   7   8   9  10  11  12
                    [ 8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], #  0
                    [ 0, 12,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], #  1
                    [ 0,  0,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], #  2
                    [ 0,  0,  0, 12,  0,  0,  0,  0,  0,  0,  0,  0,  0], #  3
                    [ 0,  0,  0,  0,  8,  0,  0,  0,  0,  0,  0,  0,  0], #  4
                    [ 0,  0,  0,  0,  0, 12,  0,  0,  0,  0,  0,  0,  0], #  5
                    [ 0,  0,  0,  0,  0,  0,  8,  0,  0,  0,  0,  0,  0], #  6
                    [ 0,  0,  0,  0,  0,  0,  0,  8,  0,  0,  0,  0,  0], #  7
                    [ 0,  0,  0,  0,  0,  0,  0,  0, 12,  0,  0,  0,  0], #  8
                    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  9,  0,  0,  0], #  9
                    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9,  0,  0], # 10
                    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  9,  0], # 11
                    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  4], # 12
                ]
            ).astype(np.float64),
        ),
    ],
)
# fmt: on
def test_build_degree_matrix(
    filename: pathlib.Path,
    adjacency_name: AdjacencyName,
    expected: np.ndarray,
):
    """
    Verify ``get_derived_skeleton(..., DEGREE, ...)`` returns the expected
    degree matrix.
    """
    mesh = graphlow.read(filename, "phlower")
    topo = mesh.topology
    degree_matrix = topo.get_derived_skeleton(
        DerivedMatrixName.DEGREE, adjacency_name
    )
    np.testing.assert_array_equal(degree_matrix.toarray(), expected)


@pytest.mark.parametrize(
    "filename, adjacency_name, expected",
    [
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            AdjacencyName.PP,
            np.array(
                [
                    # 0
                    [
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        0,
                        0,
                        0,
                        0,
                    ],
                    # 1
                    [
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        0,
                        0,
                        0,
                        0,
                    ],
                    # 2
                    [
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        0,
                        0,
                        0,
                        0,
                    ],
                    # 3
                    [
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        0,
                        0,
                        0,
                        0,
                    ],
                    # 4
                    [
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / 12.0,
                        1.0 / 12.0,
                        1.0 / 12.0,
                        1.0 / 12.0,
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                    ],
                    # 5
                    [
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / 12.0,
                        1.0 / 12.0,
                        1.0 / 12.0,
                        1.0 / 12.0,
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                    ],
                    # 6
                    [
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / 12.0,
                        1.0 / 12.0,
                        1.0 / 12.0,
                        1.0 / 12.0,
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                    ],
                    # 7
                    [
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / 12.0,
                        1.0 / 12.0,
                        1.0 / 12.0,
                        1.0 / 12.0,
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                    ],
                    # 8
                    [
                        0,
                        0,
                        0,
                        0,
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                    ],
                    # 9
                    [
                        0,
                        0,
                        0,
                        0,
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                    ],
                    # 10
                    [
                        0,
                        0,
                        0,
                        0,
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                    ],
                    # 11
                    [
                        0,
                        0,
                        0,
                        0,
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / np.sqrt(8 * 12),
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                        1.0 / 8.0,
                    ],
                ]
            ),
        ),
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            AdjacencyName.CC,
            np.array(
                [
                    [1.0 / 3.0, 1.0 / np.sqrt(6.0), 1.0 / np.sqrt(6.0)],
                    [1.0 / np.sqrt(6.0), 1.0 / 2.0, 0.0],
                    [1.0 / np.sqrt(6.0), 0.0, 1.0 / 2.0],
                ]
            ),
        ),
    ],
)
def test_build_normalized_adjacency(
    filename: pathlib.Path,
    adjacency_name: AdjacencyName,
    expected: np.ndarray,
):
    """
    Verify ``get_derived_skeleton(..., NORMALIZED, ...)`` returns the expected
    normalized adjacency matrix.
    """
    mesh = graphlow.read(filename, "phlower")
    topo = mesh.topology
    normalized_adjacency = topo.get_derived_skeleton(
        DerivedMatrixName.NORMALIZED, adjacency_name
    )
    np.testing.assert_almost_equal(normalized_adjacency.toarray(), expected)


# fmt: off
@pytest.mark.parametrize(
    "filename, adjacency_name, expected",
    [
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            AdjacencyName.CC,
            np.array(
                [
                    [ 2, -1, -1],
                    [-1,  1,  0],
                    [-1,  0,  1],
                ]
            ).astype(np.float64),
        ),
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            AdjacencyName.PP,
            np.array(
                [
                    # 0   1   2   3   4   5   6   7   8   9  10  11  12
                    [ 7, -1,  0, -1,  0, -1,  0,  0, -1, -1, -1, -1,  0], #  0
                    [-1, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0], #  1
                    [ 0, -1,  7, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0], #  2
                    [-1, -1, -1, 11, -1, -1, -1, -1, -1, -1, -1, -1,  0], #  3
                    [ 0, -1, -1, -1,  7, -1, -1, -1, -1,  0,  0,  0,  0], #  4
                    [-1, -1, -1, -1, -1, 11, -1, -1, -1, -1, -1, -1,  0], #  5
                    [ 0, -1, -1, -1, -1, -1,  7, -1, -1,  0,  0,  0,  0], #  6
                    [ 0, -1, -1, -1, -1, -1, -1,  7, -1,  0,  0,  0,  0], #  7
                    [-1, -1, -1, -1, -1, -1, -1, -1, 11, -1, -1, -1,  0], #  8
                    [-1, -1,  0, -1,  0, -1,  0,  0, -1,  8, -1, -1, -1], #  9
                    [-1, -1,  0, -1,  0, -1,  0,  0, -1, -1,  8, -1, -1], # 10
                    [-1, -1,  0, -1,  0, -1,  0,  0, -1, -1, -1,  8, -1], # 11
                    [ 0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1,  3], # 12
                ]
            ).astype(np.float64),
        ),
    ],
)
# fmt: on
def test_build_laplacian(
    filename: pathlib.Path,
    adjacency_name: AdjacencyName,
    expected: np.ndarray,
):
    """
    Verify ``get_derived_skeleton(..., LAPLACIAN, ...)`` returns the expected
    Laplacian matrix.
    """
    mesh = graphlow.read(filename, "phlower")
    topo = mesh.topology
    laplacian = topo.get_derived_skeleton(
        DerivedMatrixName.LAPLACIAN, adjacency_name
    )
    np.testing.assert_array_equal(laplacian.toarray(), expected)

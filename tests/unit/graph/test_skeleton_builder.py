"""Unit tests for graph.skeleton_builder (build_skeleton, CP/PC/CC/PP)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import scipy.sparse as sps

import graphlow
from graphlow.graph.skeleton_builder import AdjacencyName, IncidenceName


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                ]
            ),
        ),
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            np.array(
                [
                    # 0  1  2  3  4  5  6  7  8  9 10 11 12
                    [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                ]
            ),
        ),
    ],
)
def test_build_cp_and_pc(filename: pathlib.Path, expected: np.ndarray):
    """
    Verify build_skeleton(CP) and build_skeleton(PC) return correct CSR arrays.

    build_skeleton(CP) returns a csr_array of shape (n_cells, n_points).
    build_skeleton(PC) returns a csr_array of shape (n_points, n_cells).
    """
    mesh = graphlow.read(filename, "phlower")
    topo = mesh.topology
    cp = topo.get_skeleton(IncidenceName.CP)
    pc = topo.get_skeleton(IncidenceName.PC)
    np.testing.assert_array_equal(cp.toarray(), expected)
    np.testing.assert_array_equal(pc.toarray(), expected.T)


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            np.array(
                [
                    # 0  1  2  3  4  5  6  7  8  9 10 11
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  #  0
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  #  1
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  #  2
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  #  3
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #  4
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #  5
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #  6
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  #  7
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],  #  8
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],  #  9
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # 10
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # 11
                ]
            ),
        ),
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            np.array(
                [
                    # 0  1  2  3  4  5  6  7  8  9 10 11 12
                    [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0],  #  0
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #  1
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  #  2
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #  3
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  #  4
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #  5
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  #  6
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  #  7
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],  #  8
                    [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1],  #  9
                    [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1],  # 10
                    [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1],  # 11
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # 12
                ]
            ),
        ),
    ],
)
def test_build_pp(filename: pathlib.Path, expected: np.ndarray):
    """
    Verify build_skeleton(PP) returns correct CSR array.

    build_skeleton(PP) returns a csr_array of shape (n_points, n_points).
    """
    mesh = graphlow.read(filename, "phlower")
    topo = mesh.topology
    pp = topo.get_skeleton(AdjacencyName.PP)
    np.testing.assert_array_equal(pp.toarray(), expected)


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            np.array(
                [
                    [1, 1],
                    [1, 1],
                ]
            ),
        ),
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            np.array(
                [
                    [1, 1, 1],
                    [1, 1, 0],
                    [1, 0, 1],
                ]
            ),
        ),
    ],
)
def test_build_cc(filename: pathlib.Path, expected: np.ndarray):
    """
    Verify build_skeleton(CC) returns correct CSR array.

    build_skeleton(CC) returns a csr_array of shape (n_cells, n_cells).
    """
    mesh = graphlow.read(filename, "phlower")
    topo = mesh.topology
    cc = topo.get_skeleton(AdjacencyName.CC)
    np.testing.assert_array_equal(cc.toarray(), expected)


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            np.array(
                [
                    # 0  1
                    [1, 0],  #  0
                    [1, 0],  #  1
                    [1, 0],  #  2
                    [1, 0],  #  3
                    [1, 0],  #  4
                    [1, -1],  #  5
                    [0, 1],  #  6
                    [0, 1],  #  7
                    [0, 1],  #  8
                    [0, 1],  #  9
                    [0, 1],  # 10
                ]
            ),
        ),
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            np.array(
                [
                    # 0  1  2
                    [1, 0, 0],  #  0
                    [1, -1, 0],  #  1
                    [1, 0, 0],  #  2
                    [1, 0, 0],  #  3
                    [1, 0, -1],  #  4
                    [1, 0, 0],  #  5
                    [0, 1, 0],  #  6
                    [0, 1, 0],  #  7
                    [0, 1, 0],  #  8
                    [0, 1, 0],  #  9
                    [0, 1, 0],  # 10
                    [0, 0, 1],  # 11
                    [0, 0, 1],  # 12
                    [0, 0, 1],  # 13
                ]
            ),
        ),
        (
            pathlib.Path("tests/data/vtu/cube/2x2.vtu"),
            np.array(
                [
                    # 0,  1,  2,  3,  4,  5,  6,  7
                    [1, 0, 0, 0, 0, 0, 0, 0],  #  0
                    [1, -1, 0, 0, 0, 0, 0, 0],  #  1
                    [1, 0, 0, 0, 0, 0, 0, 0],  #  2
                    [1, 0, -1, 0, 0, 0, 0, 0],  #  3
                    [1, 0, 0, 0, 0, 0, 0, 0],  #  4
                    [1, 0, 0, 0, -1, 0, 0, 0],  #  5
                    [0, 1, 0, 0, 0, 0, 0, 0],  #  6
                    [0, 1, 0, 0, 0, 0, 0, 0],  #  7
                    [0, 1, 0, -1, 0, 0, 0, 0],  #  8
                    [0, 1, 0, 0, 0, 0, 0, 0],  #  9
                    [0, 1, 0, 0, 0, -1, 0, 0],  # 10
                    [0, 0, 1, 0, 0, 0, 0, 0],  # 11
                    [0, 0, 1, -1, 0, 0, 0, 0],  # 12
                    [0, 0, 1, 0, 0, 0, 0, 0],  # 13
                    [0, 0, 1, 0, 0, 0, 0, 0],  # 14
                    [0, 0, 1, 0, 0, 0, -1, 0],  # 15
                    [0, 0, 0, 1, 0, 0, 0, 0],  # 16
                    [0, 0, 0, 1, 0, 0, 0, 0],  # 17
                    [0, 0, 0, 1, 0, 0, 0, 0],  # 18
                    [0, 0, 0, 1, 0, 0, 0, -1],  # 19
                    [0, 0, 0, 0, 1, 0, 0, 0],  # 20
                    [0, 0, 0, 0, 1, -1, 0, 0],  # 21
                    [0, 0, 0, 0, 1, 0, 0, 0],  # 22
                    [0, 0, 0, 0, 1, 0, -1, 0],  # 23
                    [0, 0, 0, 0, 1, 0, 0, 0],  # 24
                    [0, 0, 0, 0, 0, 1, 0, 0],  # 25
                    [0, 0, 0, 0, 0, 1, 0, 0],  # 26
                    [0, 0, 0, 0, 0, 1, 0, -1],  # 27
                    [0, 0, 0, 0, 0, 1, 0, 0],  # 28
                    [0, 0, 0, 0, 0, 0, 1, 0],  # 29
                    [0, 0, 0, 0, 0, 0, 1, -1],  # 30
                    [0, 0, 0, 0, 0, 0, 1, 0],  # 31
                    [0, 0, 0, 0, 0, 0, 1, 0],  # 32
                    [0, 0, 0, 0, 0, 0, 0, 1],  # 33
                    [0, 0, 0, 0, 0, 0, 0, 1],  # 34
                    [0, 0, 0, 0, 0, 0, 0, 1],  # 35
                ]
            ),
        ),
    ],
)
def test_build_fc_and_cf(filename: pathlib.Path, expected: np.ndarray):
    """
    Verify build_skeleton(FC) and build_skeleton(CF) return correct CSR arrays.

    build_skeleton(FC) returns a csr_array of shape (n_faces, n_cells).
    build_skeleton(CF) returns a csr_array of shape (n_cells, n_faces).
    """
    mesh = graphlow.read(filename, "phlower")
    topo = mesh.topology
    fc = topo.get_skeleton(IncidenceName.FC)
    cf = topo.get_skeleton(IncidenceName.CF)
    np.testing.assert_array_equal(fc.toarray(), expected)
    np.testing.assert_array_equal(cf.toarray(), expected.T)


# fmt: off
@pytest.mark.parametrize(
    "filename, expected",
    [
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            np.array(
                [
                    # 0  1  2  3  4  5  6  7  8  9 10 11
                    [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],  #  0
                    [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],  #  1
                    [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  #  2
                    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],  #  3
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #  4
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],  #  5
                    [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1],  #  6
                    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0],  #  7
                    [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],  #  8
                    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],  #  9
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],  # 10
                ]
            ),
        ),
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            np.array(
                [
                    # 0  1  2  3  4  5  6  7  8  9 10 11 12
                    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],  #  0
                    [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],  #  1
                    [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],  #  2
                    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],  #  3
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],  #  4
                    [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #  5
                    [0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],  #  6
                    [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],  #  7
                    [0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],  #  8
                    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  #  9
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],  # 10
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1],  # 11
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],  # 12
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],  # 13
                ]
            ),
        ),
        (
            pathlib.Path("tests/data/vtu/cube/2x2.vtu"),
            sps.csr_array(
                (
                    np.ones(144),
                    np.array(
                        [
                            3,  0,  9, 12,
                            1,  4, 13, 10,
                            0,  1, 10,  9,
                            4,  3, 12, 13,
                            1,  0,  3,  4,
                            9, 10, 13, 12,
                            2,  5, 14, 11,
                            1,  2, 11, 10,
                            5,  4, 13, 14,
                            2,  1,  4,  5,
                            10, 11, 14, 13,
                            6,  3, 12, 15,
                            4,  7, 16, 13,
                            7,  6, 15, 16,
                            4,  3,  6,  7,
                            12, 13, 16, 15,
                            5,  8, 17, 14,
                            8,  7, 16, 17,
                            5,  4,  7,  8,
                            13, 14, 17, 16,
                            12,  9, 18, 21,
                            10, 13, 22, 19,
                            9, 10, 19, 18,
                            13, 12, 21, 22,
                            18, 19, 22, 21,
                            11, 14, 23, 20,
                            10, 11, 20, 19,
                            14, 13, 22, 23,
                            19, 20, 23, 22,
                            15, 12, 21, 24,
                            13, 16, 25, 22,
                            16, 15, 24, 25,
                            21, 22, 25, 24,
                            14, 17, 26, 23,
                            17, 16, 25, 26,
                            22, 23, 26, 25
                        ]
                    ),
                    np.array([
                          0,   4,   8,  12,
                          16, 20,  24,  28,
                          32,  36,  40,  44,
                          48,  52,  56,  60,
                          64,  68,  72,  76,
                          80,  84,  88,  92,
                          96, 100, 104, 108,
                          112, 116, 120, 124,
                          128, 132, 136, 140, 144
                    ])
                ), shape=(36, 27)).toarray()

        )
    ],
)
# fmt: on
def test_build_fp_and_pf(filename: pathlib.Path, expected: np.ndarray):
    """
    Verify build_skeleton(FP) returns correct CSR array.

    build_skeleton(FP) returns a csr_array of shape (n_faces, n_points).
    """
    mesh = graphlow.read(filename, "phlower")
    topo = mesh.topology
    fp = topo.get_skeleton(IncidenceName.FP)
    pf = topo.get_skeleton(IncidenceName.PF)
    np.testing.assert_array_equal(fp.toarray(), expected)
    np.testing.assert_array_equal(pf.toarray(), expected.T)

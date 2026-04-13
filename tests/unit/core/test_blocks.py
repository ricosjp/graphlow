"""Unit tests for graphlow.core.blocks (CellBlock and FaceBlock types)."""

import numpy as np
import pyvista as pv

from graphlow.core.blocks import (
    FixedCellBlock,
    FixedFaceBlock,
    JaggedCellBlock,
    JaggedFaceBlock,
)
from graphlow.utils.topology_helper import TopologyDim


# =============================================================================
# CellBlock / FixedCellBlock
# =============================================================================
class TestFixedCellBlock:
    def test_n_cells(self):
        """FixedCellBlock.n_cells returns length of global_indices."""
        conn = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.int64)
        block = FixedCellBlock(
            cell_type=pv.CellType.TRIANGLE,
            global_indices=np.array([0, 1, 2]),
            conn=conn,
        )
        assert block.n_cells == 3

    def test_topological_dimension(self):
        """FixedCellBlock.topological_dimension returns SURFACE for TRIANGLE."""
        conn = np.array([[0, 1, 2]], dtype=np.int64)
        block = FixedCellBlock(
            cell_type=pv.CellType.TRIANGLE,
            global_indices=np.array([0]),
            conn=conn,
        )
        assert block.topological_dimension == TopologyDim.SURFACE

    def test_conn_shape(self):
        """FixedCellBlock.conn has shape (n_cells, k)."""
        conn = np.zeros((5, 4), dtype=np.int64)
        block = FixedCellBlock(
            cell_type=pv.CellType.QUAD,
            global_indices=np.arange(5),
            conn=conn,
        )
        assert block.conn.shape == (5, 4)


# =============================================================================
# JaggedCellBlock
# =============================================================================
class TestJaggedCellBlock:
    def test_n_cells(self):
        """JaggedCellBlock.n_cells equals len(offsets)-1."""
        conn = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
        offsets = np.array([0, 3, 5, 7], dtype=np.int64)
        block = JaggedCellBlock(
            cell_type=pv.CellType.POLYGON,
            global_indices=np.array([0, 1, 2]),
            conn=conn,
            offsets=offsets,
        )
        assert block.n_cells == 3

    def test_topological_dimension(self):
        """JaggedCellBlock.topological_dimension is VOLUME for POLYHEDRON."""
        block = JaggedCellBlock(
            cell_type=pv.CellType.POLYHEDRON,
            global_indices=np.array([0]),
            conn=np.zeros(0, dtype=np.int64),
            offsets=np.array([0], dtype=np.int64),
        )
        assert block.topological_dimension == TopologyDim.VOLUME


# =============================================================================
# FaceBlock / FixedFaceBlock
# =============================================================================
class TestFixedFaceBlock:
    def test_n_faces(self):
        """FixedFaceBlock.n_faces returns length of global_indices."""
        block = FixedFaceBlock(
            cell_type=pv.CellType.TRIANGLE,
            global_indices=np.array([0, 1, 2]),
            owner=np.array([0, 0, 1]),
            neighbor=np.array([1, -1, -1]),
            conn=np.zeros((3, 3), dtype=np.int64),
        )
        assert block.n_faces == 3

    def test_topological_dimension(self):
        """FixedFaceBlock.topological_dimension returns SURFACE for QUAD."""
        block = FixedFaceBlock(
            cell_type=pv.CellType.QUAD,
            global_indices=np.array([0]),
            owner=np.array([0]),
            neighbor=np.array([-1]),
            conn=np.zeros((1, 4), dtype=np.int64),
        )
        assert block.topological_dimension == TopologyDim.SURFACE

    def test_boundary_faces_returns_only_neighbor_minus_one(self):
        """boundary_faces() keeps only faces with neighbor == -1."""
        block = FixedFaceBlock(
            cell_type=pv.CellType.TRIANGLE,
            global_indices=np.array([0, 1, 2]),
            owner=np.array([0, 0, 1]),
            neighbor=np.array([1, -1, -1]),
            conn=np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.int64),
        )
        bnd = block.boundary_faces()
        assert bnd.n_faces == 2
        np.testing.assert_array_equal(bnd.neighbor, [-1, -1])
        np.testing.assert_array_equal(bnd.global_indices, [1, 2])
        np.testing.assert_array_equal(bnd.owner, [0, 1])
        np.testing.assert_array_equal(bnd.conn, [[1, 2, 3], [2, 3, 4]])

    def test_boundary_faces_all_boundary_returns_self_sized(self):
        """boundary_faces() with all boundary returns same face count."""
        block = FixedFaceBlock(
            cell_type=pv.CellType.TRIANGLE,
            global_indices=np.array([0, 1]),
            owner=np.array([0, 0]),
            neighbor=np.array([-1, -1]),
            conn=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        )
        bnd = block.boundary_faces()
        assert bnd.n_faces == 2
        np.testing.assert_array_equal(bnd.conn, block.conn)

    def test_boundary_faces_no_boundary_returns_empty(self):
        """boundary_faces() when no boundary returns empty block."""
        block = FixedFaceBlock(
            cell_type=pv.CellType.TRIANGLE,
            global_indices=np.array([0]),
            owner=np.array([0]),
            neighbor=np.array([1]),
            conn=np.array([[0, 1, 2]], dtype=np.int64),
        )
        bnd = block.boundary_faces()
        assert bnd.n_faces == 0
        assert bnd.conn.shape[0] == 0


# =============================================================================
# JaggedFaceBlock
# =============================================================================
class TestJaggedFaceBlock:
    def test_n_faces(self):
        """JaggedFaceBlock.n_faces equals len(offsets)-1."""
        block = JaggedFaceBlock(
            cell_type=pv.CellType.POLYGON,
            global_indices=np.array([0, 1]),
            owner=np.array([0, 0]),
            neighbor=np.array([-1, 1]),
            conn=np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64),
            offsets=np.array([0, 3, 7], dtype=np.int64),
        )
        assert block.n_faces == 2

    def test_topological_dimension(self):
        """JaggedFaceBlock.topological_dimension returns SURFACE for POLYGON."""
        block = JaggedFaceBlock(
            cell_type=pv.CellType.POLYGON,
            global_indices=np.array([0]),
            owner=np.array([0]),
            neighbor=np.array([-1]),
            conn=np.array([0, 1, 2], dtype=np.int64),
            offsets=np.array([0, 3], dtype=np.int64),
        )
        assert block.topological_dimension == TopologyDim.SURFACE

    def test_boundary_faces_returns_only_neighbor_minus_one(self):
        """boundary_faces() keeps only faces with neighbor == -1."""
        # Face 0: 3 pts (bnd), 1: 4 pts (internal), 2: 3 pts (bnd)
        conn = np.array([0, 1, 2, 10, 11, 12, 13, 20, 21, 22], dtype=np.int64)
        offsets = np.array([0, 3, 7, 10], dtype=np.int64)
        block = JaggedFaceBlock(
            cell_type=pv.CellType.POLYGON,
            global_indices=np.array([0, 1, 2]),
            owner=np.array([0, 0, 1]),
            neighbor=np.array([-1, 1, -1]),
            conn=conn,
            offsets=offsets,
        )
        bnd = block.boundary_faces()
        assert bnd.n_faces == 2
        np.testing.assert_array_equal(bnd.neighbor, [-1, -1])
        np.testing.assert_array_equal(bnd.global_indices, [0, 2])
        np.testing.assert_array_equal(bnd.owner, [0, 1])
        # Kept faces: first (pts 0,1,2) and third (pts 20,21,22)
        np.testing.assert_array_equal(bnd.conn, [0, 1, 2, 20, 21, 22])
        np.testing.assert_array_equal(bnd.offsets, [0, 3, 6])

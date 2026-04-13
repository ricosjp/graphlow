import numpy as np
import pytest
import pyvista as pv

from graphlow.core.blocks import FixedFaceBlock, JaggedFaceBlock
from graphlow.core.face_registry import (
    FaceRegistry,
    FaceRegistryBuilder,
    build_face_blocks_from_registry,
    build_face_registry,
)

# =============================================================================
# Face Registry
# =============================================================================


def test_face_registry_builder_create():
    """FaceRegistryBuilder.create() returns empty builder."""
    reg = FaceRegistryBuilder.create()
    assert isinstance(reg, FaceRegistryBuilder)
    assert len(reg.owner) == 0
    assert len(reg.key_to_gid) == 0


class TestRegisterSingleFace:
    def test_triangle(self):
        """Register one triangle face: one face, boundary (neighbor == -1)."""
        reg = FaceRegistryBuilder.create()
        point_ids = np.array([4, 7, 8], dtype=np.int64)
        reg.register_face(pv.CellType.TRIANGLE, owner=0, point_ids=point_ids)
        registry = reg.finalize()
        assert registry.n_faces() == 1
        np.testing.assert_array_equal(registry.owner, [0])
        np.testing.assert_array_equal(registry.neighbor, [-1])
        np.testing.assert_array_equal(registry.boundary_mask(), [True])
        np.testing.assert_array_equal(registry.tri_gids, [0])
        np.testing.assert_array_equal(registry.tri_conn, [point_ids])

    def test_quad(self):
        """Register one quad face: one face, boundary (neighbor == -1)."""
        reg = FaceRegistryBuilder.create()
        point_ids = np.array([3, 12, 13, 20], dtype=np.int64)
        reg.register_face(pv.CellType.QUAD, owner=3, point_ids=point_ids)
        registry = reg.finalize()
        assert registry.n_faces() == 1
        np.testing.assert_array_equal(registry.owner, [3])
        np.testing.assert_array_equal(registry.neighbor, [-1])
        np.testing.assert_array_equal(registry.boundary_mask(), [True])
        np.testing.assert_array_equal(registry.quad_gids, [0])
        np.testing.assert_array_equal(registry.quad_conn, [point_ids])

    def test_pixel(self):
        """Register one pixel face: one face, boundary (neighbor == -1)."""
        reg = FaceRegistryBuilder.create()
        point_ids = np.array([0, 1, 2, 3], dtype=np.int64)
        reg.register_face(pv.CellType.PIXEL, owner=0, point_ids=point_ids)
        registry = reg.finalize()
        assert registry.n_faces() == 1
        np.testing.assert_array_equal(registry.owner, [0])
        np.testing.assert_array_equal(registry.neighbor, [-1])
        np.testing.assert_array_equal(registry.boundary_mask(), [True])
        np.testing.assert_array_equal(registry.pixel_gids, [0])
        np.testing.assert_array_equal(registry.pixel_conn, [point_ids])

    def test_polygon(self):
        """Register one polygon face: one face, boundary (neighbor == -1)."""
        reg = FaceRegistryBuilder.create()
        point_ids = np.array([10, 11, 20, 21, 22, 23], dtype=np.int64)
        reg.register_face(pv.CellType.POLYGON, owner=10, point_ids=point_ids)
        registry = reg.finalize()
        assert registry.n_faces() == 1
        np.testing.assert_array_equal(registry.owner, [10])
        np.testing.assert_array_equal(registry.neighbor, [-1])
        np.testing.assert_array_equal(registry.boundary_mask(), [True])
        np.testing.assert_array_equal(registry.poly_gids, [0])
        np.testing.assert_array_equal(registry.poly_conn, point_ids)

    def test_unsupported_face_type_raises(self):
        """Register unsupported face type raises ValueError."""
        reg = FaceRegistryBuilder.create()
        point_ids = np.array([0, 1, 2, 3], dtype=np.int64)
        with pytest.raises(ValueError, match="Unsupported face type"):
            reg.register_face(pv.CellType.TETRA, owner=0, point_ids=point_ids)


class TestRegisterMultipleFaces:
    """Register same face from two cells."""

    def test_triangle(self):
        """Register same triangle face from two cells."""
        reg = FaceRegistryBuilder.create()
        point_ids = np.array([7, 4, 8], dtype=np.int64)
        reg.register_face(pv.CellType.TRIANGLE, owner=0, point_ids=point_ids)
        reg.register_face(pv.CellType.TRIANGLE, owner=1, point_ids=point_ids)
        registry = reg.finalize()
        assert registry.n_faces() == 1
        np.testing.assert_array_equal(registry.owner, [0])
        np.testing.assert_array_equal(registry.neighbor, [1])
        np.testing.assert_array_equal(registry.boundary_mask(), [False])
        np.testing.assert_array_equal(registry.tri_gids, [0])
        np.testing.assert_array_equal(registry.tri_conn, [point_ids])

    def test_quad(self):
        """Register same quad face from two cells."""
        reg = FaceRegistryBuilder.create()
        point_ids = np.array([3, 12, 13, 20], dtype=np.int64)
        reg.register_face(pv.CellType.QUAD, owner=3, point_ids=point_ids)
        reg.register_face(pv.CellType.QUAD, owner=4, point_ids=point_ids)
        registry = reg.finalize()
        assert registry.n_faces() == 1
        np.testing.assert_array_equal(registry.owner, [3])
        np.testing.assert_array_equal(registry.neighbor, [4])
        np.testing.assert_array_equal(registry.boundary_mask(), [False])
        np.testing.assert_array_equal(registry.quad_gids, [0])
        np.testing.assert_array_equal(registry.quad_conn, [point_ids])

    def test_pixel(self):
        """Register same pixel face from two cells."""
        reg = FaceRegistryBuilder.create()
        point_ids = np.array([0, 1, 2, 3], dtype=np.int64)
        reg.register_face(pv.CellType.PIXEL, owner=0, point_ids=point_ids)
        reg.register_face(pv.CellType.PIXEL, owner=1, point_ids=point_ids)
        registry = reg.finalize()
        assert registry.n_faces() == 1
        np.testing.assert_array_equal(registry.owner, [0])
        np.testing.assert_array_equal(registry.neighbor, [1])
        np.testing.assert_array_equal(registry.boundary_mask(), [False])
        np.testing.assert_array_equal(registry.pixel_gids, [0])
        np.testing.assert_array_equal(registry.pixel_conn, [point_ids])

    def test_polygon(self):
        """Register same polygon face from two cells."""
        reg = FaceRegistryBuilder.create()
        point_ids = np.array([20, 21, 22, 23, 10, 11], dtype=np.int64)
        reg.register_face(pv.CellType.POLYGON, owner=10, point_ids=point_ids)
        reg.register_face(pv.CellType.POLYGON, owner=11, point_ids=point_ids)
        registry = reg.finalize()
        assert registry.n_faces() == 1
        np.testing.assert_array_equal(registry.owner, [10])
        np.testing.assert_array_equal(registry.neighbor, [11])
        np.testing.assert_array_equal(registry.boundary_mask(), [False])
        np.testing.assert_array_equal(registry.poly_gids, [0])
        np.testing.assert_array_equal(registry.poly_conn, point_ids)

    def test_mixed_tri_quad_polygon(self):
        """Register mixed face types: tri, quad, polygon."""
        reg = FaceRegistryBuilder.create()
        tri_pts = np.array([0, 1, 2], dtype=np.int64)
        quad_pts = np.array([3, 4, 5, 6], dtype=np.int64)
        poly_pts = np.array([10, 11, 12, 13, 14], dtype=np.int64)
        reg.register_face(pv.CellType.TRIANGLE, owner=0, point_ids=tri_pts)
        reg.register_face(pv.CellType.QUAD, owner=0, point_ids=quad_pts)
        reg.register_face(pv.CellType.POLYGON, owner=0, point_ids=poly_pts)
        registry = reg.finalize()
        assert registry.n_faces() == 3
        assert len(registry.tri_gids) == 1
        assert len(registry.quad_gids) == 1
        assert len(registry.poly_gids) == 1
        np.testing.assert_array_equal(registry.tri_conn, [tri_pts])
        np.testing.assert_array_equal(registry.quad_conn, [quad_pts])
        np.testing.assert_array_equal(registry.poly_conn, poly_pts)
        np.testing.assert_array_equal(registry.poly_offsets, [0, 5])

    def test_key_uses_sorted_vertex_order(self):
        """
        Key for triangle is sorted(point_ids)
        (0,1,2) and (2,1,0) are the same face.
        """
        reg = FaceRegistryBuilder.create()
        tri = np.array([0, 1, 2], dtype=np.int64)
        reg.register_face(pv.CellType.TRIANGLE, owner=0, point_ids=tri)
        rev = np.array([2, 1, 0], dtype=np.int64)
        reg.register_face(pv.CellType.TRIANGLE, owner=1, point_ids=rev)
        registry = reg.finalize()
        assert registry.n_faces() == 1
        np.testing.assert_array_equal(registry.owner, [0])
        np.testing.assert_array_equal(registry.neighbor, [1])
        np.testing.assert_array_equal(registry.boundary_mask(), [False])
        np.testing.assert_array_equal(registry.tri_gids, [0])
        np.testing.assert_array_equal(registry.tri_conn, [tri])


# =============================================================================
# Face Registry Build (Public API)
# =============================================================================


# fmt: off
@pytest.mark.parametrize(
    "expected_registry",
    [
        (
            FaceRegistry(
                owner=np.array(
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                    dtype=np.int64
                ),
                neighbor=np.array(
                    [-1, 1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    dtype=np.int64,
                ),  # 2 internal faces, 12 boundary faces
                face_type=np.array([
                    pv.CellType.POLYGON,
                    pv.CellType.POLYGON,
                    pv.CellType.POLYGON,
                    pv.CellType.POLYGON,
                    pv.CellType.POLYGON,
                    pv.CellType.POLYGON,
                    pv.CellType.QUAD,
                    pv.CellType.QUAD,
                    pv.CellType.QUAD,
                    pv.CellType.QUAD,
                    pv.CellType.QUAD,
                    pv.CellType.TRIANGLE,
                    pv.CellType.TRIANGLE,
                    pv.CellType.TRIANGLE,
                ]),
                tri_gids=np.array([11, 12, 13], dtype=np.int64),
                tri_conn=np.array(
                    [
                        [9, 10, 12],
                        [10, 11, 12],
                        [11, 9, 12],
                    ],
                    dtype=np.int64
                ),
                quad_gids=np.array([6, 7, 8, 9, 10], dtype=np.int64),
                quad_conn=np.array(
                    [
                        [2, 4, 7, 6],
                        [1, 2, 6, 5],
                        [3, 8, 7, 4],
                        [1, 3, 4, 2],
                        [5, 6, 7, 8],
                    ],
                    dtype=np.int64
                ),
                pixel_gids=np.array([], dtype=np.int64),
                pixel_conn=np.array([], dtype=np.int64),
                poly_gids=np.array([0, 1, 2, 3, 4, 5], dtype=np.int64),
                poly_conn=np.array(
                    [
                        0, 1, 5, 10, 9,
                        1, 3, 8, 5,
                        0, 9, 11, 8, 3,
                        5, 8, 11, 10,
                        9, 10, 11,
                        0, 3, 1
                    ],
                    dtype=np.int64
                ),
                poly_offsets=np.array(
                    [0, 5, 9, 14, 18, 21, 24],
                    dtype=np.int64
                ),
            )
        ),
    ],
)
# fmt: on
def test_build_face_registry(
    mix_poly_grid: pv.UnstructuredGrid,
    expected_registry: FaceRegistry,
) -> None:
    """
    build_face_registry collects all 3D cell faces from the grid; mix_poly
    has one polyhedron (6 faces), one hex (6 quads), one tet (4 tris).
    After deduplication we get 14 unique faces: 5 tri, 7 quad, 2 poly.
    """
    registry = build_face_registry(mix_poly_grid)
    # Compare registry with expected_registry
    assert registry.n_faces() == 14
    np.testing.assert_array_equal(registry.owner, expected_registry.owner)
    np.testing.assert_array_equal(registry.neighbor, expected_registry.neighbor)
    np.testing.assert_array_equal(
        registry.face_type, expected_registry.face_type
    )
    np.testing.assert_array_equal(registry.tri_gids, expected_registry.tri_gids)
    np.testing.assert_array_equal(registry.tri_conn, expected_registry.tri_conn)
    np.testing.assert_array_equal(
        registry.quad_gids, expected_registry.quad_gids
    )
    np.testing.assert_array_equal(
        registry.quad_conn, expected_registry.quad_conn
    )
    np.testing.assert_array_equal(
        registry.pixel_gids, expected_registry.pixel_gids
    )
    np.testing.assert_array_equal(
        registry.pixel_conn, expected_registry.pixel_conn
    )
    np.testing.assert_array_equal(
        registry.poly_gids, expected_registry.poly_gids
    )
    np.testing.assert_array_equal(
        registry.poly_conn, expected_registry.poly_conn
    )
    np.testing.assert_array_equal(
        registry.poly_offsets, expected_registry.poly_offsets
    )


def test_build_face_blocks_from_registry(
    mix_poly_grid: pv.UnstructuredGrid,
) -> None:
    """
    build_face_blocks_from_registry converts a FaceRegistry into a dict of
    FaceBlock by type; mix_poly yields TRIANGLE (FixedFaceBlock), QUAD
    (FixedFaceBlock), and POLYGON (JaggedFaceBlock) with expected lengths.
    """
    registry = build_face_registry(mix_poly_grid)
    blocks = build_face_blocks_from_registry(registry)
    assert pv.CellType.TRIANGLE in blocks
    assert pv.CellType.QUAD in blocks
    assert pv.CellType.PIXEL not in blocks
    assert pv.CellType.POLYGON in blocks

    tri_block = blocks[pv.CellType.TRIANGLE]
    assert isinstance(tri_block, FixedFaceBlock)
    assert tri_block.cell_type == pv.CellType.TRIANGLE
    assert tri_block.n_faces == 3
    np.testing.assert_array_equal(tri_block.global_indices, registry.tri_gids)
    np.testing.assert_array_equal(
        tri_block.owner, registry.owner[registry.tri_gids]
    )
    np.testing.assert_array_equal(
        tri_block.neighbor, registry.neighbor[registry.tri_gids]
    )
    np.testing.assert_array_equal(tri_block.conn, registry.tri_conn)

    quad_block = blocks[pv.CellType.QUAD]
    assert isinstance(quad_block, FixedFaceBlock)
    assert quad_block.cell_type == pv.CellType.QUAD
    assert quad_block.n_faces == 5
    np.testing.assert_array_equal(quad_block.global_indices, registry.quad_gids)
    np.testing.assert_array_equal(
        quad_block.owner, registry.owner[registry.quad_gids]
    )
    np.testing.assert_array_equal(
        quad_block.neighbor, registry.neighbor[registry.quad_gids]
    )
    np.testing.assert_array_equal(quad_block.conn, registry.quad_conn)

    poly_block = blocks[pv.CellType.POLYGON]
    assert isinstance(poly_block, JaggedFaceBlock)
    assert poly_block.cell_type == pv.CellType.POLYGON
    assert poly_block.n_faces == 6
    np.testing.assert_array_equal(poly_block.global_indices, registry.poly_gids)
    np.testing.assert_array_equal(
        poly_block.owner, registry.owner[registry.poly_gids]
    )
    np.testing.assert_array_equal(
        poly_block.neighbor, registry.neighbor[registry.poly_gids]
    )
    np.testing.assert_array_equal(poly_block.conn, registry.poly_conn)
    np.testing.assert_array_equal(poly_block.offsets, registry.poly_offsets)

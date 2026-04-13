"""Tests for surface geometry helpers on surface and volume meshes."""

import pathlib
from unittest.mock import patch

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
    Voxel,
    Wedge,
)

from graphlow import FloatPrecision, from_pyvista
from graphlow.geometry import surface


# =============================================================================
# Area
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
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
        pathlib.Path("tests/data/vtu/cube/large.vtu"),
    ],
)
class TestFaceAreas:
    """Tests for ``face_areas`` on extracted surfaces and full face sets."""

    def test_2d(self, filename: pathlib.Path):
        """Surface mesh: ``face_areas`` matches PyVista cell areas."""
        pv_volume_mesh: pv.UnstructuredGrid = pv.read(filename)
        pv_surface_mesh = pv_volume_mesh.extract_surface(algorithm=None)
        surface_mesh = from_pyvista(
            pv_surface_mesh, "phlower", FloatPrecision.FLOAT64
        )

        with patch.object(
            surface,
            "_compute_2d_surface",
            wraps=surface._compute_2d_surface,
        ) as mock_fn:
            face_areas = surface_mesh.geometry.face_areas()

        assert mock_fn.call_count == 1
        assert face_areas.dimension == phlower_dimension_tensor({"L": 2})

        actual = face_areas.to_tensor().numpy()
        expected = (
            pv_surface_mesh.compute_cell_sizes()
            .cell_data["Area"]
            .reshape(-1, 1)
        )
        np.testing.assert_almost_equal(actual, expected)

    def test_3d_all_faces(self, filename: pathlib.Path):
        """Volume mesh: ``face_areas`` matches all extracted face areas."""
        pv_volume_mesh: pv.UnstructuredGrid = pv.read(filename)
        volume_mesh = from_pyvista(
            pv_volume_mesh, "phlower", FloatPrecision.FLOAT64
        )

        with patch.object(
            surface,
            "_compute_3d_face",
            wraps=surface._compute_3d_face,
        ) as mock_fn:
            face_areas = volume_mesh.geometry.face_areas()

        assert mock_fn.call_count == 1
        assert face_areas.dimension == phlower_dimension_tensor({"L": 2})

        actual = face_areas.to_tensor().numpy()
        pv_facets_mesh = _extract_all_facets(pv_volume_mesh)
        expected = (
            pv_facets_mesh.compute_cell_sizes().cell_data["Area"].reshape(-1, 1)
        )
        np.testing.assert_almost_equal(actual, expected)


# =============================================================================
# Normals
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
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
        pathlib.Path("tests/data/vtu/cube/large.vtu"),
    ],
)
class TestFaceNormals:
    """Tests for ``face_normals`` on extracted surfaces and full face sets."""

    def test_2d(self, filename: pathlib.Path, test_device: torch.device):
        """Surface mesh: ``face_normals`` matches PyVista cell normals."""
        pv_volume_mesh: pv.UnstructuredGrid = pv.read(filename)
        pv_surface_mesh = pv_volume_mesh.extract_surface(algorithm=None)
        surface_mesh = from_pyvista(
            pv_surface_mesh,
            "phlower",
            FloatPrecision.FLOAT64,
            device=test_device,
        )

        with patch.object(
            surface,
            "_compute_2d_surface",
            wraps=surface._compute_2d_surface,
        ) as mock_fn:
            face_normals = surface_mesh.geometry.face_normals()

        assert mock_fn.call_count == 1
        assert face_normals.dimension == phlower_dimension_tensor(
            {"L": 0}, device=test_device
        )

        actual = surface_mesh.backend.to_numpy(face_normals)
        expected = (
            pv_surface_mesh.compute_normals(consistent_normals=False)
            .cell_data["Normals"]
            .reshape(-1, 3)
        )
        np.testing.assert_almost_equal(actual, expected)

    def test_3d_all_faces(self, filename: pathlib.Path):
        """Volume mesh: ``face_normals`` matches all extracted face normals."""
        pv_volume_mesh: pv.UnstructuredGrid = pv.read(filename)
        volume_mesh = from_pyvista(
            pv_volume_mesh, "phlower", FloatPrecision.FLOAT64
        )

        with patch.object(
            surface,
            "_compute_3d_face",
            wraps=surface._compute_3d_face,
        ) as mock_fn:
            face_normals = volume_mesh.geometry.face_normals()

        assert mock_fn.call_count == 1
        assert face_normals.dimension == phlower_dimension_tensor({"L": 0})

        actual = face_normals.to_tensor().numpy()
        pv_facets_mesh = _extract_all_facets(pv_volume_mesh)
        expected = (
            pv_facets_mesh.compute_normals(consistent_normals=False)
            .cell_data["Normals"]
            .reshape(-1, 3)
        )
        np.testing.assert_almost_equal(actual, expected)


# =============================================================================
# Centroids
# =============================================================================


class TestFaceCentroids:
    """Tests for ``face_centroids`` on primitive and derived surface meshes."""

    @pytest.mark.parametrize(
        "grid, expected",
        [
            (
                Tetrahedron(),
                (np.sqrt(2) / 12)
                * np.array([[1, 1, -1], [-1, -1, -1], [1, -1, 1], [-1, 1, 1]]),
            ),
            (
                Pyramid(),
                (1 / 3)
                * np.array(
                    [
                        [1, 0, 1 / np.sqrt(2)],
                        [0, 1, 1 / np.sqrt(2)],
                        [0.0, 0.0, 0.0],
                        [-1, 0, 1 / np.sqrt(2)],
                        [0, -1, 1 / np.sqrt(2)],
                    ]
                ),
            ),
            (
                Wedge(),
                np.array(
                    [
                        [1 / 2, 3 / 4, np.sqrt(3) / 4],
                        [1 / 2, 1 / 2, 0],
                        [0, 1 / 2, np.sqrt(3) / 6],
                        [1 / 2, 1 / 4, np.sqrt(3) / 4],
                        [1, 1 / 2, np.sqrt(3) / 6],
                    ]
                ),
            ),
            (
                Hexahedron(),
                np.array(
                    [
                        [1 / 2, 1 / 2, 0],
                        [1 / 2, 0, 1 / 2],
                        [0, 1 / 2, 1 / 2],
                        [1, 1 / 2, 1 / 2],
                        [1 / 2, 1, 1 / 2],
                        [1 / 2, 1 / 2, 1],
                    ]
                ),
            ),
            (
                Voxel(),
                np.array(
                    [
                        [1 / 2, 1 / 2, 0],
                        [1 / 2, 0, 1 / 2],
                        [0, 1 / 2, 1 / 2],
                        [1, 1 / 2, 1 / 2],
                        [1 / 2, 1, 1 / 2],
                        [1 / 2, 1 / 2, 1],
                    ]
                ),
            ),
            (
                Polyhedron(),
                np.array(
                    [
                        [1 / 6, 1 / 6, 1 / 3],
                        [1 / 3, 0, 1 / 3],
                        [1 / 2, 1 / 6, 0],
                        [1 / 2, 1 / 6, 1 / 3],
                    ]
                ),
            ),
        ],
    )
    def test_2d(self, grid: pv.UnstructuredGrid, expected: np.ndarray):
        """Surface mesh: ``face_centroids`` matches the analytic reference."""
        pv_surface_mesh = grid.extract_surface(algorithm=None)
        surface_mesh = from_pyvista(
            pv_surface_mesh, "phlower", FloatPrecision.FLOAT64
        )

        with patch.object(
            surface,
            "_compute_2d_surface",
            wraps=surface._compute_2d_surface,
        ) as mock_fn:
            face_centroids = surface_mesh.geometry.face_centroids()

        assert mock_fn.call_count == 1
        assert face_centroids.dimension == phlower_dimension_tensor({"L": 1})

        actual = face_centroids.to_tensor().numpy()
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.parametrize(
        "grid, expected",
        [
            (
                Tetrahedron(),
                (np.sqrt(2) / 12)
                * np.array([[1, -1, 1], [-1, 1, 1], [-1, -1, -1], [1, 1, -1]]),
            ),
            (
                Pyramid(),
                (1 / 3)
                * np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0, 1, 1 / np.sqrt(2)],
                        [-1, 0, 1 / np.sqrt(2)],
                        [0, -1, 1 / np.sqrt(2)],
                        [1, 0, 1 / np.sqrt(2)],
                    ]
                ),
            ),
            (
                Wedge(),
                np.array(
                    [
                        [0, 1 / 2, np.sqrt(3) / 6],
                        [1, 1 / 2, np.sqrt(3) / 6],
                        [1 / 2, 1 / 2, 0],
                        [1 / 2, 1 / 4, np.sqrt(3) / 4],
                        [1 / 2, 3 / 4, np.sqrt(3) / 4],
                    ]
                ),
            ),
            (
                Hexahedron(),
                np.array(
                    [
                        [0, 1 / 2, 1 / 2],
                        [1, 1 / 2, 1 / 2],
                        [1 / 2, 0, 1 / 2],
                        [1 / 2, 1, 1 / 2],
                        [1 / 2, 1 / 2, 0],
                        [1 / 2, 1 / 2, 1],
                    ]
                ),
            ),
            (
                Voxel(),
                np.array(
                    [
                        [0, 1 / 2, 1 / 2],
                        [1, 1 / 2, 1 / 2],
                        [1 / 2, 0, 1 / 2],
                        [1 / 2, 1, 1 / 2],
                        [1 / 2, 1 / 2, 0],
                        [1 / 2, 1 / 2, 1],
                    ]
                ),
            ),
            (
                Polyhedron(),
                np.array(
                    [
                        [1 / 2, 1 / 6, 0],
                        [1 / 3, 0, 1 / 3],
                        [1 / 6, 1 / 6, 1 / 3],
                        [1 / 2, 1 / 6, 1 / 3],
                    ]
                ),
            ),
        ],
    )
    def test_3d_all_faces(
        self, grid: pv.UnstructuredGrid, expected: np.ndarray
    ):
        volume_mesh = from_pyvista(grid, "phlower", FloatPrecision.FLOAT64)

        with patch.object(
            surface,
            "_compute_3d_face",
            wraps=surface._compute_3d_face,
        ) as mock_fn:
            face_centroids = volume_mesh.geometry.face_centroids()

        assert mock_fn.call_count == 1
        assert face_centroids.dimension == phlower_dimension_tensor({"L": 1})

        actual = face_centroids.to_tensor().numpy()
        np.testing.assert_almost_equal(actual, expected)


# =============================================================================
# Helper functions
# =============================================================================
def _extract_all_facets(grid: pv.UnstructuredGrid) -> pv.PolyData:
    """
    Extract all faces including internal faces from a volume mesh.

    Returns
    -------
    pyvista.PolyData
        PolyData with all internal/external faces registered as cells
    """
    polygon_cells = []
    n_cells = grid.n_cells
    facet_keys = set()

    for cell_id in range(n_cells):
        cell = grid.get_cell(cell_id)
        for j in range(cell.n_faces):
            face = cell.get_face(j)
            face_conn = face.point_ids.copy()
            key = tuple(np.sort(face_conn))

            # make PIXEL cell to QUAD cell
            if face.type == pv.CellType.PIXEL:
                face_conn[2], face_conn[3] = face_conn[3], face_conn[2]

            if key not in facet_keys:
                polygon_cells.extend([len(face_conn), *face_conn])
                facet_keys.add(key)
    return pv.PolyData(grid.points, polygon_cells)


# =============================================================================
# Tests for helper functions
# =============================================================================


# fmt: off
@pytest.mark.parametrize(
    "filename, expected_facets",
    [
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            np.array(
                [
                    4, 0, 4, 7, 3,    #  0
                    4, 1, 2, 6, 5,    #  1
                    4, 0, 1, 5, 4,    #  2
                    4, 3, 7, 6, 2,    #  3
                    4, 0, 3, 2, 1,    #  4
                    4, 4, 5, 6, 7,    #  5
                    4, 4, 8, 11, 7,   #  6
                    4, 5, 6, 10, 9,   #  7
                    4, 4, 5, 9, 8,    #  8
                    4, 7, 11, 10, 6,  #  9
                    4, 8, 9, 10, 11,  # 10
                ]
            ),
        ),
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            np.array(
                [
                    5, 0, 1, 5, 10, 9,  #  0
                    4, 1, 3, 8, 5,      #  1
                    5, 0, 9, 11, 8, 3,  #  2
                    4, 5, 8, 11, 10,    #  3
                    3, 9, 10, 11,       #  4
                    3, 0, 3, 1,         #  5
                    4, 2, 4, 7, 6,      #  6
                    4, 1, 2, 6, 5,      #  7
                    4, 3, 8, 7, 4,      #  8
                    4, 1, 3, 4, 2,      #  9
                    4, 5, 6, 7, 8,      # 10
                    3, 9, 10, 12,       # 11
                    3, 10, 11, 12,      # 12
                    3, 11, 9, 12,       # 13
                ]
            ),
        ),
        (
            pathlib.Path("tests/data/vtu/primitive_cell/voxel.vtu"),
            np.array(
                [
                    4, 2, 0, 4, 6,    #  0
                    4, 1, 3, 7, 5,    #  1
                    4, 0, 1, 5, 4,    #  2
                    4, 3, 2, 6, 7,    #  3
                    4, 1, 0, 2, 3,    #  4
                    4, 4, 5, 7, 6,    #  5
                ]
            ),
        )
    ],
)
# fmt: on
def test_extract_all_facets(
    filename: pathlib.Path,
    expected_facets: np.ndarray,
):
    grid = pv.read(filename)
    facets_grid = _extract_all_facets(grid)
    np.testing.assert_array_equal(facets_grid.faces, expected_facets)

"""Tests for geometry operators such as isoAM and its helper routines."""

import pathlib
from collections.abc import Callable
from unittest.mock import patch

import numpy as np
import pytest
import pyvista as pv
import torch

import graphlow
from graphlow.core.backend.phlower import PhlowerBackend
from graphlow.geometry.operator import (
    _compute_moment_matrix,
    _compute_normals_on_surface_points,
    _compute_rawAM_and_moment_inv,
    _create_grad_operator_from,
)


# =============================================================================
# IsoAM
# =============================================================================
class TestComputeIsoAM:
    @pytest.mark.parametrize(
        "filename, expected",
        [
            (
                pathlib.Path("tests/data/vtu/primitive_cell/tet.vtu"),
                np.array(
                    [
                        [
                            [-1.0, 1.0, 0.0, 0.0],  #  0
                            [-1.0, 2.0, -0.5, -0.5],  #  1
                            [0.0, 0.5, -0.5, 0.0],  #  2
                            [0.0, 0.5, 0.0, -0.5],  #  3
                        ],
                        [
                            [-1.0, 0.0, 1.0, 0.0],  #  0
                            [0.0, -0.5, 0.5, 0.0],  #  1
                            [-1.0, -0.5, 2.0, -0.5],  #  2
                            [0.0, 0.0, 0.5, -0.5],  #  3
                        ],
                        [
                            [-1.0, 0.0, 0.0, 1.0],  #  0
                            [0.0, -0.5, 0.0, 0.5],  #  1
                            [0.0, 0.0, -0.5, 0.5],  #  2
                            [-1.0, -0.5, -0.5, 2.0],  #  3
                        ],
                    ]
                ),
            )
        ],
    )
    def test_compute_isoAM_without_moment_matrix(
        self,
        filename: pathlib.Path,
        expected: np.ndarray,
        test_device: torch.device,
    ):
        mesh = graphlow.read(
            filename, "phlower", dtype=torch.float64, device=test_device
        )
        grad_adjs, _ = mesh.geometry.isoAM(with_moment_matrix=False)
        actual = grad_adjs.to_tensor().cpu().to_dense().numpy()
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.parametrize(
        "filename, expected",
        [
            (
                pathlib.Path("tests/data/vtu/primitive_cell/tet.vtu"),
                np.array(
                    [
                        [
                            [-1.0, 1.0, 0.0, 0.0],  #  0
                            [-1.0, 2.0, -0.5, -0.5],  #  1
                            [0.0, 0.5, -0.5, 0.0],  #  2
                            [0.0, 0.5, 0.0, -0.5],  #  3
                        ],
                        [
                            [-1.0, 0.0, 1.0, 0.0],  #  0
                            [0.0, -0.5, 0.5, 0.0],  #  1
                            [-1.0, -0.5, 2.0, -0.5],  #  2
                            [0.0, 0.0, 0.5, -0.5],  #  3
                        ],
                        [
                            [-1.0, 0.0, 0.0, 1.0],  #  0
                            [0.0, -0.5, 0.0, 0.5],  #  1
                            [0.0, 0.0, -0.5, 0.5],  #  2
                            [-1.0, -0.5, -0.5, 2.0],  #  3
                        ],
                    ]
                ),
            )
        ],
    )
    def test_compute_isoAM_consider_volume(
        self,
        filename: pathlib.Path,
        expected: np.ndarray,
        test_device: torch.device,
    ):
        mesh = graphlow.read(
            filename, "phlower", dtype=torch.float64, device=test_device
        )
        grad_adjs, _ = mesh.geometry.isoAM(
            with_moment_matrix=False, consider_volume=True
        )
        actual = grad_adjs.to_tensor().cpu().to_dense().numpy()
        np.testing.assert_almost_equal(actual, expected)

    @pytest.mark.parametrize(
        "filename, expected_grad_adjs, expected_minv",
        [
            (
                pathlib.Path("tests/data/vtu/primitive_cell/tet.vtu"),
                np.array(
                    [
                        [
                            [-1.0, 1.0, 0.0, 0.0],  #  0
                            [-1.0, 1.0, 0.0, 0.0],  #  1
                            [-1.0, 1.0, 0.0, 0.0],  #  2
                            [-1.0, 1.0, 0.0, 0.0],  #  3
                        ],
                        [
                            [-1.0, 0.0, 1.0, 0.0],  #  0
                            [-1.0, 0.0, 1.0, 0.0],  #  1
                            [-1.0, 0.0, 1.0, 0.0],  #  2
                            [-1.0, 0.0, 1.0, 0.0],  #  3
                        ],
                        [
                            [-1.0, 0.0, 0.0, 1.0],  #  0
                            [-1.0, 0.0, 0.0, 1.0],  #  1
                            [-1.0, 0.0, 0.0, 1.0],  #  2
                            [-1.0, 0.0, 0.0, 1.0],  #  3
                        ],
                    ]
                ),
                np.array(
                    [
                        [
                            [1.0, 0.0, 0.0],  #  0
                            [0.0, 1.0, 0.0],  #  1
                            [0.0, 0.0, 1.0],  #  2
                        ],
                        [
                            [1.0, 1.0, 1.0],  #  0
                            [1.0, 3.0, 1.0],  #  1
                            [1.0, 1.0, 3.0],  #  2
                        ],
                        [
                            [3.0, 1.0, 1.0],  #  0
                            [1.0, 1.0, 1.0],  #  1
                            [1.0, 1.0, 3.0],  #  2
                        ],
                        [
                            [3.0, 1.0, 1.0],  #  0
                            [1.0, 3.0, 1.0],  #  1
                            [1.0, 1.0, 1.0],  #  2
                        ],
                    ]
                ),
            )
        ],
    )
    def test_compute_isoAM_with_moment_matrix(
        self,
        filename: pathlib.Path,
        expected_grad_adjs: np.ndarray,
        expected_minv: np.ndarray,
        test_device: torch.device,
    ):
        mesh = graphlow.read(
            filename, "phlower", dtype=torch.float64, device=test_device
        )
        grad_adjs, minv = mesh.geometry.isoAM(with_moment_matrix=True)
        actual_grad_adjs = grad_adjs.to_tensor().cpu().to_dense().numpy()
        np.testing.assert_almost_equal(
            actual_grad_adjs, expected_grad_adjs, decimal=6
        )
        actual_minv = minv.to_tensor().cpu().to_dense().numpy()
        np.testing.assert_almost_equal(actual_minv, expected_minv, decimal=6)

    @pytest.mark.parametrize(
        "filename",
        [
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
        ],
    )
    def test_compute_isoAM_shapes(self, filename: pathlib.Path):
        mesh = graphlow.read(filename, "phlower", dtype=torch.float64)
        N, d = mesh.points.shape
        grad_adjs, minv = mesh.geometry.isoAM(with_moment_matrix=True)
        np.testing.assert_array_equal(grad_adjs.shape, (d, N, N))
        np.testing.assert_array_equal(minv.shape, (N, d, d))

    @pytest.mark.parametrize(
        "scalar_field, expected_grad",
        [
            (
                lambda pos: pos[:, 0] ** 2 - pos[:, 1] ** 2,
                np.array(
                    [
                        [-1.0, 1.0, 0],
                        [-1.5, 0.0, 0],
                        [-1.0, -1.0, 0],
                        [0.0, 1.5, 0],
                        [0.0, 0.0, 0],
                        [0.0, -1.5, 0],
                        [1.0, 1.0, 0],
                        [1.5, 0.0, 0],
                        [1.0, -1.0, 0],
                    ]
                ),
            )
        ],
    )
    def test_compute_isoAM_for_surface_mesh(
        self,
        scalar_field: Callable,
        expected_grad: np.ndarray,
        test_device: torch.device,
    ):
        # create a grid mesh
        ni = 3
        nj = 3
        x = np.linspace(-1, 1, ni, dtype=np.float32)
        y = np.linspace(-1, 1, nj, dtype=np.float32)
        X, Y = np.meshgrid(x, y, indexing="xy")
        Z = np.zeros([ni, nj], dtype=np.float32)
        grid = pv.StructuredGrid(X, Y, Z)
        mesh = graphlow.from_pyvista(
            grid, "phlower", dtype=torch.float64, device=test_device
        )
        grad_adjs, _ = mesh.geometry.isoAM(with_moment_matrix=True)

        phi = mesh.backend.as_tensor(
            torch.from_numpy(scalar_field(grid.points)), dimension={"Theta": 1}
        )
        actual_grad_x_phi = grad_adjs[0] @ phi
        actual_grad_y_phi = grad_adjs[1] @ phi
        actual_grad_z_phi = grad_adjs[2] @ phi
        actual_grad_vector = torch.stack(
            [actual_grad_x_phi, actual_grad_y_phi, actual_grad_z_phi], dim=1
        )
        actual_grad_vector = actual_grad_vector.to_tensor().cpu().numpy()

        np.testing.assert_almost_equal(
            actual_grad_vector, expected_grad, decimal=6
        )


# =============================================================================
# Neumann IsoAM
# =============================================================================
class TestComputeIsoAMWithNeumann:
    @pytest.mark.parametrize(
        "filename, normal_weight, \
        expected_grad_adjs, expected_normals, expected_minv",
        [
            (
                pathlib.Path("tests/data/vtu/primitive_cell/cube.vtu"),
                2.0,
                (1.0 / 18.0)
                * np.array(
                    [
                        [
                            [-7.0, 9.0, 3.0, -3.0, -3.0, 3.0, 1.0, -3.0],
                            [-9.0, 7.0, 3.0, -3.0, -3.0, 3.0, 3.0, -1.0],
                            [-3.0, 3.0, 7.0, -9.0, -1.0, 3.0, 3.0, -3.0],
                            [-3.0, 3.0, 9.0, -7.0, -3.0, 1.0, 3.0, -3.0],
                            [-3.0, 3.0, 1.0, -3.0, -7.0, 9.0, 3.0, -3.0],
                            [-3.0, 3.0, 3.0, -1.0, -9.0, 7.0, 3.0, -3.0],
                            [-1.0, 3.0, 3.0, -3.0, -3.0, 3.0, 7.0, -9.0],
                            [-3.0, 1.0, 3.0, -3.0, -3.0, 3.0, 9.0, -7.0],
                        ],
                        [
                            [-7.0, -3.0, 3.0, 9.0, -3.0, -3.0, 1.0, 3.0],
                            [-3.0, -7.0, 9.0, 3.0, -3.0, -3.0, 3.0, 1.0],
                            [-3.0, -9.0, 7.0, 3.0, -1.0, -3.0, 3.0, 3.0],
                            [-9.0, -3.0, 3.0, 7.0, -3.0, -1.0, 3.0, 3.0],
                            [-3.0, -3.0, 1.0, 3.0, -7.0, -3.0, 3.0, 9.0],
                            [-3.0, -3.0, 3.0, 1.0, -3.0, -7.0, 9.0, 3.0],
                            [-1.0, -3.0, 3.0, 3.0, -3.0, -9.0, 7.0, 3.0],
                            [-3.0, -1.0, 3.0, 3.0, -9.0, -3.0, 3.0, 7.0],
                        ],
                        [
                            [-7.0, -3.0, -3.0, -3.0, 9.0, 3.0, 1.0, 3.0],
                            [-3.0, -7.0, -3.0, -3.0, 3.0, 9.0, 3.0, 1.0],
                            [-3.0, -3.0, -7.0, -3.0, 1.0, 3.0, 9.0, 3.0],
                            [-3.0, -3.0, -3.0, -7.0, 3.0, 1.0, 3.0, 9.0],
                            [-9.0, -3.0, -1.0, -3.0, 7.0, 3.0, 3.0, 3.0],
                            [-3.0, -9.0, -3.0, -1.0, 3.0, 7.0, 3.0, 3.0],
                            [-1.0, -3.0, -9.0, -3.0, 3.0, 3.0, 7.0, 3.0],
                            [-3.0, -1.0, -3.0, -9.0, 3.0, 3.0, 3.0, 7.0],
                        ],
                    ]
                ),
                (1.0 / np.sqrt(3.0))
                * np.array(
                    [
                        [-1.0, -1.0, -1.0],
                        [1.0, -1.0, -1.0],
                        [1.0, 1.0, -1.0],
                        [-1.0, 1.0, -1.0],
                        [-1.0, -1.0, 1.0],
                        [1.0, -1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [-1.0, 1.0, 1.0],
                    ]
                ),
                (1.0 / 6.0)
                * np.array(
                    [
                        [  # 0
                            [3.0, -1.0, -1.0],
                            [-1.0, 3.0, -1.0],
                            [-1.0, -1.0, 3.0],
                        ],
                        [  # 1
                            [3.0, 1.0, 1.0],
                            [1.0, 3.0, -1.0],
                            [1.0, -1.0, 3.0],
                        ],
                        [  # 2
                            [3.0, -1.0, 1.0],
                            [-1.0, 3.0, 1.0],
                            [1.0, 1.0, 3.0],
                        ],
                        [  # 3
                            [3.0, 1.0, -1.0],
                            [1.0, 3.0, 1.0],
                            [-1.0, 1.0, 3.0],
                        ],
                        [  # 4
                            [3.0, -1.0, 1.0],
                            [-1.0, 3.0, 1.0],
                            [1.0, 1.0, 3.0],
                        ],
                        [  # 5
                            [3.0, 1.0, -1.0],
                            [1.0, 3.0, 1.0],
                            [-1.0, 1.0, 3.0],
                        ],
                        [  # 6
                            [3.0, -1.0, -1.0],
                            [-1.0, 3.0, -1.0],
                            [-1.0, -1.0, 3.0],
                        ],
                        [  # 7
                            [3.0, 1.0, 1.0],
                            [1.0, 3.0, -1.0],
                            [1.0, -1.0, 3.0],
                        ],
                    ]
                ),
            )
        ],
    )
    def test_compute_isoAM_with_neumann(
        self,
        filename: pathlib.Path,
        normal_weight: float,
        expected_grad_adjs: np.ndarray,
        expected_normals: np.ndarray,
        expected_minv: np.ndarray,
        test_device: torch.device,
    ):
        mesh = graphlow.read(
            filename, "phlower", dtype=torch.float64, device=test_device
        )
        expected_wnormals = normal_weight * expected_normals

        grad_adjs, wnormals, minv = mesh.geometry.isoAM_with_neumann(
            normal_weight=normal_weight, with_moment_matrix=True
        )
        actual_grad_adjs = grad_adjs.to_tensor().cpu().to_dense().numpy()
        np.testing.assert_almost_equal(
            actual_grad_adjs, expected_grad_adjs, decimal=6
        )
        actual_wnormals = wnormals.numpy()
        np.testing.assert_almost_equal(
            actual_wnormals, expected_wnormals, decimal=6
        )
        actual_minv = minv.to_tensor().cpu().to_dense().numpy()
        np.testing.assert_almost_equal(actual_minv, expected_minv, decimal=6)

    @pytest.mark.parametrize(
        "filename",
        [
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
        ],
    )
    def test_compute_isoAM_with_neumann_shapes(self, filename: pathlib.Path):
        mesh = graphlow.read(filename, "phlower", dtype=torch.float64)
        N, d = mesh.points.shape
        grad_adjs, wnormals, minv = mesh.geometry.isoAM_with_neumann(
            with_moment_matrix=True
        )
        np.testing.assert_array_equal(grad_adjs.shape, (d, N, N))
        np.testing.assert_array_equal(wnormals.shape, (N, d))
        np.testing.assert_array_equal(minv.shape, (N, d, d))

    def test_compute_isoAM_with_neumann_not_nan(self):
        filename = "tests/data/vtu/openedge/openedge.vtu"
        mesh = graphlow.read(filename, "phlower", dtype=torch.float64)

        grad_adjs, wnormals, minv = mesh.geometry.isoAM_with_neumann(
            normal_weight=10.0,
            with_moment_matrix=True,
            consider_volume=False,
        )

        for grad_adj in grad_adjs:
            assert not np.any(np.isnan(grad_adj.to_tensor().to_dense().numpy()))

        assert not np.any(np.isnan(wnormals.numpy()))
        assert not np.any(np.isnan(minv.to_tensor().to_dense().numpy()))


# =============================================================================
# Moment Matrix
# =============================================================================
@pytest.mark.parametrize(
    "np_adj, np_points, expected",
    [
        (
            np.array(
                [
                    # 0, 1, 2, 3, 4, 5, 6, 7, 8
                    [1, 1, 0, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 1, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 1, 1, 0, 1, 0, 0],
                    [0, 1, 0, 1, 1, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1, 1, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 0, 1, 1],
                ]
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [1.0, 2.0, 0.0],
                    [2.0, 2.0, 0.0],
                ]
            ),
            np.array(
                [
                    # 0
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    # 1
                    [
                        [2.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    # 2
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    # 3
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    # 4
                    [
                        [2.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    # 5
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    # 6
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    # 7
                    [
                        [2.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    # 8
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                ]
            ),
        )
    ],
)
def test_compute_moment_matrix(
    np_adj: np.ndarray, np_points: np.ndarray, expected: np.ndarray
):
    backend = PhlowerBackend(dtype=torch.float64)
    adj = backend.as_tensor(
        torch.from_numpy(np_adj).to_sparse_coo(), dimension={}
    )
    i_indices, _ = adj.indices()

    points = backend.as_tensor(np_points, dimension={"L": 1})
    weights = backend.ones((i_indices.shape[0],), dimension={})
    M = _compute_moment_matrix(backend, adj, points, weights)
    actual = M.to_tensor().cpu().to_dense().numpy()
    np.testing.assert_almost_equal(actual, expected)


# =============================================================================
# Compute Normals on Surface Points
# =============================================================================
class TestComputeNormalsOnSurfacePoints:
    @pytest.mark.parametrize(
        "filename, desired",
        [
            (
                pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
                np.array(
                    [
                        [-1 / np.sqrt(3), -1 / np.sqrt(3), -1 / np.sqrt(3)],
                        [1 / np.sqrt(3), -1 / np.sqrt(3), -1 / np.sqrt(3)],
                        [1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)],
                        [-1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)],
                        [-1 / np.sqrt(2), -1 / np.sqrt(2), 0.0],
                        [1 / np.sqrt(2), -1 / np.sqrt(2), 0.0],
                        [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0],
                        [-1 / np.sqrt(2), 1 / np.sqrt(2), 0.0],
                        [-1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)],
                        [1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)],
                        [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
                        [-1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)],
                    ]
                ),
            )
        ],
    )
    def test_compute_normals_on_surface_points(
        self, filename: pathlib.Path, desired: np.ndarray
    ):
        mesh = graphlow.read(filename, "phlower", dtype=torch.float64)
        normals = _compute_normals_on_surface_points(
            mesh, "conservative", 1e-12
        )
        actual = normals.to_tensor().cpu().numpy()
        np.testing.assert_almost_equal(actual, desired, decimal=6)

    def test_compute_normals_on_surface_points_not_nan(self):
        filename = "tests/data/vtp/openedge_surface/openedge_surface.vtp"
        mesh = graphlow.read(filename, "phlower", dtype=torch.float64)

        pv_mesh = pv.read(filename).compute_normals()
        pv_normals = pv_mesh.point_data["Normals"]
        filter_small_pv_normals = (
            np.linalg.vector_norm(pv_normals, axis=1) < 1e-8
        )
        normals = _compute_normals_on_surface_points(
            mesh, "conservative", 1e-12
        )
        actual = normals.to_tensor().cpu().numpy()
        assert not np.any(np.isnan(actual))
        np.testing.assert_almost_equal(actual[filter_small_pv_normals], 0.0)


# =============================================================================
# Grad Operator
# =============================================================================
@pytest.mark.parametrize(
    "rawAM, expected",
    [
        (
            np.array(
                [
                    [
                        # 0, 1, 2, 3, 4, 5, 6, 7, 8
                        [1, 1, 0, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 1, 0, 0, 0, 0],
                        [0, 1, 1, 0, 0, 1, 0, 0, 0],
                        [1, 0, 0, 1, 1, 0, 1, 0, 0],
                        [0, 1, 0, 1, 1, 1, 0, 1, 0],
                        [0, 0, 1, 0, 1, 1, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 1, 1, 0],
                        [0, 0, 0, 0, 1, 0, 1, 1, 1],
                        [0, 0, 0, 0, 0, 1, 0, 1, 1],
                    ]
                ]
            ).astype(np.float64),
            np.array(
                [
                    [
                        # 0, 1, 2, 3, 4, 5, 6, 7, 8
                        [-2, 1, 0, 1, 0, 0, 0, 0, 0],
                        [1, -3, 1, 0, 1, 0, 0, 0, 0],
                        [0, 1, -2, 0, 0, 1, 0, 0, 0],
                        [1, 0, 0, -3, 1, 0, 1, 0, 0],
                        [0, 1, 0, 1, -4, 1, 0, 1, 0],
                        [0, 0, 1, 0, 1, -3, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, -2, 1, 0],
                        [0, 0, 0, 0, 1, 0, 1, -3, 1],
                        [0, 0, 0, 0, 0, 1, 0, 1, -2],
                    ],
                ]
            ).astype(np.float64),
        )
    ],
)
def test_create_grad_operator_from(rawAM: np.ndarray, expected: np.ndarray):
    backend = PhlowerBackend(dtype=torch.float64)
    rawAM = backend.as_tensor(
        torch.from_numpy(rawAM).to_sparse_coo(), dimension={}
    )
    grad_op = _create_grad_operator_from(backend, rawAM)
    actual = grad_op.to_tensor().cpu().to_dense().numpy()
    np.testing.assert_almost_equal(actual, expected)


# =============================================================================
# Compute Raw AM and Moment Inv
# =============================================================================
class TestComputeRawAMAndMomentInv:
    @pytest.mark.parametrize(
        "mesh_filename, femio_moment_filename, femio_rawAM_filename, "
        "femio_Minv_filename",
        [
            (
                pathlib.Path("tests/data/vtu/hexbeam/mesh.vtu"),
                pathlib.Path("tests/data/femio/hexbeam/moment.npy"),
                pathlib.Path("tests/data/femio/hexbeam/rawAM.npy"),
                pathlib.Path("tests/data/femio/hexbeam/Minv.npy"),
            )
        ],
    )
    def test_compute_rawAM_and_moment_inv(
        self,
        mesh_filename: pathlib.Path,
        femio_moment_filename: pathlib.Path,
        femio_rawAM_filename: pathlib.Path,
        femio_Minv_filename: pathlib.Path,
    ):
        femio_moment = np.load(femio_moment_filename)
        femio_rawAM = np.load(femio_rawAM_filename)
        femio_Minv = np.load(femio_Minv_filename)
        mesh = graphlow.read(mesh_filename, "phlower", dtype=torch.float64)
        adj = mesh.topology.point_adjacency(layout="coo")
        points = mesh.points
        weights = mesh.backend.ones((mesh.n_points,), dimension={})
        moment = mesh.backend.as_tensor(
            torch.from_numpy(femio_moment), dimension={}
        )
        rawAM, Minv = _compute_rawAM_and_moment_inv(
            mesh.backend, adj, points, weights, moment
        )
        graphlow_rawAM = rawAM.to_tensor().cpu().to_dense().numpy()
        np.testing.assert_almost_equal(graphlow_rawAM, femio_rawAM, decimal=6)
        graphlow_Minv = Minv.to_tensor().cpu().to_dense().numpy()
        np.testing.assert_almost_equal(graphlow_Minv, femio_Minv, decimal=6)

    @pytest.mark.parametrize(
        "mesh_filename, femio_moment_filename, femio_rawAM_filename, "
        "femio_Minv_filename",
        [
            (
                pathlib.Path("tests/data/vtu/hexbeam/mesh.vtu"),
                pathlib.Path("tests/data/femio/hexbeam/moment.npy"),
                pathlib.Path("tests/data/femio/hexbeam/rawAM.npy"),
                pathlib.Path("tests/data/femio/hexbeam/Minv.npy"),
            )
        ],
    )
    def test_compute_rawAM_and_moment_inv_fallback_when_cholesky_fails(
        self,
        mesh_filename: pathlib.Path,
        femio_moment_filename: pathlib.Path,
        femio_rawAM_filename: pathlib.Path,
        femio_Minv_filename: pathlib.Path,
    ):
        """Exercise the except branch when Cholesky fails (e.g. non-SPD matrix).

        When torch.linalg.cholesky raises torch.linalg.LinAlgError,
        the code falls back to torch.linalg.inv.
        This test mocks torch.linalg.cholesky to raise so that the
        fallback path is executed and produces the same result as the reference.
        """
        femio_moment = np.load(femio_moment_filename)
        femio_rawAM = np.load(femio_rawAM_filename)
        femio_Minv = np.load(femio_Minv_filename)
        mesh = graphlow.read(mesh_filename, "phlower", dtype=torch.float64)
        adj = mesh.topology.point_adjacency(layout="coo")
        points = mesh.points
        weights = mesh.backend.ones((mesh.n_points,), dimension={})
        moment = mesh.backend.as_tensor(
            torch.from_numpy(femio_moment), dimension={}
        )

        with (
            patch(
                "torch.linalg.cholesky",
                side_effect=torch.linalg.LinAlgError,
            ),
            patch(
                "torch.linalg.inv",
                side_effect=torch.linalg.inv,
            ) as mock_torch_linalg_inv,
        ):
            rawAM, Minv = _compute_rawAM_and_moment_inv(
                mesh.backend, adj, points, weights, moment
            )
        # Guarantee the except branch ran:
        # fallback uses torch.linalg.inv, not torch.linalg.cholesky.
        mock_torch_linalg_inv.assert_called_once()

        graphlow_rawAM = rawAM.to_tensor().cpu().to_dense().numpy()
        np.testing.assert_almost_equal(graphlow_rawAM, femio_rawAM, decimal=6)
        graphlow_Minv = Minv.to_tensor().cpu().to_dense().numpy()
        np.testing.assert_almost_equal(graphlow_Minv, femio_Minv, decimal=6)

    @pytest.mark.parametrize(
        "mesh_filename, femio_moment_filename",
        [
            (
                pathlib.Path("tests/data/vtu/hexbeam/mesh.vtu"),
                pathlib.Path("tests/data/femio/hexbeam/moment.npy"),
            )
        ],
    )
    def test_compute_moment_matrix_cf_femio(
        self,
        mesh_filename: pathlib.Path,
        femio_moment_filename: pathlib.Path,
    ):
        femio_moment = np.load(femio_moment_filename)
        mesh = graphlow.read(mesh_filename, "phlower", dtype=torch.float64)
        adj = mesh.topology.point_adjacency(layout="coo")
        points = mesh.points
        weights = mesh.backend.ones((mesh.n_points,), dimension={})
        moment_matrix = _compute_moment_matrix(
            mesh.backend, adj, points, weights
        )
        graphlow_moment = moment_matrix.to_tensor().cpu().to_dense().numpy()
        np.testing.assert_almost_equal(graphlow_moment, femio_moment, decimal=6)

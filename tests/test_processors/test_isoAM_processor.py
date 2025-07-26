import pathlib
from collections.abc import Callable

import numpy as np
import pytest
import pyvista as pv
import torch

import graphlow
from graphlow.processors.isoAM_processor import IsoAMProcessor
from graphlow.util import array_handler
from graphlow.util.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.with_device
@pytest.mark.parametrize(
    "file_name, desired",
    [
        (
            pathlib.Path("tests/data/vtu/primitive_cell/octahedron.vtu"),
            np.array(
                [
                    [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    [2.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                    [2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                ]
            ),
        )
    ],
)
def test___compute_weight_from_volume(
    file_name: pathlib.Path, desired: np.ndarray, device: str
):
    mesh = graphlow.read(file_name)
    mesh.send(device=torch.device(device))
    adj = mesh.compute_point_adjacency().to_sparse_coo()
    isoAM_processor = IsoAMProcessor()

    weights_nnz = isoAM_processor._compute_weights_nnz_from_volume(mesh)
    Wij = torch.sparse_coo_tensor(
        adj.indices(), weights_nnz, (mesh.n_points, mesh.n_points)
    )
    actual = array_handler.convert_to_dense_numpy(Wij)
    np.testing.assert_almost_equal(actual, desired)


@pytest.mark.with_device
@pytest.mark.parametrize(
    "np_adj, np_points, desired",
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
                    [0.0, 0.0, 2.0],
                    [1.0, 0.0, 1.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 1.0, 2.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 1.0, 0.0],
                    [0.0, 2.0, 2.0],
                    [1.0, 2.0, 1.0],
                    [2.0, 2.0, 0.0],
                ]
            ),
            np.array(
                [
                    # x
                    [
                        # 0,  1,  2,  3,  4,  5,  6,  7,  8
                        [-1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [-1, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, -1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, -1, 1, 0, 0, 0, 0],
                        [0, 0, 0, -1, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, -1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, -1, 1, 0],
                        [0, 0, 0, 0, 0, 0, -1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, -1, 1],
                    ],
                    # y
                    [
                        # 0,  1,  2,  3,  4,  5,  6,  7,  8
                        [-1, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, -1, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, -1, 0, 0, 1, 0, 0, 0],
                        [-1, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, -1, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, -1, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, -1, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, -1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, -1, 0, 0, 1],
                    ],
                    # z
                    [
                        # 0,  1,  2,  3,  4,  5,  6,  7,  8
                        [1, -1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, -1, 0, 0, 0, 0, 0, 0],
                        [0, 1, -1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, -1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, -1, 0, 0, 0],
                        [0, 0, 0, 0, 1, -1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, -1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, -1],
                        [0, 0, 0, 0, 0, 0, 0, 1, -1],
                    ],
                ]
            ),
        )
    ],
)
def test___create_grad_operator_from(
    np_adj: np.ndarray, np_points: np.ndarray, desired: np.ndarray, device: str
):
    adj = torch.from_numpy(np_adj).to_sparse_coo().to(torch.device(device))
    i_indices, j_indices = adj.indices()

    points = torch.from_numpy(np_points).to(torch.device(device))
    n_points = points.shape[0]

    diff = points[j_indices] - points[i_indices]  # (nnz, dim)
    isoAM_processor = IsoAMProcessor()

    grad_op = isoAM_processor._create_grad_operator_from(
        i_indices, j_indices, n_points, diff
    )
    actual = array_handler.convert_to_dense_numpy(grad_op)
    np.testing.assert_almost_equal(actual, desired)


@pytest.mark.with_device
@pytest.mark.parametrize(
    "file_name, desired",
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
def test___compute_normals_on_surface_points(
    file_name: pathlib.Path, desired: np.ndarray, device: str
):
    mesh = graphlow.read(file_name)
    mesh.send(device=torch.device(device))
    isoAM_processor = IsoAMProcessor()

    normals = isoAM_processor._compute_normals_on_surface_points(mesh)
    actual = array_handler.convert_to_numpy_scipy(normals)
    np.testing.assert_almost_equal(actual, desired, decimal=6)


@pytest.mark.with_device
@pytest.mark.parametrize(
    "np_adj, np_points, desired",
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
def test__compute_moment_matrix(
    np_adj: np.ndarray, np_points: np.ndarray, desired: np.ndarray, device: str
):
    adj = torch.from_numpy(np_adj).to_sparse_coo().to(torch.device(device))
    i_indices, j_indices = adj.indices()

    points = torch.from_numpy(np_points).to(torch.device(device))
    isoAM_processor = IsoAMProcessor()

    weights = torch.ones(
        i_indices.shape[0], device=points.device, dtype=points.dtype
    )
    M = isoAM_processor._compute_moment_matrix(
        i_indices, j_indices, points, weights
    )
    actual = array_handler.convert_to_dense_numpy(M)
    np.testing.assert_almost_equal(actual, desired)


@pytest.mark.with_device
@pytest.mark.parametrize(
    "file_name, desired",
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
def test__compute_isoAM_without_moment_matrix(
    file_name: pathlib.Path, desired: np.ndarray, device: str
):
    mesh = graphlow.read(file_name)
    mesh.send(device=torch.device(device))
    grad_adjs, _ = mesh.compute_isoAM(with_moment_matrix=False)
    actual = array_handler.convert_to_dense_numpy(grad_adjs)
    np.testing.assert_almost_equal(actual, desired)


@pytest.mark.with_device
@pytest.mark.parametrize(
    "file_name, desired",
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
def test__compute_isoAM_consider_volume(
    file_name: pathlib.Path, desired: np.ndarray, device: str
):
    mesh = graphlow.read(file_name)
    mesh.send(device=torch.device(device))
    grad_adjs, _ = mesh.compute_isoAM(
        with_moment_matrix=False, consider_volume=True
    )
    actual = array_handler.convert_to_dense_numpy(grad_adjs)
    np.testing.assert_almost_equal(actual, desired)


@pytest.mark.with_device
@pytest.mark.parametrize(
    "file_name, desired_grad_adjs, desired_minv",
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
def test__compute_isoAM_with_moment_matrix(
    file_name: pathlib.Path,
    desired_grad_adjs: np.ndarray,
    desired_minv: np.ndarray,
    device: str,
):
    mesh = graphlow.read(file_name)
    mesh.send(device=torch.device(device))
    grad_adjs, minv = mesh.compute_isoAM(with_moment_matrix=True)
    actual_grad_adjs = array_handler.convert_to_dense_numpy(grad_adjs)
    np.testing.assert_almost_equal(
        actual_grad_adjs, desired_grad_adjs, decimal=6
    )
    actual_minv = array_handler.convert_to_dense_numpy(minv)
    np.testing.assert_almost_equal(actual_minv, desired_minv, decimal=6)


@pytest.mark.with_device
@pytest.mark.parametrize(
    "file_name",
    [
        pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
        pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
    ],
)
def test__compute_isoAM_shapes(file_name: pathlib.Path, device: str):
    mesh = graphlow.read(file_name)
    mesh.send(device=torch.device(device))
    N, d = mesh.points.shape
    grad_adjs, minv = mesh.compute_isoAM(with_moment_matrix=True)
    np.testing.assert_array_equal(grad_adjs.shape, (d, N, N))
    np.testing.assert_array_equal(minv.shape, (N, d, d))


@pytest.mark.with_device
@pytest.mark.parametrize(
    "file_name, normal_weight, \
    desired_grad_adjs, desired_normals, desired_minv",
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
def test__compute_isoAM_with_neumann(
    file_name: pathlib.Path,
    normal_weight: float,
    desired_grad_adjs: np.ndarray,
    desired_normals: np.ndarray,
    desired_minv: np.ndarray,
    device: str,
):
    mesh = graphlow.read(file_name)
    mesh.send(device=torch.device(device))
    desired_wnormals = normal_weight * desired_normals

    grad_adjs, wnormals, minv = mesh.compute_isoAM_with_neumann(
        normal_weight=normal_weight, with_moment_matrix=True
    )
    actual_grad_adjs = array_handler.convert_to_dense_numpy(grad_adjs)
    np.testing.assert_almost_equal(
        actual_grad_adjs, desired_grad_adjs, decimal=6
    )
    actual_wnormals = array_handler.convert_to_dense_numpy(wnormals)
    np.testing.assert_almost_equal(actual_wnormals, desired_wnormals, decimal=6)
    actual_minv = array_handler.convert_to_dense_numpy(minv)
    np.testing.assert_almost_equal(actual_minv, desired_minv, decimal=6)


@pytest.mark.with_device
@pytest.mark.parametrize(
    "file_name",
    [
        pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
        pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
    ],
)
def test__compute_isoAM_with_neumann_shapes(
    file_name: pathlib.Path, device: str
):
    mesh = graphlow.read(file_name)
    mesh.send(device=torch.device(device))
    N, d = mesh.points.shape
    grad_adjs, wnormals, minv = mesh.compute_isoAM_with_neumann(
        with_moment_matrix=True
    )
    np.testing.assert_array_equal(grad_adjs.shape, (d, N, N))
    np.testing.assert_array_equal(wnormals.shape, (N, d))
    np.testing.assert_array_equal(minv.shape, (N, d, d))


@pytest.mark.with_device
@pytest.mark.parametrize(
    "scalar_field, desired_grad",
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
def test__compute_isoAM_for_surface_mesh(
    scalar_field: Callable, desired_grad: np.ndarray, device: str
):
    # create a grid mesh
    ni = 3
    nj = 3
    x = np.linspace(-1, 1, ni, dtype=np.float32)
    y = np.linspace(-1, 1, nj, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.zeros([ni, nj], dtype=np.float32)
    grid = pv.StructuredGrid(X, Y, Z)
    mesh = graphlow.GraphlowMesh(grid)
    mesh.send(device=torch.device(device))
    grad_adjs, _ = mesh.compute_isoAM(with_moment_matrix=True)

    phi = torch.from_numpy(scalar_field(grid.points)).to(torch.device(device))
    actual_grad_x_phi = grad_adjs[0] @ phi
    actual_grad_y_phi = grad_adjs[1] @ phi
    actual_grad_z_phi = grad_adjs[2] @ phi
    actual_grad_vector = torch.stack(
        [actual_grad_x_phi, actual_grad_y_phi, actual_grad_z_phi], dim=1
    )
    actual_grad_vector = array_handler.convert_to_numpy_scipy(
        actual_grad_vector
    )

    np.testing.assert_almost_equal(actual_grad_vector, desired_grad, decimal=6)


def test___compute_normals_on_surface_points_not_nan():
    file_name = "tests/data/vtp/openedge_surface/openedge_surface.vtp"
    mesh = graphlow.read(file_name)
    isoAM_processor = IsoAMProcessor()

    pv_mesh = pv.read(file_name).compute_normals()
    pv_normals = pv_mesh.point_data["Normals"]
    filter_small_pv_normals = np.linalg.norm(pv_normals, axis=1) < 1e-8

    normals = isoAM_processor._compute_normals_on_surface_points(mesh)
    actual = array_handler.convert_to_numpy_scipy(normals)

    assert not np.any(np.isnan(actual))
    np.testing.assert_almost_equal(actual[filter_small_pv_normals], 0.0)


def test__compute_isoAM_with_neumann_not_nan():
    file_name = "tests/data/vtu/openedge/openedge.vtu"
    mesh = graphlow.read(file_name)

    grad_adjs, wnormals, minv = mesh.compute_isoAM_with_neumann(
        normal_weight=10.0,
        with_moment_matrix=True,
        consider_volume=False,
        normal_interp_mode="conservative",
    )

    for grad_adj in grad_adjs:
        assert not np.any(np.isnan(array_handler.convert_to_dense_numpy(
            grad_adj
        )))

    assert not np.any(np.isnan(wnormals.numpy()))
    assert not np.any(np.isnan(minv.numpy()))

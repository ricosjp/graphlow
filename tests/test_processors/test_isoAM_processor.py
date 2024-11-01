import pathlib

import numpy as np
import pytest
import torch

import graphlow
from graphlow.processors.isoAM_processor import IsoAMProcessor
from graphlow.util.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.parametrize(
    "file_name, desired",
    [
        (
            pathlib.Path("tests/data/vtu/primitive_cell/octahedron.vtu"),
            np.array(
                [
                    # x
                    [
                        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
                        [-1.0, -1.0, 0.0, -1.0, 0.0, -1.0, -1.0],
                        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
                    ],
                    # y
                    [
                        [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                        [-1.0, -1.0, -1.0, 0.0, -1.0, 0.0, -1.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                    ],
                    # z
                    [
                        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [-1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0],
                    ],
                ]
            ),
        )
    ],
)
def test___compute_differences(file_name: pathlib.Path, desired: np.ndarray):
    mesh = graphlow.read(file_name)
    points = mesh.points
    adj = mesh.compute_point_adjacency().to_sparse_coo()
    isoAM_processor = IsoAMProcessor()

    diff_kij = isoAM_processor._compute_differences(points, adj)
    np.testing.assert_almost_equal(diff_kij.to_dense().numpy(), desired)


@pytest.mark.parametrize(
    "np_diff_kij, desired",
    [
        (
            np.array(
                [
                    # x
                    [
                        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
                        [-1.0, -1.0, 0.0, -1.0, 0.0, -1.0, -1.0],
                        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
                    ],
                    # y
                    [
                        [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                        [-1.0, -1.0, -1.0, 0.0, -1.0, 0.0, -1.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                    ],
                    # z
                    [
                        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [-1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0],
                    ],
                ]
            ),
            np.array(
                [
                    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0],
                    [1.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.5],
                    [1.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.5],
                    [1.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.5],
                    [1.0, 0.5, 0.5, 0.0, 0.5, 0.0, 0.5],
                    [1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0],
                ]
            ),
        )
    ],
)
def test___compute_inverse_square_norm(
    np_diff_kij: np.ndarray, desired: np.ndarray
):
    diff_kij = torch.from_numpy(np_diff_kij).to_sparse_coo()
    isoAM_processor = IsoAMProcessor()

    inv_sqr_norm = isoAM_processor._compute_inverse_square_norm(diff_kij)
    np.testing.assert_almost_equal(inv_sqr_norm.to_dense().numpy(), desired)


@pytest.mark.parametrize(
    "file_name, desired",
    [
        (
            pathlib.Path("tests/data/vtu/primitive_cell/octahedron.vtu"),
            np.array(
                [
                    [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                    [2.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                    [2.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                    [2.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                    [2.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                ]
            ),
        )
    ],
)
def test___compute_weight_from_volume(
    file_name: pathlib.Path, desired: np.ndarray
):
    mesh = graphlow.read(file_name)
    adj = mesh.compute_point_adjacency().to_sparse_coo()
    isoAM_processor = IsoAMProcessor()

    Wij = isoAM_processor._compute_weight_from_volume(mesh, adj)
    np.testing.assert_almost_equal(Wij.to_dense().numpy(), desired)


@pytest.mark.parametrize(
    "np_Akij, desired",
    [
        (
            np.array(
                [
                    # x
                    [
                        [0.0, 0.0, 1.0, 0.0],
                        [1.0, 0.0, 2.0, 3.0],
                        [2.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0],
                    ],
                    # y
                    [
                        [0.0, -1.0, 0.0, 0.0],
                        [-2.0, 0.0, -2.0, -2.0],
                        [0.0, 0.0, 0.0, -1.0],
                        [1.0, 1.0, 1.0, 0.0],
                    ],
                    # z
                    [
                        [0.0, 1.0, 1.0, 1.0],
                        [1.0, 0.0, 1.0, 1.0],
                        [1.0, 1.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0, 0.0],
                    ],
                ]
            ),
            np.array(
                [
                    # x
                    [
                        [-1.0, 0.0, 1.0, 0.0],
                        [1.0, -6.0, 2.0, 3.0],
                        [2.0, 0.0, -2.0, 0.0],
                        [1.0, 1.0, 0.0, -2.0],
                    ],
                    # y
                    [
                        [1.0, -1.0, 0.0, 0.0],
                        [-2.0, 6.0, -2.0, -2.0],
                        [0.0, 0.0, 1.0, -1.0],
                        [1.0, 1.0, 1.0, -3.0],
                    ],
                    # z
                    [
                        [-3.0, 1.0, 1.0, 1.0],
                        [1.0, -3.0, 1.0, 1.0],
                        [1.0, 1.0, -3.0, 1.0],
                        [1.0, 1.0, 1.0, -3.0],
                    ],
                ]
            ),
        )
    ],
)
def test___create_grad_operator_from(np_Akij: np.ndarray, desired: np.ndarray):
    Akij = torch.from_numpy(np_Akij).to_sparse_coo()
    isoAM_processor = IsoAMProcessor()

    grad_op = isoAM_processor._create_grad_operator_from(Akij).to_dense()
    np.testing.assert_almost_equal(grad_op.numpy(), desired)


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
    file_name: pathlib.Path, desired: np.ndarray
):
    mesh = graphlow.read(file_name)
    isoAM_processor = IsoAMProcessor()

    normals = isoAM_processor._compute_normals_on_surface_points(mesh)
    np.testing.assert_almost_equal(normals.detach().numpy(), desired)


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
    file_name: pathlib.Path, desired: np.ndarray
):
    mesh = graphlow.read(file_name)
    grad_adjs, _ = mesh.compute_isoAM(with_moment_matrix=False)
    np.testing.assert_almost_equal(
        grad_adjs.detach().to_dense().numpy(), desired
    )


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
    file_name: pathlib.Path, desired: np.ndarray
):
    mesh = graphlow.read(file_name)
    grad_adjs, _ = mesh.compute_isoAM(
        with_moment_matrix=False, consider_volume=True
    )
    np.testing.assert_almost_equal(
        grad_adjs.detach().to_dense().numpy(), desired
    )


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
):
    mesh = graphlow.read(file_name)
    grad_adjs, minv = mesh.compute_isoAM(with_moment_matrix=True)
    np.testing.assert_almost_equal(
        grad_adjs.detach().to_dense().numpy(), desired_grad_adjs, decimal=6
    )
    np.testing.assert_almost_equal(
        minv.detach().numpy(), desired_minv, decimal=6
    )


@pytest.mark.parametrize(
    "file_name",
    [
        pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
        pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
    ],
)
def test__compute_isoAM_shapes(file_name: pathlib.Path):
    mesh = graphlow.read(file_name)
    N, d = mesh.points.shape
    grad_adjs, minv = mesh.compute_isoAM(with_moment_matrix=True)
    np.testing.assert_array_equal(grad_adjs.shape, (d, N, N))
    np.testing.assert_array_equal(minv.shape, (N, d, d))


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
def test__compute_isoAM_with_Neumann(
    file_name: pathlib.Path,
    normal_weight: float,
    desired_grad_adjs: np.ndarray,
    desired_normals: np.ndarray,
    desired_minv: np.ndarray,
):
    mesh = graphlow.read(file_name)
    desired_wnormals = normal_weight * desired_normals

    grad_adjs, normals, wnormals, minv = mesh.compute_isoAM_with_Neumann(
        normal_weight=normal_weight, with_moment_matrix=True
    )
    np.testing.assert_almost_equal(
        grad_adjs.detach().to_dense().numpy(), desired_grad_adjs, decimal=6
    )
    np.testing.assert_almost_equal(
        normals.detach().numpy(), desired_normals, decimal=6
    )
    np.testing.assert_almost_equal(
        wnormals.detach().numpy(), desired_wnormals, decimal=6
    )
    np.testing.assert_almost_equal(
        minv.detach().numpy(), desired_minv, decimal=6
    )


@pytest.mark.parametrize(
    "file_name",
    [
        pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
        pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
    ],
)
def test__compute_isoAM_with_Neumann_shapes(file_name: pathlib.Path):
    mesh = graphlow.read(file_name)
    N, d = mesh.points.shape
    grad_adjs, normals, wnormals, minv = mesh.compute_isoAM_with_Neumann(
        with_moment_matrix=True
    )
    np.testing.assert_array_equal(grad_adjs.shape, (d, N, N))
    np.testing.assert_array_equal(normals.shape, (N, d))
    np.testing.assert_array_equal(wnormals.shape, (N, d))
    np.testing.assert_array_equal(minv.shape, (N, d, d))

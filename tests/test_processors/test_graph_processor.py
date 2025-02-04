import pathlib

import numpy as np
import pytest
import pyvista as pv

import graphlow
from graphlow.util import array_handler


@pytest.mark.parametrize(
    "file_name, desired",
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
def test__compute_point_cell_incidence(
    file_name: pathlib.Path, desired: np.ndarray
):
    mesh = graphlow.read(file_name)
    cell_point_incidence = mesh.compute_cell_point_incidence()
    np.testing.assert_array_equal(
        cell_point_incidence.to_dense().numpy(), desired
    )


@pytest.mark.parametrize(
    "file_name, desired",
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
def test__compute_cell_adjacency(file_name: pathlib.Path, desired: np.ndarray):
    mesh = graphlow.read(file_name)
    cell_adjacency = mesh.compute_cell_adjacency()
    np.testing.assert_array_equal(
        array_handler.convert_to_dense_numpy(cell_adjacency), desired
    )


@pytest.mark.parametrize(
    "file_name, desired",
    [
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            np.array(
                [
                    [3, 0, 0],
                    [0, 2, 0],
                    [0, 0, 2],
                ]
            ),
        ),
    ],
)
def test__compute_cell_degree(file_name: pathlib.Path, desired: np.ndarray):
    mesh = graphlow.read(file_name)
    cell_degree = mesh.compute_cell_degree()
    np.testing.assert_array_equal(
        array_handler.convert_to_dense_numpy(cell_degree), desired
    )


@pytest.mark.parametrize(
    "file_name, desired",
    [
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
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
def test__compute_normalized_cell_adjacency(
    file_name: pathlib.Path, desired: np.ndarray
):
    mesh = graphlow.read(file_name)
    normalized_cell_adjacency = mesh.compute_normalized_cell_adjacency()
    np.testing.assert_almost_equal(
        array_handler.convert_to_dense_numpy(normalized_cell_adjacency), desired
    )


@pytest.mark.parametrize(
    "file_name, desired",
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
def test__compute_point_adjacency(file_name: pathlib.Path, desired: np.ndarray):
    mesh = graphlow.read(file_name)
    point_adjacency = mesh.compute_point_adjacency()
    np.testing.assert_array_equal(
        array_handler.convert_to_dense_numpy(point_adjacency), desired
    )


@pytest.mark.parametrize(
    "file_name, desired",
    [
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
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
            ),
        )
    ],
)
def test__compute_point_degree(file_name: pathlib.Path, desired: np.ndarray):
    mesh = graphlow.read(file_name)
    point_degree = mesh.compute_point_degree()
    np.testing.assert_array_equal(
        array_handler.convert_to_dense_numpy(point_degree), desired
    )


@pytest.mark.parametrize(
    "file_name, desired",
    [
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
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
        )
    ],
)
def test__compute_normalized_point_adjacency(
    file_name: pathlib.Path, desired: np.ndarray
):
    mesh = graphlow.read(file_name)
    normalized_point_adjacency = mesh.compute_normalized_point_adjacency()
    np.testing.assert_almost_equal(
        array_handler.convert_to_dense_numpy(normalized_point_adjacency),
        desired,
    )


@pytest.mark.parametrize(
    "file_name",
    [
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
    ],
)
def test__compute_point_relative_incidence(file_name: pathlib.Path):
    mesh = graphlow.read(file_name)
    mesh.dict_point_tensor.update({"feature": mesh.points[:, 0] ** 2})
    mesh.copy_features_to_pyvista()

    surface = mesh.extract_surface()
    relative_incidence = mesh.compute_point_relative_incidence(surface)

    actual_surface_points = relative_incidence.matmul(mesh.points)
    desired_surface_points = surface.points
    np.testing.assert_almost_equal(
        actual_surface_points.numpy(), desired_surface_points.numpy()
    )

    # Check reversed order also works
    relative_incidence_t = surface.compute_point_relative_incidence(mesh)
    np.testing.assert_array_equal(
        array_handler.convert_to_dense_numpy(relative_incidence),
        array_handler.convert_to_dense_numpy(relative_incidence_t).T,
    )


@pytest.mark.parametrize(
    "file_name, desired",
    [
        (
            pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
            np.array(
                [
                    # 0  1  2  3  4  5  6
                    [1, 0, 0, 0, 0, 0, 0],  #  0
                    [1, 0, 0, 0, 0, 0, 0],  #  1
                    [1, 0, 0, 0, 0, 0, 0],  #  2
                    [0, 1, 0, 0, 0, 0, 0],  #  3
                    [0, 1, 0, 0, 0, 0, 0],  #  4
                    [0, 0, 0, 1, 0, 0, 0],  #  5
                    [0, 0, 0, 1, 0, 0, 0],  #  6
                    [0, 0, 0, 1, 0, 0, 0],  #  7
                    [1, 0, 0, 0, 0, 0, 0],  #  8
                    [0, 1, 0, 0, 0, 0, 0],  #  9
                    [0, 0, 0, 1, 0, 0, 0],  # 10
                    [0, 0, 1, 0, 0, 0, 0],  # 11
                    [0, 0, 0, 0, 1, 0, 0],  # 12
                    [0, 0, 0, 0, 1, 0, 0],  # 13
                    [0, 0, 1, 0, 0, 0, 0],  # 14
                    [0, 0, 0, 0, 0, 0, 1],  # 15
                    [0, 0, 0, 0, 0, 0, 1],  # 16
                    [0, 0, 0, 0, 0, 1, 0],  # 17
                    [1, 0, 0, 0, 0, 0, 0],  # 18
                    [0, 0, 1, 0, 0, 0, 0],  # 19
                    [0, 0, 0, 0, 1, 0, 0],  # 20
                    [0, 0, 0, 0, 0, 0, 1],  # 21
                    [0, 0, 0, 0, 0, 1, 0],  # 22
                ]
            ),
        ),
    ],
)
def test__compute_cell_relative_incidence(
    file_name: pathlib.Path, desired: np.ndarray
):
    pv_mesh = pv.read(file_name)
    pv_mesh.point_data["feature"] = pv_mesh.points[:, -1] ** 2
    pv_mesh = pv_mesh.point_data_to_cell_data()
    mesh = graphlow.GraphlowMesh(pv_mesh)
    mesh.copy_features_from_pyvista()

    surface = mesh.extract_surface()
    relative_incidence = array_handler.convert_to_dense_numpy(
        mesh.compute_cell_relative_incidence(surface)
    ).astype(int)

    np.testing.assert_almost_equal(relative_incidence, desired)

    # Check reversed order also works
    relative_incidence_t = array_handler.convert_to_dense_numpy(
        surface.compute_cell_relative_incidence(mesh)
    ).astype(int)
    np.testing.assert_array_equal(relative_incidence, relative_incidence_t.T)

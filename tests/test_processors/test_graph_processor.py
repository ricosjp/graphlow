
import pathlib

import numpy as np
import pytest

import graphlow
from graphlow.util import array_handler


@pytest.mark.parametrize(
    "file_name, desired",
    [
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            ]),
        ),
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            np.array([
                # 0  1  2  3  4  5  6  7  8  9 10 11 12
                [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            ]),
        ),
    ],
)
def test__compute_point_cell_incidence(file_name, desired):
    mesh = graphlow.read(file_name)
    cell_point_incidence = mesh.compute_cell_point_incidence()
    np.testing.assert_array_equal(
        cell_point_incidence.to_dense().numpy(), desired)


@pytest.mark.parametrize(
    "file_name, desired",
    [
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            np.array([
                [1, 1],
                [1, 1],
            ]),
        ),
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            np.array([
                [1, 1, 1],
                [1, 1, 0],
                [1, 0, 1],
            ]),
        ),
    ],
)
def test__compute_cell_adjacency(file_name, desired):
    mesh = graphlow.read(file_name)
    cell_adjacency = mesh.compute_cell_adjacency()
    np.testing.assert_array_equal(
        array_handler.convert_to_dense_numpy(cell_adjacency), desired)


@pytest.mark.parametrize(
    "file_name, desired",
    [
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            np.array([
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
            ]),
        ),
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            np.array([
                # 0  1  2  3  4  5  6  7  8  9 10 11 12
                [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0], #  0
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], #  1
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], #  2
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], #  3
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], #  4
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], #  5
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], #  6
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], #  7
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], #  8
                [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1], #  9
                [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1], # 10
                [1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1], # 11
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], # 12
            ]),
        ),
    ],
)
def test__compute_point_adjacency(file_name, desired):
    mesh = graphlow.read(file_name)
    point_adjacency = mesh.compute_point_adjacency()
    np.testing.assert_array_equal(
        array_handler.convert_to_dense_numpy(point_adjacency), desired)


@pytest.mark.parametrize(
    "file_name",
    [
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
    ],
)
def test__compute_point_relative_incidence(file_name):
    mesh = graphlow.read(file_name)
    mesh.dict_point_tensor.update({'feature': mesh.points[:, 0]**2})
    mesh.copy_features_to_pyvista()

    surface = mesh.extract_surface()
    relative_incidence = mesh.compute_point_relative_incidence(surface)

    actual_surface_points = relative_incidence.matmul(mesh.points)
    desired_surface_points = surface.points
    np.testing.assert_almost_equal(
        actual_surface_points.numpy(), desired_surface_points.numpy())


import pathlib

import numpy as np
import pytest
import torch

import graphlow


@pytest.mark.parametrize(
    "file_name, desired_file",
    [
        (
            pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
            pathlib.Path("tests/data/vtu/complex/surface.vtu"),
        ),
    ],
)
def test__extract_surface(file_name, desired_file):
    mesh = graphlow.read(file_name)

    surface = mesh.extract_surface()
    desired = graphlow.read(desired_file)
    np.testing.assert_almost_equal(surface.mesh.points, desired.mesh.points)
    np.testing.assert_array_equal(surface.mesh.cells, desired.mesh.cells)


@pytest.mark.parametrize(
    "file_name",
    [
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
    ],
)
def test__to_float16(file_name):
    mesh = graphlow.read(file_name)
    mesh.to(dtype=torch.float16)

    mesh.dict_point_tensor.update({'feature': mesh.points[:, 0]**2})
    assert mesh.dict_point_tensor['feature'].dtype == torch.float16

    mesh.compute_cell_adjacency()
    assert mesh.dict_sparse_tensor['cell_adjacency'].dtype == torch.float16

    mesh.copy_features_to_pyvista()

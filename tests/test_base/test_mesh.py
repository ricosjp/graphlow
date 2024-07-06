
import pathlib

import numpy as np
import pytest
import torch

import graphlow


@pytest.mark.parametrize(
    "input_file_name",
    [
        pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
        pathlib.Path("tests/data/vtp/complex_surface/mesh.vtp"),
    ],
)
@pytest.mark.parametrize(
    "output_file_name",
    [
        pathlib.Path("tests/outputs/save/mesh.vtk"),
        pathlib.Path("tests/outputs/save/mesh.vtu"),
        pathlib.Path("tests/outputs/save/mesh.vtp"),
        pathlib.Path("tests/outputs/save/mesh.stl"),
    ],
)
def test__save(input_file_name: pathlib.Path, output_file_name: pathlib.Path):
    output_file_name.unlink(missing_ok=True)

    mesh = graphlow.read(input_file_name)
    mesh.save(output_file_name)

    # Raise ValueError when overwriting
    with pytest.raises(ValueError) as e:
        mesh.save(output_file_name)
        assert 'already exists' in str(e.value)

    # Overwrite
    mesh.save(output_file_name, overwrite_file=True, overwrite_features=True)


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
def test__send_float16(file_name):
    mesh = graphlow.read(file_name)
    mesh.send(dtype=torch.float16)

    mesh.dict_point_tensor.update({'feature': mesh.points[:, 0]**2})
    assert mesh.dict_point_tensor['feature'].dtype == torch.float16

    mesh.compute_cell_adjacency()
    assert mesh.dict_sparse_tensor['cell_adjacency'].dtype == torch.float16

    mesh.copy_features_to_pyvista()


import pathlib

import numpy as np
import pytest

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
def test__convert_to_surface(file_name, desired_file):
    mesh = graphlow.read(file_name)
    surface = mesh.convert_to_surface()
    desired = graphlow.read(desired_file)
    np.testing.assert_almost_equal(surface.mesh.points, desired.mesh.points)
    np.testing.assert_array_equal(surface.mesh.cells, desired.mesh.cells)

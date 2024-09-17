import pathlib
import shutil

import numpy as np
import pytest
import pyvista as pv
import torch

import graphlow
from graphlow.util.logger import get_logger

logger = get_logger(__name__)

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
    np.testing.assert_almost_equal(surface.pvmesh.points, desired.pvmesh.points)
    np.testing.assert_array_equal(surface.pvmesh.cells, desired.pvmesh.cells)


@pytest.mark.parametrize(
    "file_name, desired_facets, desired_fc_inc",
    [
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            np.array([
                4, 0, 4, 7, 3,   #  0
                4, 1, 2, 6, 5,   #  1
                4, 0, 1, 5, 4,   #  2
                4, 3, 7, 6, 2,   #  3
                4, 0, 3, 2, 1,   #  4
                4, 4, 5, 6, 7,   #  5
                4, 4, 8, 11, 7,  #  6
                4, 5, 6, 10, 9,  #  7
                4, 4, 5, 9, 8,   #  8
                4, 7, 11, 10, 6, #  9
                4, 8, 9, 10, 11  # 10
            ]),
            np.array([
                # 0  1
                [1,  0],  #  0
                [1,  0],  #  1
                [1,  0],  #  2
                [1,  0],  #  3
                [1,  0],  #  4
                [1, -1],  #  5
                [0,  1],  #  6
                [0,  1],  #  7
                [0,  1],  #  8
                [0,  1],  #  9
                [0,  1],  # 10
            ]),
        ),
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            np.array([
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
                3, 11, 9, 12        # 13
            ]),
            np.array([
                # 0  1  2
                [1,  0,  0],  #  0
                [1, -1,  0],  #  1
                [1,  0,  0],  #  2
                [1,  0,  0],  #  3
                [1,  0, -1],  #  4
                [1,  0,  0],  #  5
                [0,  1,  0],  #  6
                [0,  1,  0],  #  7
                [0,  1,  0],  #  8
                [0,  1,  0],  #  9
                [0,  1,  0],  # 10
                [0,  0,  1],  # 11
                [0,  0,  1],  # 12
                [0,  0,  1],  # 13
            ]),
        ),
    ],
)
def test___extract_facets_impl(file_name, desired_facets, desired_fc_inc):
    mesh = graphlow.read(file_name)
    polydata, scipy_fc_inc = mesh._extract_facets_impl()

    # polydata
    np.testing.assert_array_equal(polydata.faces, desired_facets)

    # fc inc matrix
    np.testing.assert_array_equal(
        scipy_fc_inc.toarray().astype(int), desired_fc_inc
    )


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


@pytest.mark.parametrize(
    "file_name",
    [
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
    ],
)
def test__optimize(file_name):
    # Optimization setting
    n_optimization = 500
    print_period = 10
    target_lz = 3.
    weight_l2 = 1e-6
    desired_coeff = np.array([0., 0., 2.])

    output_directory = pathlib.Path("tests/outputs/optimization")
    if output_directory.exists():
        shutil.rmtree(output_directory)

    def cost_function(deformed_points, deformation):
        z = deformed_points[:, -1]
        lz = torch.max(z) - torch.min(z)
        loss_lz = (lz - target_lz)**2
        norm_deformation = torch.einsum('ip,ip->', deformation, deformation)
        return loss_lz + weight_l2 * norm_deformation

    # Initialize
    mesh = graphlow.read(file_name)
    points = mesh.points
    deform_coeff = torch.nn.Parameter(torch.rand(3))
    optimizer = torch.optim.Adam([deform_coeff], lr=1e-2)

    # Optimization loop
    logger.info("\n   i,       cx,       cy,       cz,        cost")
    for i in range(1, n_optimization + 1):
        optimizer.zero_grad()

        deformation = torch.einsum('np,p->np', points, deform_coeff)
        deformed_points = points + deformation

        cost = cost_function(deformed_points, deformation)

        if i % print_period == 0:
            cx = deform_coeff[0]
            cy = deform_coeff[1]
            cz = deform_coeff[2]
            logger.info(f"{i:4d}, {cx:8.5f}, {cy:8.5f}, {cz:8.5f}, {cost:.5e}")
            mesh.dict_point_tensor.update(
                {"deformation": deformation}, overwrite=True)
            mesh.save(
                output_directory / f"mesh.{i:08d}.vtu",
                overwrite_file=True, overwrite_features=True)

        cost.backward()
        optimizer.step()

    np.testing.assert_almost_equal(
        deform_coeff.detach().numpy(), desired_coeff, decimal=2)


@pytest.mark.parametrize(
    "file_name",
    [
        pathlib.Path("tests/data/vts/cube/mesh.vts"),
    ],
)
def test__optimize_ball(file_name):
    # Optimization setting
    n_optimization = 10000
    print_period = 100
    weight_norm_constraint = 1e-2
    n_hidden = 16
    desired_radius = .7

    output_directory = pathlib.Path("tests/outputs/ball_optimization")
    if output_directory.exists():
        shutil.rmtree(output_directory)

    # Initialize
    pv_mesh = pv.read(file_name)
    pv_mesh.points = pv_mesh.points - np.mean(
        pv_mesh.points, axis=0, keepdims=True)  # Center mesh position

    mesh = graphlow.GraphlowMesh(pv_mesh)
    initial_points = mesh.points.clone()

    surface = mesh.extract_surface()
    surface_initial_points = surface.points.clone()

    w1 = torch.nn.Parameter(torch.rand(3, n_hidden))
    w2 = torch.nn.Parameter(torch.rand(n_hidden, 3))
    optimizer = torch.optim.Adam([w1, w2], lr=1e-2)

    def cost_function(deformed_points, surface_deformed_points):
        mesh.dict_point_tensor.update({
            'points': deformed_points}, overwrite=True)
        surface.dict_point_tensor.update({
            'points': surface_deformed_points}, overwrite=True)
        norms = torch.linalg.norm(surface_deformed_points, dim=1)
        deformation = deformed_points - initial_points
        norm_constraint = torch.mean(deformation * deformation)
        return torch.mean((norms - desired_radius)**2) \
            + weight_norm_constraint * norm_constraint

    def compute_deformed_points(points):
        hidden = torch.tanh(torch.einsum('np,pq->nq', points, w1))
        deformation = torch.einsum('np,pq->nq', hidden, w2)
        return points + deformation

    # Optimization loop
    logger.info("\n   i,         cost")
    for i in range(1, n_optimization + 1):
        optimizer.zero_grad()

        deformed_points = compute_deformed_points(initial_points)
        surface_deformed_points = compute_deformed_points(
            surface_initial_points)

        cost = cost_function(deformed_points, surface_deformed_points)

        if i % print_period == 0:
            logger.info(f"{i:4d}, {cost:.5e}")
            mesh.dict_point_tensor.update(
                {"deformation": deformed_points - initial_points},
                overwrite=True)
            mesh.save(
                output_directory / f"mesh.{i:08d}.vtu",
                overwrite_file=True, overwrite_features=True)

        cost.backward()
        optimizer.step()

    actual_radius = torch.mean(
        torch.norm(surface_deformed_points, dim=1)).detach().numpy()
    np.testing.assert_almost_equal(actual_radius, desired_radius, decimal=3)

@pytest.mark.parametrize(
    "file_name",
    [
        pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
        pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
    ],
)
def test__use_cache(file_name):
    mesh = graphlow.read(file_name)
    cell_point_incidence = mesh.compute_cell_point_incidence()
    cached_cell_point_incidence = mesh.compute_cell_point_incidence()
    np.testing.assert_array_equal(cell_point_incidence.to_dense().numpy(),
                                  cached_cell_point_incidence.to_dense().numpy())

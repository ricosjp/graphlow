
import pathlib
import shutil

import numpy as np
import pytest
import pyvista as pv
import torch

import graphlow


@pytest.mark.parametrize(
    "file_name",
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
    ],
)
def test__compute_area(file_name):
    pv_vol = graphlow.read(file_name)
    pv_surf = pv_vol.extract_surface()
    cell_areas = pv_surf.compute_area()
    desired = pv_surf.mesh.compute_cell_sizes().cell_data["Area"]
    np.testing.assert_almost_equal(cell_areas.detach().numpy(), desired, decimal=4)


@pytest.mark.parametrize(
    "file_name",
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
    ],
)
def test__compute_volume(file_name):
    pv_vol = graphlow.read(file_name)
    cell_volumes = pv_vol.compute_volume()

    # See below for why `compute_cell_quality` is used instead of `compute_cell_sizes`
    # https://colab.research.google.com/drive/1ZkMbVfN-74ZXbDFO2ocva-JEYin6Ux4b?usp=sharing
    desired = np.abs(pv_vol.mesh.compute_cell_quality(quality_measure='volume').cell_data["CellQuality"])

    # fix desired for polyhedron cell 
    # because vtkCellQuality doesn't support vtkPolyhedron
    for i in range(pv_vol.mesh.n_cells):
        cell = pv_vol.mesh.get_cell(i)
        celltype = cell.type
        if celltype == pv.CellType.POLYHEDRON:
            tet_cell_grid = pv_vol.tetrahedralize_cell(cell)
            tet_cell_volumes = np.abs(tet_cell_grid.compute_cell_quality(quality_measure='volume').cell_data["CellQuality"])
            desired[i] = np.sum(tet_cell_volumes)
    np.testing.assert_almost_equal(cell_volumes.detach().numpy(), desired, decimal=4)


@pytest.mark.parametrize(
    "file_name",
    [
        pathlib.Path("tests/data/vts/cube/mesh.vts"),
        pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
    ],
)
def test__optimize_area_volume(file_name):
    # Optimization setting
    n_optimization = 10000
    print_period = 100
    weight_norm_constraint = 1e-2
    weight_volume_constraint = 1e-2
    n_hidden = 16

    output_directory = pathlib.Path("tests/outputs/geometry_optimization")
    if output_directory.exists():
        shutil.rmtree(output_directory)

    # Initialize
    pv_mesh = pv.read(file_name)
    pv_mesh.points = pv_mesh.points - np.mean(
        pv_mesh.points, axis=0, keepdims=True)  # Center mesh position

    mesh = graphlow.GraphlowMesh(pv_mesh)
    initial_volume = mesh.compute_volume().clone()
    initial_total_volume = torch.sum(initial_volume)
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

        volume = mesh.compute_volume()  # TODO Define
        area = surface.compute_area()  # TODO Define
        deformation = deformed_points - initial_points

        cost_area = torch.sum(area)
        volume_constraint = torch.sum((volume - initial_volume)**2)
        norm_constraint = torch.mean(deformation * deformation)

        return cost_area + weight_volume_constraint * volume_constraint \
            + weight_norm_constraint * norm_constraint

    def compute_deformed_points(points):
        hidden = torch.tanh(torch.einsum('np,pq->nq', points, w1))
        deformation = torch.einsum('np,pq->nq', hidden, w2)
        return points + deformation

    # Optimization loop
    print("\n   i,         cost")
    for i in range(1, n_optimization + 1):
        optimizer.zero_grad()

        deformed_points = compute_deformed_points(initial_points)
        surface_deformed_points = compute_deformed_points(
            surface_initial_points)

        cost = cost_function(deformed_points, surface_deformed_points)

        if i % print_period == 0:
            print(
                f"{i:4d}, {cost:.5e}")
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
    desired_radius = (initial_total_volume.numpy() * 3 / 4 / np.pi)**(1/3)
    np.testing.assert_almost_equal(actual_radius, desired_radius, decimal=3)

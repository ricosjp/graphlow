
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
    np.testing.assert_almost_equal(
        cell_areas.detach().numpy(), desired, decimal=4)


@pytest.mark.parametrize(
    "file_name",
    [
        pathlib.Path("tests/data/vts/cube/mesh.vts"),
        pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
    ],
)
def test__compute_volume(file_name):
    pv_vol = graphlow.read(file_name)
    cell_volumes = pv_vol.compute_volume()
    desired = pv_vol.mesh.compute_cell_sizes().cell_data["Volume"]
    np.testing.assert_almost_equal(
        cell_volumes.detach().numpy(), desired, decimal=4)


@pytest.mark.parametrize(
    "file_name, n_optimization, use_bias, threshold",
    [
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            1000,
            False,
            1.,
        ),
        (
            pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
            10000,
            False,
            .05,
        ),
        (
            pathlib.Path("tests/data/vts/cube/mesh.vts"),
            2000,
            False,
            .05,
        ),
    ],
)
def test__optimize_area_volume(file_name, n_optimization, use_bias, threshold):
    # Optimization setting
    print_period = int(n_optimization / 100)
    weight_norm_constraint = 1.
    weight_volume_constraint = 10.
    n_hidden = 64
    deformation_factor = 1.
    lr = 1e-2
    output_activation = torch.nn.Identity()

    output_directory = pathlib.Path("tests/outputs/geometry_optimization") \
        / file_name.parent.name
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
    surface_initial_area = surface.compute_area().clone()
    surface_initial_total_area = torch.sum(surface_initial_area)
    surface_initial_points = surface.points.clone()

    w1 = torch.nn.Parameter(torch.randn(3, n_hidden) / n_hidden**.5)
    w2 = torch.nn.Parameter(torch.randn(n_hidden, 3) / n_hidden**.5)
    if use_bias:
        b1 = torch.nn.Parameter(torch.randn(n_hidden) / n_hidden)
        b2 = torch.nn.Parameter(torch.randn(3) / 3)
        params = [w1, b1, w2, b2]
    else:
        b1 = 0.
        b2 = 0.
        params = [w1, w2]
    optimizer = torch.optim.Adam(params, lr=lr)

    def cost_function(deformed_points, surface_deformed_points):
        mesh.dict_point_tensor.update({
            'points': deformed_points}, overwrite=True)
        surface.dict_point_tensor.update({
            'points': surface_deformed_points}, overwrite=True)

        volume = mesh.compute_volume()
        total_volume = torch.sum(volume)
        area = surface.compute_area()
        total_area = torch.sum(area)
        deformation = deformed_points - initial_points

        if torch.any(volume < 1e-3 * initial_total_volume / mesh.n_cells):
            return None, None, None

        cost_area = total_area / surface_initial_total_area
        volume_constraint = (
            (total_volume - initial_total_volume) / initial_total_volume)**2
        norm_constraint = torch.exp(torch.mean(
            deformation * deformation / initial_total_volume**(2/3)))

        return cost_area + weight_volume_constraint * volume_constraint \
            + weight_norm_constraint * norm_constraint, \
            total_area, total_volume

    def compute_deformed_points(points):
        hidden = torch.tanh(torch.einsum('np,pq->nq', points, w1) + b1)
        deformation = output_activation(
            torch.einsum('np,pq->nq', hidden, w2) + b2)
        return points + deformation * deformation_factor

    deformed_points = compute_deformed_points(initial_points)
    mesh.dict_point_tensor.update(
        {"deformation": deformed_points - initial_points},
        overwrite=True)
    mesh.save(
        output_directory / f"mesh.{0:08d}.vtu",
        overwrite_file=True, overwrite_features=True)

    # Optimization loop
    print(f"\ninitial volume: {initial_total_volume:.5f}")
    print("     i,        area, volume ratio,        cost")
    for i in range(1, n_optimization + 1):
        optimizer.zero_grad()

        deformed_points = compute_deformed_points(initial_points)
        surface_deformed_points = compute_deformed_points(
            surface_initial_points)

        cost, area, volume = cost_function(
            deformed_points, surface_deformed_points)
        if cost is None:
            deformation_factor = deformation_factor * .9
            print(f"update deformation_factor: {deformation_factor}")
            continue

        if i % print_period == 0:
            volume_ratio = volume / initial_total_volume
            print(f"{i:6d}, {area:.5e},  {volume_ratio:.5e}, {cost:.5e}")
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
    relative_error = (actual_radius - desired_radius) ** 2 / desired_radius**2
    assert relative_error < threshold

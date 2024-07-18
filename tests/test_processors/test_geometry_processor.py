
import pathlib
import shutil

import numpy as np
import pytest
import pyvista as pv
import torch

import graphlow


def tetrahedralize_cell_for_test(cell):
    pid_map = {global_pid: local_pid for local_pid, global_pid in enumerate(cell.point_ids)}

    cell_n_points = cell.n_points

    cell_points = np.vstack((cell.points, cell.center))
    cell_center_pid = cell_n_points
    cell_n_points += 1

    tet_cells_list = []
    for face in cell.faces:
        cell_points = np.vstack((cell_points, face.center))
        face_center_pid = cell_n_points
        cell_n_points += 1
        for edge in face.edges:
            edge_global_pids = edge.point_ids
            edge_local_pids = [pid_map[global_pid] for global_pid in edge_global_pids]
            tet_cell = np.array([4, edge_local_pids[1], edge_local_pids[0], face_center_pid, cell_center_pid])
            tet_cells_list.append(tet_cell)

    tet_cells = np.asarray(tet_cells_list)
    tet_celltypes = np.full(tet_cells.shape[0], pv.CellType.TETRA)
    return pv.UnstructuredGrid(tet_cells, tet_celltypes, cell_points)


@pytest.mark.parametrize(
    "points, unit_cell, unit_celltype, desired_cells",
    [
        (
            np.array([
                [0.0, 0.0, 0.0], # 0
                [2.0, 0.0, 0.0], # 1
                [2.0, 2.0, 0.0], # 2
                [0.0, 2.0, 0.0], # 3
                [0.0, 0.0, 2.0], # 4
                [2.0, 0.0, 2.0], # 5
                [2.0, 2.0, 2.0], # 6
                [0.0, 2.0, 2.0], # 7
                [1.0, 1.0, 1.0], # 8  cell center
                [0.0, 1.0, 1.0], # 9  face center x=0 yz
                [2.0, 1.0, 1.0], # 10 face center x=2 yz
                [1.0, 0.0, 1.0], # 11 face center y=0 zx
                [1.0, 2.0, 1.0], # 12 face center y=2 zx
                [1.0, 1.0, 0.0], # 13 face center z=0 xy
                [1.0, 1.0, 2.0], # 14 face center z=2 xy
            ]),
            np.array([8, 0, 1, 2, 3, 4, 5, 6, 7]),
            np.array([pv.CellType.HEXAHEDRON]),
            np.array([
                [4, 4, 0, 9, 8], # face x=0 yz
                [4, 7, 4, 9, 8],
                [4, 3, 7, 9, 8],
                [4, 0, 3, 9, 8],
                [4, 2, 1, 10, 8], # face x=2 yz
                [4, 6, 2, 10, 8],
                [4, 5, 6, 10, 8],
                [4, 1, 5, 10, 8],
                [4, 1, 0, 11, 8], # face y=0 zx
                [4, 5, 1, 11, 8],
                [4, 4, 5, 11, 8],
                [4, 0, 4, 11, 8],
                [4, 7, 3, 12, 8], # face y=2 zx
                [4, 6, 7, 12, 8],
                [4, 2, 6, 12, 8],
                [4, 3, 2, 12, 8],
                [4, 3, 0, 13, 8], # face z=0 xy
                [4, 2, 3, 13, 8],
                [4, 1, 2, 13, 8],
                [4, 0, 1, 13, 8],
                [4, 5, 4, 14, 8], # face z=2 xy
                [4, 6, 5, 14, 8],
                [4, 7, 6, 14, 8],
                [4, 4, 7, 14, 8],
            ]),
        )
    ],
)
def test__tetrahedralize_cell(points, unit_cell, unit_celltype, desired_cells):
    desired_celltypes = np.full(desired_cells.shape[0], pv.CellType.TETRA)
    desired_grid = pv.UnstructuredGrid(desired_cells, desired_celltypes, points)

    unit_grid = pv.UnstructuredGrid(unit_cell, unit_celltype, points)
    tet_cell_grid = tetrahedralize_cell_for_test(unit_grid.get_cell(0))
    np.testing.assert_almost_equal(tet_cell_grid.points, desired_grid.points)
    np.testing.assert_array_equal(tet_cell_grid.cells, desired_grid.cells)


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
            tet_cell_grid = tetrahedralize_cell_for_test(cell)
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

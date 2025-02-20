import pathlib
import shutil

import numpy as np
import pytest
import pyvista as pv
import torch

import graphlow
from graphlow.util.logger import get_logger

logger = get_logger(__name__)


def tetrahedralize_cell_for_test(cell: pv.Cell) -> pv.UnstructuredGrid:
    pid_map = {
        global_pid: local_pid
        for local_pid, global_pid in enumerate(cell.point_ids)
    }

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
            edge_local_pids = [
                pid_map[global_pid] for global_pid in edge_global_pids
            ]
            tet_cell = np.array(
                [
                    4,
                    edge_local_pids[1],
                    edge_local_pids[0],
                    face_center_pid,
                    cell_center_pid,
                ]
            )
            tet_cells_list.append(tet_cell)

    tet_cells = np.asarray(tet_cells_list)
    tet_celltypes = np.full(tet_cells.shape[0], pv.CellType.TETRA)
    return pv.UnstructuredGrid(tet_cells, tet_celltypes, cell_points)


@pytest.mark.parametrize(
    "points, unit_cell, unit_celltype, desired_cells",
    [
        (
            np.array(
                [
                    [0.0, 0.0, 0.0],  # 0
                    [2.0, 0.0, 0.0],  # 1
                    [2.0, 2.0, 0.0],  # 2
                    [0.0, 2.0, 0.0],  # 3
                    [0.0, 0.0, 2.0],  # 4
                    [2.0, 0.0, 2.0],  # 5
                    [2.0, 2.0, 2.0],  # 6
                    [0.0, 2.0, 2.0],  # 7
                    [1.0, 1.0, 1.0],  # 8  cell center
                    [0.0, 1.0, 1.0],  # 9  face center x=0 yz
                    [2.0, 1.0, 1.0],  # 10 face center x=2 yz
                    [1.0, 0.0, 1.0],  # 11 face center y=0 zx
                    [1.0, 2.0, 1.0],  # 12 face center y=2 zx
                    [1.0, 1.0, 0.0],  # 13 face center z=0 xy
                    [1.0, 1.0, 2.0],  # 14 face center z=2 xy
                ]
            ),
            np.array([8, 0, 1, 2, 3, 4, 5, 6, 7]),
            np.array([pv.CellType.HEXAHEDRON]),
            np.array(
                [
                    [4, 4, 0, 9, 8],  # face x=0 yz
                    [4, 7, 4, 9, 8],
                    [4, 3, 7, 9, 8],
                    [4, 0, 3, 9, 8],
                    [4, 2, 1, 10, 8],  # face x=2 yz
                    [4, 6, 2, 10, 8],
                    [4, 5, 6, 10, 8],
                    [4, 1, 5, 10, 8],
                    [4, 1, 0, 11, 8],  # face y=0 zx
                    [4, 5, 1, 11, 8],
                    [4, 4, 5, 11, 8],
                    [4, 0, 4, 11, 8],
                    [4, 7, 3, 12, 8],  # face y=2 zx
                    [4, 6, 7, 12, 8],
                    [4, 2, 6, 12, 8],
                    [4, 3, 2, 12, 8],
                    [4, 3, 0, 13, 8],  # face z=0 xy
                    [4, 2, 3, 13, 8],
                    [4, 1, 2, 13, 8],
                    [4, 0, 1, 13, 8],
                    [4, 5, 4, 14, 8],  # face z=2 xy
                    [4, 6, 5, 14, 8],
                    [4, 7, 6, 14, 8],
                    [4, 4, 7, 14, 8],
                ]
            ),
        )
    ],
)
def test__tetrahedralize_cell(
    points: np.ndarray,
    unit_cell: np.ndarray,
    unit_celltype: np.ndarray,
    desired_cells: np.ndarray,
):
    desired_celltypes = np.full(desired_cells.shape[0], pv.CellType.TETRA)
    desired_grid = pv.UnstructuredGrid(desired_cells, desired_celltypes, points)

    unit_grid = pv.UnstructuredGrid(unit_cell, unit_celltype, points)
    tet_cell_grid = tetrahedralize_cell_for_test(unit_grid.get_cell(0))
    np.testing.assert_almost_equal(tet_cell_grid.points, desired_grid.points)
    np.testing.assert_array_equal(tet_cell_grid.cells, desired_grid.cells)


@pytest.mark.parametrize(
    "file_name",
    [
        pathlib.Path("tests/data/vtu/primitive_cell/cube.vtu"),
    ],
)
def test__convert_elemental2nodal_mean(file_name: pathlib.Path):
    volmesh = graphlow.read(file_name)
    cell_volumes = volmesh.compute_volumes()
    nodal_vols = volmesh.convert_elemental2nodal(cell_volumes)
    desired = (
        volmesh.pvmesh.compute_cell_quality(quality_measure="volume")
        .cell_data_to_point_data()
        .point_data["CellQuality"]
    )
    np.testing.assert_almost_equal(nodal_vols.detach().numpy(), desired)


@pytest.mark.parametrize(
    "file_name, desired",
    [
        (
            pathlib.Path("tests/data/vtu/primitive_cell/cube.vtu"),
            0.125 * np.ones(8),
        )
    ],
)
def test__convert_elemental2nodal_effective(
    file_name: pathlib.Path, desired: np.ndarray
):
    volmesh = graphlow.read(file_name)
    cell_volumes = volmesh.compute_volumes()
    nodal_vols = volmesh.convert_elemental2nodal(cell_volumes, mode="effective")
    np.testing.assert_almost_equal(nodal_vols.detach().numpy(), desired)


@pytest.mark.parametrize(
    "file_name",
    [
        pathlib.Path("tests/data/vtu/primitive_cell/cube.vtu"),
    ],
)
def test__convert_nodal2elemental_mean(file_name: pathlib.Path):
    volmesh = graphlow.read(file_name)
    elem_points = volmesh.convert_nodal2elemental(volmesh.points)
    desired = volmesh.pvmesh.cell_centers().points
    np.testing.assert_almost_equal(elem_points.detach().numpy(), desired)


@pytest.mark.parametrize(
    "file_name, desired",
    [
        (
            pathlib.Path("tests/data/vtu/primitive_cell/cube.vtu"),
            np.full((1, 3), 4.0),
        )
    ],
)
def test__convert_nodal2elemental_effective(
    file_name: pathlib.Path, desired: np.ndarray
):
    volmesh = graphlow.read(file_name)
    elem_points = volmesh.convert_nodal2elemental(
        volmesh.points, mode="effective"
    )
    np.testing.assert_almost_equal(elem_points.detach().numpy(), desired)


@pytest.mark.parametrize(
    "file_name, desired",
    [
        (
            pathlib.Path("tests/data/vtu/primitive_cell/cuboid.vtu"),
            np.array(
                [
                    [0.0, -3.0, 0.0],
                    [0.0, 0.0, -6.0],
                    [-2.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 6.0],
                ]
            ),
        )
    ],
)
def test__compute_area_vecs(file_name: pathlib.Path, desired: np.ndarray):
    volmesh = graphlow.read(file_name)
    surfmesh = volmesh.extract_surface()
    area_vecs = surfmesh.compute_area_vecs()
    np.testing.assert_almost_equal(area_vecs.detach().numpy(), desired)


@pytest.mark.parametrize(
    "file_name, data, n_hop, desired",
    [
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            np.array(
                # 0  1  2  3  4  5  6  7  8  9 10 11
                [1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 5, 1]
            ),
            1,
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ),
        (
            pathlib.Path("tests/data/vtk/hex/mesh.vtk"),
            np.array(
                # 0  1  2  3  4  5  6  7  8  9 10 11
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            ),
            4,
            np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]),
        ),
    ],
)
def test__compute_median_nodal(
    file_name: pathlib.Path, data: np.ndarray, n_hop: int, desired: np.ndarray
):
    volmesh = graphlow.read(file_name)
    filtered_data = volmesh.compute_median(
        torch.from_numpy(data), mode="nodal", n_hop=n_hop
    )
    np.testing.assert_almost_equal(filtered_data.detach().numpy(), desired)


@pytest.mark.parametrize(
    "file_name, data, n_hop, desired",
    [
        (
            pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
            np.array(
                # 0  1  2  3  4  5  6
                [1, 1, 1, 1, 3, 1, 1]
            ),
            1,
            np.array([1, 1, 1, 1, 1, 1, 1]),
        ),
        (
            pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
            np.array(
                # 0  1  2  3  4  5  6
                [0, 1, 2, 3, 4, 5, 6]
            ),
            3,
            np.array([3, 3, 3, 3, 3, 3, 3]),
        ),
    ],
)
def test__compute_median_for_elemental(
    file_name: pathlib.Path, data: np.ndarray, n_hop: int, desired: np.ndarray
):
    volmesh = graphlow.read(file_name)
    filtered_data = volmesh.compute_median(
        torch.from_numpy(data), mode="elemental", n_hop=n_hop
    )
    np.testing.assert_almost_equal(filtered_data.detach().numpy(), desired)


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
        pathlib.Path("tests/data/vtu/cube/large.vtu"),
    ],
)
def test__compute_areas(file_name: pathlib.Path):
    volmesh = graphlow.read(file_name)
    surfmesh = volmesh.extract_surface()
    cell_areas = surfmesh.compute_areas()
    desired = surfmesh.pvmesh.compute_cell_sizes().cell_data["Area"]
    np.testing.assert_almost_equal(
        cell_areas.detach().numpy(), desired, decimal=4
    )


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
        pathlib.Path("tests/data/vtu/cube/large.vtu"),
    ],
)
def test__compute_volumes(file_name: pathlib.Path):
    volmesh = graphlow.read(file_name)
    cell_volumes = volmesh.compute_volumes()

    # See below for why `compute_cell_quality`is used
    # instead of `compute_cell_sizes`
    # https://colab.research.google.com/drive/1ZkMbVfN-74ZXbDFO2ocva-JEYin6Ux4b?usp=sharing
    desired = np.abs(
        volmesh.pvmesh.compute_cell_quality(quality_measure="volume").cell_data[
            "CellQuality"
        ]
    )

    # fix desired for polyhedron cell
    # because vtkCellQuality doesn't support vtkPolyhedron
    for i in range(volmesh.pvmesh.n_cells):
        cell = volmesh.pvmesh.get_cell(i)
        celltype = cell.type
        if celltype == pv.CellType.POLYHEDRON:
            tet_cell_grid = tetrahedralize_cell_for_test(cell)
            tet_cell_volumes = np.abs(
                tet_cell_grid.compute_cell_quality(
                    quality_measure="volume"
                ).cell_data["CellQuality"]
            )
            desired[i] = np.sum(tet_cell_volumes)
    np.testing.assert_almost_equal(
        cell_volumes.detach().numpy(), desired, decimal=4
    )


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
def test__compute_signed_volumes(file_name: pathlib.Path):
    volmesh = graphlow.read(file_name)
    cell_volumes = volmesh.compute_volumes()

    # See below for why `compute_cell_quality`is used
    # instead of `compute_cell_sizes`
    # https://colab.research.google.com/drive/1ZkMbVfN-74ZXbDFO2ocva-JEYin6Ux4b?usp=sharing
    desired = volmesh.pvmesh.compute_cell_quality(
        quality_measure="volume"
    ).cell_data["CellQuality"]

    # fix desired for polyhedron cell
    # because vtkCellQuality doesn't support vtkPolyhedron
    celltypes = volmesh.pvmesh.celltypes
    poly_mask = celltypes == pv.CellType.POLYHEDRON
    if np.any(poly_mask):
        polys = volmesh.pvmesh.extract_cells(
            poly_mask
        ).cast_to_unstructured_grid()
        # HACK `compute_cell_quality` should be used here,
        # but since it would follow the same implementation as we developed,
        # `compute_cell_sizes` is being used instead.
        # `compute_cell_sizes` returns correct results for non-twisted cells,
        # so there is generally no issue.
        desired[poly_mask] = polys.compute_cell_sizes().cell_data["Volume"]
    np.testing.assert_almost_equal(
        cell_volumes.detach().numpy(), desired, decimal=4
    )


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
def test__compute_normals(file_name: pathlib.Path):
    volmesh = graphlow.read(file_name)
    surf = volmesh.extract_surface()
    facets = volmesh.extract_facets()

    # implemented
    surf_normals = surf.compute_normals().detach().numpy()
    facets_normals = facets.compute_normals().detach().numpy()

    # desired
    desired_surf_normals = (
        surf.pvmesh.extract_surface()
        .compute_normals(
            cell_normals=True,
        )
        .cell_data["Normals"]
    )

    desired_facets_normals = (
        facets.pvmesh.extract_surface()
        .compute_normals(
            cell_normals=True,
            # consistent_normals is invalid for internal face
            consistent_normals=False,
        )
        .cell_data["Normals"]
    )

    # check
    np.testing.assert_almost_equal(
        surf_normals, desired_surf_normals, decimal=4
    )
    np.testing.assert_almost_equal(
        facets_normals, desired_facets_normals, decimal=4
    )


@pytest.mark.parametrize(
    "file_name",
    [
        # primitives
        pathlib.Path("tests/data/vtu/primitive_cell/cuboid.vtu"),
    ],
)
def test__volume_gradient(file_name: pathlib.Path):
    volmesh = graphlow.read(file_name)
    volmesh.points.requires_grad_(True)
    cell_volumes = volmesh.compute_volumes()
    total_volume = torch.sum(cell_volumes)
    total_volume.backward()

    vol_grad = volmesh.points.grad

    for i in range(volmesh.pvmesh.n_cells):
        cell = volmesh.pvmesh.get_cell(i)
        for face in cell.faces:
            pids = face.point_ids
            # TODO: To obtain results for any plane,
            # take the dot product with the normal and then sum the results.
            dV = torch.abs(torch.sum(vol_grad[pids]))
            area = (
                face.cast_to_unstructured_grid()
                .compute_cell_quality("area")
                .cell_data["CellQuality"]
            )
            np.testing.assert_equal(dV, area)


@pytest.mark.parametrize(
    "file_name, n_optimization, use_bias, threshold",
    [
        (
            pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
            1000,
            False,
            1.0,
        ),
        (
            pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
            10000,
            False,
            0.05,
        ),
        (
            pathlib.Path("tests/data/vts/cube/mesh.vts"),
            2000,
            False,
            0.05,
        ),
    ],
)
def test__optimize_area_volume(
    file_name: pathlib.Path,
    n_optimization: int,
    use_bias: bool,
    threshold: float,
    pytestconfig: pytest.Config,
):
    # Optimization setting
    print_period = int(n_optimization / 100)
    weight_norm_constraint = 1.0
    weight_volume_constraint = 10.0
    n_hidden = 64
    deformation_factor = 1.0
    lr = 1e-2
    output_activation = torch.nn.Identity()

    output_directory = (
        pathlib.Path("tests/outputs/geometry_optimization")
        / file_name.parent.name
    )
    if output_directory.exists():
        shutil.rmtree(output_directory)

    # Initialize
    pv_mesh = pv.read(file_name)
    pv_mesh.points = pv_mesh.points - np.mean(
        pv_mesh.points, axis=0, keepdims=True
    )  # Center mesh position

    mesh = graphlow.GraphlowMesh(pv_mesh)
    initial_volumes = mesh.compute_volumes().clone()
    initial_total_volume = torch.sum(initial_volumes)
    initial_points = mesh.points.clone()

    surface = mesh.extract_surface()
    surface_initial_areas = surface.compute_areas().clone()
    surface_initial_total_area = torch.sum(surface_initial_areas)
    surface_initial_points = surface.points.clone()

    w1 = torch.nn.Parameter(torch.randn(3, n_hidden) / n_hidden**0.5)
    w2 = torch.nn.Parameter(torch.randn(n_hidden, 3) / n_hidden**0.5)
    if use_bias:
        b1 = torch.nn.Parameter(torch.randn(n_hidden) / n_hidden)
        b2 = torch.nn.Parameter(torch.randn(3) / 3)
        params = [w1, b1, w2, b2]
    else:
        b1 = 0.0
        b2 = 0.0
        params = [w1, w2]
    optimizer = torch.optim.Adam(params, lr=lr)

    def cost_function(
        deformed_points: torch.Tensor, surface_deformed_points: torch.Tensor
    ) -> torch.Tensor:
        mesh.dict_point_tensor.update(
            {"points": deformed_points}, overwrite=True
        )
        surface.dict_point_tensor.update(
            {"points": surface_deformed_points}, overwrite=True
        )

        volumes = mesh.compute_volumes()
        total_volume = torch.sum(volumes)
        areas = surface.compute_areas()
        total_area = torch.sum(areas)
        deformation = deformed_points - initial_points

        if torch.any(volumes < 1e-3 * initial_total_volume / mesh.n_cells):
            return None, None, None

        cost_area = total_area / surface_initial_total_area
        volume_constraint = (
            (total_volume - initial_total_volume) / initial_total_volume
        ) ** 2
        norm_constraint = torch.exp(
            torch.mean(
                deformation * deformation / initial_total_volume ** (2 / 3)
            )
        )

        return (
            cost_area
            + weight_volume_constraint * volume_constraint
            + weight_norm_constraint * norm_constraint,
            total_area,
            total_volume,
        )

    def compute_deformed_points(points: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(torch.einsum("np,pq->nq", points, w1) + b1)
        deformation = output_activation(
            torch.einsum("np,pq->nq", hidden, w2) + b2
        )
        return points + deformation * deformation_factor

    deformed_points = compute_deformed_points(initial_points)
    mesh.dict_point_tensor.update(
        {"deformation": deformed_points - initial_points}, overwrite=True
    )
    if pytestconfig.getoption("save"):
        mesh.save(
            output_directory / f"mesh.{0:08d}.vtu",
            overwrite_file=True,
            overwrite_features=True,
        )

    # Optimization loop
    logger.info(f"\ninitial volume: {initial_total_volume:.5f}")
    logger.info("     i,        area, volume ratio,        cost")
    for i in range(1, n_optimization + 1):
        optimizer.zero_grad()

        deformed_points = compute_deformed_points(initial_points)
        surface_deformed_points = compute_deformed_points(
            surface_initial_points
        )

        cost, area, volume = cost_function(
            deformed_points, surface_deformed_points
        )
        if cost is None:
            deformation_factor = deformation_factor * 0.9
            logger.info(f"update deformation_factor: {deformation_factor}")
            continue

        if i % print_period == 0:
            volume_ratio = volume / initial_total_volume
            logger.info(f"{i:6d}, {area:.5e},  {volume_ratio:.5e}, {cost:.5e}")
            mesh.dict_point_tensor.update(
                {"deformation": deformed_points - initial_points},
                overwrite=True,
            )
            if pytestconfig.getoption("save"):
                mesh.save(
                    output_directory / f"mesh.{i:08d}.vtu",
                    overwrite_file=True,
                    overwrite_features=True,
                )

        cost.backward()
        optimizer.step()

    actual_radius = (
        torch.mean(torch.norm(surface_deformed_points, dim=1)).detach().numpy()
    )
    desired_radius = (initial_total_volume.numpy() * 3 / 4 / np.pi) ** (1 / 3)
    relative_error = (actual_radius - desired_radius) ** 2 / desired_radius**2
    assert relative_error < threshold

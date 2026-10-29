"""Solve Poisson's equation through the generic FEM API on a TET10 mesh."""

from pathlib import Path

import numpy as np
import pytest
import pyvista as pv
import scipy.sparse as sp
import torch
from scipy.sparse import linalg

import graphlow


def _promote_tetrahedra_to_tet10(
    linear_grid: pv.UnstructuredGrid,
) -> pv.UnstructuredGrid:
    """Add one shared midpoint per edge, in VTK's TET10 node order."""
    # Nodes 4 through 9 follow edges 01, 12, 20, 03, 13, 23.
    edges = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
    points = list(linear_grid.points.astype(np.float64))
    midpoint_ids: dict[tuple[int, int], int] = {}
    connectivity = []
    for vertices in linear_grid.cells_dict[pv.CellType.TETRA]:
        nodes = vertices.tolist()
        for a, b in edges:
            first, second = sorted((int(vertices[a]), int(vertices[b])))
            edge = (first, second)
            if edge not in midpoint_ids:
                midpoint_ids[edge] = len(points)
                points.append((points[first] + points[second]) / 2)
            nodes.append(midpoint_ids[edge])
        connectivity.append(nodes)
    return pv.UnstructuredGrid(
        {pv.CellType.QUADRATIC_TETRA: np.asarray(connectivity)},
        np.asarray(points),
    )


def test_tet10_poisson_matches_quadratic_solution(
    data_dir: Path, tmp_path: Path, pytestconfig: pytest.Config
):
    """Recover a quadratic solution of -Δu = 12 on the cube [0, 2]^3.

    Prescribe the analytic solution on all six boundary faces. TET10 can
    represent this quadratic exactly, so the error should be at solver
    tolerance, including at unconstrained vertices and edge midpoints.
    """
    device = torch.device(pytestconfig.getoption("device"))
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requested --device=cuda, but CUDA is not available")

    linear_grid = pv.read(data_dir / "vtu/cube/2x2_tet.vtu")
    quadratic_grid = _promote_tetrahedra_to_tet10(linear_grid)
    mesh_path = tmp_path / "tet10_cube.vtu"
    quadratic_grid.save(mesh_path)
    mesh = graphlow.read(
        mesh_path, "phlower", dtype=torch.float64, device=device
    )
    backend = mesh.backend
    cell_type = pv.CellType.QUADRATIC_TETRA
    np.testing.assert_array_equal(
        mesh.topology.unique_cell_types(), [cell_type]
    )
    assert mesh.n_cells == linear_grid.n_cells > 1
    assert mesh.topology.cell_block(cell_type).conn.shape == (mesh.n_cells, 10)

    xyz = backend.to_numpy(mesh.xyz)
    x, y, z = xyz.T
    exact = (
        x * (2 - x)
        + 2 * y * (2 - y)
        + 3 * z * (2 - z)
        + x * y
        + 2 * y * z
        - x * z
    )
    boundary = np.any(np.isclose(xyz, 0) | np.isclose(xyz, 2), axis=1)
    interior = ~boundary
    # Keep unconstrained original vertices and midside nodes in the problem.
    assert interior[: linear_grid.n_points].any()
    assert interior[linear_grid.n_points :].any()
    assert boundary[linear_grid.n_points :].any()
    assert np.any(exact[boundary] == 0)
    assert np.any(exact[boundary] != 0)

    # The source is analytic, independent of the assembled stiffness matrix.
    source = 12 * backend.ones((mesh.n_points, 1), dimension={"L": -2})
    mass = mesh.geometry.cell_local_fem_mass()
    rhs = mesh.geometry.apply_cell_local_fem_matrix(mass, source)
    stiffness = mesh.geometry.cell_local_fem_rigidity()
    global_matrix = mesh.geometry.global_fem_matrix(stiffness)

    boundary_ids = backend.as_index_tensor(np.flatnonzero(boundary))
    boundary_values = backend.as_tensor(exact[boundary], dimension={})
    # Explicitly store prescribed zeros as well as nonzero boundary values.
    dirichlet = backend.as_tensor(
        torch.sparse_coo_tensor(
            indices=torch.stack((boundary_ids, torch.zeros_like(boundary_ids))),
            values=backend.to_torch(boundary_values),
            size=(mesh.n_points, 1),
            check_invariants=False,
        ).coalesce(),
        dimension={},
    )
    matrix, constrained_rhs = (
        mesh.geometry.apply_dirichlet_to_global_fem_matrix(
            global_matrix, rhs, dirichlet
        )
    )
    assert matrix.dimension == constrained_rhs.dimension

    indices = matrix.indices().cpu().numpy()
    sparse_matrix = sp.coo_array(
        (matrix.values().cpu().numpy(), (indices[0], indices[1])),
        shape=tuple(matrix.shape),
    ).tocsr()
    solution, info = linalg.cg(
        sparse_matrix,
        backend.to_numpy(constrained_rhs)[:, 0],
        rtol=1e-12,
        atol=0.0,
        maxiter=10 * mesh.n_points,
    )
    assert info == 0, f"Conjugate gradient did not converge: {info}"
    np.testing.assert_allclose(solution, exact, rtol=1e-10, atol=1e-10)

    # Verify equilibrium on free nodes using the system before imposing BCs.
    numerical = backend.as_tensor(solution[:, None], dimension={})
    residual = backend.to_numpy(global_matrix @ numerical - rhs)[:, 0]
    np.testing.assert_allclose(residual[interior], 0.0, rtol=0.0, atol=1e-10)

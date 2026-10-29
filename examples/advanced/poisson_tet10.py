r"""
Poisson equation with quadratic tetrahedra
==========================================

Solve the same problem as ``tests/e2e/test_fem_poisson.py`` using graphlow's
generic FEM API and a mesh of ten-node tetrahedra (TET10):

.. math::

   -\Delta u = 12 \quad \text{in } \Omega = [0, 2]^3,
   \qquad u = u_{\mathrm{exact}} \quad \text{on } \partial\Omega,

where the prescribed solution is

.. math::

   u_{\mathrm{exact}} = x(2-x) + 2y(2-y) + 3z(2-z) + xy + 2yz - xz.

The source follows analytically from the Laplacian. The boundary values
include both zero and nonzero values on the six faces of the cube.
Coordinates in this expression are numerical values in the mesh's length
unit; ``u`` is dimensionless and the source has dimension ``L^-2``.

TET10 represents this quadratic field exactly on straight-sided cells.
The nodal error should therefore be close to the linear solver tolerance,
even on this coarse mesh. The example reports that error and the residual
at unconstrained nodes, then evaluates the quadratic FEM field throughout
the central ``z = 1`` section. Three panels show:

* the exact solution and the numerical solution on a shared color scale;
* the absolute error on its own color scale, with scientific notation and
  the maximum absolute error over the sampled section.

The display evaluates TET10 shape functions at each sample point. It does
not interpolate a coarse, triangulated slice or interpolate nodal errors.
Thin lines on all three panels mark the original element boundaries in
the section, showing how the field and error relate to the mesh.

Run from the repository with
``uv run python examples/advanced/poisson_tet10.py``. For a calculation
without a plot, import this module and call ``main(plot=False)``. The returned
PyVista grid contains ``solution``, ``exact``, and ``absolute_error`` point
arrays and can be saved with ``grid.save("poisson_tet10.vtu")``.
"""

###############################################################################
# Imports and repository data
# ---------------------------
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scipy.sparse as sp
import torch
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.sparse import linalg

import graphlow
from graphlow.geometry.elements import get_shape_functions


def _repo_root() -> Path:
    """Locate mesh data for both standalone and Sphinx-Gallery execution."""
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        # Sphinx-Gallery does not define __file__ while executing examples.
        for candidate in (Path.cwd(), *Path.cwd().parents):
            if (candidate / "pyproject.toml").is_file() and (
                candidate / "src/graphlow"
            ).is_dir():
                return candidate
    raise FileNotFoundError("Could not locate the graphlow repository root")


###############################################################################
# Step 1: Add shared edge midpoints to the tetrahedral cube
# ---------------------------------------------------------
# Start with the repository's 48-cell cube mesh. Each TET10 has four vertex
# nodes followed by six edge nodes in VTK order. A shared edge must reuse the
# same midpoint ID in every incident cell so the field remains continuous.
def create_tet10_cube() -> pv.UnstructuredGrid:
    """Promote the cube's linear tetrahedra to straight-sided TET10 cells."""
    linear_grid = pv.read(_repo_root() / "tests/data/vtu/cube/2x2_tet.vtu")
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


###############################################################################
# Step 2: Define the exact solution
# ---------------------------------
# All squared terms contribute to the source: -Delta(u) = 2 + 4 + 6 = 12.
# The mixed terms affect the solution and boundary values but have zero
# Laplacian. Interior vertices and edge midpoints are left unconstrained.
def exact_solution(points: np.ndarray) -> np.ndarray:
    """Evaluate the quadratic solution at coordinates shaped (n_points, 3)."""
    x, y, z = points.T
    return (
        x * (2 - x)
        + 2 * y * (2 - y)
        + 3 * z * (2 - z)
        + x * y
        + 2 * y * z
        - x * z
    )


###############################################################################
# Step 3: Assemble, apply boundary conditions, and solve
# ------------------------------------------------------
# Local mass and stiffness matrices are dictionaries keyed by cell type.
# The default quadrature handles quadratic elements. The load is M f and
# the global system is K u = M f. graphlow forms these operators; SciPy's
# conjugate-gradient solver solves the constrained sparse system on the CPU.
def solve_poisson(grid: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    """Solve the cube problem, report errors, and return a grid with results."""
    mesh = graphlow.from_pyvista(grid, "phlower", dtype=torch.float64)
    backend = mesh.backend
    xyz = backend.to_numpy(mesh.xyz)
    exact = exact_solution(xyz)
    boundary = np.any(np.isclose(xyz, 0) | np.isclose(xyz, 2), axis=1)

    source = 12 * backend.ones((mesh.n_points, 1), dimension={"L": -2})
    mass = mesh.geometry.cell_local_fem_mass()
    rhs = mesh.geometry.apply_cell_local_fem_matrix(mass, source)
    stiffness = mesh.geometry.cell_local_fem_rigidity()
    global_matrix = mesh.geometry.global_fem_matrix(stiffness)

    boundary_ids = backend.as_index_tensor(np.flatnonzero(boundary))
    boundary_values = backend.as_tensor(exact[boundary], dimension={})
    # Every stored entry is constrained, including entries whose value is zero.
    # Omitted entries represent the unknown interior values.
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
    if info != 0:
        raise RuntimeError(f"Conjugate gradient did not converge: info={info}")

    error = np.abs(solution - exact)
    numerical = backend.as_tensor(solution[:, None], dimension={})
    residual = backend.to_numpy(global_matrix @ numerical - rhs)[:, 0]
    print(f"TET10 mesh: {mesh.n_cells} cells, {mesh.n_points} nodes")
    print(f"Unconstrained nodes: {np.count_nonzero(~boundary)}")
    print(f"Maximum nodal error: {np.max(error):.3e}")
    print(f"Nodal RMSE: {np.sqrt(np.mean(error**2)):.3e}")
    print(
        f"Maximum interior residual: {np.max(np.abs(residual[~boundary])):.3e}"
    )

    result = grid.copy()
    result.point_data["solution"] = solution
    result.point_data["exact"] = exact
    result.point_data["absolute_error"] = error
    return result


###############################################################################
# Step 4: Evaluate the quadratic field inside each element
# --------------------------------------------------------
# The geometry is affine because the edge nodes are exact midpoints. The
# four vertices therefore determine the physical-to-reference coordinate
# map, while all ten nodal values contribute to the quadratic field.
# Curved TET10 geometry would require an inverse isoparametric mapping.
def evaluate_solution(
    grid: pv.UnstructuredGrid, points: np.ndarray
) -> np.ndarray:
    """Evaluate u_h at (n_samples, 3) points inside this straight TET10 mesh."""
    cell_ids = grid.find_containing_cell(points)
    if np.any(cell_ids < 0):
        raise ValueError("All sample points must lie inside the TET10 mesh")
    connectivity = grid.cells_dict[pv.CellType.QUADRATIC_TETRA][cell_ids]
    vertices = grid.points[connectivity[:, :4]]
    jacobian = (vertices[:, 1:] - vertices[:, :1]).transpose(0, 2, 1)
    reference_points = np.linalg.solve(
        jacobian, (points - vertices[:, 0])[..., None]
    )[..., 0]
    shape_values = get_shape_functions(pv.CellType.QUADRATIC_TETRA)(
        torch.as_tensor(reference_points, dtype=torch.float64)
    ).numpy()
    return np.einsum("qa,qa->q", shape_values, grid["solution"][connectivity])


###############################################################################
# Step 5: Compare exact, numerical, and absolute error sections
# -------------------------------------------------------------
# Surface values are prescribed, so display an interior plane. Use the
# same color scale for both solutions and a separate scale for the error.
# Evaluate the analytical solution directly at the same coordinates.
def draw(grid: pv.UnstructuredGrid, *, screenshot: Path | None = None) -> None:
    """Compare exact, numerical, and error sections; optionally save a PNG."""
    coordinates = np.linspace(0.0, 2.0, 161)
    x, y = np.meshgrid(coordinates, coordinates)
    section_points = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
    numerical = evaluate_solution(grid, section_points).reshape(x.shape)
    analytical = exact_solution(section_points).reshape(x.shape)
    section_error = np.abs(numerical - analytical)
    max_error = float(section_error.max())

    # Straight-sided TET10 cells have the same geometry as their four vertices.
    # Slice that geometry as polygons to draw the original element boundaries.
    # Field values above still use all ten nodes and quadratic shape functions.
    vertices = grid.cells_dict[pv.CellType.QUADRATIC_TETRA][:, :4]
    geometry = pv.UnstructuredGrid({pv.CellType.TETRA: vertices}, grid.points)
    section = geometry.slice(
        normal="z", origin=(0, 0, 1), generate_triangles=False
    ).clean()
    edges = section.extract_all_edges()
    segments = edges.points[edges.lines.reshape(-1, 3)[:, 1:], :2]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(14, 5.5),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    figure.suptitle(
        f"Poisson on TET10 | Section z = 1 | "
        f"{grid.n_cells} cells, {grid.n_points} nodes",
        fontsize=16,
    )

    limits = (
        min(numerical.min(), analytical.min()),
        max(numerical.max(), analytical.max()),
    )
    for ax, values, title in zip(
        axes[:2],
        (analytical, numerical),
        ("Exact\nAnalytical solution", "Numerical\nQuadratic FEM evaluation"),
        strict=True,
    ):
        filled = ax.contourf(
            x, y, values, levels=np.linspace(*limits, 25), cmap="viridis"
        )
        contours = ax.contour(
            x,
            y,
            values,
            levels=np.arange(2, 10),
            colors="white",
            linewidths=0.8,
        )
        ax.clabel(contours, inline=True, fontsize=9, fmt="%g")
        ax.set_title(title)
    figure.colorbar(
        filled,
        ax=axes[:2],
        label="u (shared scale)",
        ticks=np.arange(1, 10, 2),
        orientation="horizontal",
        fraction=0.07,
        aspect=40,
        pad=0.08,
    )

    # Scale the error independently to reveal its spatial distribution.
    # Keep a nonzero range even if all sampled errors happen to be zero.
    error_limit = max(max_error, np.finfo(float).eps)
    error_filled = axes[2].contourf(
        x,
        y,
        section_error,
        levels=np.linspace(0.0, error_limit, 25),
        cmap="magma",
    )
    axes[2].set_title(f"Absolute error\nMaximum: {max_error:.2e}")
    error_formatter = ScalarFormatter(useMathText=True)
    error_formatter.set_powerlimits((0, 0))
    figure.colorbar(
        error_filled,
        ax=axes[2],
        label=r"$|u_h-u_{\mathrm{exact}}|$",
        format=error_formatter,
        ticks=MaxNLocator(nbins=4),
        orientation="horizontal",
        fraction=0.07,
        aspect=20,
        pad=0.08,
    )

    for ax, edge_color in zip(
        axes, ("#242b33", "#242b33", "white"), strict=True
    ):
        ax.add_collection(
            LineCollection(
                segments,
                colors=edge_color,
                linewidths=0.6,
                alpha=0.4,
                zorder=1.5,
            )
        )
        ax.set(
            xlabel="x",
            aspect="equal",
            xlim=(0, 2),
            ylim=(0, 2),
            xticks=np.linspace(0, 2, 5),
            yticks=np.linspace(0, 2, 5),
        )
    axes[0].set_ylabel("y")
    axes[2].set_ylabel("y")
    axes[2].tick_params(axis="y", labelleft=True)

    if screenshot is not None:
        figure.savefig(screenshot, dpi=180, facecolor="white")
        plt.close(figure)
    else:
        plt.show()


###############################################################################
# Run the example
# ----------------
def main(*, plot: bool = True) -> pv.UnstructuredGrid:
    """Solve the TET10 Poisson problem and optionally display the result."""
    result = solve_poisson(create_tet10_cube())
    if plot:
        draw(result)
    return result


if __name__ == "__main__":
    main()

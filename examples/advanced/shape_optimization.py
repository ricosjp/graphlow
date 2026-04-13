"""
Shape optimization: minimize surface area under volume constraint
=================================================================
Use *graphlow* to deform a volumetric mesh with a small MLP and gradient
descent: minimize surface area while keeping total volume fixed and
penalizing large deformations. The result approaches a sphere (fixed
volume, minimal area).

This example walks through: loading a mesh, building a deformation model,
defining a differentiable cost, and running the optimization loop. Helper
functions are defined at module level with clear roles.
"""

###############################################################################
# Imports and constants
# ---------------------
import itertools
import pathlib
from collections.abc import Callable

import numpy as np
import phlower_tensor as pt
import pyvista as pv
import torch

import graphlow
from graphlow.utils.topology_helper import TopologyDim

# Optimization weights and hyperparameters
WEIGHT_VOLUME_CONSTRAINT = 10.0
WEIGHT_NORM_CONSTRAINT = 1.0
MLP_HIDDEN = 64
DEFORMATION_FACTOR = 1.0
LEARNING_RATE = 1e-2
# Reject step if any cell volume < this * (total / n_cells)
TINY_CELL_RATIO = 1e-3


###############################################################################
# Step 1: Deformation model (MLP)
# --------------------------------
# The mesh is deformed by adding a learned delta to each point:
# delta = MLP(points). We use a small 2-layer MLP with tanh and optional bias.
def make_deformation_mlp(
    hidden_size: int,
    use_bias: bool = False,
) -> tuple[
    Callable[[pt.PhlowerTensor, float], pt.PhlowerTensor],
    list[torch.nn.Parameter],
]:
    """
    Create the deformation MLP and its trainable parameters.

    Returns
    -------
    deform : collections.abc.Callable
        Function that maps points to deformed points.
    params : list[torch.nn.Parameter]
        Trainable parameters for the optimizer.
    """
    scale = hidden_size**0.5
    w1 = torch.nn.Parameter(torch.randn(3, hidden_size) / scale)
    w2 = torch.nn.Parameter(torch.randn(hidden_size, 3) / scale)
    if use_bias:
        b1 = torch.nn.Parameter(torch.randn(hidden_size) / hidden_size)
        b2 = torch.nn.Parameter(torch.randn(3) / 3)
        params = [w1, b1, w2, b2]
    else:
        b1 = 0.0
        b2 = 0.0
        params = [w1, w2]
    activation = torch.nn.Identity()

    def deform(
        points: pt.PhlowerTensor, factor: float = 1.0
    ) -> pt.PhlowerTensor:
        hidden = torch.tanh(points.to_tensor() @ w1 + b1)
        delta = pt.phlower_tensor(
            activation(hidden @ w2 + b2),
            dimension=points.dimension,
        )
        return points + delta * factor

    return deform, params


###############################################################################
# Step 2: Cost function
# ---------------------
# Differentiable cost: normalized surface area + volume penalty + deformation
# regularizer. Returns ``None`` for cost/area/volume when any cell volume
# becomes too small (step rejected).
def compute_cost(
    mesh: graphlow.TensorMesh[pt.PhlowerTensor],
    surface: graphlow.TensorMesh[pt.PhlowerTensor],
    initial_points: pt.PhlowerTensor,
    initial_total_volume: pt.PhlowerTensor,
    surface_initial_total_area: pt.PhlowerTensor,
    deformed_points: pt.PhlowerTensor,
    weight_volume: float,
    weight_norm: float,
    tiny_cell_ratio: float,
) -> tuple[
    pt.PhlowerTensor | None,
    pt.PhlowerTensor | None,
    pt.PhlowerTensor | None,
]:
    """
    Evaluate cost and optional area/volume for the current deformation.

    Updates mesh and surface points in place from deformed_points, then
    computes cell volumes and surface face areas. If any cell volume is
    below a small fraction of the mean, returns ``(None, None, None)`` to
    signal step rejection.

    Returns
    -------
    cost : pt.PhlowerTensor or None
        Scalar objective value, or None when the step is rejected.
    total_area : pt.PhlowerTensor or None
        Total surface area, or None when the step is rejected.
    total_volume : pt.PhlowerTensor or None
        Total volume, or None when the step is rejected.
    """
    mesh.points = deformed_points
    volumes = mesh.geometry.cell_volumes()
    total_volume = torch.sum(volumes)
    surface.points = surface.gather_parent_point_data(deformed_points)
    areas = surface.geometry.face_areas()
    total_area = torch.sum(areas)
    deformation = deformed_points - initial_points

    min_vol = initial_total_volume.to_tensor() / mesh.n_cells * tiny_cell_ratio
    if torch.any(volumes.to_tensor() < min_vol):
        return None, None, None

    cost_area = total_area / surface_initial_total_area
    volume_penalty = (
        (total_volume - initial_total_volume) / initial_total_volume
    ) ** 2
    scale_vol = initial_total_volume ** (2 / 3)
    norm_penalty = torch.exp(torch.mean(deformation * deformation / scale_vol))
    cost = (
        cost_area + weight_volume * volume_penalty + weight_norm * norm_penalty
    )
    return cost, total_area, total_volume


###############################################################################
# Step 3: Build example mesh (hex grid)
# -------------------------------------
# Create a small hexahedral grid. You can replace with your own mesh
# (e.g. :func:`graphlow.read` or :func:`graphlow.from_pyvista`).
def generate_hex_grid(ni: int, nj: int, nk: int) -> pv.UnstructuredGrid:
    """
    Build a hexahedral grid for the example.

    Returns
    -------
    pv.UnstructuredGrid
        Hexahedral grid.
    """
    n_cells = (ni - 1) * (nj - 1) * (nk - 1)
    x = np.arange(ni, dtype=np.float32)
    y = np.arange(nj, dtype=np.float32)
    z = np.arange(nk, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T
    indices = np.arange(ni * nj * nk).reshape(ni, nj, nk)
    hex_lists = []
    for k in range(nk - 1):
        for j in range(nj - 1):
            for i in range(ni - 1):
                v0 = indices[i, j, k]
                v1 = indices[i + 1, j, k]
                v2 = indices[i + 1, j + 1, k]
                v3 = indices[i, j + 1, k]
                v4 = indices[i, j, k + 1]
                v5 = indices[i + 1, j, k + 1]
                v6 = indices[i + 1, j + 1, k + 1]
                v7 = indices[i, j + 1, k + 1]
                hex_lists.append([8, v0, v1, v2, v3, v4, v5, v6, v7])
    flat = list(itertools.chain.from_iterable(hex_lists))
    celltypes = np.full(n_cells, pv.CellType.HEXAHEDRON)
    return pv.UnstructuredGrid(flat, celltypes, points)


###############################################################################
# Step 4: Visualization
# ---------------------
# Color by vertex index (turbo cmap) so that vertex movement is easy to see.
def create_plotter(grid: pv.UnstructuredGrid) -> pv.Plotter:
    """
    Build a plotter for GIF animation.

    Returns
    -------
    pv.Plotter
        Plotter configured for frame capture.
    """
    plotter = pv.Plotter(window_size=[800, 600])
    init_grid = grid.copy()
    grid["index"] = np.arange(grid.n_points)
    plotter.add_mesh(init_grid, show_edges=True, color="white", opacity=0.1)
    plotter.add_mesh(
        grid,
        scalars="index",
        show_edges=True,
        lighting=False,
        cmap="turbo",
        opacity=0.8,
    )
    plotter.open_gif("shape_optimization.gif")
    plotter.show_bounds(mesh=init_grid, location="outer")
    plotter.camera_position = "iso"
    return plotter


###############################################################################
# Step 5: Run optimization
# ------------------------
# Load (or create) mesh, center it, enable gradients, build MLP and cost,
# then run gradient descent. Optionally save meshes and report final
# mean radius vs. sphere radius.
def main(
    mesh_path: pathlib.Path | None = None,
    n_steps: int = 2000,
    use_bias: bool = False,
    save_dir: pathlib.Path | None = None,
    log_interval: int = 100,
) -> None:
    """
    Run shape optimization: minimize area under fixed volume.

    If mesh_path is None, use a small generated hex grid. Otherwise load
    the mesh from mesh_path. When save_dir is set, write mesh.VTU every
    log_interval steps.
    """
    # ----- Load or create mesh -----
    if mesh_path is not None and mesh_path.exists():
        pv_mesh: pv.DataSet = pv.read(mesh_path)
        pv_mesh: pv.UnstructuredGrid = pv_mesh.cast_to_unstructured_grid()
    else:
        pv_mesh = generate_hex_grid(10, 10, 10)

    pv_mesh.points = pv_mesh.points - np.mean(
        pv_mesh.points, axis=0, keepdims=True
    )
    mesh = graphlow.from_pyvista(pv_mesh, "phlower")
    mesh.requires_grad(True)

    if mesh.topology.mesh_dim() < TopologyDim.VOLUME:
        raise RuntimeError(
            "Volume mesh required (face_registry / cell_volumes)"
        )

    initial_volumes = mesh.geometry.cell_volumes().clone()
    initial_total_volume = torch.sum(initial_volumes).detach().clone()
    initial_points = mesh.points.clone()

    surface = mesh.extract_surface()
    surface_initial_areas = surface.geometry.face_areas().detach().clone()
    surface_initial_total_area = torch.sum(surface_initial_areas)

    # ----- Deformation model and optimizer -----
    deform_fn, params = make_deformation_mlp(MLP_HIDDEN, use_bias=use_bias)
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    deform_factor = [DEFORMATION_FACTOR]

    def cost_fn(
        deformed_points: pt.PhlowerTensor,
    ) -> tuple[
        pt.PhlowerTensor | None,
        pt.PhlowerTensor | None,
        pt.PhlowerTensor | None,
    ]:
        return compute_cost(
            mesh,
            surface,
            initial_points,
            initial_total_volume,
            surface_initial_total_area,
            deformed_points,
            WEIGHT_VOLUME_CONSTRAINT,
            WEIGHT_NORM_CONSTRAINT,
            TINY_CELL_RATIO,
        )

    if save_dir is not None:
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    plotter = create_plotter(mesh.pvmesh)
    plotter.write_frame()

    print("Step  |     area | vol ratio |     cost")
    print("------|----------|-----------|----------")

    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        deformed_points = deform_fn(initial_points, deform_factor[0])

        cost, area, volume = cost_fn(deformed_points)
        if cost is None:
            deform_factor[0] *= 0.9
            continue

        if step % log_interval == 0:
            vol_ratio = volume / initial_total_volume
            print(
                f"{step:5d} | {area.to_tensor():.3e} | "
                f"{vol_ratio.to_tensor():.3e} | {cost.to_tensor():.3e}"
            )
            mesh.pvmesh.points = (
                mesh.backend.to_torch(mesh.points).detach().numpy()
            )
            plotter.write_frame()
            if save_dir is not None:
                mesh.pvmesh.save(str(save_dir / f"mesh.{step:06d}.vtu"))

        cost.backward()
        optimizer.step()
    plotter.close()

    # ----- Final shape: mean radius vs. sphere radius -----
    final_deformed = deform_fn(initial_points, deform_factor[0])
    surface_deformed_points = surface.gather_parent_point_data(final_deformed)
    actual_mean_radius = (
        torch.mean(
            torch.linalg.norm(surface_deformed_points.to_tensor(), dim=1)
        )
        .detach()
        .numpy()
    )
    v0 = initial_total_volume.to_tensor().numpy()
    desired_radius = (v0 * 3 / 4 / np.pi) ** (1 / 3)
    print(f"\nInitial total volume: {v0:.5f}")
    print(f"Desired sphere radius (same volume): {desired_radius:.5f}")
    print(f"Actual mean boundary radius:        {actual_mean_radius:.5f}")


###############################################################################
# Step 6: Run the example
# -----------------------
if __name__ == "__main__":
    main(n_steps=2000, log_interval=100)

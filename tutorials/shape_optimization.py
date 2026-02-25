"""
Optimize the shape of volumetric mesh
=====================================
*graphlow* is a powerful tool for shape optimization,
leveraging differentiable tensor-based geometric computations.

This example shows how to minimize surface area
while constraining volume changes and vertex deformation.

.. image:: ./images/sphx_glr_shape_optimization_001.gif
    :width: 300
    :align: center
"""

###############################################################################
# Import necessary modules including :mod:`graphlow`.
# ---------------------------------------------------
import itertools

import numpy as np
import phlower_tensor as pt
import pyvista as pv
import torch

import graphlow


###############################################################################
# Mesh: create a hexahedral grid
# ------------------------------
# Helper to build an example volumetric mesh. You can replace this with
# your own mesh (e.g. loaded from a file).
def generate_grid(ni: int, nj: int, nk: int) -> pv.UnstructuredGrid:
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
                # hex
                v0 = indices[i, j, k]
                v1 = indices[i + 1, j, k]
                v2 = indices[i + 1, j + 1, k]
                v3 = indices[i, j + 1, k]
                v4 = indices[i, j, k + 1]
                v5 = indices[i + 1, j, k + 1]
                v6 = indices[i + 1, j + 1, k + 1]
                v7 = indices[i, j + 1, k + 1]
                hex_lists.append([8, v0, v1, v2, v3, v4, v5, v6, v7])

    hex_cells = list(itertools.chain.from_iterable(hex_lists))
    celltypes = np.full((n_cells), pv.CellType.HEXAHEDRON)
    return pv.UnstructuredGrid(hex_cells, celltypes, points)


###############################################################################
# Cost function
# -------------
# We minimize **surface area** and penalize:
# - change in total volume (keep volume roughly constant),
# - large vertex deformations (keep the shape smooth).
WEIGHT_VOLUME = 10.0
WEIGHT_DEFORMATION = 1.0


def cost_function(
    mesh: graphlow.GraphlowMesh,
    init_total_volume: float,
    init_surface_total_area: float,
) -> pt.PhlowerTensor | None:
    """
    Cost = (area term) + (volume penalty) + (deformation penalty).

    Returns None if any cell volume is too small (numerical safeguard).
    """
    deformation = mesh.dict_point_tensor["deformation"]
    surface = mesh.extract_surface(pass_point_data=True)

    volumes = mesh.compute_volumes()
    areas = surface.compute_areas()

    total_volume = torch.sum(volumes)
    total_area = torch.sum(areas)

    # Avoid inverting nearly-degenerate cells
    min_volume_ratio = 1e-3
    min_vol = min_volume_ratio * init_total_volume / mesh.n_cells
    if torch.any(volumes.to_tensor() < min_vol.to_tensor()):
        return None

    # Normalized terms (order of magnitude ~1)
    area_term = total_area / init_surface_total_area
    volume_penalty = (
        (total_volume - init_total_volume) / init_total_volume
    ) ** 2
    def_t = deformation.to_tensor()
    deformation_penalty = torch.mean(def_t * def_t)

    return (
        area_term
        + WEIGHT_VOLUME * volume_penalty
        + WEIGHT_DEFORMATION * deformation_penalty
    )


###############################################################################
# Visualization: export optimization progress as GIF
# --------------------------------------------------
def create_gif_plotter(mesh: pv.UnstructuredGrid) -> pv.Plotter:
    plotter = pv.Plotter(window_size=[800, 600])
    init_mesh = mesh.copy()
    plotter.add_mesh(init_mesh, show_edges=True, color="white", opacity=0.1)
    plotter.add_mesh(
        mesh, show_edges=True, lighting=False, cmap="turbo", opacity=0.8
    )

    plotter.open_gif("shape_optimization_result.gif")
    plotter.show_bounds(init_mesh, location="outer")
    plotter.camera_position = "iso"
    return plotter


###############################################################################
# Optimization loop
# -----------------
# 1. Build mesh and keep initial geometry (points, total volume, surface area).
# 2. Deformation = small MLP: W2 @ tanh(points @ W1).
# 3. Deformed mesh: points_new = points_init + deformation(points_init).
# 4. Minimize cost (area + volume/deformation penalties) by gradient descent.
def optimize_shape(input_mesh: pv.UnstructuredGrid) -> None:
    # --- Hyperparameters ---
    n_steps = 2000
    print_every = max(1, n_steps // 100)
    n_hidden = 64
    learning_rate = 1e-2
    deformation_scale = 1.0  # Reduced if cells become too flat

    # --- Center the mesh at the origin ---
    input_mesh.points = input_mesh.points - np.mean(
        input_mesh.points, axis=0, keepdims=True
    )

    mesh = graphlow.GraphlowMesh(input_mesh)

    # --- Reference values (used in the cost) ---
    init_volumes = mesh.compute_volumes().clone()
    init_total_volume = torch.sum(init_volumes)
    init_points = mesh.points.clone()

    init_surface = mesh.extract_surface()
    init_surface_areas = init_surface.compute_areas().clone()
    init_surface_total_area = torch.sum(init_surface_areas)

    # --- Deformation network: (N,3) -> (N,n_hidden) -> (N,3) ---
    w1 = torch.nn.Parameter(torch.randn(3, n_hidden) / n_hidden**0.5)
    w2 = torch.nn.Parameter(torch.randn(n_hidden, 3) / n_hidden**0.5)
    optimizer = torch.optim.Adam([w1, w2], lr=learning_rate)
    output_activation = torch.nn.Identity()

    def compute_deformation(points: pt.PhlowerTensor) -> pt.PhlowerTensor:
        hidden = torch.tanh(points.to_tensor() @ w1)
        out = pt.phlower_tensor(
            output_activation(hidden @ w2), dimension=points.dimension
        )
        return deformation_scale * out

    # --- Initial frame for GIF ---
    deformation = compute_deformation(init_points)
    mesh.dict_point_tensor.update({"deformation": deformation}, overwrite=True)
    mesh.copy_features_to_pyvista(overwrite=True)
    mesh.pvmesh.points = mesh.points.numpy()
    plotter = create_gif_plotter(mesh.pvmesh)
    plotter.write_frame()

    # --- Minimize cost ---
    init_vol_display = torch.sum(mesh.compute_volumes()).to_tensor()
    print(f"\nInitial total volume: {init_vol_display:.5f}")
    print("     i,        cost")
    for i in range(1, n_steps + 1):
        optimizer.zero_grad()

        deformation = compute_deformation(init_points)
        deformed_points = init_points + deformation

        mesh.dict_point_tensor.update(
            {"deformation": deformation}, overwrite=True
        )
        mesh.dict_point_tensor.update(
            {"points": deformed_points}, overwrite=True
        )
        cost = cost_function(mesh, init_total_volume, init_surface_total_area)

        if cost is None:
            # Cell too thin -> reduce deformation scale and retry
            deformation_scale *= 0.9
            print(f"  Scale reduced to {deformation_scale:.3f}; retry step.")
            continue

        if i % print_every == 0:
            print(f"{i:6d}, {cost.to_tensor().item():.5e}")
            mesh.copy_features_to_pyvista(overwrite=True)
            mesh.pvmesh.points = mesh.points.numpy()
            plotter.write_frame()

        cost.backward()
        optimizer.step()

    plotter.close()


###############################################################################
# Run the optimization
# --------------------
# Finally, we define the main function to run the optimization.
def main():
    mesh = generate_grid(10, 10, 10)
    optimize_shape(mesh)


if __name__ == "__main__":
    main()

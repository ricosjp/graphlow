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
import itertools

import numpy as np
import pyvista as pv
import torch

import graphlow


###############################################################################
# Prepare a volumetric mesh
# --------------------------
# First, we define a function to generate grid data as the example mesh.
# If you wish to use your own mesh, you can skip this step.
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
# Define the cost function
# ------------------------
# To minimize surface area with constraints,
# we defined the cost function as follows:
weight_deformation_constraint = 1.0
weight_volume_constraint = 10.0


def cost_function(
    mesh: graphlow.GraphlowMesh,
    init_total_volume: float,
    init_surface_total_area: float,
) -> torch.Tensor:
    deformation = mesh.dict_point_tensor["deformation"]
    surface = mesh.extract_surface(pass_point_data=True)

    volumes = mesh.compute_volumes()
    areas = surface.compute_areas()

    total_volume = torch.sum(volumes)
    total_area = torch.sum(areas)

    if torch.any(volumes < 1e-3 * init_total_volume / mesh.n_cells):
        return None

    cost_area = total_area / init_surface_total_area
    volume_constraint = (
        (total_volume - init_total_volume) / init_total_volume
    ) ** 2
    deformation_constraint = torch.mean(deformation * deformation)
    return (
        cost_area
        + weight_volume_constraint * volume_constraint
        + weight_deformation_constraint * deformation_constraint
    )


###############################################################################
# Visualize
# ---------
# Following code generate gif plotter to visualize the result.
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
# Once the cost function and the visualizing function are determined,
# the remaining step is to write the optimization code
# that updates the points and deformations
#
# This is the example of the optimization code.
def optimize_shape(input_mesh: pv.UnstructuredGrid):
    # Optimization setting
    n_optimization = 2000
    print_period = int(n_optimization / 100)
    n_hidden = 64
    deformation_factor = 1.0
    lr = 1e-2
    output_activation = torch.nn.Identity()

    # Initialize
    input_mesh.points = input_mesh.points - np.mean(
        input_mesh.points, axis=0, keepdims=True
    )  # Center mesh position

    mesh = graphlow.GraphlowMesh(input_mesh)

    init_volumes = mesh.compute_volumes().clone()
    init_total_volume = torch.sum(init_volumes)
    init_points = mesh.points.clone()

    init_surface = mesh.extract_surface()
    init_surface_areas = init_surface.compute_areas().clone()
    init_surface_total_area = torch.sum(init_surface_areas)

    w1 = torch.nn.Parameter(torch.randn(3, n_hidden) / n_hidden**0.5)
    w2 = torch.nn.Parameter(torch.randn(n_hidden, 3) / n_hidden**0.5)
    params = [w1, w2]
    optimizer = torch.optim.Adam(params, lr=lr)

    def compute_deformation(points: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(torch.einsum("np,pq->nq", points, w1))
        deformation = output_activation(torch.einsum("np,pq->nq", hidden, w2))
        return deformation_factor * deformation

    deformation = compute_deformation(init_points)
    mesh.dict_point_tensor.update({"deformation": deformation}, overwrite=True)

    mesh.copy_features_to_pyvista(overwrite=True)
    mesh.pvmesh.points = mesh.points.detach().numpy()
    plotter = create_gif_plotter(mesh.pvmesh)
    plotter.write_frame()

    # Optimization loop
    print(f"\ninitial volume: {torch.sum(mesh.compute_volumes()):.5f}")
    print("     i,        cost")
    for i in range(1, n_optimization + 1):
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
            deformation_factor = deformation_factor * 0.9
            print(f"update deformation_factor: {deformation_factor}")
            continue

        if i % print_period == 0:
            print(f"{i:6d}, {cost:.5e}")
            mesh.copy_features_to_pyvista(overwrite=True)
            mesh.pvmesh.points = mesh.points.detach().numpy()
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

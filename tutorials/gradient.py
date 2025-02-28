"""
Compute gradient of scalar field on a mesh
==========================================
:mod:`graphlow` can treat a mesh as a graph and
compute gradients of physical quantities given on the graph.

This tutorial shows how to compute the gradient of a scalar field on a mesh.

.. image:: ./images/sphx_glr_gradient_001.png
    :width: 300
    :align: center
"""

###############################################################################
# Import necessary modules including :mod:`graphlow`.
import numpy as np
import pyvista as pv
import torch

import graphlow


###############################################################################
# Prepare grid mesh
# -----------------
# First, we define a function to generate grid data as the example mesh.
# If you wish to use your own mesh, you can skip this step.
def generate_grid(ni: int, nj: int) -> pv.StructuredGrid:
    x = np.linspace(-1, 1, ni, dtype=np.float32)
    y = np.linspace(-1, 1, nj, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.zeros([ni, nj], dtype=np.float32)
    grid = pv.StructuredGrid(X, Y, Z)
    return grid


###############################################################################
# Define a scalar field
# ---------------------
# Next, we define a scalar field that depends on the coordinates.
# We define a scalar field as follows:
#
# .. math::
#    \phi = x^2 - y^2
#
# You can modify the scalar field as you like.
def scalar_field(pos: np.ndarray) -> np.ndarray:
    x = pos[:, 0]
    y = pos[:, 1]
    return x * x - y * y


###############################################################################
# Compute gradient
# ----------------
# To compute the gradient of a physical quantity using graphlow,
# we use :func:`compute_isoAM`.
def compute_gradient(
    mesh: graphlow.GraphlowMesh, phi: np.ndarray
) -> torch.Tensor:
    grad_adjs, _ = mesh.compute_isoAM(with_moment_matrix=True)
    grad_x = grad_adjs[0] @ phi
    grad_y = grad_adjs[1] @ phi
    grad_z = grad_adjs[2] @ phi
    grad_vectors = torch.stack((grad_x, grad_y, grad_z), dim=-1)
    return grad_vectors


###############################################################################
# Visualize
# ---------
# To visualize the the scalar field and its gradient,
# we define the following function.
def draw(grid: pv.StructuredGrid):
    plotter = pv.Plotter(window_size=[800, 600])
    plotter.add_mesh(grid, scalars="phi", show_edges=True)
    plotter.add_arrows(
        grid.points, grid["grad_phi"], mag=0.1, show_scalar_bar=False
    )
    plotter.show_bounds(grid, location="outer")
    plotter.show()


###############################################################################
# Main
# ----
# Finally, we define the main function to run the tutorial.
ni = 11
nj = 11


def main():
    grid = generate_grid(ni, nj)
    mesh = graphlow.GraphlowMesh(grid)

    phi = scalar_field(mesh.points)
    grad_phi = compute_gradient(mesh, phi)

    grid["phi"] = phi.numpy()
    grid["grad_phi"] = grad_phi.numpy()

    draw(grid)


if __name__ == "__main__":
    main()

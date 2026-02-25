"""
Compute gradient of scalar field on a mesh
==========================================
:mod:`graphlow` treats a mesh as a graph and can compute
gradients of scalar fields defined on the nodes.

This tutorial: define φ(x,y), build the isoAM gradient operator,
then compute ∇φ and visualize it.

.. image:: ./images/sphx_glr_gradient_001.png
    :width: 300
    :align: center
"""

###############################################################################
# Import necessary modules including :mod:`graphlow`.
# ---------------------------------------------------
import numpy as np
import phlower_tensor as pt
import pyvista as pv
import torch

import graphlow


###############################################################################
# Mesh: create a 2D grid
# ----------------------
# Flat grid in the xy-plane. You can replace this with your own mesh.
def generate_grid(ni: int, nj: int) -> pv.StructuredGrid:
    x = np.linspace(-1, 1, ni, dtype=np.float32)
    y = np.linspace(-1, 1, nj, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = np.zeros([ni, nj], dtype=np.float32)
    grid = pv.StructuredGrid(X, Y, Z)
    return grid


###############################################################################
# Scalar field φ
# --------------
# Example: φ = x² − y². You can change the formula.
#
# .. math::
#    \phi = x^2 - y^2
#
def scalar_field(pos: np.ndarray) -> pt.PhlowerTensor:
    """Evaluate φ at each point. pos: (n_points, 3) or (n_points, 2)."""
    x = pos[:, 0]
    y = pos[:, 1]
    phi = x * x - y * y
    return pt.phlower_tensor(phi, dimension={"Theta": 1})


###############################################################################
# Gradient via isoAM
# ------------------
# :func:`compute_isoAM` returns a gradient operator (one matrix per dimension).
# Applying it to nodal values φ gives ∂φ/∂x, ∂φ/∂y, ∂φ/∂z; we stack into
# (n_points, 3) gradient vectors.
def compute_gradient(
    mesh: graphlow.GraphlowMesh,
    phi: pt.PhlowerTensor,
) -> pt.PhlowerTensor:
    """
    Compute ∇φ at each node.

    Parameters
    ----------
    mesh : GraphlowMesh
        Mesh with points and connectivity.
    phi : array or tensor
        Scalar field values, shape (n_points,).

    Returns
    -------
    torch.Tensor
        Gradient vectors, shape (n_points, 3).
    """
    isoAM, _ = mesh.compute_isoAM(with_moment_matrix=True)

    # isoAM[k] @ phi -> k-th component of gradient
    gx = isoAM[0] @ phi
    gy = isoAM[1] @ phi
    gz = isoAM[2] @ phi
    grad_vectors = torch.stack((gx, gy, gz), dim=-1)
    return grad_vectors


###############################################################################
# Visualization: scalar field + gradient arrows
# ---------------------------------------------
def draw(grid: pv.StructuredGrid) -> None:
    plotter = pv.Plotter(window_size=[800, 600])
    plotter.add_mesh(grid, scalars="phi", show_edges=True)
    plotter.add_arrows(
        grid.points, grid["grad_phi"], mag=0.1, show_scalar_bar=False
    )
    plotter.show_bounds(grid, location="outer")
    plotter.show()


###############################################################################
# Run the tutorial
# ----------------
GRID_NI = 11
GRID_NJ = 11


def main() -> None:
    # 1. Build mesh
    grid = generate_grid(GRID_NI, GRID_NJ)
    mesh = graphlow.GraphlowMesh(grid)

    # 2. φ at each node (from coordinates)
    phi = scalar_field(mesh.points.numpy())

    # 3. ∇φ
    grad_phi = compute_gradient(mesh, phi)

    # 4. Attach to grid for visualization
    grid["phi"] = np.asarray(phi)
    grid["grad_phi"] = grad_phi.numpy()

    draw(grid)


if __name__ == "__main__":
    main()

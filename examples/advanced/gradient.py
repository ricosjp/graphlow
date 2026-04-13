"""
Compute gradient of scalar field on a mesh
==========================================
:mod:`graphlow` treats a mesh as a graph and can compute
gradients of scalar fields defined on the nodes.

This tutorial: define φ(x,y), build the isoAM gradient operator,
then compute ∇φ and visualize it.
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
    """
    Build a flat structured grid in the xy-plane.

    Parameters
    ----------
    ni : int
        Number of points along x.
    nj : int
        Number of points along y.

    Returns
    -------
    pv.StructuredGrid
        Structured grid with points of shape ``(ni * nj, 3)``.
    """
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
    """
    Evaluate ``phi = x^2 - y^2`` at each point.

    Parameters
    ----------
    pos : np.ndarray
        Point coordinates of shape ``(n_points, 2)`` or ``(n_points, 3)``.
        Only the x and y columns are used.

    Returns
    -------
    pt.PhlowerTensor
        Scalar field values of shape ``(n_points,)``.
    """
    x = pos[:, 0]
    y = pos[:, 1]
    phi = x * x - y * y
    return pt.phlower_tensor(phi, dimension={"Theta": 1})


###############################################################################
# Gradient via isoAM
# ------------------
# :meth:`~graphlow.core.geometry.MeshGeometry.isoAM` returns gradient operators
# (one sparse matrix per dimension). Applying them to nodal values φ gives
# ∂φ/∂x, ∂φ/∂y, ∂φ/∂z; we stack into (n_points, 3) gradient vectors.
def compute_gradient(
    mesh: graphlow.TensorMesh[pt.PhlowerTensor],
    phi: pt.PhlowerTensor,
) -> pt.PhlowerTensor:
    """
    Compute ∇φ at each node.

    Parameters
    ----------
    mesh : graphlow.TensorMesh[pt.PhlowerTensor]
        Mesh with points and connectivity.
    phi : pt.PhlowerTensor
        Scalar field values of shape ``(n_points,)`` or ``(n_points, 1)``.

    Returns
    -------
    pt.PhlowerTensor
        Gradient vectors of shape ``(n_points, 3)``.
    """
    isoAM, _ = mesh.geometry.isoAM(with_moment_matrix=True)

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
    """
    Visualize the scalar field and gradient vectors.

    Parameters
    ----------
    grid : pv.StructuredGrid
        Grid containing ``phi`` and ``grad_phi`` arrays.

    Returns
    -------
    None
    """
    plotter = pv.Plotter(window_size=[800, 600])
    plotter.add_mesh(grid, scalars="phi", show_edges=True)
    plotter.add_arrows(
        grid.points, grid["grad_phi"], mag=0.1, show_scalar_bar=False
    )
    plotter.show_bounds(mesh=grid, location="outer")
    plotter.show()


###############################################################################
# Run the tutorial
# ----------------
GRID_NI = 11
GRID_NJ = 11


def main() -> None:
    """
    Run the gradient tutorial.

    Returns
    -------
    None
    """
    # 1. Build mesh
    grid = generate_grid(GRID_NI, GRID_NJ)
    mesh = graphlow.from_pyvista(grid, "phlower")

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

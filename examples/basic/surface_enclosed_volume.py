"""
Compute enclosed volume on a surface mesh
=========================================

This tutorial extracts a closed surface and computes its enclosed volume with
``surface_volume()``. The result can be compared with the sum of cell volumes
from the parent volume mesh.
"""

###############################################################################
# Imports
# -------
import numpy as np
import pyvista as pv
import torch

import graphlow


###############################################################################
# Step 1: Build a small volume mesh
# ---------------------------------
def make_volume_grid() -> pv.UnstructuredGrid:
    """Create a small warped hexahedral grid."""
    x = np.array([0.0, 1.0, 2.2, 3.5], dtype=np.float32)
    y = np.array([0.0, 0.8, 1.7], dtype=np.float32)
    z = np.array([0.0, 0.6, 1.4], dtype=np.float32)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    Z = Z + 0.12 * X * Y
    grid = pv.StructuredGrid(X, Y, Z)
    return grid.cast_to_unstructured_grid()


###############################################################################
# Step 2: Compare volume from cells and from the closed surface
# -------------------------------------------------------------
def main() -> None:
    """Run the tutorial."""
    volume_mesh = graphlow.from_pyvista(make_volume_grid(), backend="torch")
    surface_mesh = volume_mesh.extract_surface()

    total_cell_volume = torch.sum(volume_mesh.geometry.cell_volumes()).item()
    enclosed_volume = surface_mesh.geometry.surface_volume().item()

    print("surface_volume() is defined for watertight surface meshes.")
    print(f"Volume from cell_volumes().sum(): {total_cell_volume:.6f}")
    print(f"Enclosed volume from surface_volume(): {enclosed_volume:.6f}")
    print(
        f"Absolute difference: {abs(total_cell_volume - enclosed_volume):.6e}"
    )

    plotter = pv.Plotter(window_size=[800, 600])
    plotter.add_mesh(surface_mesh.pvmesh, show_edges=True, color="royalblue")
    plotter.show_bounds(mesh=surface_mesh.pvmesh, location="outer")
    plotter.camera_position = "iso"
    plotter.show()


if __name__ == "__main__":
    main()

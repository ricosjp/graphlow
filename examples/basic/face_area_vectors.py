"""
Inspect oriented face area vectors
==================================

This tutorial computes oriented area vectors on a surface mesh and compares
them with ``face_areas() * face_normals()``. It is a good first example when
you want to understand what ``face_area_vectors()`` represents geometrically.
"""

###############################################################################
# Imports
# -------
import numpy as np
import pyvista as pv
import torch

import graphlow


###############################################################################
# Step 1: Build a small volume mesh and extract its surface
# ---------------------------------------------------------
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
# Step 2: Compute area vectors, normals, and areas
# ------------------------------------------------
def main() -> None:
    """Run the tutorial."""
    volume_mesh = graphlow.from_pyvista(make_volume_grid(), backend="torch")
    surface_mesh = volume_mesh.extract_surface()

    face_centroids = surface_mesh.geometry.face_centroids()
    face_area_vectors = surface_mesh.geometry.face_area_vectors()
    face_normals = surface_mesh.geometry.face_normals()
    face_areas = surface_mesh.geometry.face_areas()

    reconstructed = face_normals * face_areas

    print(f"Surface face count: {surface_mesh.n_cells}")
    print(f"face_area_vectors shape: {tuple(face_area_vectors.shape)}")
    print(
        "face_area_vectors == face_normals * face_areas: "
        f"{torch.allclose(face_area_vectors, reconstructed)}"
    )
    print(f"First area vector: {face_area_vectors[0].tolist()}")

    plotter = pv.Plotter(window_size=[900, 700])
    plotter.add_mesh(
        surface_mesh.pvmesh,
        show_edges=True,
        color="white",
        opacity=0.85,
    )
    plotter.add_arrows(
        face_centroids.detach().cpu().numpy(),
        face_area_vectors.detach().cpu().numpy(),
        mag=0.35,
        color="tomato",
    )
    plotter.show_bounds(mesh=surface_mesh.pvmesh, location="outer")
    plotter.camera_position = "iso"
    plotter.show()


if __name__ == "__main__":
    main()

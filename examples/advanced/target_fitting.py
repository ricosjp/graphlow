"""
Target fitting: minimize Chamfer distance to a point cloud
==========================================================
Deform mesh points with gradient descent so that
:func:`graphlow.TensorMesh.geometry.chamfer_distance` to a target point set
decreases. The target is a **dense** point cloud on an **ellipsoid**; the mesh
(an icosphere) is centered, scaled, then updated in place to fit the target.

Using an ellipsoid (different radii per axis) and many target points makes the
optimization non-trivial and shows the mesh deforming toward the target. The
same topology (sphere-like) allows the loss to converge to a low value. Soft-min
Chamfer ensures gradients reach all vertices for stable convergence.

This example walks through: sampling target points, loading the mesh,
defining the loss, and running the optimization with visualization.
"""

###############################################################################
# Imports and constants
# ---------------------
import pathlib

import numpy as np
import pyvista as pv
import torch

import graphlow

LEARNING_RATE = 0.06
N_STEPS = 2000
LOG_INTERVAL = 100
# Soft-min temperature for Chamfer: gradient flows to all vertices.
SOFTMIN_TEMPERATURE = 0.08
# Target shape: ellipsoid semi-axes (a, b, c). Non-uniform = non-trivial fit.
ELLIPSOID_AXES = (1.4, 0.8, 0.5)
# Dense target point cloud (independent of mesh vertex count).
N_TARGET_POINTS = 2000
# Icosphere subdivision level (higher = more vertices, smoother fit).
ICOSPHERE_NSUB = 2
# Fixed RNG seed for reproducible target sampling (not part of the algorithm).
_TARGET_SAMPLE_SEED = 0


###############################################################################
# Step 1: Target point cloud
# --------------------------
# Sample points on an ellipsoid (uniform in solid angle, then scale by axes).
# A dense cloud with many more points than mesh vertices gives a clear
# target shape and stable Chamfer gradients.
def points_on_ellipsoid(
    n: int,
    axes: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Sample n points on an ellipsoid with semi-axes (a, b, c). Returns (n, 3).
    """
    rng = np.random.default_rng(_TARGET_SAMPLE_SEED)
    phi = np.arccos(2.0 * rng.uniform(0, 1, n) - 1.0)
    theta = rng.uniform(0, 2 * np.pi, n)
    a, b, c = axes
    x = a * np.sin(phi) * np.cos(theta)
    y = b * np.sin(phi) * np.sin(theta)
    z = c * np.cos(phi)
    return np.stack([x, y, z], axis=1).astype(np.float32)


###############################################################################
# Step 2: Load and prepare mesh
# -----------------------------
# Use an icosphere (many vertices) so the mesh can deform into the target
# shape. Center and scale so initial points sit in a comparable range to
# the target (e.g. ellipsoid extent).
def load_and_center_mesh(
    mesh_path: pathlib.Path | None,
    icosphere_nsub: int = ICOSPHERE_NSUB,
    target_scale: float | None = None,
) -> graphlow.TensorMesh[torch.Tensor]:
    """Load mesh, require_grad, center and scale to sit near target extent."""
    if mesh_path is not None and mesh_path.exists():
        pv_mesh: pv.DataSet = pv.read(mesh_path)
        pv_mesh: pv.UnstructuredGrid = pv_mesh.cast_to_unstructured_grid()
    else:
        pv_mesh = pv.Icosphere(radius=1.0, nsub=icosphere_nsub)
        pv_mesh = pv_mesh.cast_to_unstructured_grid()

    pv_mesh.points = pv_mesh.points.astype(np.float32)
    mesh = graphlow.from_pyvista(pv_mesh, "torch")
    mesh.requires_grad(True)

    with torch.no_grad():
        torch.sub(
            mesh.points, mesh.points.mean(dim=0, keepdim=True), out=mesh.points
        )
        r = torch.linalg.norm(mesh.points, dim=1).mean().item()
        if r > 1e-8:
            torch.mul(mesh.points, 1.0 / r, out=mesh.points)
        if target_scale is not None and target_scale > 0:
            torch.mul(mesh.points, target_scale, out=mesh.points)

    return mesh


###############################################################################
# Step 3: Loss and optimizer step
# -------------------------------
# Loss is Chamfer distance. We update mesh.points in place with gradient
# descent (no separate parameter tensor).
def loss_fn(
    mesh: graphlow.TensorMesh[torch.Tensor],
    target_points: torch.Tensor,
    softmin_temperature: float | None = SOFTMIN_TEMPERATURE,
) -> torch.Tensor:
    """Chamfer distance; soft min lets gradient flow to all vertices."""
    cd = mesh.geometry.chamfer_distance(
        target_points, softmin_temperature=softmin_temperature
    )
    return cd.squeeze()


###############################################################################
# Step 4: Visualization
# ---------------------
# Color by per-vertex distance to the target (nearest point in the cloud).
# Dark = close, bright = far; shows fitting quality over the mesh.
def _distance_to_target(
    mesh_pts: np.ndarray, target_pts: np.ndarray
) -> np.ndarray:
    """Per-vertex min distance to target point cloud. Shape (n_points,)."""
    diff = mesh_pts[:, None, :] - target_pts[None, :, :]
    return np.linalg.norm(diff, axis=2).min(axis=1).astype(np.float32)


def create_plotter(
    mesh_grid: pv.UnstructuredGrid,
    target_pts: np.ndarray,
) -> pv.Plotter:
    """Build a plotter for GIF: initial mesh + target point cloud."""
    plotter = pv.Plotter(window_size=[800, 600])
    init_grid = mesh_grid.copy()
    mesh_grid["distance_to_target"] = _distance_to_target(
        mesh_grid.points, target_pts
    )

    plotter.add_mesh(init_grid, show_edges=True, color="white", opacity=0.1)
    plotter.add_mesh(
        mesh_grid,
        scalars="distance_to_target",
        show_edges=True,
        lighting=False,
        cmap="viridis",
        opacity=0.8,
    )
    target_poly = pv.PolyData(target_pts)
    plotter.add_mesh(
        target_poly,
        color="white",
        point_size=4,
        opacity=0.4,
        render_points_as_spheres=True,
    )
    plotter.open_gif("target_fitting.gif")
    plotter.show_bounds(mesh=init_grid, location="outer")
    plotter.camera_position = "iso"
    return plotter


###############################################################################
# Step 5: Run optimization
# ------------------------
# Gradient descent on mesh.points; log and write frame every log_interval.
def main(
    mesh_path: pathlib.Path | None = None,
    n_steps: int = N_STEPS,
    lr: float = LEARNING_RATE,
    log_interval: int = LOG_INTERVAL,
    softmin_temperature: float | None = SOFTMIN_TEMPERATURE,
    n_target_points: int = N_TARGET_POINTS,
    ellipsoid_axes: tuple[float, float, float] | None = None,
    icosphere_nsub: int = ICOSPHERE_NSUB,
) -> None:
    """
    Fit mesh to target point cloud by minimizing Chamfer distance.

    Target is a dense point cloud on an ellipsoid; mesh is an icosphere so it
    has enough vertices to deform into the target shape. Same topology allows
    the loss to converge to a low value. softmin_temperature > 0 makes
    gradient flow to all vertices (recommended). Frames are written to
    target_fitting.gif every log_interval steps.
    """
    if ellipsoid_axes is None:
        ellipsoid_axes = ELLIPSOID_AXES
    # Scale initial mesh to sit inside the target envelope.
    target_scale = sum(ellipsoid_axes) / 3.0
    mesh = load_and_center_mesh(
        mesh_path,
        icosphere_nsub=icosphere_nsub,
        target_scale=target_scale,
    )
    device = mesh.points.device
    dtype = mesh.points.dtype

    target_np = points_on_ellipsoid(n_target_points, axes=ellipsoid_axes)
    target_points = torch.tensor(
        target_np, device=device, dtype=dtype, requires_grad=False
    )

    plotter = create_plotter(mesh.pvmesh, target_np)
    plotter.write_frame()

    initial_loss = (
        loss_fn(mesh, target_points, softmin_temperature=softmin_temperature)
        .detach()
        .item()
    )
    print("Mesh vertices:", mesh.n_points, "| Target points:", n_target_points)
    print("Initial Chamfer distance:", f"{initial_loss:.6e}")
    print("Step  | Chamfer distance")
    print("------|------------------")

    for step in range(1, n_steps + 1):
        loss = loss_fn(
            mesh, target_points, softmin_temperature=softmin_temperature
        )
        loss.backward()
        with torch.no_grad():
            torch.sub(mesh.points, mesh.points.grad * lr, out=mesh.points)
        torch.zero_(mesh.points.grad)

        if step % log_interval == 0:
            loss_val = loss.detach().item()
            print(f"{step:5d} | {loss_val:.6e}")
            pts = mesh.points.detach().numpy()
            mesh.pvmesh.points = pts
            mesh.pvmesh["distance_to_target"] = _distance_to_target(
                pts, target_np
            )
            plotter.write_frame()

    plotter.close()
    final_loss = (
        loss_fn(mesh, target_points, softmin_temperature=softmin_temperature)
        .detach()
        .item()
    )
    print(f"\nFinal Chamfer distance: {final_loss:.6e}")
    print(f"Relative reduction: {(1 - final_loss / initial_loss) * 100:.1f}%")


###############################################################################
# Step 6: Run the example
# -----------------------
if __name__ == "__main__":
    main(n_steps=N_STEPS, log_interval=LOG_INTERVAL)

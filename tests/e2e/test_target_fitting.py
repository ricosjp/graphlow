"""E2E tests: target shape fitting via Chamfer distance (roadmap 13.2).

Deform mesh points to minimize chamfer_distance(mesh, target_points).
Aligns with examples/target_fitting.py: ellipsoid target, dense point cloud,
softmin Chamfer, and centered/scaled mesh.
"""

from __future__ import annotations

import pathlib
import shutil

import numpy as np
import pytest
import pyvista as pv
import torch

import graphlow

_TARGET_SAMPLE_SEED = 0

# -----------------------------------------------------------------------------
# Test config: mesh_path (under data_dir), target, optimization, assertions
# -----------------------------------------------------------------------------
FITTING_CASES = [
    {
        "id": "icosphere",
        "mesh_path": pathlib.Path("vtp/icosphere_surface/mesh.vtp"),
        "ellipsoid_axes": (1.4, 0.8, 0.5),
        "softmin_temperature": 0.08,
        "n_target": 2000,
        "n_steps": 2000,
        "lr": 0.06,
        "final_loss_threshold": 1.0,
    },
]


def _points_on_ellipsoid(
    n: int,
    axes: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Sample n points on an ellipsoid with semi-axes (a, b, c)."""
    rng = np.random.default_rng(_TARGET_SAMPLE_SEED)
    phi = np.arccos(2.0 * rng.uniform(0, 1, n) - 1.0)
    theta = rng.uniform(0, 2 * np.pi, n)
    a, b, c = axes
    x = a * np.sin(phi) * np.cos(theta)
    y = b * np.sin(phi) * np.sin(theta)
    z = c * np.cos(phi)
    return np.stack([x, y, z], axis=1).astype(np.float32)


@pytest.mark.slow
@pytest.mark.parametrize("config", FITTING_CASES, ids=lambda c: c["id"])
def test_fit_mesh_to_target_points(
    config: dict,
    data_dir: pathlib.Path,
    pytestconfig: pytest.Config,
) -> None:
    """Minimize Chamfer distance to ellipsoid target point cloud."""
    mesh_path = config["mesh_path"]
    ellipsoid_axes = config["ellipsoid_axes"]
    softmin_temperature = config["softmin_temperature"]
    n_target = config["n_target"]
    n_steps = config["n_steps"]
    lr = config["lr"]
    final_loss_threshold = config["final_loss_threshold"]

    # ----- 1. Output directory (optional; only if --save) -----
    save_enabled = pytestconfig.getoption("save", default=False)
    output_dir = pathlib.Path("tests/outputs/target_fitting") / config["id"]
    if save_enabled and output_dir.exists():
        shutil.rmtree(output_dir)
    if save_enabled:
        output_dir.mkdir(parents=True, exist_ok=True)

    path = data_dir / mesh_path
    if not path.exists():
        pytest.skip(f"Data not found: {path}")
    pv_mesh = pv.read(path).cast_to_unstructured_grid()
    pv_mesh.points = pv_mesh.points.astype(np.float32)
    mesh = graphlow.from_pyvista(pv_mesh, "torch")
    mesh.requires_grad(True)
    device = mesh.points.device
    dtype = mesh.points.dtype

    # Dense target on ellipsoid (independent of mesh vertex count).
    target_np = _points_on_ellipsoid(n_target, axes=ellipsoid_axes)
    target_points = torch.tensor(
        target_np, device=device, dtype=dtype, requires_grad=False
    )

    # Center and scale mesh to sit near target (same as example).
    target_scale = sum(ellipsoid_axes) / 3.0
    with torch.no_grad():
        mesh.points.sub_(mesh.points.mean(dim=0, keepdim=True))
        r = torch.linalg.norm(mesh.points, dim=1).mean().item()
        if r > 1e-8:
            mesh.points.mul_(1.0 / r)
        mesh.points.mul_(target_scale)

    def loss_fn() -> torch.Tensor:
        cd = mesh.geometry.chamfer_distance(
            target_points, softmin_temperature=softmin_temperature
        )
        return cd.squeeze()

    log_interval = max(1, n_steps // 10)
    initial_loss = loss_fn().detach().item()
    for step in range(1, n_steps + 1):
        loss = loss_fn()
        loss.backward()
        with torch.no_grad():
            mesh.points.sub_(mesh.points.grad * lr)
        mesh.points.grad.zero_()

        if save_enabled and step % log_interval == 0:
            mesh.pvmesh.points = mesh.points.detach().numpy()
            mesh.pvmesh.save(str(output_dir / f"mesh.{step:08d}.vtu"))

    final_loss = loss_fn().detach().item()
    assert final_loss < initial_loss, (
        f"Chamfer distance should decrease: "
        f"initial={initial_loss:.6f}, final={final_loss:.6f}"
    )
    assert final_loss < final_loss_threshold, (
        f"Final Chamfer distance should be < {final_loss_threshold}, "
        f"got {final_loss:.6f}"
    )

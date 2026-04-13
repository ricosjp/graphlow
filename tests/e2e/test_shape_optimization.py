"""E2E tests: shape optimization.

Minimize surface area under a volume constraint by deforming mesh points
with a small MLP and running gradient descent. Verifies that the pipeline
(load mesh → deform → volume/area → cost → backward → step) runs and that
the final shape approaches a sphere (fixed volume, minimal area).

Boundary surface points are used only for the final assertion (mean radius
of the deformed boundary). Collected via FP skeleton + boundary_mask.
"""

from __future__ import annotations

import logging
import pathlib
import shutil

import numpy as np
import phlower_tensor as pt
import pytest
import pyvista as pv
import torch

import graphlow
from graphlow.utils.topology_helper import TopologyDim

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Test config: mesh file, iteration count, and convergence threshold
# -----------------------------------------------------------------------------
OPTIMIZATION_CASES = [
    {
        "id": "mix_poly",
        "mesh_path": pathlib.Path("tests/data/vtu/mix_poly/mesh.vtu"),
        "n_steps": 1000,
        "use_bias": False,
        "relative_error_threshold": 1.0,
    },
    {
        "id": "complex",
        "mesh_path": pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
        "n_steps": 10000,
        "use_bias": False,
        "relative_error_threshold": 0.05,
    },
    {
        "id": "cube",
        "mesh_path": pathlib.Path("tests/data/vts/cube/mesh.vts"),
        "n_steps": 2000,
        "use_bias": False,
        "relative_error_threshold": 0.05,
    },
]

# -----------------------------------------------------------------------------
# Hyperparameters for the optimization (fixed for all cases)
# -----------------------------------------------------------------------------
WEIGHT_VOLUME_CONSTRAINT = 10.0
WEIGHT_NORM_CONSTRAINT = 1.0
MLP_HIDDEN = 64
DEFORMATION_FACTOR = 1.0
LEARNING_RATE = 1e-2
# Reject step if any cell volume < this * (total / n_cells)
TINY_CELL_RATIO = 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("config", OPTIMIZATION_CASES, ids=lambda c: c["id"])
def test_optimize_area_volume(
    config: dict,
    pytestconfig: pytest.Config,
) -> None:
    """Minimize area under fixed volume; assert final shape near sphere."""
    mesh_path = config["mesh_path"]
    n_steps = config["n_steps"]
    use_bias = config["use_bias"]
    threshold = config["relative_error_threshold"]
    log_interval = max(1, n_steps // 100)
    logger.info(
        "Running shape optimization: mesh=%s, n_steps=%d",
        mesh_path,
        n_steps,
    )

    # ----- 1. Output directory (optional; only if --save) -----
    save_enabled = pytestconfig.getoption("save", default=False)
    output_dir = pathlib.Path("tests/outputs/geometry_optimization") / (
        mesh_path.parent.name
    )
    if save_enabled and output_dir.exists():
        shutil.rmtree(output_dir)
    if save_enabled:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ----- 2. Load mesh and center it -----
    pv_mesh = pv.read(mesh_path).cast_to_unstructured_grid()
    pv_mesh.points = pv_mesh.points - np.mean(
        pv_mesh.points, axis=0, keepdims=True
    )
    mesh = graphlow.from_pyvista(pv_mesh, "phlower")
    mesh.requires_grad(True)

    if mesh.topology.mesh_dim() < TopologyDim.VOLUME:
        logger.info("Skipping: volume mesh required for boundary surface")
        pytest.skip("Volume mesh required for boundary surface (face_registry)")

    initial_volumes = mesh.geometry.cell_volumes().clone()
    initial_total_volume = torch.sum(initial_volumes).detach().clone()
    initial_points = mesh.points.clone()

    surface = mesh.extract_surface()
    surface_initial_areas = surface.geometry.face_areas().detach().clone()
    surface_initial_total_area = torch.sum(surface_initial_areas)

    # ----- 3. Deformation model: MLP (points -> delta) -----
    scale = MLP_HIDDEN**0.5
    w1 = torch.nn.Parameter(torch.randn(3, MLP_HIDDEN) / scale)
    w2 = torch.nn.Parameter(torch.randn(MLP_HIDDEN, 3) / scale)
    if use_bias:
        b1 = torch.nn.Parameter(torch.randn(MLP_HIDDEN) / MLP_HIDDEN)
        b2 = torch.nn.Parameter(torch.randn(3) / 3)
        params = [w1, b1, w2, b2]
    else:
        b1 = 0.0
        b2 = 0.0
        params = [w1, w2]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    output_activation = torch.nn.Identity()

    # ----- 4. Cost: area + volume penalty + deformation regularizer -----
    def cost_fn(
        deformed_points: pt.PhlowerTensor,
    ) -> tuple[
        pt.PhlowerTensor | None,
        pt.PhlowerTensor | None,
        pt.PhlowerTensor | None,
    ]:
        mesh.points = deformed_points
        volumes = mesh.geometry.cell_volumes()
        total_volume = torch.sum(volumes)
        # Surface mesh points are a snapshot
        # sync from volume so area is current
        surface.points = surface.gather_parent_point_data(deformed_points)
        areas = surface.geometry.face_areas()
        total_area = torch.sum(areas)
        deformation = deformed_points - initial_points

        # Reject steps that collapse any cell
        min_vol = (
            initial_total_volume.to_tensor() / mesh.n_cells * TINY_CELL_RATIO
        )
        if torch.any(volumes.to_tensor() < min_vol):
            return None, None, None

        cost_area = total_area / surface_initial_total_area
        volume_penalty = (
            (total_volume - initial_total_volume) / initial_total_volume
        ) ** 2
        scale_vol = initial_total_volume ** (2 / 3)
        norm_penalty = torch.exp(
            torch.mean(deformation * deformation / scale_vol)
        )
        cost = (
            cost_area
            + WEIGHT_VOLUME_CONSTRAINT * volume_penalty
            + WEIGHT_NORM_CONSTRAINT * norm_penalty
        )
        return cost, total_area, total_volume

    # ----- 5. Optimization loop -----
    deform_factor = [DEFORMATION_FACTOR]  # mutable so cost_fn can reduce it

    def deform_with_factor(points: pt.PhlowerTensor) -> pt.PhlowerTensor:
        hidden = torch.tanh(points.to_tensor() @ w1 + b1)
        delta = pt.phlower_tensor(
            output_activation(hidden @ w2 + b2), dimension=points.dimension
        )
        return points + delta * deform_factor[0]

    logger.info("initial volume: %s", f"{initial_total_volume.to_tensor():.5f}")
    logger.info("  step |     area | vol ratio |     cost")

    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        deformed_points = deform_with_factor(initial_points)

        cost, area, volume = cost_fn(deformed_points)
        if cost is None:
            deform_factor[0] *= 0.9
            logger.info("reduced deformation_factor: %s", deform_factor[0])
            continue

        if step % log_interval == 0:
            vol_ratio = volume / initial_total_volume
            logger.info(
                "%6d | %.3e | %.3e | %.3e",
                step,
                area.to_tensor(),
                vol_ratio.to_tensor(),
                cost.to_tensor(),
            )
            if save_enabled:
                mesh.pvmesh.points = (
                    mesh.backend.to_torch(mesh.points).detach().numpy()
                )
                mesh.pvmesh.save(str(output_dir / f"mesh.{step:08d}.vtu"))

        cost.backward()
        optimizer.step()

    # ----- 6. Assert: mean radius of boundary points ~ sphere radius -----
    final_deformed = deform_with_factor(initial_points)
    surface_deformed_points = surface.gather_parent_point_data(final_deformed)
    actual_mean_radius = (
        torch.mean(
            torch.linalg.norm(surface_deformed_points.to_tensor(), dim=1)
        )
        .detach()
        .numpy()
    )
    # Sphere same volume: V = (4/3)*pi*r^3  =>  r = (3*V/(4*pi))^(1/3)
    v0 = initial_total_volume.to_tensor().numpy()
    desired_radius = (v0 * 3 / 4 / np.pi) ** (1 / 3)
    relative_error = (actual_mean_radius - desired_radius) ** 2 / (
        desired_radius**2
    )
    assert relative_error < threshold, (
        f"relative radius error {relative_error} >= {threshold}"
    )

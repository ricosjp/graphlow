"""
Continuous sphere eversion with graphlow geometry APIs
======================================================
This example demonstrates a *continuous* sphere eversion, i.e. an inside-out
flip without changing the surface mesh connectivity, based on the PyVista
reference implementation:
https://docs.pyvista.org/ja/examples/99-advanced/sphere_eversion

The deformation is parameterized as a continuous transform chain, where only
vertex positions evolve while the underlying mesh connectivity remains fixed.

1) Total area via ``mesh.geometry.face_areas()``
2) Face normals via ``mesh.geometry.face_normals()``
3) Enclosed volume via ``mesh.geometry.surface_volume()``

This example highlights several key properties of ``graphlow``:

- **Topology-geometry decoupling**
  The mesh connectivity is kept unchanged throughout the entire eversion,
  while all geometric quantities are recomputed solely from vertex positions.

- **Robustness under extreme deformation**
  The eversion passes through highly distorted and self-intersecting states,
  yet geometric evaluations (areas, normals, volume) remain stable.

- **Support for non-manifold configurations**
  Intermediate states include self-intersections that violate manifold
  assumptions, demonstrating that ``graphlow`` does not rely on strict
  manifold constraints.

- **Continuous and differentiable geometry**
  The deformation is continuous and compatible with autodiff through the
  backend, enabling gradient-based analysis of geometric quantities.

Overall, this demo serves as a stress test of geometry processing under
extreme, topology-preserving deformation, showcasing the flexibility and
robustness of ``graphlow``'s geometry APIs.
"""

###############################################################################
# Imports and constants
# ---------------------
import argparse
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyvista as pv
import torch

import graphlow

DEFAULT_N_STEPS = 30
DEFAULT_THETA_RESOLUTION = 200
DEFAULT_PHI_RESOLUTION = 400
DEFAULT_NORMAL_SCALE = 0.08
DEFAULT_NORMAL_STRIDE = 20
SURFACE_COLOR = "aquamarine"
BACKFACE_COLOR = "forestgreen"
NORMAL_COLOR = "blue"
PLOTTER_WINDOW_SIZE = (1200, 450)
_PARAM_W = 2.0
_PARAM_N_LOBES = 2
_PARAM_Q = 2.0 / 3.0
_PARAM_BETA = 1.0
_PARAM_ALPHA_FINAL = 1.0
_PARAM_ETA_FINAL = 2.0

# Read this file in order:
# 1. CLI / reference mesh / eversion schedule
# 2. Continuous deformation maps
# 3. Geometry measurements
# 4. Plotting / scene updates


###############################################################################
# Small containers
# ----------------
@dataclass(frozen=True)
class EversionState:
    """One eversion stage state evaluated on the fixed surface mesh."""

    mode: Literal["unfold", "close"]
    t: float
    q: float
    p: float
    xi: float
    alpha: float
    eta: float
    lam: float


@dataclass
class HistoricalData:
    """Historical data for the metrics charts."""

    frame_indices: list[int]
    area: list[float]
    volume: list[float]


@dataclass
class SceneActors:
    """PyVista plotter and current actors."""

    plotter: pv.Plotter
    surface_actor: object
    normals_actor: object
    area_chart: pv.Chart2D
    volume_chart: pv.Chart2D


@dataclass
class FrameMetrics:
    """Geometry values extracted from the current graphlow mesh."""

    area: float
    signed_volume: float
    normal_origins: np.ndarray
    normal_vectors: np.ndarray
    normal_latitude_index: np.ndarray
    normal_longitude_index: np.ndarray


###############################################################################
# Step 1: Parse options
# ---------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line options for the example."""
    parser = argparse.ArgumentParser(
        description="Continuous sphere eversion with graphlow metrics.",
    )
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument(
        "--theta-resolution",
        type=int,
        default=DEFAULT_THETA_RESOLUTION,
    )
    parser.add_argument(
        "--phi-resolution",
        type=int,
        default=DEFAULT_PHI_RESOLUTION,
    )
    parser.add_argument(
        "--normal-scale",
        type=float,
        default=DEFAULT_NORMAL_SCALE,
    )
    parser.add_argument(
        "--normal-stride",
        type=int,
        default=DEFAULT_NORMAL_STRIDE,
        help="Sample normal arrows every N latitude/longitude lines (>=1).",
    )
    return parser.parse_args()


###############################################################################
# Step 2: Build one fixed graphlow surface mesh
# ---------------------------------------------
def build_reference_surface_mesh(
    *,
    theta_resolution: int,
    phi_resolution: int,
) -> graphlow.TensorMesh[torch.Tensor]:
    """
    Build the reference surface mesh and attach spherical parameters.

    A single TensorMesh is created once. All later frames only update
    ``mesh.points`` while keeping connectivity fixed.
    """
    pv_mesh = pv.Sphere(
        radius=1.0,
        theta_resolution=theta_resolution,
        phi_resolution=phi_resolution,
    )
    pv_mesh.rotate_z(90.0, inplace=True)
    pv_mesh = reverse_triangle_winding(pv_mesh)
    mesh = graphlow.from_pyvista(pv_mesh, "torch", torch.float64)
    mesh.requires_grad(True)

    reference_points = mesh.points.clone()
    radius = torch.linalg.vector_norm(reference_points, dim=1)
    theta = torch.asin(torch.clamp(reference_points[:, 2] / radius, -1.0, 1.0))
    phi = torch.atan2(reference_points[:, 1], reference_points[:, 0])
    h, _ = sphere_to_cylinder(
        theta,
        phi,
        w=_PARAM_W,
        n_lobes=_PARAM_N_LOBES,
    )
    latitude_index, longitude_index = build_sphere_point_indices(
        theta_resolution=theta_resolution,
        phi_resolution=phi_resolution,
    )

    mesh.point_data["theta"] = theta[:, None]
    mesh.point_data["phi"] = phi[:, None]
    mesh.point_data["h"] = h[:, None]
    mesh.point_data["latitude_index"] = latitude_index[:, None]
    mesh.point_data["longitude_index"] = longitude_index[:, None]
    return mesh


###############################################################################
# Step 2a: Reference mesh helpers
# -------------------------------
def reverse_triangle_winding(surface: pv.PolyData) -> pv.PolyData:
    """
    Reverse triangle winding to match the PyVista eversion orientation.

    The eversion formulas define a consistent surface orientation. When we use
    ``pv.Sphere(...)`` as the fixed reference topology, its default face
    winding is opposite to the orientation assumed by the reference eversion
    demo. Reversing the triangle order once at construction time aligns the
    initial normals and signed volume with the demo.
    """
    faces = surface.faces.reshape(-1, 4).copy()
    if not np.all(faces[:, 0] == 3):
        raise ValueError("Expected a triangulated sphere surface.")
    faces[:, [2, 3]] = faces[:, [3, 2]]
    flipped = pv.PolyData(surface.points.copy(), faces.reshape(-1))
    return flipped


def build_sphere_point_indices(
    *,
    theta_resolution: int,
    phi_resolution: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build integer latitude/longitude indices matching ``pv.Sphere`` order."""
    n_points = 2 + theta_resolution * (phi_resolution - 2)
    latitude_index = torch.zeros(n_points, dtype=torch.int64)
    longitude_index = torch.zeros(n_points, dtype=torch.int64)

    latitude_index[0] = phi_resolution - 1
    latitude_index[1] = 0

    ring_size = phi_resolution - 2
    point_id = 2
    for longitude_id in range(theta_resolution):
        for ring_id in range(ring_size):
            latitude_index[point_id] = phi_resolution - 2 - ring_id
            longitude_index[point_id] = longitude_id
            point_id += 1

    return latitude_index, longitude_index


###############################################################################
# Step 3: Eversion parameter schedule
# -----------------------------------
def generate_eversion_schedule(
    *,
    n_steps: int,
) -> list[EversionState]:
    """Generate the eversion stage schedule as scalar parameter states."""
    states: list[EversionState] = []
    t = -1.0 / _PARAM_Q
    q = _PARAM_Q
    p = 0.0
    xi = 0.0
    alpha = 0.0
    eta = 1.0

    # Stage 1: sphere -> inverted wormhole
    for lam in np.linspace(0.0, 1.0, n_steps, endpoint=False):
        states.append(
            EversionState(
                mode="unfold",
                t=t,
                q=q,
                p=0.0,
                xi=0.0,
                alpha=0.0,
                eta=eta,
                lam=float(lam),
            )
        )

    # Stage 2: inverted wormhole -> unfolded wormhole
    xi_values = np.linspace(0.0, 1.0, n_steps)
    alpha_values = np.linspace(0.0, _PARAM_ALPHA_FINAL, n_steps)
    eta_values = np.linspace(1.0, _PARAM_ETA_FINAL, n_steps)
    for xi, alpha, eta in zip(xi_values, alpha_values, eta_values, strict=True):
        states.append(
            EversionState(
                mode="close",
                t=t,
                q=q,
                p=0.0,
                xi=float(xi),
                alpha=float(alpha),
                eta=float(eta),
                lam=0.0,
            )
        )
    xi = 1.0
    alpha = _PARAM_ALPHA_FINAL
    eta = _PARAM_ETA_FINAL

    # Stage 3: unfolded wormhole -> closed wormhole
    for q in np.linspace(_PARAM_Q, 0.0, n_steps):
        p = 1.0 - abs(q * t)
        states.append(
            EversionState(
                mode="close",
                t=t,
                q=float(q),
                p=float(p),
                xi=xi,
                alpha=alpha,
                eta=eta,
                lam=0.0,
            )
        )
    q = 0.0
    p = 1.0

    # Stage 4: closed wormhole turns inside out
    for t in np.linspace(-1.0 / _PARAM_Q, 1.0 / _PARAM_Q, n_steps):
        p = 1.0 - abs(q * t)
        states.append(
            EversionState(
                mode="close",
                t=float(t),
                q=q,
                p=float(p),
                xi=xi,
                alpha=alpha,
                eta=eta,
                lam=0.0,
            )
        )
    t = 1.0 / _PARAM_Q
    p = 0.0

    # Stage 5: closed wormhole -> unfolded wormhole
    for q in np.linspace(0.0, _PARAM_Q, n_steps + 1)[1:]:
        p = 1.0 - abs(q * t)
        states.append(
            EversionState(
                mode="close",
                t=t,
                q=float(q),
                p=float(p),
                xi=xi,
                alpha=alpha,
                eta=eta,
                lam=0.0,
            )
        )
    q = _PARAM_Q
    p = 0.0

    # Stage 6: unfolded wormhole -> inverted wormhole
    # Reverse the Stage 2 closing parameters back to the inverted wormhole.
    # eta must also return from eta_final to 1.0; otherwise Stage 7 starts
    # from a different surface and a visible discontinuity appears.
    xi_values = np.linspace(1.0, 0.0, n_steps + 1)[1:]
    alpha_values = np.linspace(_PARAM_ALPHA_FINAL, 0.0, n_steps + 1)[1:]
    eta_values = np.linspace(_PARAM_ETA_FINAL, 1.0, n_steps + 1)[1:]
    for xi, alpha, eta in zip(
        xi_values,
        alpha_values,
        eta_values,
        strict=True,
    ):
        states.append(
            EversionState(
                mode="close",
                t=t,
                q=q,
                p=p,
                xi=float(xi),
                alpha=float(alpha),
                eta=float(eta),
                lam=0.0,
            )
        )

    # Stage 7: inverted wormhole -> sphere
    for lam in np.linspace(1.0, 0.0, n_steps + 1)[1:]:
        states.append(
            EversionState(
                mode="unfold",
                t=t,
                q=q,
                p=0.0,
                xi=0.0,
                alpha=0.0,
                eta=1.0,
                lam=float(lam),
            )
        )

    return states


###############################################################################
# Step 4: Deformation maps evaluated on TensorMesh point_data
# -----------------------------------------------------------
# The eversion is split into small reusable transforms:
# 1. Sphere coordinates -> cylindrical variables
# 2. Cylindrical variables -> open wormhole
# 3. Open wormhole -> closed / flipped states
# 4. Dispatcher that evaluates one frame from the current schedule state
def sphere_to_cylinder(
    theta: torch.Tensor,
    phi: torch.Tensor,
    *,
    w: float,
    n_lobes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map spherical coordinates to intermediate cylindrical variables."""
    h = w * torch.sin(theta) / torch.cos(theta) ** n_lobes
    return h, phi


def cylinder_to_wormhole(
    h: torch.Tensor,
    phi: torch.Tensor,
    *,
    t: float,
    p: float,
    q: float,
    n_lobes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map the cylindrical representation to an open wormhole."""
    x = (
        t * torch.cos(phi)
        + p * torch.sin((n_lobes - 1) * phi)
        - h * torch.sin(phi)
    )
    y = (
        t * torch.sin(phi)
        + p * torch.cos((n_lobes - 1) * phi)
        + h * torch.cos(phi)
    )
    z = (
        h * torch.sin(n_lobes * phi)
        - t / n_lobes * torch.cos(n_lobes * phi)
        - q * t * h
    )
    return x, y, z


def close_wormhole(
    x0: torch.Tensor,
    y0: torch.Tensor,
    z0: torch.Tensor,
    *,
    eta: float,
    xi: float,
    alpha: float,
    beta: float,
    kappa: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the wormhole-closing transform with torch tensors."""
    denominator = xi + eta * (x0**2 + y0**2)
    x1 = x0 / denominator**kappa
    y1 = y0 / denominator**kappa
    z1 = z0 / denominator

    gamma = 2.0 * math.sqrt(alpha * beta)
    if np.isclose(gamma, 0.0):
        denominator = x1**2 + y1**2
        x2 = x1 / denominator
        y2 = y1 / denominator
        z2 = -z1
        return x2, y2, z2

    exponential = torch.exp(gamma * z1)
    numerator = alpha - beta * (x1**2 + y1**2)
    denominator = alpha + beta * (x1**2 + y1**2)
    x2 = x1 * exponential / denominator
    y2 = y1 * exponential / denominator
    z2 = (
        numerator / denominator * exponential / gamma
        - (alpha - beta) / (alpha + beta) / gamma
    )
    return x2, y2, z2


def unfold_sphere(
    theta: torch.Tensor,
    phi: torch.Tensor,
    *,
    t: float,
    q: float,
    eta: float,
    lam: float,
    w: float,
    n_lobes: int,
    kappa: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the unfold / recover transform with torch tensors."""
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    x = (
        t * (1.0 - lam + lam * cos_theta**n_lobes) * cos_phi
        - lam * w * sin_theta * sin_phi
    )
    x = x / cos_theta**n_lobes
    y = (
        t * (1.0 - lam + lam * cos_theta**n_lobes) * sin_phi
        + lam * w * sin_theta * cos_phi
    )
    y = y / cos_theta**n_lobes
    z = lam * (
        w * sin_theta * (torch.sin(n_lobes * phi) - q * t) / cos_theta**n_lobes
        - t / n_lobes * torch.cos(n_lobes * phi)
    ) - (1.0 - lam) * eta ** (1.0 + kappa) * t * abs(t) ** (
        2.0 * kappa
    ) * sin_theta / cos_theta ** (2 * n_lobes)

    denominator = x**2 + y**2
    x2 = x * eta**kappa / denominator ** (1.0 - kappa)
    y2 = y * eta**kappa / denominator ** (1.0 - kappa)
    z2 = -z / eta / denominator
    return x2, y2, z2


###############################################################################
# Step 4b: Evaluate one schedule state on the fixed mesh
# ------------------------------------------------------
def evaluate_points_from_state(
    mesh: graphlow.TensorMesh[torch.Tensor],
    state: EversionState,
) -> torch.Tensor:
    """Evaluate the current vertex positions from the stored parameters."""
    theta = mesh.point_data["theta"].squeeze(-1)
    phi = mesh.point_data["phi"].squeeze(-1)
    h = mesh.point_data["h"].squeeze(-1)
    kappa = (_PARAM_N_LOBES - 1) / (2 * _PARAM_N_LOBES)

    if state.mode == "unfold":
        x, y, z = unfold_sphere(
            theta,
            phi,
            t=state.t,
            q=state.q,
            eta=state.eta,
            lam=state.lam,
            w=_PARAM_W,
            n_lobes=_PARAM_N_LOBES,
            kappa=kappa,
        )
    else:
        x0, y0, z0 = cylinder_to_wormhole(
            h,
            phi,
            t=state.t,
            p=state.p,
            q=state.q,
            n_lobes=_PARAM_N_LOBES,
        )
        x, y, z = close_wormhole(
            x0,
            y0,
            z0,
            eta=state.eta,
            xi=state.xi,
            alpha=state.alpha,
            beta=_PARAM_BETA,
            kappa=kappa,
        )

    return torch.stack([x, y, z], dim=1)


###############################################################################
# Step 5: Measure geometry from the current TensorMesh
# ----------------------------------------------------
# These helpers convert the current geometry into display-ready diagnostics:
# 1. Sparse normals for the 3D scene
# 2. Area / signed volume for the metric plots
def sample_normals(
    origins: np.ndarray,
    vectors: np.ndarray,
    latitude_index: np.ndarray,
    longitude_index: np.ndarray,
    *,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample point normals on a latitude-longitude grid."""
    stride = max(1, int(stride))
    latitude_keys = latitude_index.reshape(-1).astype(np.int64)
    longitude_keys = longitude_index.reshape(-1).astype(np.int64)
    if latitude_keys.size <= 2:
        return origins, vectors

    min_latitude = int(latitude_keys.min())
    max_latitude = int(latitude_keys.max())
    pole_mask = (latitude_keys == min_latitude) | (
        latitude_keys == max_latitude
    )
    ring_mask = ~pole_mask

    sample_mask = pole_mask | (
        ring_mask
        & (latitude_keys % stride == 0)
        & (longitude_keys % 2 * stride == 0)
    )
    return origins[sample_mask], vectors[sample_mask]


def map_face_normals_to_points(
    mesh: graphlow.TensorMesh[torch.Tensor],
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Conservatively map face area vectors onto points and normalize for display.

    Each cell area vector is conservatively distributed to incident points
    using ``mesh.topology.map_cell_to_point(..., "conservative", "segment")``.
    """
    point_area_vectors = mesh.topology.map_cell_to_point(
        mesh.geometry.face_area_vectors(),
        "conservative",
        "segment",
    )

    point_normal_norms = torch.linalg.vector_norm(
        point_area_vectors,
        dim=1,
        keepdim=True,
    )
    point_normals = point_area_vectors / torch.clamp(
        point_normal_norms,
        min=eps,
    )
    normal_origins = mesh.points.detach().cpu().numpy()
    normal_vectors = point_normals.detach().cpu().numpy()
    return normal_origins, normal_vectors


def extract_frame_metrics(
    mesh: graphlow.TensorMesh[torch.Tensor],
) -> FrameMetrics:
    """Compute area, volume, and point normals with graphlow APIs."""
    face_areas = mesh.geometry.face_areas()
    signed_volume = mesh.geometry.surface_volume()

    area = float(torch.sum(face_areas).detach().cpu().item())
    volume = float(signed_volume.detach().cpu().item())
    normal_origins, normal_vectors = map_face_normals_to_points(mesh)
    normal_latitude_index = (
        mesh.point_data["latitude_index"].detach().cpu().numpy()
    )
    normal_longitude_index = (
        mesh.point_data["longitude_index"].detach().cpu().numpy()
    )
    return FrameMetrics(
        area=area,
        signed_volume=volume,
        normal_origins=normal_origins,
        normal_vectors=normal_vectors,
        normal_latitude_index=normal_latitude_index,
        normal_longitude_index=normal_longitude_index,
    )


def sync_pyvista_points(mesh: graphlow.TensorMesh[torch.Tensor]) -> None:
    """Copy the current torch points into the PyVista mesh for rendering."""
    mesh.pvmesh.points = mesh.points.detach().cpu().numpy()


###############################################################################
# Step 6: Build the metrics plot
# ------------------------------
def format_metric_chart(
    chart: pv.Chart2D,
    *,
    title: str,
    y_label: str,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    show_x_label: bool,
) -> None:
    """Apply consistent styling to a metric chart."""
    chart.title = title
    chart.x_label = "Frame" if show_x_label else ""
    chart.y_label = y_label
    chart.x_range = list(x_range)
    chart.y_range = list(y_range)
    chart.background_color = (1.0, 1.0, 1.0, 1.0)
    chart.border_color = (0.75, 0.75, 0.75, 1.0)
    chart.grid = True


def create_metrics_charts(
    historical_data: HistoricalData,
    max_frame: int,
) -> tuple[pv.Chart2D, pv.Chart2D]:
    """Create two PyVista ``Chart2D`` panels for area and signed volume."""
    frame_indices = np.array(historical_data.frame_indices)
    area = np.array(historical_data.area)
    volume = np.array(historical_data.volume)

    area_chart = pv.Chart2D()
    area_chart.scatter(frame_indices, area, size=3)
    format_metric_chart(
        area_chart,
        title="Total area",
        y_label="Area",
        x_range=(0, max_frame),
        y_range=(0.0, 15.0),
        show_x_label=False,
    )
    volume_chart = pv.Chart2D()
    volume_chart.scatter(frame_indices, volume, size=3)
    format_metric_chart(
        volume_chart,
        title="Signed volume",
        y_label="Volume",
        x_range=(0, max_frame),
        y_range=(-4.0, 4.0),
        show_x_label=True,
    )
    return area_chart, volume_chart


###############################################################################
# Step 7: Build and update the PyVista scene
# ------------------------------------------
# The scene is created once, then its actors are replaced frame by frame.
def create_scene(
    mesh: graphlow.TensorMesh[torch.Tensor],
    metrics: FrameMetrics,
    *,
    max_frame: int,
    normal_stride: int,
    normal_scale: float,
) -> tuple[SceneActors, HistoricalData]:
    """Create the off-screen PyVista scene for GIF rendering."""
    sync_pyvista_points(mesh)
    historical_data = HistoricalData(
        frame_indices=[],
        area=[],
        volume=[],
    )
    historical_data.frame_indices.append(0)
    historical_data.area.append(metrics.area)
    historical_data.volume.append(metrics.signed_volume)

    shape = (2, 4)
    groups = [([0, 1], [0, 1]), (0, [2, 3]), (1, [2, 3])]
    plotter = pv.Plotter(
        window_size=list(PLOTTER_WINDOW_SIZE),
        off_screen=True,
        shape=shape,
        groups=groups,
    )

    plotter.open_gif("sphere_eversion_continuous_metrics.gif")
    plotter.add_text(
        "Continuous eversion",
        position="upper_left",
        font_size=12,
    )
    plotter.subplot(0, 0)
    plotter.camera_position = "iso"
    surface_actor = plotter.add_mesh(
        mesh.pvmesh,
        color=SURFACE_COLOR,
        specular=1.0,
        specular_power=50.0,
        show_edges=False,
        smooth_shading=True,
        backface_params={"color": BACKFACE_COLOR},
    )

    normal_origins, normal_vectors = sample_normals(
        metrics.normal_origins,
        metrics.normal_vectors,
        metrics.normal_latitude_index,
        metrics.normal_longitude_index,
        stride=normal_stride,
    )
    normals_actor = plotter.add_arrows(
        normal_origins,
        normal_vectors,
        mag=normal_scale,
        color=NORMAL_COLOR,
    )

    area_chart, volume_chart = create_metrics_charts(
        historical_data,
        max_frame=max_frame,
    )

    plotter.subplot(0, 2)
    plotter.add_chart(area_chart)
    plotter.subplot(1, 2)
    plotter.add_chart(volume_chart)

    scene_actors = SceneActors(
        plotter=plotter,
        surface_actor=surface_actor,
        normals_actor=normals_actor,
        area_chart=area_chart,
        volume_chart=volume_chart,
    )
    return scene_actors, historical_data


def update_scene(
    scene: SceneActors,
    mesh: graphlow.TensorMesh[torch.Tensor],
    metrics: FrameMetrics,
    historical_data: HistoricalData,
    *,
    normal_stride: int,
    normal_scale: float,
) -> None:
    """Replace the current surface mesh and normal arrows."""
    sync_pyvista_points(mesh)
    historical_data.frame_indices.append(len(historical_data.frame_indices))
    historical_data.area.append(metrics.area)
    historical_data.volume.append(metrics.signed_volume)

    frame_indices = np.array(historical_data.frame_indices)
    area = np.array(historical_data.area)
    volume = np.array(historical_data.volume)

    scene.plotter.subplot(0, 0)
    scene.plotter.camera_position = "iso"

    scene.plotter.remove_actor(scene.surface_actor)
    scene.surface_actor = scene.plotter.add_mesh(
        mesh.pvmesh,
        color=SURFACE_COLOR,
        specular=1.0,
        specular_power=50.0,
        show_edges=False,
        smooth_shading=True,
        backface_params={"color": BACKFACE_COLOR},
    )

    scene.plotter.remove_actor(scene.normals_actor)
    normal_origins, normal_vectors = sample_normals(
        metrics.normal_origins,
        metrics.normal_vectors,
        metrics.normal_latitude_index,
        metrics.normal_longitude_index,
        stride=normal_stride,
    )
    scene.normals_actor = scene.plotter.add_arrows(
        normal_origins,
        normal_vectors,
        mag=normal_scale,
        color=NORMAL_COLOR,
    )
    scene.plotter.subplot(0, 2)
    for plot in scene.area_chart.plots():
        plot.update(frame_indices, area)

    scene.plotter.subplot(1, 2)
    for plot in scene.volume_chart.plots():
        plot.update(frame_indices, volume)
    scene.plotter.write_frame()


###############################################################################
# Step 8: Frame loop
# ------------------
def render_animation(
    *,
    mesh: graphlow.TensorMesh[torch.Tensor],
    schedule: list[EversionState],
    scene: SceneActors,
    historical_data: HistoricalData,
    normal_stride: int,
    normal_scale: float,
) -> None:
    """Render all frames by updating one TensorMesh in place."""
    for state in schedule:
        mesh.points = evaluate_points_from_state(mesh, state)
        metrics = extract_frame_metrics(mesh)
        update_scene(
            scene,
            mesh,
            metrics,
            historical_data,
            normal_stride=normal_stride,
            normal_scale=normal_scale,
        )


###############################################################################
# Step 9: Run the example
# -----------------------
def main() -> None:
    """
    Run the example.

    Step 1
        Parse CLI options.
    Step 2
        Build the eversion schedule.
    Step 3
        Build one fixed surface ``TensorMesh``.
    Step 4
        Measure initial geometry; create metric charts and off-screen scene
        (GIF opened).
    Step 5
        For each schedule state, update ``mesh.points`` and ``write_frame``.
    Step 6
        Close the plotter (finalize GIF).
    """
    args = parse_args()
    pv.global_theme.background = "white"

    # Build the eversion schedule.
    schedule = generate_eversion_schedule(n_steps=args.n_steps)
    max_frame = len(schedule)

    # Build the fixed reference topology.
    mesh = build_reference_surface_mesh(
        theta_resolution=args.theta_resolution,
        phi_resolution=args.phi_resolution,
    )
    initial_metrics = extract_frame_metrics(mesh)

    scene, historical_data = create_scene(
        mesh,
        initial_metrics,
        max_frame=max_frame,
        normal_stride=max(1, args.normal_stride),
        normal_scale=args.normal_scale,
    )

    # Render all frames.
    render_animation(
        mesh=mesh,
        schedule=schedule,
        scene=scene,
        historical_data=historical_data,
        normal_stride=max(1, args.normal_stride),
        normal_scale=args.normal_scale,
    )

    scene.plotter.close()


if __name__ == "__main__":
    main()

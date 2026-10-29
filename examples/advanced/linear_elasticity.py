"""
Linear elasticity: tension and torsion
======================================

This example assembles isotropic linear elasticity with graphlow's
quadrature FEM API (material rank 4), applies Dirichlet conditions, and
solves with SciPy.

Two classic problems are solved:

1. **Uniaxial tension** on a rectangular tet beam. The exact field is uniform
   axial stretch with Poisson contraction about the cross-section center
   ``(y_c, z_c)``

   .. math::

      u_x = \\varepsilon x,\\quad
      u_y = -\\nu\\varepsilon (y - y_c),\\quad
      u_z = -\\nu\\varepsilon (z - z_c)

2. **End torsion** on a tetrahedral cylinder. The top face rotates while the
   bottom is fixed; the analytical rotation grows linearly with ``z``.
   The FEM problem is the linearized (small-angle) model. A linearized twist
   displacement lengthens the radius slightly
   (``|r'| = |r|\\sqrt{1+\\theta^2}``); the GIF therefore morphs with the
   matching **finite rotation** so the cylinder twists in place without that
   artificial expansion.

Local cell rigidity comes from
:meth:`~graphlow.core.geometry.MeshGeometry.cell_local_fem_rigidity`, and
:meth:`~graphlow.core.geometry.MeshGeometry.global_fem_matrix` assembles
the global vector stiffness matrix.
Gallery GIFs morph each mesh from the undeformed geometry (semi-transparent
white) to the deformed state. The tension GIF amplifies lateral displacement
for display so Poisson contraction is easy to see.
"""

###############################################################################
# Imports
# -------
import pathlib
from collections.abc import Callable

import numpy as np
import phlower_tensor as pt
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

import graphlow


def _repo_root() -> pathlib.Path:
    """Locate the repository root without relying on ``__file__``.

    Sphinx-Gallery executes examples without defining ``__file__``, so walk
    upward from the current working directory and look for project markers.
    """
    try:
        return pathlib.Path(__file__).resolve().parents[2]
    except NameError:
        pass

    for candidate in (pathlib.Path.cwd(), *pathlib.Path.cwd().parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "graphlow"
        ).is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the graphlow repository root "
        f"from cwd={pathlib.Path.cwd()}"
    )


REPO_ROOT = _repo_root()
TENSION_MESH_PATH = REPO_ROOT / "tests/data/vtu/tetbeam/mesh.vtu"
TORSION_MESH_PATH = REPO_ROOT / "tests/data/vtu/cylinder/tet.vtu"

YOUNG_MODULUS = 1.0e6
POISSON_RATIO = 0.3
MAX_DISPLACEMENT = 0.1
# Larger than the unit-test 0.1-turn load so the twist is obvious in the GIF.
MAX_TWIST_TURNS = 0.35
N_FRAMES = 30
# Poisson thinning is ~3% at the test strain; amplify uy/uz in the GIF only.
TENSION_LATERAL_VIS_SCALE = 12.0


###############################################################################
# Step 1: Build the isotropic elasticity tensor
# ---------------------------------------------
# Lamé parameters define the rank-4 stiffness used by the local rigidity API.
def isotropic_stiffness(mesh: graphlow.TensorMesh) -> pt.PhlowerTensor:
    """Return isotropic elasticity tensor shaped ``(1, 3, 3, 3, 3, 1)``."""
    backend = mesh.backend
    lam = (YOUNG_MODULUS * POISSON_RATIO) / (
        (1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO)
    )
    mu = YOUNG_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
    stiffness_dimension = {"L": -1, "M": 1, "T": -2}
    delta = backend.as_tensor(torch.eye(3, dtype=backend.dtype))
    return (
        lam
        * backend.einsum(
            "ij,kl->ijkl", delta, delta, dimension=stiffness_dimension
        )
        + mu
        * (
            backend.einsum(
                "ik,jl->ijkl", delta, delta, dimension=stiffness_dimension
            )
            + backend.einsum(
                "il,jk->ijkl", delta, delta, dimension=stiffness_dimension
            )
        )
    )[None, ..., None].to(backend.device)


def to_sparse_dirichlet(
    mesh: graphlow.TensorMesh,
    dirichlet: torch.Tensor,
) -> pt.PhlowerTensor:
    """Convert a dense NaN-masked Dirichlet array to a COO sparse tensor."""
    ax0, ax1, ax2 = torch.where(~torch.isnan(dirichlet))
    return mesh.backend.as_tensor(
        torch.sparse_coo_tensor(
            values=dirichlet[ax0, ax1, ax2].to(dtype=mesh.backend.dtype),
            indices=torch.stack([ax0, ax1, ax2], dim=0),
            size=dirichlet.shape,
            check_invariants=False,
        ),
        dimension=mesh.points.dimension,
    )


###############################################################################
# Step 2: Prescribe Dirichlet data
# --------------------------------
# Tension pulls the right face; torsion rotates the top face about ``z``.
def uniaxial_dirichlet(
    mesh: graphlow.TensorMesh,
) -> tuple[pt.PhlowerTensor, np.ndarray]:
    """Return sparse Dirichlet data and the analytical nodal displacement.

    Lateral contraction is centered on the cross-section mid-plane so the
    beam thins symmetrically about ``(y_c, z_c)``.
    """
    backend = mesh.backend
    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    z = mesh.points[:, 2]

    mask_xmin = torch.abs(x - torch.min(x)).to_tensor() < 1e-5
    mask_xmax = torch.abs(x - torch.max(x)).to_tensor() < 1e-5
    x_min = torch.min(x)
    y_c = 0.5 * (torch.min(y) + torch.max(y))
    z_c = 0.5 * (torch.min(z) + torch.max(z))
    length = float((torch.max(x) - x_min).to_tensor())
    strain = MAX_DISPLACEMENT / length

    exact = np.stack(
        [
            ((x - x_min) * strain).to("cpu").numpy(),
            (-(y - y_c) * strain * POISSON_RATIO).to("cpu").numpy(),
            (-(z - z_c) * strain * POISSON_RATIO).to("cpu").numpy(),
        ],
        axis=-1,
    )

    dirichlet = (
        torch.ones(
            (mesh.n_points, 3, 1),
            dtype=backend.dtype,
            device=backend.device,
        )
        * torch.nan
    )
    # Prescribe the exact end-face displacement (free lateral surfaces).
    end_values = torch.as_tensor(
        exact, dtype=backend.dtype, device=backend.device
    )
    dirichlet[mask_xmin, :, 0] = end_values[mask_xmin]
    dirichlet[mask_xmax, :, 0] = end_values[mask_xmax]
    return to_sparse_dirichlet(mesh, dirichlet), exact


def torsion_dirichlet(
    mesh: graphlow.TensorMesh,
) -> tuple[pt.PhlowerTensor, np.ndarray]:
    """Return sparse Dirichlet data and the analytical twist field."""
    backend = mesh.backend
    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    z = mesh.points[:, 2]

    mask_zmin = torch.abs(z - torch.min(z)).to_tensor() < 1e-5
    mask_zmax = torch.abs(z - torch.max(z)).to_tensor() < 1e-5
    max_angle = MAX_TWIST_TURNS * 2.0 * torch.pi

    dirichlet = (
        torch.ones(
            (mesh.n_points, 3, 1),
            dtype=backend.dtype,
            device=backend.device,
        )
        * torch.nan
    )
    dirichlet[mask_zmin, :] = 0.0
    dirichlet[mask_zmax, 0, 0] = -max_angle * y[mask_zmax].to_tensor()
    dirichlet[mask_zmax, 1, 0] = max_angle * x[mask_zmax].to_tensor()
    dirichlet[mask_zmax, 2, 0] = 0.0

    z_max = torch.max(z)
    exact = np.stack(
        [
            -(max_angle * y * z / z_max).to("cpu").numpy(),
            (max_angle * x * z / z_max).to("cpu").numpy(),
            np.zeros(mesh.n_points),
        ],
        axis=-1,
    )
    return to_sparse_dirichlet(mesh, dirichlet), exact


###############################################################################
# Step 3: Assemble, constrain, and solve
# --------------------------------------
def solve_elasticity(
    mesh: graphlow.TensorMesh,
    sparse_dirichlet: pt.PhlowerTensor,
) -> np.ndarray:
    """Assemble rank-4 elasticity, apply Dirichlet BCs, and solve Ku = f."""
    backend = mesh.backend
    stiffness = isotropic_stiffness(mesh)
    local_rigidity = mesh.geometry.cell_local_fem_rigidity(
        rank=4, cell_material_coeff=stiffness
    )
    rigidity = mesh.geometry.global_fem_matrix(local_rigidity)
    load = backend.zeros(
        (mesh.n_points * 3, 1), dimension={"L": 1, "M": 1, "T": -2}
    ).to(dtype=backend.dtype)
    rigidity, load = mesh.geometry.apply_dirichlet_to_global_fem_matrix(
        rigidity, load, sparse_dirichlet=sparse_dirichlet
    )

    matrix = sp.coo_matrix(
        (
            rigidity.values().to("cpu").numpy(),
            (
                rigidity.indices().to("cpu").numpy()[0],
                rigidity.indices().to("cpu").numpy()[1],
            ),
        ),
        shape=tuple(rigidity.shape),
    ).tocsr()
    displacement, _ = spla.cg(matrix, load.to("cpu").numpy().ravel(), rtol=1e-8)
    return displacement.reshape(-1, 3)


###############################################################################
# Step 4: Animate with a semi-transparent undeformed reference
# ------------------------------------------------------------
# Oblique perspective, ghost original, edged mesh, rainbow scalars. Tension
# amplifies uy/uz; torsion morphs with finite rotation (no fake radial growth).
def _add_ghost_and_deformed(
    plotter: pv.Plotter,
    undeformed: pv.UnstructuredGrid,
    deformed: pv.UnstructuredGrid,
    *,
    scalar_name: str,
    scalar_clim: list[float],
    scalar_bar_title: str,
) -> None:
    """Add a translucent ghost mesh and a colored deforming mesh."""
    plotter.add_mesh(
        undeformed,
        color="lightgray",
        opacity=0.35,
        show_edges=True,
        edge_color="dimgray",
        lighting=False,
    )
    plotter.add_mesh(
        deformed,
        scalars=scalar_name,
        show_edges=True,
        edge_color="black",
        lighting=False,
        cmap="turbo",
        opacity=1.0,
        clim=scalar_clim,
        scalar_bar_args={"title": scalar_bar_title},
    )
    plotter.add_axes(line_width=2, labels_off=False)


def _finite_twist_points(
    original: np.ndarray,
    *,
    factor: float,
    max_turns: float = MAX_TWIST_TURNS,
) -> np.ndarray:
    """Rotate each ring by ``factor * max_turns * 2π * z / z_max``."""
    points = original.copy()
    z = original[:, 2]
    z_max = float(np.max(z))
    theta = factor * max_turns * 2.0 * np.pi * (z / z_max)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x = original[:, 0]
    y = original[:, 1]
    points[:, 0] = cos_t * x - sin_t * y
    points[:, 1] = sin_t * x + cos_t * y
    return points


def animate_tension(
    grid: pv.UnstructuredGrid,
    displacement: np.ndarray,
    *,
    gif_name: str = "linear_elasticity_tension.gif",
    n_frames: int = N_FRAMES,
    lateral_vis_scale: float = TENSION_LATERAL_VIS_SCALE,
) -> None:
    """Show axial stretch and exaggerated Poisson thinning vs a gray ghost."""
    original = np.asarray(grid.points, dtype=np.float64).copy()
    # Display-only: keep ux, amplify uy/uz so cross-section shrinkage is clear.
    display_u = displacement.copy()
    display_u[:, 1:] *= lateral_vis_scale

    # Discrete bands along the beam (matches the reference rainbow segments).
    x = original[:, 0]
    n_bands = 8
    band = np.floor(
        n_bands * (x - x.min()) / (x.max() - x.min() + 1e-12)
    ).astype(np.float64)
    band = np.clip(band, 0, n_bands - 1)

    undeformed = grid.copy(deep=True)
    deformed = grid.copy(deep=True)
    deformed["band"] = band

    plotter = pv.Plotter(window_size=[900, 600])
    plotter.add_text(
        f"Tension (uy,uz x{lateral_vis_scale:g})",
        font_size=11,
        position="upper_edge",
    )
    _add_ghost_and_deformed(
        plotter,
        undeformed,
        deformed,
        scalar_name="band",
        scalar_clim=[0.0, float(n_bands - 1)],
        scalar_bar_title="x-band",
    )
    # Oblique view similar to the reference beam screenshot.
    plotter.camera_position = [
        (1.55, -0.95, 0.55),
        (0.6, 0.05, 0.05),
        (0.0, 0.0, 1.0),
    ]
    plotter.open_gif(gif_name)
    plotter.write_frame()
    for factor in np.linspace(0.0, 1.0, n_frames):
        deformed.points = original + factor * display_u
        plotter.write_frame()
    plotter.close()


def animate_torsion(
    grid: pv.UnstructuredGrid,
    displacement: np.ndarray,
    *,
    gif_name: str = "linear_elasticity_torsion.gif",
    n_frames: int = N_FRAMES,
) -> None:
    """Show finite end torsion; color by |u| (rings on top, bands on side)."""
    _ = displacement  # FEM u is linearized; the GIF uses finite rotation.
    original = np.asarray(grid.points, dtype=np.float64).copy()
    final_points = _finite_twist_points(original, factor=1.0)
    magnitude_max = float(np.linalg.norm(final_points - original, axis=1).max())

    undeformed = grid.copy(deep=True)
    deformed = grid.copy(deep=True)
    deformed["magnitude"] = np.zeros(deformed.n_points)

    plotter = pv.Plotter(window_size=[900, 600])
    plotter.add_text(
        f"Torsion ({MAX_TWIST_TURNS:g} turns, finite rotation)",
        font_size=11,
        position="upper_edge",
    )
    _add_ghost_and_deformed(
        plotter,
        undeformed,
        deformed,
        scalar_name="magnitude",
        scalar_clim=[0.0, magnitude_max],
        scalar_bar_title="|u|",
    )
    # Elevated oblique view so the top face (and its |u| rings) is readable.
    plotter.camera_position = [
        (2.6, -2.6, 10.5),
        (0.0, 0.0, 4.0),
        (0.0, 0.0, 1.0),
    ]
    plotter.open_gif(gif_name)
    plotter.write_frame()
    for factor in np.linspace(0.0, 1.0, n_frames):
        warped = _finite_twist_points(original, factor=float(factor))
        deformed.points = warped
        deformed["magnitude"] = np.linalg.norm(warped - original, axis=1)
        plotter.write_frame()
    plotter.close()


###############################################################################
# Step 5: Run both problems
# -------------------------
def run_case(
    *,
    name: str,
    mesh_path: pathlib.Path,
    dirichlet_fn: Callable[
        [graphlow.TensorMesh], tuple[pt.PhlowerTensor, np.ndarray]
    ],
    animate_fn: Callable[[pv.UnstructuredGrid, np.ndarray], None],
    plot: bool,
) -> None:
    """Solve one elasticity problem, report RMSE, and optionally animate."""
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    mesh = graphlow.read(mesh_path, "phlower", dtype=torch.float64)
    sparse_dirichlet, exact = dirichlet_fn(mesh)
    displacement = solve_elasticity(mesh, sparse_dirichlet)
    rmse = float(np.sqrt(np.mean((displacement - exact) ** 2)))

    print(f"[{name}] points={mesh.n_points} cells={mesh.n_cells}")
    print(f"[{name}] RMSE vs analytical field: {rmse:.3e}")

    if plot:
        animate_fn(mesh.pvmesh, displacement)


def main(*, plot: bool = True) -> None:
    """Solve tension and torsion, writing a GIF for each case."""
    run_case(
        name="tension",
        mesh_path=TENSION_MESH_PATH,
        dirichlet_fn=uniaxial_dirichlet,
        animate_fn=animate_tension,
        plot=plot,
    )
    run_case(
        name="torsion",
        mesh_path=TORSION_MESH_PATH,
        dirichlet_fn=torsion_dirichlet,
        animate_fn=animate_torsion,
        plot=plot,
    )


if __name__ == "__main__":
    main()

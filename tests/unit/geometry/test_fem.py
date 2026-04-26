"""Tests for FEM operators."""

import pathlib

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from scipy.sparse import linalg

import graphlow
from graphlow.utils import functionals


@pytest.mark.parametrize(
    "file_path",
    [
        pathlib.Path("tests/data/vtu/primitive_cell/tet.vtu"),
        pathlib.Path("tests/data/vtu/tetbeam/mesh.vtu"),
    ],
)
@pytest.mark.parametrize("has_material", [True, False])
@pytest.mark.parametrize(
    "rank, patterns",
    [
        (0, ["c a b f -> c b a f"]),
        (2, ["c a b f -> c b a f"]),
        (4, ["c a b i j f -> c b a i j f", "c a b i j f -> c a b j i f"]),
    ],
)
def test_cell_local_fem_rigidity_tet_symmetry(
    file_path: pathlib.Path,
    rank: int,
    has_material: bool,
    patterns: list[str],
    test_device: torch.device,
):
    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )
    if has_material:
        rand = torch.rand(3, 3, dtype=mesh.backend.dtype)
        rand = mesh.backend.as_tensor(rand @ rand.transpose(0, 1), dimension={})
        e = mesh.backend.ones((mesh.n_cells, 1), dimension={})
        if rank == 0:
            cell_material_coeff = e
        elif rank == 2:
            cell_material_coeff = functionals.einsum(
                "ef,ij->eijf", e, rand, dimension="auto"
            )
        elif rank == 4:
            cell_material_coeff = functionals.einsum(
                "ef,ik,jl->eijklf", e, rand, rand, dimension="auto"
            )
        else:
            raise ValueError(f"Unexpected rank: {rank}")
    else:
        cell_material_coeff = None

    c_rigidity = mesh.geometry.cell_local_fem_rigidity_tet(
        cell_material_coeff=cell_material_coeff, rank=rank
    )

    for pattern in patterns:
        np.testing.assert_almost_equal(
            (c_rigidity - functionals.rearrange(c_rigidity, pattern))
            .to("cpu")
            .numpy(),
            0.0,
            decimal=5,
        )


@pytest.mark.parametrize(
    "file_path, rank, desired",
    [
        (
            pathlib.Path("tests/data/vtu/primitive_cell/tet.vtu"),
            0,
            np.array(
                [
                    [0.5, -1 / 6, -1 / 6, -1 / 6],
                    [-1 / 6, 1 / 6, 0, 0],
                    [-1 / 6, 0, 1 / 6, 0],
                    [-1 / 6, 0, 0, 1 / 6],
                ]
            )[None, ..., None],
        ),
    ],
)
def test_cell_local_fem_rigidity_tet_component(
    file_path: pathlib.Path,
    rank: int,
    desired: np.ndarray,
    test_device: torch.device,
):
    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )
    c_rigidity = mesh.geometry.cell_local_fem_rigidity_tet(rank=rank)
    np.testing.assert_almost_equal(c_rigidity.to("cpu").numpy(), desired)


@pytest.mark.parametrize(
    "file_path",
    [
        pathlib.Path("tests/data/vtu/hexbeam/mesh.vtu"),
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
    ],
)
def test_cell_local_fem_rigidity_tet_raises_when_cell_type_not_supported(
    file_path: pathlib.Path,
):
    mesh = graphlow.read(file_path, "phlower", dtype=torch.float64)
    with pytest.raises(
        ValueError, match="cell_tet_conn not supported for cell types"
    ):
        mesh.geometry.cell_local_fem_rigidity_tet()


@pytest.mark.parametrize(
    "file_path, desired",
    [
        (
            pathlib.Path("tests/data/vtu/primitive_cell/tet.vtu"),
            np.array(
                [
                    [2, 1, 1, 1],
                    [1, 2, 1, 1],
                    [1, 1, 2, 1],
                    [1, 1, 1, 2],
                ]
            )[None, ..., None]
            * 0.5
            / 3
            / 20,
        ),
    ],
)
def test_cell_local_fem_mass_tet_component(
    file_path: pathlib.Path,
    desired: np.ndarray,
    test_device: torch.device,
):
    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )
    c_mass = mesh.geometry.cell_local_fem_mass_tet()
    np.testing.assert_almost_equal(c_mass.to("cpu").numpy(), desired)


@pytest.mark.parametrize(
    "file_path",
    [
        pathlib.Path("tests/data/vtu/tetbeam/mesh.vtu"),
    ],
)
def test_apply_cell_local_fem_mass_tet(
    file_path: pathlib.Path, test_device: torch.device
):
    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )
    u = mesh.backend.ones((mesh.n_points, 1), dimension={})
    c_mass = mesh.geometry.cell_local_fem_mass_tet()
    mass_u = mesh.geometry.apply_cell_local_fem_matrix_tet(c_mass, u)

    total_volume = mesh.backend.to_numpy(
        torch.sum(torch.abs(mesh.geometry.cell_volumes()))
    )
    np.testing.assert_almost_equal(
        mesh.backend.to_numpy(torch.sum(mass_u)), total_volume
    )


@pytest.mark.parametrize(
    "file_path",
    [
        pathlib.Path("tests/data/vtu/tetbeam/mesh.vtu"),
    ],
)
@pytest.mark.parametrize("direction", [0, 1, 2])
@pytest.mark.parametrize(
    "function",
    ["linear", "square", "cos"],
)
def test_apply_cell_local_fem_rigidity_tet(
    file_path: pathlib.Path,
    direction: int,
    function: str,
    test_device: torch.device,
):
    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )
    surface = mesh.extract_surface()
    mask_internal = torch.ones(mesh.n_points, dtype=bool)
    mask_internal[surface.parent_point_ids] = False
    x = mesh.points[:, [direction]]
    l_max = mesh.backend.as_tensor([1.0], dimension={"L": 1})

    if function == "linear":
        u = x / torch.max(mesh.points)
        v = u * 0
    elif function == "square":
        u = 0.1 * (x / l_max) ** 2
        v = 0.1 * 2 * mesh.backend.ones(u.shape, dimension={})
    elif function == "cos":
        u = torch.cos(x / l_max * 2 * torch.pi)
        v = -((2 * torch.pi) ** 2) * u
    else:
        raise ValueError(f"Unexpected function: {function}")
    c_mass = mesh.geometry.cell_local_fem_mass_tet()
    desired = (
        mesh.geometry.apply_cell_local_fem_matrix_tet(c_mass, v)[mask_internal]
        .to("cpu")
        .numpy()
    )

    c_rigidity = mesh.geometry.cell_local_fem_rigidity_tet(rank=0)
    lap_u = -mesh.geometry.apply_cell_local_fem_matrix_tet(c_rigidity, u)
    scale = np.sqrt(np.mean(desired**2))
    assert (
        np.sqrt(
            np.mean((lap_u[mask_internal].to("cpu").numpy() - desired) ** 2)
        )
        < scale * 0.3 + 1e-8
    )


@pytest.mark.parametrize(
    "file_path",
    [
        pathlib.Path("tests/data/vtu/primitive_cell/tet.vtu"),
        pathlib.Path("tests/data/vtu/tetbeam/mesh.vtu"),
    ],
)
@pytest.mark.parametrize("rank", [0])
def test_global_fem_rigidity_tet_symmetry_conservation(
    file_path: pathlib.Path,
    rank: int,
    test_device: torch.device,
):
    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )
    c_rigidity = mesh.geometry.cell_local_fem_rigidity_tet(rank=rank)

    lap = mesh.geometry.global_fem_matrix_tet(c_rigidity)
    diff = (lap - lap.transpose(0, 1)).coalesce().values().to("cpu").numpy()
    np.testing.assert_almost_equal(diff, 0)

    # Check conservation
    sum_ = lap.to_tensor().sum(dim=0).values().to("cpu").numpy()
    np.testing.assert_almost_equal(sum_, 0)


@pytest.mark.parametrize(
    "file_path",
    [
        pathlib.Path("tests/data/vtu/primitive_cell/tet.vtu"),
        pathlib.Path("tests/data/vtu/cube/2x2_tet.vtu"),
        pathlib.Path("tests/data/vtu/tetbeam/mesh.vtu"),
    ],
)
@pytest.mark.parametrize("rank", [0, 2, 4])
def test_global_fem_rigidity_tet_consistent(
    file_path: pathlib.Path, rank: int, test_device: torch.device
):
    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )
    if rank == 4:
        randn = torch.randn((mesh.n_points, 3, 1), dtype=mesh.backend.dtype)
    else:
        randn = torch.randn((mesh.n_points, 1), dtype=mesh.backend.dtype)
    u = mesh.backend.as_tensor(
        randn + torch.rand(1),
        dimension={},
    )
    reshaped_u = u.reshape((-1, 1))
    c_rigidity = mesh.geometry.cell_local_fem_rigidity_tet(rank=rank)
    desired_lap_u = mesh.geometry.apply_cell_local_fem_matrix_tet(c_rigidity, u)

    rigidity = mesh.geometry.global_fem_matrix_tet(c_rigidity)
    actual_lap_u = (rigidity @ reshaped_u).reshape(u.shape)
    np.testing.assert_almost_equal(
        actual_lap_u.to("cpu").numpy(), desired_lap_u.to("cpu").numpy()
    )


@pytest.mark.parametrize(
    "file_path",
    [
        pathlib.Path("tests/data/vtu/tetbeam/mesh.vtu"),
    ],
)
def test_implicit_heat(file_path: pathlib.Path, test_device: torch.device):
    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )

    delta_t = mesh.backend.as_tensor([[0.04]], dimension={"T": 1})
    diffusion = mesh.backend.as_tensor([[0.1]], dimension={"L": 2, "T": -1})

    x = mesh.points[:, [0]]
    u = torch.cos(x / torch.max(x) * 2 * torch.pi)

    c_mass = mesh.geometry.cell_local_fem_mass_tet()
    global_mat = (delta_t * diffusion).to(dtype=mesh.backend.dtype)
    c_rigidity = mesh.geometry.cell_local_fem_rigidity_tet(
        rank=0, cell_material_coeff=global_mat
    )

    f = mesh.geometry.apply_cell_local_fem_matrix_tet(c_mass, u)
    # Check dimension is compatible
    assert (
        mesh.geometry.apply_cell_local_fem_matrix_tet(c_rigidity, u).dimension
        == f.dimension
    )

    coeff = torch.squeeze((delta_t * diffusion).to_tensor()).to("cpu").numpy()

    # Solve (M + dt nu L) U^{n+1} = M U^n
    def matvec(v: np.ndarray) -> np.ndarray:
        return (
            mesh.geometry.apply_cell_local_fem_matrix_tet(
                c_mass,
                mesh.backend.as_tensor(v[:, None], dimension=u.dimension).to(
                    dtype=mesh.backend.dtype
                ),
            )
            .to("cpu")
            .numpy()[..., 0]
            + mesh.geometry.apply_cell_local_fem_matrix_tet(
                c_rigidity,
                mesh.backend.as_tensor(v[:, None], dimension=u.dimension).to(
                    dtype=mesh.backend.dtype
                ),
            )
            .to("cpu")
            .numpy()[..., 0]
        )

    op = linalg.LinearOperator(
        shape=(mesh.n_points, mesh.n_points), matvec=matvec
    )

    res, _ = linalg.cg(op, f.to("cpu").numpy(), rtol=1e-8)

    desired = u.to("cpu").numpy()[:, 0] * np.exp(-coeff * (2 * np.pi) ** 2)
    assert np.sqrt(np.mean((res - desired) ** 2)) < 0.01


@pytest.mark.parametrize(
    "file_path",
    [
        pathlib.Path("tests/data/vtu/cube/2x2_tet.vtu"),
        pathlib.Path("tests/data/vtu/tetbeam/mesh.vtu"),
    ],
)
def test_laplace(file_path: pathlib.Path, test_device: torch.device):
    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )
    backend = mesh.backend

    x = mesh.points[:, 0]
    b = backend.zeros((mesh.n_points, 1), dimension={"L": 1})

    mask_xmin = torch.abs(x - torch.min(x)).to_tensor() < 1e-5
    mask_xmax = torch.abs(x - torch.max(x)).to_tensor() < 1e-5
    dirichlet = torch.ones((mesh.n_points, 1)) * torch.nan
    dirichlet[mask_xmin, 0] = 0
    dirichlet[mask_xmax, 0] = 0.5
    ax0, ax1 = torch.where(~torch.isnan(dirichlet))
    dirichlet_values = dirichlet[ax0, ax1]
    sparse_dirichlet = backend.as_tensor(
        torch.sparse_coo_tensor(
            values=dirichlet_values.to(dtype=backend.dtype),
            indices=torch.stack([ax0, ax1], dim=0),
            size=dirichlet.shape,
        ),
        dimension=mesh.points.dimension,
    )

    c_rigidity = mesh.geometry.cell_local_fem_rigidity_tet(rank=0)
    rigidity = mesh.geometry.global_fem_matrix_tet(c_rigidity)
    rigidity, b = mesh.geometry.apply_dirichlet_to_global_fem_matrix(
        rigidity, b, sparse_dirichlet
    )

    # Check dimension is compatible
    assert c_rigidity.dimension == b.dimension

    sp_rigidity = sp.coo_array(
        (
            rigidity.values().to("cpu").numpy(),
            (
                rigidity.indices().to("cpu").numpy()[0],
                rigidity.indices().to("cpu").numpy()[1],
            ),
        ),
        shape=rigidity.shape,
    )
    res, _ = linalg.cg(sp_rigidity, b.to("cpu").numpy(), rtol=1e-8)

    desired = (x / torch.max(x)).to("cpu").numpy() * 0.5
    assert np.sqrt(np.mean((res - desired) ** 2)) < 1e-6


@pytest.mark.parametrize(
    "file_path",
    [
        pathlib.Path("tests/data/vtu/tetbeam/mesh.vtu"),
    ],
)
def test_poisson(file_path: pathlib.Path, test_device: torch.device):
    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )
    backend = mesh.backend

    x = mesh.points[:, 0]
    f = backend.ones((mesh.n_points, 1), dimension={"L": -2})
    c_mass = mesh.geometry.cell_local_fem_mass_tet()
    b = mesh.geometry.apply_cell_local_fem_matrix_tet(c_mass, f)

    mask_xmin = torch.abs(x - torch.min(x)).to_tensor() < 1e-5
    mask_xmax = torch.abs(x - torch.max(x)).to_tensor() < 1e-5
    dirichlet = torch.ones((mesh.n_points, 1)) * torch.nan
    dirichlet[mask_xmin, 0] = 0
    dirichlet[mask_xmax, 0] = 0.5
    ax0, ax1 = torch.where(~torch.isnan(dirichlet))
    dirichlet_values = dirichlet[ax0, ax1]
    sparse_dirichlet = backend.as_tensor(
        torch.sparse_coo_tensor(
            values=dirichlet_values.to(dtype=backend.dtype),
            indices=torch.stack([ax0, ax1], dim=0),
            size=dirichlet.shape,
        ),
        dimension=mesh.points.dimension,
    )

    c_rigidity = mesh.geometry.cell_local_fem_rigidity_tet(rank=0)
    rigidity = mesh.geometry.global_fem_matrix_tet(c_rigidity)
    rigidity, b = mesh.geometry.apply_dirichlet_to_global_fem_matrix(
        rigidity, b, sparse_dirichlet
    )

    # Check dimension is compatible
    assert c_rigidity.dimension == b.dimension

    sp_rigidity = sp.coo_array(
        (
            rigidity.values().to("cpu").numpy(),
            (
                rigidity.indices().to("cpu").numpy()[0],
                rigidity.indices().to("cpu").numpy()[1],
            ),
        ),
        shape=rigidity.shape,
    )
    res, _ = linalg.cg(sp_rigidity, b.to("cpu").numpy(), rtol=1e-8)

    x_ = (x / torch.max(x)).to("cpu").numpy()
    desired = 0.5 * x_ * (1 - x_) + x_ * 0.5
    assert np.sqrt(np.mean((res - desired) ** 2)) < 1e-4


@pytest.mark.parametrize(
    "file_path",
    [
        pathlib.Path("tests/data/vtu/tetbeam/mesh.vtu"),
    ],
)
def test_structural_analysis(
    file_path: pathlib.Path, test_device: torch.device
):
    modulus = 1.0e6
    nu = 0.3
    max_disp = 0.1

    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )
    backend = mesh.backend

    lam = (modulus * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = modulus / (2 * (1 + nu))
    stiffness_dimension = {"L": -1, "M": 1, "T": -2}
    delta = backend.as_tensor(torch.eye(3, dtype=backend.dtype))

    stiffness = (
        lam
        * functionals.einsum(
            "ij,kl->ijkl", delta, delta, dimension=stiffness_dimension
        )
        + mu
        * (
            functionals.einsum(
                "ik,jl->ijkl", delta, delta, dimension=stiffness_dimension
            )
            + functionals.einsum(
                "il,jk->ijkl", delta, delta, dimension=stiffness_dimension
            )
        )
    )[None, ..., None].to(backend.device)
    np.testing.assert_almost_equal(
        stiffness.to("cpu").numpy(),
        stiffness.rearrange("g i j k l f -> g j i k l f").to("cpu").numpy(),
    )
    np.testing.assert_almost_equal(
        stiffness.to("cpu").numpy(),
        stiffness.rearrange("g i j k l f -> g i j l k f").to("cpu").numpy(),
    )

    x = mesh.points[:, 0]
    y = mesh.points[:, 1]
    z = mesh.points[:, 2]
    u_dimension = mesh.points.dimension

    mask_xmin = torch.abs(x - torch.min(x)).to_tensor() < 1e-5
    mask_xmax = torch.abs(x - torch.max(x)).to_tensor() < 1e-5
    mask_ymin = torch.abs(y - torch.min(y)).to_tensor() < 1e-5
    mask_zmin = torch.abs(z - torch.min(z)).to_tensor() < 1e-5
    dirichlet = torch.ones((mesh.n_points, 3, 1)) * torch.nan
    dirichlet[mask_xmin, 0] = 0
    dirichlet[mask_xmin & mask_ymin & mask_zmin, :] = 0
    dirichlet[mask_xmax, 0] = max_disp
    ax0, ax1, ax2 = torch.where(~torch.isnan(dirichlet))
    dirichlet_values = dirichlet[ax0, ax1, ax2]
    sparse_dirichlet = backend.as_tensor(
        torch.sparse_coo_tensor(
            values=dirichlet_values.to(dtype=backend.dtype),
            indices=torch.stack([ax0, ax1, ax2], dim=0),
            size=dirichlet.shape,
        ),
        dimension=mesh.points.dimension,
    )

    u = backend.zeros((mesh.n_points * 3, 1), dimension=u_dimension).to(
        dtype=backend.dtype
    )

    c_rigidity = mesh.geometry.cell_local_fem_rigidity_tet(
        rank=4, cell_material_coeff=stiffness
    )
    rigidity = mesh.geometry.global_fem_matrix_tet(c_rigidity)
    f = backend.zeros(
        (mesh.n_points * 3, 1), dimension={"L": 1, "M": 1, "T": -2}
    ).to(dtype=backend.dtype)  # [force/volume] * [volume]
    rigidity, f = mesh.geometry.apply_dirichlet_to_global_fem_matrix(
        rigidity, f, sparse_dirichlet=sparse_dirichlet
    )

    # Check dimension is compatible
    assert (rigidity @ u).dimension == f.dimension
    sp_rigidity = sp.coo_array(
        (
            rigidity.values().to("cpu").numpy(),
            (
                rigidity.indices().to("cpu").numpy()[0],
                rigidity.indices().to("cpu").numpy()[1],
            ),
        ),
        shape=rigidity.shape,
    )

    # Solve K u = f
    res, _ = linalg.cg(sp_rigidity, f.to("cpu").numpy(), rtol=1e-8)
    u = res.reshape(-1, 3)

    desired = np.stack(
        [
            x.to("cpu").numpy() * max_disp,
            -y.to("cpu").numpy() * max_disp * nu,
            -z.to("cpu").numpy() * max_disp * nu,
        ],
        axis=-1,
    )
    assert np.sqrt(np.mean((u - desired) ** 2)) < 1e-8

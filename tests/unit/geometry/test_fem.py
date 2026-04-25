"""Tests for geometry operators such as isoAM and its helper routines."""

import pathlib

import numpy as np
import phlower_tensor
import pytest
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
        (1, ["c a b i j f -> c b a i j f", "c a b i j f -> c a b j i f"]),
    ],
)
def test_cell_local_rigidity_tet_symmetry(
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
        rand = mesh.backend.as_tensor(rand @ rand.transpose(0, 1))
        e = mesh.backend.ones((mesh.n_cells, 1)).to(dtype=mesh.backend.dtype)
        if rank == 0:
            cell_material_coeff = functionals.einsum(
                "ef,ij->eijf", e, rand, dimension={}
            )
        else:
            cell_material_coeff = functionals.einsum(
                "ef,ik,jl->eijklf", e, rand, rand, dimension={}
            )
    else:
        cell_material_coeff = None

    c_rigidity = mesh.geometry.cell_local_rigidity_tet(
        cell_material_coeff=cell_material_coeff, rank=rank
    )

    for pattern in patterns:
        np.testing.assert_almost_equal(
            (c_rigidity - functionals.rearrange(c_rigidity, pattern)).numpy(),
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
def test_cell_local_rigidity_tet_component(
    file_path: pathlib.Path,
    rank: int,
    desired: np.ndarray,
    test_device: torch.device,
):
    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )
    c_rigidity = mesh.geometry.cell_local_rigidity_tet(rank=rank)
    np.testing.assert_almost_equal(c_rigidity.numpy(), desired)


@pytest.mark.parametrize(
    "file_path",
    [
        pathlib.Path("tests/data/vtu/hexbeam/mesh.vtu"),
        pathlib.Path("tests/data/vtu/complex/mesh.vtu"),
    ],
)
def test_cell_local_rigidity_tet_raises_when_cell_type_not_supported(
    file_path: pathlib.Path,
):
    mesh = graphlow.read(file_path, "phlower", dtype=torch.float64)
    with pytest.raises(
        ValueError, match="fem_tet not supported for cell types"
    ):
        mesh.geometry.cell_local_rigidity_tet()


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
def test_cell_local_mass_tet_component(
    file_path: pathlib.Path,
    desired: np.ndarray,
    test_device: torch.device,
):
    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )
    c_mass = mesh.geometry.cell_local_mass_tet()
    np.testing.assert_almost_equal(c_mass.numpy(), desired)


@pytest.mark.parametrize(
    "file_path",
    [
        pathlib.Path("tests/data/vtu/tetbeam/mesh.vtu"),
    ],
)
def test_apply_cell_local_rigidity_tet(
    file_path: pathlib.Path, test_device: torch.device
):
    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )
    x = mesh.points[:, [0]]
    u = x / torch.max(x)
    c_rigidity = mesh.geometry.cell_local_rigidity_tet(rank=0)
    lap_u = mesh.geometry.apply_cell_local_matrix_tet(c_rigidity, u)
    assert torch.sum(lap_u).numpy() < 1e-8


@pytest.mark.parametrize(
    "file_path",
    [
        pathlib.Path("tests/data/vtu/tetbeam/mesh.vtu"),
    ],
)
def test_apply_cell_local_mass_tet(
    file_path: pathlib.Path, test_device: torch.device
):
    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )
    u = mesh.backend.ones((mesh.n_points, 1), dimension={})
    c_mass = mesh.geometry.cell_local_mass_tet()
    mass_u = mesh.geometry.apply_cell_local_matrix_tet(c_mass, u)
    raise ValueError(torch.sum(mass_u).numpy())


@pytest.mark.parametrize(
    "file_path",
    [
        pathlib.Path("tests/data/vtu/tetbeam/mesh.vtu"),
    ],
)
def test_implicit_heat(file_path: pathlib.Path, test_device: torch.device):
    delta_t = phlower_tensor.phlower_tensor([[0.1]], dimension={"T": 1})
    diffusion = phlower_tensor.phlower_tensor(
        [[0.1]], dimension={"L": 2, "T": -1}
    )

    mesh = graphlow.read(
        file_path, "phlower", dtype=torch.float64, device=test_device
    )
    x = mesh.points[:, [0]]
    u = torch.cos(x / torch.max(x) * 2 * torch.pi)

    c_mass = mesh.geometry.cell_local_mass_tet()
    global_mat = functionals.einsum(
        "gf,gf,ij->gijf",
        delta_t,
        diffusion,
        phlower_tensor.phlower_tensor(torch.eye(3), dimension={}),
        dimension=delta_t.dimension * diffusion.dimension,
    ).to(dtype=mesh.backend.dtype)
    c_rigidity = mesh.geometry.cell_local_rigidity_tet(
        rank=0, cell_material_coeff=global_mat
    )

    f = mesh.geometry.apply_cell_local_matrix_tet(c_mass, u)

    coeff = torch.squeeze((delta_t * diffusion).to_tensor()).numpy()

    # Solve (M + dt nu L) U^{n+1} = M U^n
    def matvec(v: np.ndarray) -> np.ndarray:
        return (
            mesh.geometry.apply_cell_local_matrix_tet(
                c_mass,
                mesh.backend.as_tensor(v[:, None]).to(dtype=mesh.backend.dtype),
            ).numpy()[..., 0]
            + mesh.geometry.apply_cell_local_matrix_tet(
                c_rigidity,
                mesh.backend.as_tensor(v[:, None]).to(dtype=mesh.backend.dtype),
            ).numpy()[..., 0]
        )

    op = linalg.LinearOperator(
        shape=(mesh.n_points, mesh.n_points), matvec=matvec
    )

    res, _ = linalg.cg(op, f, rtol=1e-8)

    desired = u.numpy()[:, 0] * np.exp(-coeff * (2 * np.pi) ** 2)
    assert np.sqrt(np.mean((res - desired) ** 2)) < 0.02

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from phlower_tensor._tensor import PhlowerDimensionTensor

from graphlow.core.backend.base import Backend, TensorLike
from graphlow.utils import functionals
from graphlow.utils.dimension import get_dimension

if TYPE_CHECKING:
    from graphlow.core.mesh import TensorMesh

logger = logging.getLogger(__name__)


def apply_cell_local_matrix_tet[T: TensorLike](
    mesh: TensorMesh[T], cell_local_matrix_tet: T, u: T
) -> T:
    c_u = u[mesh.topology.cell_tet_conn()]
    c_f = functionals.einsum("cabf,eb...->ea...", cell_local_matrix_tet, c_u)

    p_res = mesh.backend.zeros_like(u)
    p_res.index_put_(
        (torch.from_numpy(mesh.topology.cell_tet_conn()).to(torch.int64),),
        c_f,
        accumulate=True,
    )
    return p_res


def cell_local_rigidity_tet[T: TensorLike](
    mesh: TensorMesh[T], cell_material_coeff: T | None = None, rank: int = 0
) -> T:
    backend = mesh.backend

    if cell_material_coeff is None:
        cell_material_coeff = _generate_global_identity_tensor(
            backend=backend, rank=rank, dtype=backend.dtype
        )
    connectivity = mesh.topology.cell_tet_conn()
    c_x = mesh.points[connectivity]

    # x^i = C^i_a L_a, where L_a is the volume coordinate as in
    # eq 9.11 of Liu and Quek 2013
    c_total = backend.ones((mesh.n_cells, 1, 4), dimension=get_dimension(c_x))
    c_coeff = torch.cat(
        [c_total, functionals.rearrange(c_x, "e a j -> e j a")], dim=1
    )

    # NOTE: We use pinv instead of inv
    #       because degenerated cells are ignored anyway due to volume == 0
    c_inv_coeff = torch.linalg.pinv(c_coeff)
    c_grad_shape = c_inv_coeff[:, :, 1:]  # [c, a, d]

    c_volume = torch.abs(mesh.geometry.cell_volumes())

    if cell_material_coeff.shape[0] == 1:
        str_n_mat = "g"
    else:
        str_n_mat = "e"

    if get_dimension(c_volume) is None:
        rigidity_dimension = None
    else:
        rigidity_dimension = (
            get_dimension(c_grad_shape)
            * get_dimension(cell_material_coeff)
            * get_dimension(c_grad_shape)
            * get_dimension(c_volume)
        )
    if rank == 0:
        # Element rigidity matrix: [c, a, a, f]
        c_rigidity = functionals.einsum(
            f"eai,{str_n_mat}ijf,ebj,ef->eabf",
            c_grad_shape,
            cell_material_coeff,
            c_grad_shape,
            c_volume,
            dimension=rigidity_dimension,
        )
    elif rank == 1:
        # Element rigidity matrix: [c, a, a, d, d, f]
        c_rigidity = functionals.einsum(
            f"eai,{str_n_mat}ijklf,ebk,ef->eabjlf",
            c_grad_shape,
            cell_material_coeff,
            c_grad_shape,
            c_volume,
            dimension=rigidity_dimension,
        )
    else:
        raise NotImplementedError(f"Unsupported rank: {rank}")

    return c_rigidity


def cell_local_mass_tet[T: TensorLike](
    mesh: TensorMesh[T],
    cell_density: T | None = None,
    density_demension: dict[str, float] | PhlowerDimensionTensor | None = None,
) -> T:
    backend = mesh.backend

    if cell_density is None:
        cell_density = backend.ones((1, 1), dimension=density_demension)
    if cell_density.shape[0] == 1:
        str_n_density = "g"
    else:
        str_n_density = "c"
    c_volume = torch.abs(mesh.geometry.cell_volumes())

    # Rank 0 version of eq 9.23 of Liu and Quek 2013
    mass_coeff = backend.as_tensor(torch.ones((4, 4)) + torch.eye(4)) / 20

    if get_dimension(c_volume) is None:
        mass_dimension = None
    else:
        if density_demension is None:
            mass_dimension = get_dimension(c_volume)
        else:
            mass_dimension = get_dimension(c_volume) * get_dimension(
                cell_density
            )
    c_mass = functionals.einsum(
        f"{str_n_density}f,cf,ab->cabf",
        cell_density,
        c_volume,
        mass_coeff,
        dimension=mass_dimension,
    )
    return c_mass


def global_matrix_tet[T: TensorLike](
    mesh: TensorMesh[T], cell_local_matrix_tet: T | None = None, rank: int = 0
) -> T:
    backend = mesh.backend

    connectivity = mesh.topology.cell_tet_conn()
    list_n_c = [
        torch.sparse_coo_tensor(
            indices=torch.stack(
                [
                    torch.from_numpy(connectivity[:, a]),
                    torch.arange(mesh.n_cells),
                ],
                dim=0,
            ),
            values=torch.ones(mesh.n_cells, dtype=backend.dtype),
            size=(mesh.n_points, mesh.n_cells),
        )
        for a in range(connectivity.shape[-1])
    ]
    list_k_c_a = [
        sum_mul(cell_local_matrix_tet[:, a, :, 0], list_n_c).transpose(0, 1)
        for a in range(4)
    ]
    lap = (
        list_n_c[0] @ list_k_c_a[0]
        + list_n_c[1] @ list_k_c_a[1]
        + list_n_c[2] @ list_k_c_a[2]
        + list_n_c[3] @ list_k_c_a[3]
    ).coalesce()

    # Check symmetry
    diff = (lap - lap.transpose(0, 1)).coalesce().values()
    torch.testing.assert_close(
        diff, torch.zeros_like(diff), atol=1e-3, rtol=1e-3
    )
    sum_ = lap.sum(dim=0).values()
    torch.testing.assert_close(
        sum_, torch.zeros_like(sum_), atol=1e-2, rtol=1e-2
    )

    # Scale Laplacian to be non-dimensional
    lap: torch.Tensor = -lap
    return lap


def _generate_global_identity_tensor[T: TensorLike](
    backend: Backend, rank: int, dtype: torch.dtype
) -> T:
    delta = backend.as_tensor(torch.eye(3, dtype=dtype), dimension={})
    if rank == 0:
        cell_material_coeff = delta[None, ..., None]
    elif rank == 1:
        cell_material_coeff = functionals.einsum(
            "ik,jl->ijkl", delta, delta, dimension={}
        )[None, ..., None]
    else:
        raise NotImplementedError(f"Unsupported rank: {rank}")
    return cell_material_coeff


def sum_mul(
    dense: torch.Tensor, list_sparse: list[torch.Tensor]
) -> torch.Tensor:
    assert dense.shape[1] == len(list_sparse)
    return (
        dense[:, 0] * list_sparse[0]
        + dense[:, 1] * list_sparse[1]
        + dense[:, 2] * list_sparse[2]
        + dense[:, 3] * list_sparse[3]
    )

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from graphlow.core.backend.base import Backend, TensorLike
from graphlow.utils.dimension import get_dimension

if TYPE_CHECKING:
    from graphlow.core.mesh import TensorMesh

logger = logging.getLogger(__name__)


def apply_cell_local_fem_matrix_tet[T: TensorLike](
    mesh: TensorMesh[T],
    cell_local_matrix_tet: T,
    u: T,
    vector_rank: int | None = None,
    matrix_rank: int | None = None,
) -> T:
    """
    Apply the given local FEM matrix on the tet mesh.

    Parameters
    ----------
    mesh: TensorMesh[T]
        Mesh with differentiable points.
    cell_local_matrix_tet: T
        Cell-wise local FEM matrix.
    u: T
        Point-wise physical variable to be multiplied by the matrix.
    vector_rank: int | None
        Rank of the vector. If not fed, the rank is inferred when possible.
    matrix_rank: int | None
        Rank of the matrix. If not fed, the rank is inferred when possible.

    Returns
    -------
    matvec: T
        matvec results.
    """
    _validate_mesh(mesh)

    vector_rank = _get_rank(u, rank=vector_rank)
    matrix_rank = _get_rank(cell_local_matrix_tet, rank=matrix_rank, offset=2)

    cell_tet_conn = mesh.backend.as_index_tensor(mesh.topology.cell_tet_conn())

    c_u = u[cell_tet_conn]
    if matrix_rank == 0:
        if vector_rank == 0:
            additional_string = ""
        else:
            additional_string = "..."
        c_f = mesh.backend.einsum(
            f"cabf,cb{additional_string}f->ca{additional_string}f",
            cell_local_matrix_tet,
            c_u,
            dimension="auto",
        )
    elif matrix_rank == 2 and vector_rank == 1:
        c_f = mesh.backend.einsum(
            "cabijf,cbjf->caif", cell_local_matrix_tet, c_u, dimension="auto"
        )
    else:
        raise NotImplementedError(
            f"Unexpected combination of {vector_rank = } and {matrix_rank = }"
        )

    p_res = mesh.backend.zeros(u.shape, dimension=c_f.dimension)
    p_res.index_put_((cell_tet_conn,), c_f, accumulate=True)
    return p_res


def cell_local_fem_rigidity_tet[T: TensorLike](
    mesh: TensorMesh[T], cell_material_coeff: T | None = None, rank: int = 0
) -> T:
    """
    Generate the cell-wise local FEM rigidity matrix on the tet mesh.

    Parameters
    ----------
    mesh: TensorMesh[T]
        Mesh with differentiable points.
    cell_material_coeff: T
        Cell-wise (or global) material coefficient.
    rank: int
        Rank of the material coefficient. Typically, 0 for isotropic heat,
        2 for anisotropic heat, and 4 for structural analysis.

    Returns
    -------
    cell_local_rigidity: T
        Cell-wise local rigidity matrix.
    """
    _validate_mesh(mesh)
    backend = mesh.backend

    if cell_material_coeff is None:
        cell_material_coeff = _generate_global_identity_tensor(
            backend=backend, rank=rank, dtype=backend.dtype
        )
    connectivity = backend.as_index_tensor(mesh.topology.cell_tet_conn())
    c_x = mesh.points[connectivity]

    # x^i = C^i_a L_a, where L_a is the volume coordinate as in
    # eq 9.11 of Liu and Quek 2013
    c_total = backend.ones((mesh.n_cells, 1, 4), dimension=get_dimension(c_x))
    c_coeff = torch.cat(
        [c_total, mesh.backend.rearrange(c_x, "e a j -> e j a")], dim=1
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

    if rank == 0:
        c_rigidity = mesh.backend.einsum(
            f"eai,{str_n_mat}f,ebi,ef->eabf",
            c_grad_shape,
            cell_material_coeff,
            c_grad_shape,
            c_volume,
            dimension="auto",
        )
    elif rank == 2:
        c_rigidity = mesh.backend.einsum(
            f"eai,{str_n_mat}ijf,ebj,ef->eabf",
            c_grad_shape,
            cell_material_coeff,
            c_grad_shape,
            c_volume,
            dimension="auto",
        )
    elif rank == 4:
        c_rigidity = mesh.backend.einsum(
            f"eai,{str_n_mat}ijklf,ebk,ef->eabjlf",
            c_grad_shape,
            cell_material_coeff,
            c_grad_shape,
            c_volume,
            dimension="auto",
        )
    else:
        raise NotImplementedError(f"Unsupported rank: {rank}")

    return c_rigidity


def cell_local_fem_mass_tet[T: TensorLike](
    mesh: TensorMesh[T],
    cell_density: T | None = None,
) -> T:
    """
    Generate the cell-wise local FEM mass matrix on the tet mesh.

    Parameters
    ----------
    mesh: TensorMesh[T]
        Mesh with differentiable points.
    cell_density: T
        Cell-wise (or global) density.

    Returns
    -------
    cell_local_mass: T
        Cell-wise local mass matrix.
    """
    _validate_mesh(mesh)
    backend = mesh.backend

    if cell_density is None:
        cell_density = backend.ones((1, 1), dimension={})
    if cell_density.shape[0] == 1:
        str_n_density = "g"
    else:
        str_n_density = "c"
    c_volume = torch.abs(mesh.geometry.cell_volumes())

    # Rank 0 version of eq 9.23 of Liu and Quek 2013
    mass_coeff = (
        backend.as_tensor(torch.ones((4, 4)) + torch.eye(4), dimension={}) / 20
    )

    c_mass = mesh.backend.einsum(
        f"{str_n_density}f,cf,ab->cabf",
        cell_density,
        c_volume,
        mass_coeff,
        dimension="auto",
    )
    return c_mass


def global_fem_matrix_tet[T: TensorLike](
    mesh: TensorMesh[T],
    cell_local_matrix_tet: T,
) -> T:
    """
    Compute the global FEM matrix on the tet mesh
    from the given cell-wise local FEM matrix.

    Parameters
    ----------
    mesh: TensorMesh[T]
        Mesh with differentiable points.
    cell_local_matrix_tet: T
        Cell-wise local FEM matrix.

    Returns
    -------
    global_matrix: T
        Assembled global FEM matrix with COO layout.
    """
    _validate_mesh(mesh)
    backend = mesh.backend

    rank = len(cell_local_matrix_tet.shape) - 4  # exclude (c, a, b, f)
    connectivity = backend.as_index_tensor(mesh.topology.cell_tet_conn())
    list_n_c = [
        torch.sparse_coo_tensor(
            indices=torch.stack(
                [
                    connectivity[:, a],
                    torch.arange(mesh.n_cells, device=backend.device),
                ],
                dim=0,
            ),
            values=torch.ones(
                mesh.n_cells, dtype=backend.dtype, device=backend.device
            ),
            size=(mesh.n_points, mesh.n_cells),
        )
        for a in range(connectivity.shape[-1])
    ]

    if rank == 0:
        global_rigidity = _generate_global_fem_matrix_rank0(
            backend, list_n_c, cell_local_matrix_tet
        )
    elif rank == 2:
        list_global_rigidities = [
            [
                _generate_global_fem_matrix_rank0(
                    backend,
                    list_n_c,
                    cell_local_matrix_tet[:, ..., i_row, i_col, :],
                )
                for i_col in range(3)
            ]
            for i_row in range(3)
        ]
        n_dof = mesh.n_points * 3
        global_rigidity = torch.empty(
            (n_dof, n_dof),
            layout=torch.sparse_coo,
            dtype=backend.dtype,
            device=backend.device,
        )
        for i_row in range(3):
            for i_col in range(3):
                partial_rigidity = list_global_rigidities[i_row][i_col]
                indices = torch.stack(
                    [
                        partial_rigidity.indices()[0] * 3 + i_row,
                        partial_rigidity.indices()[1] * 3 + i_col,
                    ],
                    dim=0,
                )
                rearranged_rigidity = torch.sparse_coo_tensor(
                    values=partial_rigidity.values(),
                    indices=indices,
                    size=(n_dof, n_dof),
                )
                global_rigidity += rearranged_rigidity
    else:
        raise NotImplementedError(f"Unexpected rank: {rank}")

    return backend.as_tensor(
        global_rigidity.coalesce(),
        dimension=get_dimension(cell_local_matrix_tet),
    )


def apply_dirichlet_to_global_fem_matrix[T: TensorLike](
    mesh: TensorMesh[T], global_matrix: T, global_b: T, sparse_dirichlet: T
) -> tuple[T, T]:
    """
    Apply the given Dirichlet boundary condition to the linear problem.

    Parameters
    ----------
    mesh: TensorMesh[T]
        Mesh with differentiable points.
    global_matrix: T
        Sparse global matrix.
    global_b: T
        Dense global tensor.
    sparse_dirichlet: T
        Sparse tensor describing the Dirichlet boundary condition.

    Returns
    -------
    global_matrix_with_dirichlet: T
        Sparse global matrix after taking into account the Dirichlet bc.
    global_b_with_dirichlet: T
        Global b after taking into account the Dirichlet bc.
    """
    _validate_mesh(mesh)
    if sparse_dirichlet.shape[-1] != 1:
        raise NotImplementedError(
            f"The last dim should be 1 but given: {sparse_dirichlet.shape}"
        )
    backend = mesh.backend

    sparse_dirichlet = sparse_dirichlet.coalesce()
    if len(sparse_dirichlet.shape) == 2:
        # Vector rank 0
        dirichlet_indices = sparse_dirichlet.indices()[0]
    elif len(sparse_dirichlet.shape) == 3:
        # Vector rank 1
        dirichlet_indices = (
            3 * sparse_dirichlet.indices()[0] + sparse_dirichlet.indices()[1]
        )
    else:
        raise NotImplementedError(f"Unexpected rank: {sparse_dirichlet.shape}")
    dirichlet_values = sparse_dirichlet.values()

    # Move effect of the Dirichlet to RHS (Dirichlet force)
    dirichlet_source = torch.zeros_like(global_b.to_tensor())
    dirichlet_source[dirichlet_indices] = dirichlet_values[:, None]
    dirichlet_force = -global_matrix.to_tensor() @ dirichlet_source
    dirichlet_force[dirichlet_indices] = 0

    # Symmetrically remove off-diagonals for Dirichlet
    values = global_matrix.values().clone()
    indices = global_matrix.indices()
    row = indices[0]
    col = indices[1]
    mask_row = torch.isin(row, dirichlet_indices)
    mask_col = torch.isin(col, dirichlet_indices)
    mask_diag = row == col
    values[mask_row] = 0.0
    values[mask_col] = 0.0
    values[mask_diag & mask_row] = 1.0

    new_global_b = global_b + backend.as_tensor(
        dirichlet_force, dimension=global_b.dimension
    )
    new_global_b[dirichlet_indices] = dirichlet_values[:, None]
    return backend.as_tensor(
        torch.sparse_coo_tensor(
            indices=indices, values=values, size=global_matrix.size()
        ).coalesce(),
        dimension=global_matrix.dimension,
    ), new_global_b


def _validate_mesh[T: TensorLike](mesh: TensorMesh[T]):
    if not mesh.topology.is_unique_cell():
        raise NotImplementedError(
            "Mixed cell type is not supported. Given: "
            f"{mesh.topology.unique_cell_types()}"
        )


def _generate_global_fem_matrix_rank0[T: TensorLike](
    backend: Backend, list_n_c: list[torch.Tensor], cell_local_matrix: T
) -> torch.Tensor:
    if cell_local_matrix.shape[-1] != 1:
        raise NotImplementedError(
            f"The last dim should be 1 but given: {cell_local_matrix.shape}"
        )

    n_points_per_cell = cell_local_matrix.shape[-2]
    list_k_c_a = [
        torch.sum(
            torch.stack(
                [
                    backend.to_torch(cell_local_matrix[:, a, b, 0])
                    * list_n_c[b]
                    for b in range(n_points_per_cell)
                ],
                dim=0,
            ),
            dim=0,
        ).transpose(0, 1)
        for a in range(n_points_per_cell)
    ]
    global_rigidity = torch.sum(
        torch.stack(
            [list_n_c[a] @ list_k_c_a[a] for a in range(n_points_per_cell)],
            dim=0,
        ),
        dim=0,
    )
    return global_rigidity


def _generate_global_identity_tensor[T: TensorLike](
    backend: Backend, rank: int, dtype: torch.dtype
) -> T:
    delta = backend.as_tensor(torch.eye(3, dtype=dtype), dimension={})
    if rank == 0:
        cell_material_coeff = backend.as_tensor(
            torch.ones((1, 1)), dimension={}
        )
    elif rank == 2:
        cell_material_coeff = delta[None, ..., None]
    elif rank == 4:
        cell_material_coeff = backend.einsum(
            "ik,jl->ijkl", delta, delta, dimension="auto"
        )[None, ..., None]
    else:
        raise NotImplementedError(f"Unsupported rank: {rank}")
    return cell_material_coeff


def _get_rank[T: TensorLike](
    t: T, rank: int | None = None, offset: int = 0
) -> int:
    if rank is not None:
        return rank
    if hasattr(t, "rank"):
        rank = t.rank() - offset
        if rank < 0:
            raise ValueError(f"Rank is negative for {t} with offset {offset}")
        return rank
    raise ValueError(f"Feed rank when using {t.__class__}")

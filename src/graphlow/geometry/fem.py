"""Build, apply, and assemble finite element (FEM) matrices.

``cell_local_fem_rigidity`` and ``cell_local_fem_mass`` compute one local
matrix per cell and return a dictionary keyed by cell type. Each value
follows the connectivity order of ``mesh.topology.cell_block(cell_type)``.
Pass this dictionary to ``apply_cell_local_fem_matrix`` to multiply a nodal
field, or to ``global_fem_matrix`` to assemble a global sparse matrix.

Tensor layouts use ``c`` for cells, ``a`` and ``b`` for local node indices,
``i``, ``j``, ``k``, and ``l`` for spatial components, and ``f`` for features.
Features are independent channels stored on the last axis, even when there
is only one. Tensor rank counts component axes, excluding cell, node, and
feature axes; for example, a scalar field has rank 0 and a vector has rank 1.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import pyvista as pv
import torch

from graphlow.core.backend.base import Backend, TensorLike
from graphlow.core.blocks import FixedCellBlock
from graphlow.geometry.elements.jacobian import (
    compute_jacobian,
    jacobian_measure,
    physical_shape_function_gradients,
)
from graphlow.geometry.elements.shape_functions import (
    get_shape_function_gradients,
    get_shape_functions,
)
from graphlow.geometry.quadrature.rules import get_quadrature_rule
from graphlow.utils.dimension import get_dimension

if TYPE_CHECKING:
    from phlower_tensor import PhlowerDimensionTensor

    from graphlow.core.mesh import TensorMesh

logger = logging.getLogger(__name__)


def apply_cell_local_fem_matrix[T: TensorLike](
    mesh: TensorMesh[T],
    cell_local_matrix: T | Mapping[pv.CellType, T],
    u: T,
    vector_rank: int | None = None,
    matrix_rank: int | None = None,
) -> T:
    """Multiply a nodal field by local FEM matrices and sum at mesh points.

    Gather ``u`` at each cell's nodes, multiply by that cell's matrix, and
    add the results at shared mesh points. This computes the assembled
    operator's action without constructing a global sparse matrix.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh whose cell connectivity maps local nodes to mesh points.
    cell_local_matrix : T or Mapping[pyvista.CellType, T]
        Local matrices with shape ``(c, a, b, f)`` for scalar coefficients
        or ``(c, a, b, i, j, f)`` for matrices that couple vector components.
        Here ``c`` is the number of cells in the block, ``a`` and ``b`` are
        its local node axes, and ``f`` is the feature axis.

        A single tensor is accepted only when the mesh has one cell type.
        For a mixed mesh, pass a nonempty mapping from cell type to tensor;
        only the supplied blocks contribute. Each tensor must follow
        ``mesh.topology.cell_block(cell_type).conn`` row and node order.
    u : T
        Nodal values with shape ``(mesh.n_points, ..., f)``, where ``...``
        contains the component axes. Scalars use ``(mesh.n_points, f)``
        and vectors use ``(mesh.n_points, n_components, f)``. Features are
        multiplied independently, with singleton feature broadcasting.
    vector_rank : int or None, optional
        Number of component axes in ``u``: 0 for scalars, 1 for vectors.
        If omitted, use ``u.rank()`` for Phlower tensors and ``u.ndim - 2``
        otherwise.
    matrix_rank : int or None, optional
        Number of component axes in each local matrix. Rank 0 applies the
        same scalar operator to every component of ``u``. Rank 2 contracts
        the matrix's ``j`` axis with a rank-1 vector's component axis.
        If omitted, use ``local.rank() - 2`` for Phlower tensors and
        ``local.ndim - 4`` otherwise.

    Returns
    -------
    T
        Sum of all supplied cell contributions, with the same shape as
        ``u``. Matrix component and feature sizes must be compatible with
        that output shape. Points outside the supplied blocks receive zero.

    Raises
    ------
    ValueError
        If a single tensor is supplied for a mesh without exactly one cell
        type, local node axes do not match the block, or an inferred rank
        is negative.
    NotImplementedError
        If a block has variable node counts, or the rank combination is
        neither ``matrix_rank == 0`` nor
        ``(matrix_rank, vector_rank) == (2, 1)``.
    """
    vector_rank = _get_rank(u, rank=vector_rank)
    # The product's physical dimension is known only after the first matvec.
    result: T | None = None
    matrices = _local_matrices_by_cell_type(mesh, cell_local_matrix)
    for cell_type, local in matrices.items():
        block = _require_fixed_cell_block(mesh, cell_type)
        _validate_node_count_matches_block(block, local)
        connectivity = mesh.backend.as_index_tensor(block.conn)
        local_product = _multiply_on_cell_nodes(
            mesh,
            local,
            u[connectivity],
            vector_rank,
            _get_rank(local, rank=matrix_rank, offset=2),
        )
        if result is None:
            result = mesh.backend.zeros(
                u.shape, dimension=get_dimension(local_product)
            )
        result.index_put_((connectivity,), local_product, accumulate=True)
    return result


def cell_local_fem_rigidity[T: TensorLike](
    mesh: TensorMesh[T],
    cell_material_coeff: T | Mapping[pv.CellType, T] | None = None,
    rank: int = 0,
    *,
    quadrature_order: int | None = None,
) -> dict[pv.CellType, T]:
    r"""Integrate a local stiffness (rigidity) matrix for each mesh cell.

    The material coefficient is constant within each cell. Its tensor rank
    selects a scalar diffusion operator or an operator that couples vector
    components; see the integrands in Notes.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh with supported finite elements and a fixed node count within
        each cell type. Different cell types may have different node counts.
    cell_material_coeff : T, Mapping[pyvista.CellType, T], or None, optional
        Material coefficients with shape ``(c, f)`` for rank 0,
        ``(c, i, j, f)`` for rank 2, or ``(c, i, j, k, l, f)`` for rank 4.
        Component axes have the size of the physical coordinate dimension;
        ``f`` is the number of independent material features.

        The first axis and cell ordering depend on how coefficients are
        supplied:

        * A tensor with ``c == 1`` supplies one material for the whole mesh.
        * A tensor with ``c == mesh.n_cells`` supplies one material per cell
          in mesh order. Rows for each block are selected using
          ``block.global_indices``.
        * A mapping must contain exactly the mesh's cell types. Each value
          uses ``c == 1`` for a uniform material within that block, or
          ``c == block.n_cells`` with rows in ``block.conn`` order.

        ``None`` uses one dimensionless identity coefficient: ``1`` for
        rank 0, :math:`\delta_{ij}` for rank 2, or
        :math:`\delta_{ik}\delta_{jl}` for rank 4. The default component
        axes have size 3 and the feature axis has size 1.
    rank : {0, 2, 4}, default=0
        Material tensor rank, supplied explicitly rather than inferred.
        Use 0 for isotropic diffusion, 2 for anisotropic diffusion, or 4
        for a vector problem such as linear elasticity. Material ranks 0
        and 2 produce rank-0 local matrices; rank 4 produces rank-2 matrices.
    quadrature_order : int or None, optional
        Minimum polynomial degree integrated exactly by the reference
        quadrature rule, not the number of quadrature points. ``None``
        requests degree 2 for linear elements and 4 for quadratic elements.
        The selected rule may have a higher degree.

    Returns
    -------
    dict[pyvista.CellType, T]
        Local matrices keyed by cell type. For each block, entry ``r``
        corresponds to mesh cell ``block.global_indices[r]`` and uses the
        node order in ``block.conn[r]``. Material ranks 0 and 2 return shape
        ``(n_block_cells, n_nodes, n_nodes, n_features)``. Rank 4 returns
        ``(n_block_cells, n_nodes, n_nodes, n_components, n_components,
        n_features)``, with output component ``j`` and input component ``l``.

    Raises
    ------
    ValueError
        If coefficient mapping keys or leading sizes do not match the mesh.
    NotImplementedError
        If the material rank, cell type, or quadrature order is unsupported.

    Notes
    -----
    For shape functions :math:`N_a` and :math:`N_b`, integrate the following
    expressions over each physical cell. Repeated spatial indices are
    summed; features are independent.

    * Rank 0: :math:`\kappa \nabla N_a \cdot \nabla N_b`.
    * Rank 2: :math:`\partial_i N_a C_{ij} \partial_j N_b`.
    * Rank 4: :math:`\partial_i N_a C_{ijkl} \partial_k N_b`, leaving
      component indices :math:`j` and :math:`l` in the local matrix.

    The result can be passed directly to :func:`apply_cell_local_fem_matrix`
    or, for a single feature, :func:`global_fem_matrix`.
    """
    if rank not in (0, 2, 4):
        raise NotImplementedError(f"Unsupported rank: {rank}")
    backend = mesh.backend
    if cell_material_coeff is None:
        cell_material_coeff = _generate_global_identity_tensor(
            backend, rank, backend.dtype
        )
    blocks = _require_fixed_cell_blocks_by_type(mesh)
    coefficients = _coefficients_for_blocks(mesh, blocks, cell_material_coeff)
    matrices: dict[pv.CellType, T] = {}
    for cell_type, block in blocks.items():
        coeff = coefficients[cell_type]
        rule = get_quadrature_rule(cell_type, quadrature_order, backend=backend)
        reference_gradients = get_shape_function_gradients(cell_type)(rule.xi)
        nodes = mesh.xyz[backend.as_index_tensor(block.conn)]
        jacobian = compute_jacobian(nodes, reference_gradients, backend)
        grad = physical_shape_function_gradients(
            jacobian, reference_gradients, backend
        )
        _, abs_det_j = jacobian_measure(jacobian)
        match rank:
            case 0:
                equation = "eqai,ef,eqbi,q,eq->eabf"
            case 2:
                equation = "eqai,eijf,eqbj,q,eq->eabf"
            case 4:
                equation = "eqai,eijklf,eqbk,q,eq->eabjlf"
            case _:
                raise NotImplementedError(f"Unsupported rank: {rank}")
        matrices[cell_type] = backend.einsum(
            equation,
            grad,
            coeff,
            grad,
            rule.weights,
            abs_det_j,
            dimension="auto",
        )
    return matrices


def cell_local_fem_mass[T: TensorLike](
    mesh: TensorMesh[T],
    cell_density: T | Mapping[pv.CellType, T] | None = None,
    *,
    quadrature_order: int | None = None,
) -> dict[pv.CellType, T]:
    r"""Integrate a consistent mass matrix for each mesh cell.

    Each entry is :math:`M_{ab} = \int \rho N_a N_b \, dV`, with density
    constant within the cell. The full local matrix is returned; no diagonal
    mass lumping is performed.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh with supported finite elements and a fixed node count within
        each cell type. Different cell types may have different node counts.
    cell_density : T, Mapping[pyvista.CellType, T], or None, optional
        Density with shape ``(c, f)``, where ``f`` is the number of independent
        features. Supply one of the following:

        * A tensor with ``c == 1`` for a uniform density over the whole mesh.
        * A tensor with ``c == mesh.n_cells`` for densities in mesh cell
          order. Each block selects its rows using ``block.global_indices``.
        * A mapping containing exactly the mesh's cell types. Each value
          has ``c == 1`` for a uniform density within the block, or
          ``c == block.n_cells`` with rows in ``block.conn`` order.

        ``None`` uses dimensionless unit density with shape ``(1, 1)``.
    quadrature_order : int or None, optional
        Minimum polynomial degree integrated exactly by the reference
        quadrature rule, not the number of quadrature points. ``None``
        requests degree 2 for linear elements and 4 for quadratic elements.
        The selected rule may have a higher degree.

    Returns
    -------
    dict[pyvista.CellType, T]
        Local mass matrices keyed by cell type, each with shape
        ``(n_block_cells, n_nodes, n_nodes, n_features)``. Entry ``r``
        corresponds to mesh cell ``block.global_indices[r]`` and uses the
        node order in ``block.conn[r]``. These rank-0 matrices can also be
        applied independently to each component of a vector field.

    Raises
    ------
    ValueError
        If density mapping keys or leading sizes do not match the mesh.
    NotImplementedError
        If the cell type or quadrature order is unsupported.
    """
    backend = mesh.backend
    if cell_density is None:
        cell_density = backend.ones((1, 1), dimension={})
    blocks = _require_fixed_cell_blocks_by_type(mesh)
    densities = _coefficients_for_blocks(mesh, blocks, cell_density)
    matrices: dict[pv.CellType, T] = {}
    for cell_type, block in blocks.items():
        density = densities[cell_type]
        rule = get_quadrature_rule(cell_type, quadrature_order, backend=backend)
        shape = get_shape_functions(cell_type)(rule.xi)
        reference_gradients = get_shape_function_gradients(cell_type)(rule.xi)
        nodes = mesh.xyz[backend.as_index_tensor(block.conn)]
        jacobian = compute_jacobian(nodes, reference_gradients, backend)
        _, abs_det_j = jacobian_measure(jacobian)
        matrices[cell_type] = backend.einsum(
            "qa,qb,ef,q,eq->eabf",
            shape,
            shape,
            density,
            rule.weights,
            abs_det_j,
            dimension="auto",
        )
    return matrices


def global_fem_matrix[T: TensorLike](
    mesh: TensorMesh[T],
    cell_local_matrix: T | Mapping[pv.CellType, T],
) -> T:
    """Assemble local FEM matrices into a global sparse COO matrix.

    Map each cell's local node indices to mesh points and sum entries that
    share a global row and column, including contributions from different
    cell types.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh whose cell connectivity maps local nodes to mesh points.
    cell_local_matrix : T or Mapping[pyvista.CellType, T]
        Local matrices with shape ``(c, a, b, 1)`` for a scalar operator or
        ``(c, a, b, d, d, 1)`` for a vector operator with ``d`` components.
        The final feature axis must have size 1.

        A single tensor is accepted only when the mesh has one cell type.
        For a mixed mesh, pass a nonempty mapping from cell type to tensor;
        only the supplied blocks contribute. Each tensor must follow
        ``mesh.topology.cell_block(cell_type).conn`` row and node order,
        and all blocks must use the same number of components and physical
        dimension. An unspecified dimension (``None``) is accepted only if
        unspecified for every block.

    Returns
    -------
    T
        Coalesced sparse COO matrix with duplicate contributions summed.
        Its shape is ``(mesh.n_points, mesh.n_points)`` for scalar operators
        or ``(mesh.n_points * d, mesh.n_points * d)`` for vector operators.
        Vector degrees of freedom are ordered by point, then component:
        component ``i`` at point ``p`` has index ``d * p + i``. Reshape a
        nodal vector field to ``(mesh.n_points * d, 1)`` before multiplication.

    Raises
    ------
    ValueError
        If a single tensor is supplied for a mesh without exactly one cell
        type, the mapping is empty, block dimensions differ, or local node
        axes do not match the block.
    NotImplementedError
        If a block has variable node counts, the feature axis is not size 1,
        or the local matrix component rank is neither 0 nor 2.
    """
    assembled = None
    matrices = _local_matrices_by_cell_type(mesh, cell_local_matrix)
    # Scattering calls to_torch and drops dimension metadata, so validate
    # every block before converting any of them.
    dimension = _require_shared_dimension(matrices)
    for cell_type, local in matrices.items():
        partial = _scatter_block_to_uncoalesced_coo(
            mesh, _require_fixed_cell_block(mesh, cell_type), local
        )
        assembled = partial if assembled is None else assembled + partial
    return mesh.backend.as_tensor(assembled.coalesce(), dimension=dimension)


def apply_dirichlet_to_global_fem_matrix[T: TensorLike](
    mesh: TensorMesh[T], global_matrix: T, global_b: T, sparse_dirichlet: T
) -> tuple[T, T]:
    """Impose prescribed nodal values on the linear system ``A u = b``.

    Move constrained-column contributions to the right-hand side, zero
    constrained rows and columns, and set their stored diagonal entries to
    one. Set the corresponding right-hand-side entries to the prescribed
    values. This preserves symmetry when the input matrix is symmetric.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh whose backend wraps the resulting tensors. This operation
        requires the Phlower tensor interface on the matrix and right-hand
        side, including ``to_tensor()`` and ``dimension``.
    global_matrix : T
        Coalesced sparse COO matrix ``A`` with shape ``(n_dof, n_dof)``,
        where ``n_dof`` is the number of global degrees of freedom. Every
        constrained diagonal entry must already be stored in the matrix.
    global_b : T
        Dense right-hand side ``b`` with shape ``(n_dof, 1)``. For vectors,
        flatten point and component axes in the order used by
        :func:`global_fem_matrix`.
    sparse_dirichlet : T
        Sparse COO tensor with all axes sparse and shape
        ``(mesh.n_points, 1)`` for scalars or ``(mesh.n_points, 3, 1)`` for
        three-component vectors. Each stored entry prescribes a value;
        missing entries are unconstrained. A prescribed zero must therefore
        be stored explicitly. Vector indices map to ``3 * point + component``.

    Returns
    -------
    global_matrix_with_dirichlet : T
        New coalesced sparse COO matrix with the boundary conditions applied.
    global_b_with_dirichlet : T
        New dense right-hand side with the boundary conditions applied.

    Raises
    ------
    NotImplementedError
        If ``sparse_dirichlet`` has a feature axis other than size 1 or
        its vector rank is neither 0 nor 1.
    """
    if sparse_dirichlet.shape[-1] != 1:
        raise NotImplementedError(
            f"The last dim should be 1 but given: {sparse_dirichlet.shape}"
        )
    backend = mesh.backend

    sparse_dirichlet = sparse_dirichlet.coalesce()
    # Exclude the point and feature axes.
    vector_rank = len(sparse_dirichlet.shape) - 2
    match vector_rank:
        case 0:
            dirichlet_indices = sparse_dirichlet.indices()[0]
        case 1:
            dirichlet_indices = (
                3 * sparse_dirichlet.indices()[0]
                + sparse_dirichlet.indices()[1]
            )
        case _:
            raise NotImplementedError(
                f"Unexpected rank: {vector_rank} "
                f"(shape: {sparse_dirichlet.shape})"
            )
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
            indices=indices,
            values=values,
            size=global_matrix.size(),
            check_invariants=False,
        ).coalesce(),
        dimension=global_matrix.dimension,
    ), new_global_b


# =============================================================================
# Deprecated tet-specific wrappers
# =============================================================================
def cell_local_fem_rigidity_tet[T: TensorLike](
    mesh: TensorMesh[T], cell_material_coeff: T | None = None, rank: int = 0
) -> T:
    """Return the linear-tetrahedron stiffness matrices through the old API.

    Arguments follow :func:`cell_local_fem_rigidity`. This wrapper returns
    only the tensor under ``pyvista.CellType.TETRA``; the replacement returns
    a dictionary of tensors keyed by cell type.

    .. deprecated::
        Use ``cell_local_fem_rigidity(mesh, ...)[pyvista.CellType.TETRA]``
        to retain the tensor return value.
    """
    logger.warning(
        "cell_local_fem_rigidity_tet is deprecated; use cell_local_fem_rigidity"
    )
    return cell_local_fem_rigidity(
        mesh, cell_material_coeff=cell_material_coeff, rank=rank
    )[pv.CellType.TETRA]


def cell_local_fem_mass_tet[T: TensorLike](
    mesh: TensorMesh[T],
    cell_density: T | None = None,
) -> T:
    """Return the linear-tetrahedron mass matrices through the old API.

    Arguments follow :func:`cell_local_fem_mass`. This wrapper returns only
    the tensor under ``pyvista.CellType.TETRA``; the replacement returns a
    dictionary of tensors keyed by cell type.

    .. deprecated::
        Use ``cell_local_fem_mass(mesh, ...)[pyvista.CellType.TETRA]``
        to retain the tensor return value.
    """
    logger.warning(
        "cell_local_fem_mass_tet is deprecated; use cell_local_fem_mass"
    )
    return cell_local_fem_mass(mesh, cell_density=cell_density)[
        pv.CellType.TETRA
    ]


def apply_cell_local_fem_matrix_tet[T: TensorLike](
    mesh: TensorMesh[T],
    cell_local_matrix_tet: T,
    u: T,
    vector_rank: int | None = None,
    matrix_rank: int | None = None,
) -> T:
    """Apply local matrices through the deprecated tetrahedron API.

    Delegates to :func:`apply_cell_local_fem_matrix`, passing
    ``cell_local_matrix_tet`` as ``cell_local_matrix``. Tensor layouts, rank
    handling, and the returned nodal field follow that function. A single
    tensor requires a mesh containing exactly one cell type.

    .. deprecated::
        Use :func:`apply_cell_local_fem_matrix` and rename the
        ``cell_local_matrix_tet`` keyword to ``cell_local_matrix``.
    """
    logger.warning(
        "apply_cell_local_fem_matrix_tet is deprecated; "
        "use apply_cell_local_fem_matrix"
    )
    return apply_cell_local_fem_matrix(
        mesh,
        cell_local_matrix=cell_local_matrix_tet,
        u=u,
        vector_rank=vector_rank,
        matrix_rank=matrix_rank,
    )


def global_fem_matrix_tet[T: TensorLike](
    mesh: TensorMesh[T],
    cell_local_matrix_tet: T,
) -> T:
    """Assemble a global matrix through the deprecated tetrahedron API.

    Delegates to :func:`global_fem_matrix`. Tensor layouts and the returned
    sparse COO matrix follow that function. A single tensor requires a mesh
    containing exactly one cell type.

    .. deprecated::
        Use :func:`global_fem_matrix` and rename the ``cell_local_matrix_tet``
        keyword to ``cell_local_matrix``.
    """
    logger.warning("global_fem_matrix_tet is deprecated; use global_fem_matrix")
    return global_fem_matrix(mesh, cell_local_matrix_tet)


# =============================================================================
# Internal helpers
# =============================================================================


def _require_fixed_cell_blocks_by_type[T: TensorLike](
    mesh: TensorMesh[T],
) -> dict[pv.CellType, FixedCellBlock]:
    """Collect fixed-connectivity blocks for all cell types in the mesh.

    Each block must have a constant number of nodes per cell. Different
    blocks may have different node counts.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh that owns the blocks.

    Returns
    -------
    dict[pyvista.CellType, FixedCellBlock]
        One block per cell type, in ``mesh.topology.unique_cell_types()``
        order.

    Raises
    ------
    NotImplementedError
        If any cell type uses variable-length connectivity.
    """
    blocks: dict[pv.CellType, FixedCellBlock] = {}
    for cell_type_id in mesh.topology.unique_cell_types():
        cell_type = pv.CellType(int(cell_type_id))
        blocks[cell_type] = _require_fixed_cell_block(mesh, cell_type)
    return blocks


def _require_fixed_cell_block[T: TensorLike](
    mesh: TensorMesh[T], cell_type: pv.CellType
) -> FixedCellBlock:
    """Fetch one cell-type block and require fixed-length connectivity.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh that owns the block.
    cell_type : pyvista.CellType
        Block to select.

    Returns
    -------
    FixedCellBlock
        Block with connectivity of shape ``(n_block_cells, n_nodes)`` and
        ``global_indices`` mapping its rows to mesh cell indices.

    Raises
    ------
    NotImplementedError
        If the selected block is not a ``FixedCellBlock``.
    """
    block = mesh.topology.cell_block(cell_type)
    if not isinstance(block, FixedCellBlock):
        raise NotImplementedError(
            "FEM operators require a fixed-connectivity cell type, "
            f"got {cell_type.name}."
        )
    return block


def _local_matrices_by_cell_type[T: TensorLike](
    mesh: TensorMesh[T],
    cell_local_matrix: T | Mapping[pv.CellType, T],
) -> Mapping[pv.CellType, T]:
    """Return local matrices keyed by cell type.

    A mapping is returned unchanged, without checking its keys or values.
    A single tensor is wrapped in a dictionary under the mesh's only cell
    type. Raise ``ValueError`` if a tensor is supplied and the mesh does not
    contain exactly one cell type.
    """
    if isinstance(cell_local_matrix, Mapping):
        return cell_local_matrix
    types = mesh.topology.unique_cell_types()
    if len(types) != 1:
        raise ValueError(
            "cell_type is required when the mesh contains more than one "
            f"cell type: {types}"
        )
    return {pv.CellType(int(types[0])): cell_local_matrix}


def _require_shared_dimension[T: TensorLike](
    matrices: Mapping[pv.CellType, T],
) -> PhlowerDimensionTensor | None:
    """Require a nonempty mapping with the same dimension on every block.

    Return the shared dimension before any block loses its metadata during
    assembly. ``None`` represents an unspecified dimension and must match
    across all blocks; it is distinct from a dimensionless value.
    """
    if not matrices:
        raise ValueError("Local matrix mapping must not be empty.")
    entries = iter(matrices.items())
    first_cell_type, first_local = next(entries)
    dimension = get_dimension(first_local)
    for cell_type, local in entries:
        local_dimension = get_dimension(local)
        if local_dimension != dimension:
            raise ValueError(
                "Local matrix blocks must have the same physical dimension: "
                f"{first_cell_type.name} has {dimension}, "
                f"but {cell_type.name} has {local_dimension}."
            )
    return dimension


def _coefficients_for_blocks[T: TensorLike](
    mesh: TensorMesh[T],
    blocks: Mapping[pv.CellType, FixedCellBlock],
    coeff: T | Mapping[pv.CellType, T],
) -> dict[pv.CellType, T]:
    """Resolve uniform, mesh-ordered, or block-ordered material coefficients.

    A tensor with leading size 1 is one value for every cell. A tensor with
    leading size ``mesh.n_cells`` is mesh cell order and is gathered with
    ``block.global_indices``. A mapping holds one tensor per cell type,
    already in that block's order, with leading size 1 or ``block.n_cells``.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh whose cell count determines the expected size of a tensor
        containing one coefficient per mesh cell.
    blocks : Mapping[pyvista.CellType, FixedCellBlock]
        Blocks that receive coefficients, keyed by cell type.
    coeff : T or Mapping[pyvista.CellType, T]
        Mesh-ordered coefficient, or one block-ordered coefficient per
        cell type.

    Returns
    -------
    dict[pyvista.CellType, T]
        One coefficient tensor per block. Per-cell tensors follow
        ``block.conn`` row order; uniform tensors keep their leading size
        of 1 for broadcasting rather than being expanded to every cell.

    Raises
    ------
    ValueError
        If mapping keys differ from ``blocks``, a mapping value has leading
        size other than 1 or ``block.n_cells``, or a tensor has leading size
        other than 1 or ``mesh.n_cells``.
    """
    if isinstance(coeff, Mapping):
        present = set(blocks)
        supplied = set(coeff)
        missing = present - supplied
        extra = supplied - present
        if missing or extra:
            raise ValueError(
                "Coefficient mapping keys must match the mesh cell types. "
                f"Missing {[cell_type.name for cell_type in missing]}, "
                f"unexpected {[cell_type.name for cell_type in extra]}."
            )
        resolved: dict[pv.CellType, T] = {}
        for cell_type, block in blocks.items():
            local = coeff[cell_type]
            if int(local.shape[0]) not in (1, block.n_cells):
                raise ValueError(
                    f"Coefficient for {cell_type.name} has leading "
                    f"size {int(local.shape[0])}; expected 1 or "
                    f"{block.n_cells}."
                )
            resolved[cell_type] = local
        return resolved

    leading = int(coeff.shape[0])
    if leading == 1:
        return dict.fromkeys(blocks, coeff)
    if leading == mesh.n_cells:
        return {
            cell_type: coeff[mesh.backend.as_index_tensor(block.global_indices)]
            for cell_type, block in blocks.items()
        }
    raise ValueError(
        f"Coefficient leading size {leading} must be 1 or {mesh.n_cells}. "
        "Block-ordered coefficients are a mapping from cell type to tensor."
    )


def _multiply_on_cell_nodes[T: TensorLike](
    mesh: TensorMesh[T],
    cell_local_matrix: T,
    cell_u: T,
    vector_rank: int,
    matrix_rank: int,
) -> T:
    """Multiply local matrices by gathered nodal values within one cell block.

    ``cell_u`` has layout ``(c, b, ..., f)``. A rank-0 matrix with layout
    ``(c, a, b, f)`` sums over ``b`` and preserves all component axes. A
    rank-2 matrix with layout ``(c, a, b, i, j, f)`` requires a rank-1 field
    ``(c, b, j, f)`` and sums over both ``b`` and ``j``. Other combinations
    raise ``NotImplementedError``.

    The result has layout ``(c, a, ..., f)``. Contributions at shared mesh
    points remain separate until the caller scatters and sums them.
    """
    if matrix_rank == 0:
        extra = "" if vector_rank == 0 else "..."
        return mesh.backend.einsum(
            f"cabf,cb{extra}f->ca{extra}f",
            cell_local_matrix,
            cell_u,
            dimension="auto",
        )
    if matrix_rank == 2 and vector_rank == 1:
        return mesh.backend.einsum(
            "cabijf,cbjf->caif",
            cell_local_matrix,
            cell_u,
            dimension="auto",
        )
    raise NotImplementedError(
        f"Unexpected combination of {vector_rank = } and {matrix_rank = }"
    )


def _validate_node_count_matches_block[T: TensorLike](
    block: FixedCellBlock, local: T
) -> None:
    """Require both local node axes to match the block's nodes per cell.

    Compare ``local.shape[1:3]`` with the width of ``block.conn`` and raise
    ``ValueError`` on a mismatch. Cell and component axes are not checked.
    """
    n_nodes = block.conn.shape[1]
    node_axes = (int(local.shape[1]), int(local.shape[2]))
    if node_axes != (n_nodes, n_nodes):
        raise ValueError(
            f"Local matrix node axes {node_axes} do not match "
            f"the block connectivity width {n_nodes}."
        )


def _scatter_block_to_uncoalesced_coo[T: TensorLike](
    mesh: TensorMesh[T], block: FixedCellBlock, cell_local_matrix: T
) -> torch.Tensor:
    """Scatter one block's local matrices into an uncoalesced global COO.

    Accept scalar matrices ``(c, a, b, 1)`` or vector matrices
    ``(c, a, b, d, d, 1)``. Validate the local node counts, remove the single
    feature axis, and return a plain ``torch.Tensor``. Duplicate global
    entries remain unsummed; the caller combines blocks and coalesces them.
    """
    _validate_node_count_matches_block(block, cell_local_matrix)
    local = mesh.backend.to_torch(cell_local_matrix)
    if local.shape[-1] != 1:
        raise NotImplementedError(
            f"The last dim should be 1 but given: {tuple(local.shape)}"
        )
    # Rank counts axes between the node pair ``(a, b)`` and the feature axis.
    rank = local.ndim - 4
    if rank not in (0, 2):
        raise NotImplementedError(f"Unexpected rank: {rank}")
    return _coo_from_local_entries(
        mesh.backend.as_index_tensor(block.conn),
        local[..., 0],
        mesh.n_points,
    )


def _coo_from_local_entries(
    connectivity: torch.Tensor, entries: torch.Tensor, n_points: int
) -> torch.Tensor:
    """Scatter local entries into one uncoalesced COO matrix.

    Parameters
    ----------
    connectivity : torch.Tensor
        Point indices of shape ``(n_cells, n_nodes)``.
    entries : torch.Tensor
        Local entries, without a feature axis, of shape
        ``(n_cells, n_nodes, n_nodes)`` for matrix rank 0 or
        ``(n_cells, n_nodes, n_nodes, n_vector, n_vector)`` for matrix rank 2.
        Vector entry
        ``entries[c, a, b, i, j]`` maps to row
        ``n_vector * connectivity[c, a] + i`` and column
        ``n_vector * connectivity[c, b] + j``.
    n_points : int
        Number of mesh points.

    Returns
    -------
    torch.Tensor
        Sparse COO matrix of shape ``(n_points, n_points)`` for scalars or
        ``(n_points * n_vector, n_points * n_vector)`` for vectors. Duplicate
        contributions are not yet summed.
    """
    n_cells, n_nodes = connectivity.shape
    # Exclude the cell axis and the two local node axes.
    matrix_rank = entries.ndim - 3
    match matrix_rank:
        case 0:
            rows = connectivity[:, :, None].expand(n_cells, n_nodes, n_nodes)
            cols = connectivity[:, None, :].expand(n_cells, n_nodes, n_nodes)
            size = n_points
        case 2:
            n_vector = int(entries.shape[-1])
            component = torch.arange(n_vector, device=connectivity.device)
            rows = connectivity[
                :, :, None, None, None
            ] * n_vector + component.view(1, 1, 1, n_vector, 1)
            cols = connectivity[
                :, None, :, None, None
            ] * n_vector + component.view(1, 1, 1, 1, n_vector)
            shape = (n_cells, n_nodes, n_nodes, n_vector, n_vector)
            rows = rows.expand(shape)
            cols = cols.expand(shape)
            size = n_points * n_vector
        case _:
            raise NotImplementedError(
                f"Unexpected local entry shape: {tuple(entries.shape)}"
            )
    return torch.sparse_coo_tensor(
        indices=torch.stack((rows.reshape(-1), cols.reshape(-1))),
        values=entries.reshape(-1),
        size=(size, size),
        check_invariants=False,
    )


def _generate_global_identity_tensor[T: TensorLike](
    backend: Backend, rank: int, dtype: torch.dtype
) -> T:
    r"""Return a dimensionless identity material shared by all mesh cells.

    Rank 0 returns ``1`` with shape ``(1, 1)``. Rank 2 returns
    :math:`\delta_{ij}` with shape ``(1, 3, 3, 1)``. Rank 4 returns
    :math:`\delta_{ik}\delta_{jl}` with shape ``(1, 3, 3, 3, 3, 1)``.
    The leading size 1 broadcasts over cells; the last axis is one feature.
    Other ranks raise ``NotImplementedError``.
    """
    delta = backend.as_tensor(torch.eye(3, dtype=dtype), dimension={})
    match rank:
        case 0:
            return backend.as_tensor(torch.ones((1, 1)), dimension={})
        case 2:
            return delta[None, ..., None]
        case 4:
            return backend.einsum(
                "ik,jl->ijkl", delta, delta, dimension="auto"
            )[None, ..., None]
        case _:
            raise NotImplementedError(f"Unsupported rank: {rank}")


def _get_rank[T: TensorLike](
    t: T, rank: int | None = None, offset: int = 0
) -> int:
    """Return the number of component axes, excluding storage-layout axes.

    A supplied ``rank`` is returned unchanged. Otherwise a Phlower tensor
    uses ``t.rank() - offset``, which keeps time and voxel axes out of the
    count. Any other tensor is laid out as one entity axis, ``offset``
    structural axes, the component axes, and one feature axis, so the
    component rank is ``t.ndim - 2 - offset``.

    Parameters
    ----------
    t : T
        Tensor whose rank is required.
    rank : int or None, optional
        Explicit component rank. Returned unchanged without inference or
        validation when supplied.
    offset : int, default=0
        Structural axes included in ``t.rank()`` or ``t.ndim`` that are not
        part of the component rank. Local matrices use 2 for the node axes.

    Returns
    -------
    int
        Component rank: for example, 0 for a scalar field ``(points, f)``,
        1 for a vector field ``(points, i, f)``, or 2 for a local matrix
        ``(cells, a, b, i, j, f)`` when ``offset=2``.

    Raises
    ------
    ValueError
        If the inferred rank is negative.
    """
    if rank is not None:
        return rank
    if hasattr(t, "rank"):
        rank = t.rank() - offset
    else:
        rank = t.ndim - 2 - offset
    if rank < 0:
        raise ValueError(f"Rank is negative for {t} with offset {offset}")
    return rank

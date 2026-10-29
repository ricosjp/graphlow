"""Element-generic FEM integration, material layouts, and assembly contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest
import pyvista as pv
import torch
from phlower_tensor import phlower_tensor

import graphlow
from graphlow.core.backend.base import Backend
from graphlow.core.backend.phlower import PhlowerBackend
from graphlow.core.element_definitions import get_element
from graphlow.core.mesh import TensorMesh
from graphlow.geometry.fem import _get_rank

_REFERENCE_MEASURES = {
    pv.CellType.TRIANGLE: 0.5,
    pv.CellType.QUAD: 4.0,
    pv.CellType.TETRA: 1 / 6,
    pv.CellType.HEXAHEDRON: 8.0,
    pv.CellType.QUADRATIC_TRIANGLE: 0.5,
    pv.CellType.QUADRATIC_QUAD: 4.0,
    pv.CellType.QUADRATIC_TETRA: 1 / 6,
    pv.CellType.QUADRATIC_HEXAHEDRON: 8.0,
}
_QUADRATIC_TYPES = tuple(_REFERENCE_MEASURES)[4:]
_AFFINE_MAP = np.array([[2.0, 0.2, 0.0], [0.0, 3.0, 0.4], [0.1, 0.0, 4.0]])
_TRANSLATION = np.array([0.3, -0.2, 0.7])


def _from_grid(grid: pv.UnstructuredGrid, backend: Backend) -> TensorMesh:
    return graphlow.from_pyvista(
        grid, backend.name, dtype=backend.dtype, device=backend.device
    )


def _element_mesh(cell_type: pv.CellType, backend: Backend) -> TensorMesh:
    reference = np.asarray(get_element(cell_type).reference_coords)
    points = reference @ _AFFINE_MAP[: reference.shape[1]] + _TRANSLATION
    grid = pv.UnstructuredGrid(
        {cell_type: np.arange(len(points))[None, :]}, points
    )
    return _from_grid(grid, backend)


def _physical_measure(cell_type: pv.CellType) -> float:
    dimension = get_element(cell_type).reference_dimension
    basis = _AFFINE_MAP[:dimension]
    return _REFERENCE_MEASURES[cell_type] * np.sqrt(
        np.linalg.det(basis @ basis.T)
    )


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize(
    "cell_type",
    tuple(_REFERENCE_MEASURES),
    ids=lambda cell_type: cell_type.name,
)
class TestLocalMatrices:
    """Check cell integrals against analytic fields on affine geometries."""

    def test_mass_integrates_constant_and_linear_fields(
        self, cell_type: pv.CellType, backend: Backend
    ):
        mesh = _element_mesh(cell_type, backend)
        density = backend.as_tensor([[1.25, 3.0]], dimension={})
        matrices = mesh.geometry.cell_local_fem_mass(density)
        assert set(matrices) == {cell_type}
        mass = backend.to_torch(matrices[cell_type])[0]
        assert mass.shape == (mesh.n_points, mesh.n_points, 2)
        _assert_close(mass, mass.transpose(0, 1))

        dimension = get_element(cell_type).reference_dimension
        simplex = cell_type in (
            pv.CellType.TRIANGLE,
            pv.CellType.QUADRATIC_TRIANGLE,
            pv.CellType.TETRA,
            pv.CellType.QUADRATIC_TETRA,
        )
        reference_centroid = np.full(
            dimension, 1 / (dimension + 1) if simplex else 0.0
        )
        centroid = reference_centroid @ _AFFINE_MAP[:dimension] + _TRANSLATION
        measure = _physical_measure(cell_type)
        total = mass.new_tensor([1.25, 3.0]) * measure
        _assert_close(mass.sum(dim=(0, 1)), total)
        # Partition of unity and linear completeness give exact first moments.
        xyz = backend.to_torch(mesh.xyz)
        moments = torch.einsum("abf,bi->if", mass, xyz)
        _assert_close(moments, mass.new_tensor(centroid)[:, None] * total)

    @pytest.mark.parametrize("rank", [0, 2, 4])
    def test_rigidity_matches_affine_field_energy(
        self, cell_type: pv.CellType, rank: int, backend: Backend
    ):
        """Compare u.T K u with the continuum energy, including cross terms."""
        mesh = _element_mesh(cell_type, backend)
        basis = _AFFINE_MAP[: get_element(cell_type).reference_dimension]
        tangent_projection = basis.T @ np.linalg.solve(basis @ basis.T, basis)
        if rank == 0:
            material = np.asarray(2.5)
        else:
            size = 3 if rank == 2 else 9
            seed = np.arange(1, size * size + 1).reshape(size, size) / size**2
            material = (seed.T @ seed + np.eye(size)).reshape((3,) * rank)
        coeff = backend.as_tensor(
            material[None, ..., None] * np.array([1.0, 2.5]), dimension={}
        )
        matrices = mesh.geometry.cell_local_fem_rigidity(coeff, rank=rank)
        assert set(matrices) == {cell_type}
        local = backend.to_torch(matrices[cell_type])[0]
        if rank == 4:
            gradient = np.array(
                [[1.0, 0.4, -0.2], [0.3, 2.0, 0.7], [-1, 0.6, 1]]
            )
            physical_gradient = (tangent_projection @ gradient).reshape(-1)
            energy_density = (
                physical_gradient @ material.reshape(9, 9) @ physical_gradient
            )
            assert local.shape == (mesh.n_points, mesh.n_points, 3, 3, 2)
            local = local.permute(0, 2, 1, 3, 4).reshape(
                mesh.n_points * 3, mesh.n_points * 3, 2
            )
        else:
            gradient = np.array([1.5, -2.0, 0.75])
            physical_gradient = tangent_projection @ gradient
            energy_density = (
                2.5 * (physical_gradient @ physical_gradient)
                if rank == 0
                else physical_gradient @ material @ physical_gradient
            )
            assert local.shape == (mesh.n_points, mesh.n_points, 2)

        _assert_close(local, local.transpose(0, 1))
        field = (
            backend.to_torch(mesh.xyz) @ local.new_tensor(gradient)
        ).reshape(-1)
        energy = torch.einsum("a,abf,b->f", field, local, field)
        expected = (
            local.new_tensor([1.0, 2.5])
            * _physical_measure(cell_type)
            * energy_density
        )
        _assert_close(energy, expected)

    @pytest.mark.parametrize("operator", ["mass", "rigidity"])
    def test_reflection_preserves_scalar_local_matrices(
        self, cell_type: pv.CellType, operator: str, backend: Backend
    ):
        mesh = _element_mesh(cell_type, backend)
        reflected_grid = mesh.pvmesh.copy()
        reflected_grid.points[:, 0] *= -1
        reflected = _from_grid(reflected_grid, backend)
        original = getattr(mesh.geometry, f"cell_local_fem_{operator}")()[
            cell_type
        ]
        actual = getattr(reflected.geometry, f"cell_local_fem_{operator}")()[
            cell_type
        ]
        _assert_close(backend.to_torch(actual), backend.to_torch(original))


@pytest.mark.parametrize(
    "cell_type", _QUADRATIC_TYPES, ids=lambda cell_type: cell_type.name
)
def test_quadratic_mass_integrates_squared_quadratic_field(
    cell_type: pv.CellType, backend: Backend
):
    """Full integration resolves degree four; one-point integration does not."""
    mesh = _element_mesh(cell_type, backend)
    reference = np.asarray(get_element(cell_type).reference_coords)
    field = backend.to_torch(backend.as_tensor(reference[:, 0] ** 2))
    matrices = {
        order: backend.to_torch(
            mesh.geometry.cell_local_fem_mass(quadrature_order=order)[cell_type]
        )[0, ..., 0]
        for order in (None, 1, 4)
    }
    dimension = reference.shape[1]
    mean_fourth_power = (
        math.factorial(dimension)
        * math.factorial(4)
        / math.factorial(dimension + 4)
        if cell_type
        in (pv.CellType.QUADRATIC_TRIANGLE, pv.CellType.QUADRATIC_TETRA)
        else 1 / 5
    )
    expected = field.new_tensor(
        _physical_measure(cell_type) * mean_fourth_power
    )
    _assert_close(field @ matrices[None] @ field, expected)
    _assert_close(matrices[None], matrices[4])
    assert not torch.isclose(
        field @ matrices[1] @ field, expected, rtol=1e-3, atol=1e-6
    )


def test_warped_hex_integrals_and_geometry_gradients(backend: Backend):
    """Raising one top vertex by h gives volume 8 + h and derivative 1."""
    backend = backend.to(dtype=torch.float64)
    points = np.array(get_element(pv.CellType.HEXAHEDRON).reference_coords)
    points[6, 2] += 0.4
    grid = pv.UnstructuredGrid(
        {pv.CellType.HEXAHEDRON: np.arange(8)[None]}, points
    )
    mesh = _from_grid(grid, backend)
    mesh.requires_grad()
    raw_points = backend.to_torch(mesh.points)
    mass = mesh.geometry.cell_local_fem_mass()[pv.CellType.HEXAHEDRON]
    stiffness = mesh.geometry.cell_local_fem_rigidity()[pv.CellType.HEXAHEDRON]
    volume = backend.to_torch(mass).sum()
    x = backend.to_torch(mesh.xyz)[:, 0]
    energy = x @ backend.to_torch(stiffness)[0, ..., 0] @ x
    _assert_close(volume, volume.new_tensor(8.4))
    _assert_close(energy, energy.new_tensor(8.4))
    for integral in (volume, energy):
        (gradient,) = torch.autograd.grad(
            integral, raw_points, retain_graph=True
        )
        gradient = gradient.reshape(8, 3)
        assert torch.isfinite(gradient).all()
        _assert_close(gradient[6, 2], gradient.new_tensor(1.0))
        _assert_close(gradient.sum(dim=0), gradient.new_zeros(3))


@pytest.mark.parametrize(
    "cell_type", [pv.CellType.TRIANGLE, pv.CellType.HEXAHEDRON]
)
@pytest.mark.parametrize(
    "operator,rank",
    [("mass", 0), ("rigidity", 0), ("rigidity", 2), ("rigidity", 4)],
)
def test_phlower_dimensions_survive_assembly(
    cell_type: pv.CellType,
    operator: str,
    rank: int,
    backend_phlower: PhlowerBackend,
):
    mesh = _element_mesh(cell_type, backend_phlower)
    backend = mesh.backend
    coefficient = backend.ones((1,) + (3,) * rank + (1,), dimension={"M": 1})
    kwargs = {"rank": rank} if operator == "rigidity" else {}
    local = getattr(mesh.geometry, f"cell_local_fem_{operator}")(
        coefficient, **kwargs
    )
    length_power = int(get_element(cell_type).reference_dimension)
    if operator == "rigidity":
        length_power -= 2
    expected_dimension = backend.ones(
        (1, 1), dimension={"L": length_power, "M": 1}
    ).dimension
    assert local[cell_type].dimension == expected_dimension
    global_matrix = mesh.geometry.global_fem_matrix(local)
    assert global_matrix.dimension == expected_dimension
    shape = (mesh.n_points, 3, 1) if rank == 4 else (mesh.n_points, 1)
    field = backend.as_tensor(
        np.arange(math.prod(shape), dtype=np.float64).reshape(shape),
        dimension={"T": 1},
    )
    direct = mesh.geometry.apply_cell_local_fem_matrix(local, field)
    assembled = global_matrix @ field.reshape((-1, 1))
    expected_product_dimension = backend.ones(
        (1, 1), dimension={"L": length_power, "M": 1, "T": 1}
    ).dimension
    assert direct.dimension == assembled.dimension == expected_product_dimension
    _assert_close(direct.to_tensor().reshape(-1, 1), assembled.to_tensor())


@pytest.fixture
def mixed_grid() -> pv.UnstructuredGrid:
    """Interleave two hexes and two attached tets, plus an unused point."""
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
            [2, 0, 0],
            [2, 1, 0],
            [2, 0, 1],
            [2, 1, 1],
            [0, -1, 0],
            [2, -1, 0],
            [3, 3, 3],
        ],
        dtype=np.float64,
    )
    cells = np.array(
        [
            8,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            4,
            0,
            1,
            4,
            12,
            8,
            1,
            8,
            9,
            2,
            5,
            10,
            11,
            6,
            4,
            1,
            8,
            5,
            13,
        ]
    )
    types = [pv.CellType.HEXAHEDRON, pv.CellType.TETRA] * 2
    return pv.UnstructuredGrid(cells, types, points)


@pytest.fixture
def mixed_mesh(mixed_grid: pv.UnstructuredGrid, backend: Backend) -> TensorMesh:
    return _from_grid(mixed_grid, backend)


@pytest.mark.parametrize(
    "dimensions",
    [
        pytest.param(({"L": 1}, {"M": 1}), id="length-mass"),
        pytest.param(({"M": 1}, {"L": 1}), id="mass-length"),
        pytest.param((None, {"L": 1}), id="unspecified-length"),
        pytest.param(({"L": 1}, None), id="length-unspecified"),
        pytest.param((None, {}), id="unspecified-dimensionless"),
        pytest.param(({}, None), id="dimensionless-unspecified"),
    ],
)
def test_global_assembly_rejects_inconsistent_dimensions(
    mixed_grid: pv.UnstructuredGrid,
    backend_phlower: PhlowerBackend,
    dimensions: tuple[dict[str, int] | None, dict[str, int] | None],
):
    mesh = _from_grid(mixed_grid, backend_phlower)
    backend = mesh.backend
    local = {}
    for cell_type, dimension in zip(
        (pv.CellType.HEXAHEDRON, pv.CellType.TETRA), dimensions, strict=True
    ):
        block = mesh.topology.cell_block(cell_type)
        n_nodes = block.conn.shape[1]
        local[cell_type] = backend.ones(
            (block.n_cells, n_nodes, n_nodes, 1), dimension=dimension
        )

    with pytest.raises(ValueError, match="same physical dimension"):
        mesh.geometry.global_fem_matrix(local)


def test_global_assembly_rejects_empty_mapping(mixed_mesh: TensorMesh):
    with pytest.raises(ValueError, match="mapping must not be empty"):
        mixed_mesh.geometry.global_fem_matrix({})


@pytest.mark.parametrize(
    "operator,rank",
    [("mass", 0), ("rigidity", 0), ("rigidity", 2), ("rigidity", 4)],
)
@pytest.mark.parametrize(
    "layout", ["uniform", "mesh", "blocks", "block_uniform"]
)
def test_mixed_material_layouts(
    mixed_mesh: TensorMesh, operator: str, rank: int, layout: str
):
    """Coefficient rows follow global cell IDs, even for interleaved types."""
    mesh = mixed_mesh
    backend = mesh.backend
    integrate = getattr(mesh.geometry, f"cell_local_fem_{operator}")
    kwargs = {"rank": rank} if operator == "rigidity" else {}
    unit = integrate(**kwargs)
    assert set(unit) == {pv.CellType.HEXAHEDRON, pv.CellType.TETRA}
    cell_ids = {pv.CellType.HEXAHEDRON: [0, 2], pv.CellType.TETRA: [1, 3]}
    scales = np.array([[2, 3], [5, 7], [11, 13], [17, 19]], dtype=np.float64)
    if layout == "uniform":
        scales[:] = scales[0]
    elif layout == "block_uniform":
        scales[2:] = scales[:2]
    identity = (
        np.asarray(1.0)
        if rank == 0
        else np.eye(3)
        if rank == 2
        else np.eye(9).reshape(3, 3, 3, 3)
    )
    coefficients = (
        scales.reshape((4,) + (1,) * rank + (2,)) * identity[None, ..., None]
    )
    if layout == "uniform":
        supplied = backend.as_tensor(coefficients[:1], dimension={})
    elif layout == "mesh":
        supplied = backend.as_tensor(coefficients, dimension={})
    else:
        supplied = {
            cell_type: backend.as_tensor(
                coefficients[ids[:1] if layout == "block_uniform" else ids],
                dimension={},
            )
            for cell_type, ids in cell_ids.items()
        }
    actual = integrate(supplied, **kwargs)
    for cell_type, ids in cell_ids.items():
        block = mesh.topology.cell_block(cell_type)
        np.testing.assert_array_equal(block.global_indices, ids)
        base = backend.to_torch(unit[cell_type])
        scale = base.new_tensor(scales[ids]).reshape(
            (2,) + (1,) * (base.ndim - 2) + (2,)
        )
        _assert_close(backend.to_torch(actual[cell_type]), base * scale)


@pytest.mark.parametrize("n_components", [1, 3])
@pytest.mark.parametrize("subset", [False, True])
def test_assembly_and_matvec_match_dense_reference(
    mixed_mesh: TensorMesh, n_components: int, subset: bool
):
    """Dense assembly checks shared points and component ordering."""
    mesh = mixed_mesh
    backend = mesh.backend
    local = {}
    expected = np.zeros((mesh.n_points * n_components,) * 2)
    types = (
        (pv.CellType.HEXAHEDRON,)
        if subset
        else (pv.CellType.HEXAHEDRON, pv.CellType.TETRA)
    )
    for cell_type in types:
        block = mesh.topology.cell_block(cell_type)
        n_nodes = block.conn.shape[1]
        shape = (block.n_cells, n_nodes, n_nodes)
        if n_components > 1:
            shape += (n_components, n_components)
        # Nonsymmetric entries expose swapped node and component axes.
        values = (
            np.arange(1, math.prod(shape) + 1, dtype=np.float64).reshape(shape)
            / 100
        )
        local[cell_type] = backend.as_tensor(values[..., None], dimension={})
        for cell, nodes in enumerate(block.conn):
            dofs = (
                nodes[:, None] * n_components + np.arange(n_components)
            ).reshape(-1)
            matrix = values[cell]
            if n_components > 1:
                matrix = matrix.transpose(0, 2, 1, 3).reshape(
                    len(dofs), len(dofs)
                )
            expected[np.ix_(dofs, dofs)] += matrix

    assembled = backend.to_torch(mesh.geometry.global_fem_matrix(local))
    assert assembled.layout == torch.sparse_coo
    assert assembled.is_coalesced()
    _assert_close(assembled.to_dense(), assembled.new_tensor(expected))
    shape = (
        (mesh.n_points, 1)
        if n_components == 1
        else (mesh.n_points, n_components, 1)
    )
    field = backend.as_tensor(
        np.linspace(-1, 2, math.prod(shape)).reshape(shape), dimension={}
    )
    actual = backend.to_torch(
        mesh.geometry.apply_cell_local_fem_matrix(local, field)
    )
    target = assembled.new_tensor(expected) @ backend.to_torch(field).reshape(
        -1, 1
    )
    _assert_close(actual.reshape(-1, 1), target)
    _assert_close(actual[-1], torch.zeros_like(actual[-1]))
    if subset:
        _assert_close(actual[12:14], torch.zeros_like(actual[12:14]))


@pytest.mark.parametrize(
    "cell_type", [pv.CellType.HEXAHEDRON, pv.CellType.QUADRATIC_TETRA]
)
@pytest.mark.parametrize("vector_rank", [0, 1, 2])
def test_scalar_mass_applies_to_each_component_and_feature(
    cell_type: pv.CellType, vector_rank: int, backend: Backend
):
    mesh = _element_mesh(cell_type, backend)
    # A singleton material feature broadcasts over the field's two features.
    local = mesh.geometry.cell_local_fem_mass()[cell_type]
    shape = (mesh.n_points,) + (3,) * vector_rank + (2,)
    field = backend.as_tensor(
        np.arange(math.prod(shape), dtype=np.float64).reshape(shape),
        dimension={},
    )
    matrix = backend.to_torch(local)[0, ..., 0]
    expected = (
        matrix @ backend.to_torch(field).reshape(mesh.n_points, -1)
    ).reshape(shape)
    for ranks in ({}, {"vector_rank": vector_rank, "matrix_rank": 0}):
        actual = mesh.geometry.apply_cell_local_fem_matrix(
            local, field, **ranks
        )
        _assert_close(backend.to_torch(actual), expected)
    assembled = mesh.geometry.global_fem_matrix(local)
    _assert_close(backend.to_torch(assembled).to_dense(), matrix)


@pytest.mark.parametrize("operator", ["mass", "rigidity"])
@pytest.mark.parametrize(
    "invalid", ["missing", "extra", "block_size", "mesh_size"]
)
def test_material_layout_validation(
    mixed_mesh: TensorMesh, operator: str, invalid: str
):
    mesh = mixed_mesh
    backend = mesh.backend
    supplied = {
        cell_type: backend.ones((1, 1), dimension={})
        for cell_type in (pv.CellType.TETRA, pv.CellType.HEXAHEDRON)
    }
    if invalid == "missing":
        del supplied[pv.CellType.TETRA]
    elif invalid == "extra":
        supplied[pv.CellType.TRIANGLE] = backend.ones((1, 1), dimension={})
    elif invalid == "block_size":
        supplied[pv.CellType.TETRA] = backend.ones(
            (mesh.n_cells, 1), dimension={}
        )
    else:
        supplied = backend.ones((2, 1), dimension={})
    match = (
        "mapping keys" if invalid in ("missing", "extra") else "leading size"
    )
    with pytest.raises(ValueError, match=match):
        getattr(mesh.geometry, f"cell_local_fem_{operator}")(supplied)


@pytest.mark.parametrize("assemble", [False, True])
def test_mixed_assembly_requires_mapping(
    mixed_mesh: TensorMesh, assemble: bool
):
    mesh = mixed_mesh
    local = mesh.geometry.cell_local_fem_mass()[pv.CellType.TETRA]
    with pytest.raises(ValueError, match="more than one cell type"):
        if assemble:
            mesh.geometry.global_fem_matrix(local)
        else:
            mesh.geometry.apply_cell_local_fem_matrix(
                local, mesh.backend.ones((mesh.n_points, 1))
            )


@pytest.mark.parametrize("assemble", [False, True])
@pytest.mark.parametrize("node_axes", [(4, 4), (8, 4), (4, 8)])
def test_assembly_rejects_wrong_node_axes(
    backend: Backend, assemble: bool, node_axes: tuple[int, int]
):
    mesh = _element_mesh(pv.CellType.HEXAHEDRON, backend)
    local = backend.ones((1, *node_axes, 1), dimension={})
    with pytest.raises(ValueError, match="node axes"):
        if assemble:
            mesh.geometry.global_fem_matrix(local)
        else:
            mesh.geometry.apply_cell_local_fem_matrix(
                local, backend.ones((mesh.n_points, 1))
            )


@pytest.mark.parametrize("operator", ["mass", "rigidity"])
@pytest.mark.parametrize("name", ["complex", "mix_poly"])
def test_variable_connectivity_is_unsupported(
    operator: str, name: str, backend: Backend
):
    mesh = graphlow.read(
        f"tests/data/vtu/{name}/mesh.vtu",
        backend.name,
        dtype=backend.dtype,
        device=backend.device,
    )
    with pytest.raises(
        NotImplementedError, match="fixed-connectivity.*POLYHEDRON"
    ):
        getattr(mesh.geometry, f"cell_local_fem_{operator}")()


@pytest.mark.parametrize("operator", ["mass", "rigidity"])
def test_unsupported_quadrature_order(operator: str, backend: Backend):
    mesh = _element_mesh(pv.CellType.TRIANGLE, backend)
    with pytest.raises(
        NotImplementedError, match="Unsupported quadrature order"
    ):
        getattr(mesh.geometry, f"cell_local_fem_{operator}")(quadrature_order=5)


@pytest.mark.parametrize("operator", ["mass", "rigidity"])
def test_fixed_connectivity_without_quadrature_is_unsupported(
    operator: str, backend: Backend
):
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
        dtype=np.float64,
    )
    mesh = _from_grid(
        pv.UnstructuredGrid({pv.CellType.WEDGE: np.arange(6)[None]}, points),
        backend,
    )
    with pytest.raises(NotImplementedError, match="No quadrature rule"):
        getattr(mesh.geometry, f"cell_local_fem_{operator}")()


@pytest.mark.parametrize("rank", [1, 3, 5])
def test_unsupported_material_rank(rank: int, backend: Backend):
    mesh = _element_mesh(pv.CellType.HEXAHEDRON, backend)
    with pytest.raises(NotImplementedError, match=f"Unsupported rank: {rank}"):
        mesh.geometry.cell_local_fem_rigidity(rank=rank)


@pytest.mark.parametrize(
    "shape,error",
    [
        ((1, 8, 8, 2), "last dim should be 1"),
        ((1, 8, 8, 3, 1), "Unexpected rank"),
    ],
)
def test_global_assembly_rejects_unsupported_layout(
    shape: tuple[int, ...], error: str, backend: Backend
):
    mesh = _element_mesh(pv.CellType.HEXAHEDRON, backend)
    with pytest.raises(NotImplementedError, match=error):
        mesh.geometry.global_fem_matrix(backend.ones(shape, dimension={}))


@pytest.mark.parametrize("vector_rank,matrix_rank", [(0, 2), (2, 2), (1, 1)])
def test_matvec_rejects_unsupported_rank_combination(
    vector_rank: int, matrix_rank: int, backend: Backend
):
    mesh = _element_mesh(pv.CellType.HEXAHEDRON, backend)
    local = backend.ones((1, 8, 8) + (3,) * matrix_rank + (1,), dimension={})
    field = backend.ones(
        (mesh.n_points,) + (3,) * vector_rank + (1,), dimension={}
    )
    with pytest.raises(NotImplementedError, match="Unexpected combination"):
        mesh.geometry.apply_cell_local_fem_matrix(local, field)


@pytest.mark.parametrize(
    "shape,offset,expected",
    [
        ((5, 2), 0, 0),
        ((5, 3, 2), 0, 1),
        ((2, 8, 8, 1), 2, 0),
        ((2, 8, 8, 3, 3, 1), 2, 2),
    ],
)
def test_rank_inference(
    shape: tuple[int, ...], offset: int, expected: int, backend: Backend
):
    assert _get_rank(backend.ones(shape), offset=offset) == expected


def test_rank_respects_phlower_time_axis():
    series = phlower_tensor(torch.ones(2, 5, 3, 1), is_time_series=True)
    assert _get_rank(series) == 1
    assert _get_rank(series.to_tensor()) == 2


def test_explicit_rank_overrides_inference(backend: Backend):
    assert _get_rank(backend.ones((5,)), rank=1, offset=2) == 1


def test_negative_inferred_rank_is_rejected(backend: Backend):
    with pytest.raises(ValueError, match="Rank is negative"):
        _get_rank(backend.ones((5, 3)), offset=2)

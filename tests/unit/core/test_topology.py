"""Unit tests for MeshTopology (core/topology.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import phlower_tensor as pt
import pytest
import pyvista as pv
import torch
from tests.unit.conftest import BackendParams

import graphlow
from graphlow.core.mesh import TensorMesh
from graphlow.graph.skeleton_builder import AdjacencyName, IncidenceName
from graphlow.graph.skeleton_derived import DerivedMatrixName
from graphlow.io.pyvista import from_pyvista
from graphlow.utils.topology_helper import TopologyDim

EitherTensor = torch.Tensor | pt.PhlowerTensor


@pytest.fixture
def triangle_mesh(bparam: BackendParams) -> TensorMesh[EitherTensor]:
    """Single triangle: 3 points, 1 cell (2D, no volume)."""
    pytest.importorskip("phlower_tensor")
    pts = np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]], dtype=np.float64)
    cells = np.array([3, 0, 1, 2])
    ctypes = np.array([5])
    grid = pv.UnstructuredGrid(cells, ctypes, pts)
    return from_pyvista(grid, bparam.name, bparam.dtype, device=bparam.device)


@pytest.fixture
def complex_mesh(
    data_dir: Path, bparam: BackendParams
) -> TensorMesh[EitherTensor]:
    """Complex mesh: 6 polyhedrons."""
    pytest.importorskip("phlower_tensor")
    path = data_dir / "vtu" / "complex" / "mesh.vtu"
    mesh = graphlow.read(
        path, bparam.name, dtype=bparam.dtype, device=bparam.device
    )
    return mesh


# =============================================================================
# Test MeshTopology properties / basic info
# =============================================================================
def test_topology_n_points_n_cells_delegate_to_mesh(
    tet_mesh: TensorMesh[EitherTensor],
):
    """MeshTopology.n_points and n_cells are defined."""
    topo = tet_mesh.topology
    assert topo.n_points == tet_mesh.n_points == 4
    assert topo.n_cells == tet_mesh.n_cells == 1


def test_connectivity(tet_mesh: TensorMesh[EitherTensor]):
    """MeshTopology.cell_conn() and cell_offsets() delegate to mesh."""
    topo = tet_mesh.topology
    expected_conn = np.array([[0, 1, 2, 3]])
    expected_offsets = np.array([0, 4])
    expected_types = np.array([pv.CellType.TETRA.value])
    assert np.all(topo.cell_conn() == expected_conn)
    assert np.all(topo.cell_offsets() == expected_offsets)
    assert np.all(topo.cell_types() == expected_types)


def test_tet_connectivity(tet_mesh: TensorMesh[EitherTensor]):
    """MeshTopology.cell_tet_conn() returns 2-dim connectivity."""
    topo = tet_mesh.topology
    actual_tet_conn = topo.cell_tet_conn()
    desired_tet_conn = np.array([[0, 1, 2, 3]])
    assert len(actual_tet_conn.shape) == 2
    np.testing.assert_array_equal(actual_tet_conn, desired_tet_conn)


def test_tet_conn_raises_when_not_tet(complex_mesh: TensorMesh[EitherTensor]):
    """MeshTopology.cell_tet_conn() raises when the mesh is not pure tet."""
    topo = complex_mesh.topology
    with pytest.raises(
        ValueError, match="cell_tet_conn not supported for cell types"
    ):
        topo.cell_tet_conn()


def test_unique_cell_types(complex_mesh: TensorMesh[EitherTensor]):
    """MeshTopology.unique_cell_types() returns unique cell types."""
    topo = complex_mesh.topology
    expected_types = np.array([pv.CellType.POLYHEDRON.value])
    assert np.all(topo.unique_cell_types() == expected_types)


def test_mesh_dim(complex_mesh: TensorMesh[EitherTensor]):
    """MeshTopology.mesh_dim() returns the maximum topological dimension."""
    topo = complex_mesh.topology
    assert topo.mesh_dim() == TopologyDim.VOLUME


def test_face_registry_raises_for_less_than_volume(
    triangle_mesh: TensorMesh[EitherTensor],
):
    """face_registry() raises when mesh_dim < VOLUME."""
    topo = triangle_mesh.topology
    with pytest.raises(ValueError, match="less than volume"):
        topo.face_registry()


def test_invalidate_clears_caches(tet_mesh: TensorMesh[EitherTensor]):
    """invalidate() clears topology caches."""
    topo = tet_mesh.topology
    a = topo.unique_cell_types()
    b = topo.unique_cell_types()
    assert a is b  # same cached array
    topo.invalidate()
    c = topo.unique_cell_types()
    assert c is not a  # cache was cleared, so recomputed and new array
    assert int(c[0]) == pv.CellType.TETRA


def test_cell_block_returns_none_for_missing_cell_type(
    tet_mesh: TensorMesh[EitherTensor],
):
    """cell_block(cell_type) returns None when cell type not in mesh."""
    topo = tet_mesh.topology
    # Tet mesh has no hexahedron.
    assert topo.cell_block(pv.CellType.HEXAHEDRON) is None
    assert topo.cell_block(pv.CellType.TETRA) is not None


def test_get_derived_skeleton_raises_for_unknown_op(
    tet_mesh: TensorMesh[EitherTensor],
):
    """get_derived_skeleton(op, name) raises for unknown DerivedMatrixName."""
    topo = tet_mesh.topology
    with pytest.raises(ValueError, match="Unknown derived"):
        topo.get_derived_skeleton(
            "UNKNOWN_OP",
            AdjacencyName.PP,
        )


# =============================================================================
# Host skeleton / block caches
# =============================================================================
def test_get_skeleton_returns_cached_matrix(tet_mesh: TensorMesh[EitherTensor]):
    """Second get_skeleton(CP) returns the same scipy CSR object."""
    topo = tet_mesh.topology
    a = topo.get_skeleton(IncidenceName.CP)
    b = topo.get_skeleton(IncidenceName.CP)
    assert a is b


def test_get_derived_skeleton_returns_cached_matrix(
    tet_mesh: TensorMesh[EitherTensor],
):
    """Derived skeleton cache reuses the same matrix for repeated calls."""
    topo = tet_mesh.topology
    a = topo.get_derived_skeleton(DerivedMatrixName.DEGREE, AdjacencyName.PP)
    b = topo.get_derived_skeleton(DerivedMatrixName.DEGREE, AdjacencyName.PP)
    assert a is b


def test_cell_block_returns_cached_block(tet_mesh: TensorMesh[EitherTensor]):
    """cell_block(TETRA) returns the same CellBlock instance when cached."""
    topo = tet_mesh.topology
    a = topo.cell_block(pv.CellType.TETRA)
    b = topo.cell_block(pv.CellType.TETRA)
    assert a is not None and a is b


def test_face_registry_returns_cached_registry(
    tet_mesh: TensorMesh[EitherTensor],
):
    """face_registry() returns the same FaceRegistry on second call."""
    topo = tet_mesh.topology
    a = topo.face_registry()
    b = topo.face_registry()
    assert a is b


def test_face_block_cached(tet_mesh: TensorMesh[EitherTensor]):
    """face_block(TRIANGLE) returns the same FaceBlock when cached."""
    topo = tet_mesh.topology
    a = topo.face_block(pv.CellType.TRIANGLE, boundary_only=False)
    b = topo.face_block(pv.CellType.TRIANGLE, boundary_only=False)
    assert a is not None and a is b


# =============================================================================
# free_host_skeletons
# =============================================================================
def test_free_host_skeletons_none_rebuilds_skeleton(
    tet_mesh: TensorMesh[EitherTensor],
):
    """free_host_skeletons(None) clears host skeletons; next get builds anew."""
    topo = tet_mesh.topology
    first = topo.get_skeleton(IncidenceName.CP)
    topo.free_host_skeletons(None)
    second = topo.get_skeleton(IncidenceName.CP)
    assert first is not second


def test_free_host_skeletons_single_name_keeps_other_skeletons(
    tet_mesh: TensorMesh[EitherTensor],
):
    """free_host_skeletons('cp') drops CP only; PC remains cached."""
    topo = tet_mesh.topology
    cp0 = topo.get_skeleton(IncidenceName.CP)
    pc = topo.get_skeleton(IncidenceName.PC)
    topo.free_host_skeletons(str(IncidenceName.CP))
    cp1 = topo.get_skeleton(IncidenceName.CP)
    pc_again = topo.get_skeleton(IncidenceName.PC)
    assert cp0 is not cp1
    assert pc is pc_again


# =============================================================================
# Backend sparse cache (bcache) and free_backend_sparse
# =============================================================================
def test_cell_point_incidence_reuses_backend_cache(
    tet_mesh: TensorMesh[EitherTensor],
):
    """Repeated cell_point_incidence returns the same cached tensor."""
    topo = tet_mesh.topology
    a = topo.cell_point_incidence(layout="coo")
    b = topo.cell_point_incidence(layout="coo")
    assert a is b


def test_free_backend_sparse_named_drops_entry_materializes_fresh(
    tet_mesh: TensorMesh[EitherTensor],
):
    """Named free_backend_sparse drops cache; next materialization is new."""
    topo = tet_mesh.topology
    first = topo.cell_point_incidence(layout="coo")
    topo.free_backend_sparse(str(IncidenceName.CP), layout="coo")
    second = topo.cell_point_incidence(layout="coo")
    assert first is not second


def test_free_backend_sparse_none_clears_all_backend_materializations(
    tet_mesh: TensorMesh[EitherTensor],
):
    """free_backend_sparse(None) invalidates the whole backend sparse cache."""
    topo = tet_mesh.topology
    cp = topo.cell_point_incidence(layout="coo")
    pc = topo.point_cell_incidence(layout="coo")
    topo.free_backend_sparse(None)
    cp2 = topo.cell_point_incidence(layout="coo")
    pc2 = topo.point_cell_incidence(layout="coo")
    assert cp is not cp2
    assert pc is not pc2


# =============================================================================
# invalidate vs skeletons and backend cache
# =============================================================================
def test_invalidate_clears_host_skeleton_and_backend_sparse(
    tet_mesh: TensorMesh[EitherTensor],
):
    """invalidate() forces new host skeletons and new backend sparse tensors."""
    topo = tet_mesh.topology
    sk0 = topo.get_skeleton(IncidenceName.CP)
    t0 = topo.cell_point_incidence(layout="coo")
    topo.invalidate()
    sk1 = topo.get_skeleton(IncidenceName.CP)
    t1 = topo.cell_point_incidence(layout="coo")
    assert sk0 is not sk1
    assert t0 is not t1

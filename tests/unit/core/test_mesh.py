"""Unit tests for TensorMesh (core/mesh.py): only what is defined there."""

import pathlib

import numpy as np
import phlower_tensor as pt
import pytest
import pyvista as pv
import torch
from tests.unit.conftest import BackendParams

import graphlow
from graphlow.core.mesh import TensorMesh
from graphlow.utils.enums import FeatureName


# =============================================================================
# Test TensorMesh properties
# =============================================================================
def test_n_points(tet_mesh: TensorMesh):
    """TensorMesh.n_points returns points.shape[0] (defined in mesh.py)."""
    assert tet_mesh.n_points == 4


def test_n_cells(tet_mesh: TensorMesh):
    """TensorMesh.n_cells returns pvmesh.n_cells (defined in mesh.py)."""
    assert tet_mesh.n_cells == 1


# =============================================================================
# Test requires_grad
# =============================================================================
EitherTensor = torch.Tensor | pt.PhlowerTensor


@pytest.fixture
def tet_mesh_with_data(
    tet_mesh: TensorMesh[EitherTensor],
) -> TensorMesh[EitherTensor]:
    tet_mesh.point_data["scalar"] = tet_mesh.backend.ones(
        (tet_mesh.n_points, 1)
    )
    tet_mesh.point_data["point_ids"] = tet_mesh.backend.as_tensor(
        np.arange(tet_mesh.n_points, dtype=np.int32)
    )
    tet_mesh.cell_data["value"] = tet_mesh.backend.ones((tet_mesh.n_cells, 1))
    tet_mesh.cell_data["cell_ids"] = tet_mesh.backend.as_tensor(
        np.arange(tet_mesh.n_cells, dtype=np.int32)
    )
    return tet_mesh


def test_requires_grad_points_only(tet_mesh: TensorMesh[EitherTensor]):
    """requires_grad(True/False) sets points.requires_grad."""
    backend = tet_mesh.backend
    tet_mesh.point_data["scalar"] = tet_mesh.backend.ones(
        (tet_mesh.n_points, 1)
    )
    tet_mesh.cell_data["value"] = tet_mesh.backend.ones((tet_mesh.n_cells, 1))
    assert not backend.to_torch(tet_mesh.points).requires_grad
    assert not backend.to_torch(tet_mesh.point_data["scalar"]).requires_grad
    assert not backend.to_torch(tet_mesh.cell_data["value"]).requires_grad

    tet_mesh.requires_grad(True)
    assert backend.to_torch(tet_mesh.points).requires_grad
    assert not backend.to_torch(tet_mesh.point_data["scalar"]).requires_grad
    assert not backend.to_torch(tet_mesh.cell_data["value"]).requires_grad


def test_requires_grad_all(tet_mesh_with_data: TensorMesh[EitherTensor]):
    """
    requires_grad(True, point_data=True, cell_data=True) sets all tensors.
    """
    tet_mesh = tet_mesh_with_data
    backend = tet_mesh.backend
    assert not backend.to_torch(tet_mesh.points).requires_grad
    assert not backend.to_torch(tet_mesh.point_data["scalar"]).requires_grad
    assert not backend.to_torch(tet_mesh.point_data["point_ids"]).requires_grad
    assert not backend.to_torch(tet_mesh.cell_data["value"]).requires_grad
    assert not backend.to_torch(tet_mesh.cell_data["cell_ids"]).requires_grad

    tet_mesh.requires_grad(True, point_data=True, cell_data=True)
    assert backend.to_torch(tet_mesh.points).requires_grad
    assert backend.to_torch(tet_mesh.point_data["scalar"]).requires_grad
    assert not backend.to_torch(tet_mesh.point_data["point_ids"]).requires_grad
    assert backend.to_torch(tet_mesh.cell_data["value"]).requires_grad
    assert not backend.to_torch(tet_mesh.cell_data["cell_ids"]).requires_grad


def test_to_updates_dtype(
    tet_mesh_with_data: TensorMesh[EitherTensor],
) -> None:
    """to updates dtype in-place."""
    tet_mesh = tet_mesh_with_data
    backend = tet_mesh.backend
    # check that the cell adjacency's dtype is the same as the backend
    assert tet_mesh_with_data.topology.cell_adjacency().dtype == backend.dtype

    # switch the dtype
    target_dtype = (
        torch.float64 if backend.dtype == torch.float32 else torch.float32
    )
    returned = tet_mesh.to(
        device=backend.device,
        non_blocking=True,
        dtype=target_dtype,
    )

    assert returned is tet_mesh
    assert tet_mesh.backend.dtype == target_dtype
    assert tet_mesh.points.dtype == target_dtype
    assert tet_mesh.point_data["scalar"].dtype == target_dtype
    assert tet_mesh.cell_data["value"].dtype == target_dtype
    # check that integer tensors are not cast to float
    assert tet_mesh.point_data["point_ids"].dtype == torch.int32
    assert tet_mesh.cell_data["cell_ids"].dtype == torch.int32
    # check that backend cache is also updated
    assert tet_mesh.topology.cell_adjacency().dtype == target_dtype


def test_to_updates_device_cycle(
    tet_mesh_with_data: TensorMesh[EitherTensor],
) -> None:
    """to updates device in-place, cpu -> cuda -> cpu."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    tet_mesh = tet_mesh_with_data
    backend = tet_mesh_with_data.backend
    # check that the cell adjacency's device is the same as the backend
    assert tet_mesh.topology.cell_adjacency().device.type == backend.device.type

    # switch the device
    target_device = (
        torch.device("cuda")
        if backend.device.type == "cpu"
        else torch.device("cpu")
    )
    returned = tet_mesh.to(device=target_device, non_blocking=True)
    assert returned is tet_mesh_with_data
    assert tet_mesh.backend.device == target_device
    assert tet_mesh.points.device.type == target_device.type
    assert tet_mesh.point_data["scalar"].device.type == target_device.type
    assert tet_mesh.cell_data["value"].device.type == target_device.type
    assert tet_mesh.point_data["point_ids"].device.type == target_device.type
    assert tet_mesh.cell_data["cell_ids"].device.type == target_device.type
    # check that backend cache is also updated
    assert tet_mesh.topology.cell_adjacency().device.type == target_device.type

    # switch the device back
    returned = tet_mesh.to(device=backend.device, non_blocking=True)
    assert returned is tet_mesh
    assert tet_mesh.backend.device == backend.device
    assert tet_mesh.points.device.type == backend.device.type
    assert tet_mesh.point_data["scalar"].device.type == backend.device.type
    assert tet_mesh.cell_data["value"].device.type == backend.device.type
    assert tet_mesh.point_data["point_ids"].device.type == backend.device.type
    assert tet_mesh.cell_data["cell_ids"].device.type == backend.device.type
    # check that backend cache is also updated
    assert tet_mesh.topology.cell_adjacency().device.type == backend.device.type


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
def test_to_invalid_dtype(
    tet_mesh_with_data: TensorMesh[EitherTensor],
    dtype: torch.dtype,
) -> None:
    """to raises ValueError for invalid dtype."""
    with pytest.raises(ValueError):
        tet_mesh_with_data.to(dtype=dtype)


# =============================================================================
# Test extract_surface
# =============================================================================
@pytest.fixture
def volume_mesh(
    data_dir: pathlib.Path, bparam: BackendParams
) -> TensorMesh[EitherTensor]:
    """Complex volume mesh (phlower backend)."""
    path = data_dir / "vtu" / "complex" / "mesh.vtu"
    mesh = graphlow.read(
        path,
        bparam.name,
        dtype=bparam.dtype,
        device=bparam.device,
    )
    mesh.point_data.update(
        {
            "T": mesh.backend.as_tensor(
                np.random.rand(mesh.n_points, 1), dimension={"Theta": 1}
            ),
        }
    )
    return mesh


def test_extract_surface_has_parent_point_map(
    volume_mesh: TensorMesh[EitherTensor],
) -> None:
    """extract_surface has parent_point_map."""
    assert volume_mesh._parent_point_map is None
    surface = volume_mesh.extract_surface()
    assert surface._parent_point_map is not None
    assert FeatureName.ORIGINAL_INDEX not in volume_mesh.point_data
    assert FeatureName.ORIGINAL_INDEX not in volume_mesh.pvmesh.point_data
    assert FeatureName.ORIGINAL_INDEX not in surface.point_data
    assert FeatureName.ORIGINAL_INDEX not in surface.pvmesh.point_data


def test_extract_surface_gradient_flows_to_volume(
    volume_mesh: TensorMesh[EitherTensor],
) -> None:
    """Backward through surface face_areas flows to volume points."""
    volume_mesh.requires_grad(True)
    surface = volume_mesh.extract_surface()
    area = surface.geometry.face_areas()
    loss = torch.sum(area)
    loss.backward()
    grad = volume_mesh.backend.to_torch(volume_mesh.points).grad
    assert grad is not None
    assert grad.shape == volume_mesh.points.shape


def test_extract_surface_keep_graph_false_stops_gradient(
    volume_mesh: TensorMesh[EitherTensor],
) -> None:
    """If keep_graph=False, gradients do not flow back to the volume mesh."""
    volume_mesh.requires_grad(True)
    surface = volume_mesh.extract_surface(keep_graph=False)
    assert volume_mesh.backend.to_torch(surface.points).requires_grad is False

    loss = torch.sum(surface.points)
    with pytest.raises(RuntimeError):
        loss.backward()
    assert volume_mesh.backend.to_torch(volume_mesh.points).grad is None


# =============================================================================
# Test copy_features_to_pyvista
# =============================================================================
def test_copy_features_to_pyvista(
    tet_mesh: TensorMesh[EitherTensor],
) -> None:
    """
    copy_features_to_pyvista transfers backend tensors to the PyVista mesh.
    """
    backend = tet_mesh.backend
    point_values = np.arange(tet_mesh.n_points, dtype=np.float64)[:, None]
    cell_values = np.array([[3.0]], dtype=np.float64)
    tet_mesh.point_data["temperature"] = backend.as_tensor(
        point_values, dimension={"Theta": 1}
    )
    tet_mesh.cell_data["quality"] = backend.as_tensor(
        cell_values, dimension={"L": 1}
    )

    tet_mesh.copy_features_to_pyvista()

    np.testing.assert_allclose(
        tet_mesh.pvmesh.point_data["temperature"], point_values[:, 0]
    )
    np.testing.assert_allclose(
        tet_mesh.pvmesh.cell_data["quality"], cell_values[:, 0]
    )


def test_copy_features_to_pyvista_raises_on_conflicting_keys(
    tet_mesh: TensorMesh[EitherTensor],
) -> None:
    """copy_features_to_pyvista rejects collisions unless overwrite=True."""
    backend = tet_mesh.backend
    tet_mesh.point_data["temperature"] = backend.as_tensor(
        np.ones((tet_mesh.n_points, 1), dtype=np.float64),
        dimension={"Theta": 1},
    )
    tet_mesh.pvmesh.point_data["temperature"] = np.zeros(
        (tet_mesh.n_points, 1), dtype=np.float64
    )

    with pytest.raises(ValueError, match="Keys already exist"):
        tet_mesh.copy_features_to_pyvista()


def test_copy_features_to_pyvista_overwrite_existing_arrays(
    tet_mesh: TensorMesh[EitherTensor],
) -> None:
    """copy_features_to_pyvista(overwrite=True) replaces PyVista arrays."""
    backend = tet_mesh.backend
    expected = np.arange(tet_mesh.n_points, dtype=np.float64)[:, None]
    tet_mesh.point_data["temperature"] = backend.as_tensor(
        expected, dimension={"Theta": 1}
    )
    tet_mesh.pvmesh.point_data["temperature"] = np.full(
        (tet_mesh.n_points, 1), -1.0, dtype=np.float64
    )

    tet_mesh.copy_features_to_pyvista(overwrite=True)

    np.testing.assert_allclose(
        tet_mesh.pvmesh.point_data["temperature"], expected[:, 0]
    )


# =============================================================================
# Test copy_features_from_pyvista
# =============================================================================
def test_copy_features_from_pyvista_with_dimensions(
    tet_mesh: TensorMesh[EitherTensor],
) -> None:
    """copy_features_from_pyvista preserves provided dimension metadata."""
    backend = tet_mesh.backend
    point_values = np.arange(tet_mesh.n_points, dtype=np.float64)[:, None]
    cell_values = np.array([[2.5]], dtype=np.float64)
    tet_mesh.pvmesh.point_data["temperature"] = point_values
    tet_mesh.pvmesh.cell_data["quality"] = cell_values

    tet_mesh.copy_features_from_pyvista(
        dimension_collection={
            "temperature": {"Theta": 1},
            "quality": {"L": 1},
        }
    )

    np.testing.assert_allclose(
        backend.to_numpy(tet_mesh.point_data["temperature"]), point_values[:, 0]
    )
    np.testing.assert_allclose(
        backend.to_numpy(tet_mesh.cell_data["quality"]), cell_values[:, 0]
    )

    if backend.name == "phlower":
        temperature_dimension = tet_mesh.point_data["temperature"].dimension
        quality_dimension = tet_mesh.cell_data["quality"].dimension

        expected_temperature_dimension = pt.phlower_dimension_tensor(
            {"Theta": 1},
            device=backend.device,
        )
        expected_quality_dimension = pt.phlower_dimension_tensor(
            {"L": 1},
            device=backend.device,
        )

        assert temperature_dimension == expected_temperature_dimension
        assert quality_dimension == expected_quality_dimension


def test_copy_features_from_pyvista_raises_on_conflicting_keys(
    tet_mesh: TensorMesh[EitherTensor],
) -> None:
    """copy_features_from_pyvista rejects collisions unless overwrite=True."""
    backend = tet_mesh.backend
    tet_mesh.point_data["temperature"] = backend.as_tensor(
        np.ones((tet_mesh.n_points, 1), dtype=np.float64),
        dimension={"Theta": 1},
    )
    tet_mesh.pvmesh.point_data["temperature"] = np.zeros(
        (tet_mesh.n_points, 1), dtype=np.float64
    )

    with pytest.raises(ValueError, match="Keys already exist"):
        tet_mesh.copy_features_from_pyvista()


def test_copy_features_from_pyvista_overwrite_existing_arrays(
    tet_mesh: TensorMesh[EitherTensor],
) -> None:
    """copy_features_from_pyvista(overwrite=True) updates backend tensors."""
    backend = tet_mesh.backend
    tet_mesh.point_data["temperature"] = backend.as_tensor(
        np.full((tet_mesh.n_points, 1), -1.0, dtype=np.float64),
        dimension={"Theta": 1},
    )
    tet_mesh.pvmesh.point_data["temperature"] = np.arange(
        tet_mesh.n_points, dtype=np.float64
    )[:, None]

    tet_mesh.copy_features_from_pyvista(
        overwrite=True,
        dimension_collection={"temperature": {"Theta": 1}},
    )

    np.testing.assert_allclose(
        backend.to_numpy(tet_mesh.point_data["temperature"]),
        np.arange(tet_mesh.n_points, dtype=np.float64),
    )


# =============================================================================
# Test save
# =============================================================================
def test_save_raises_on_overwrite_file(
    tet_mesh: TensorMesh[EitherTensor], tmp_path: pathlib.Path
) -> None:
    """
    save raises if the target file already exists and overwrite is disabled.
    """
    output = tmp_path / "mesh.vtu"
    output.write_text("test file")

    with pytest.raises(ValueError, match="already exists"):
        tet_mesh.save(output)


def test_save_without_cast(
    tet_mesh: TensorMesh[EitherTensor], tmp_path: pathlib.Path
) -> None:
    """
    save(..., cast=False) writes the current grid.
    """
    output = tmp_path / "nested" / "mesh.vtu"
    tet_mesh.save(output, cast=False)
    assert output.exists()
    saved = pv.read(output)
    assert isinstance(saved, pv.UnstructuredGrid)
    assert saved.n_cells == tet_mesh.n_cells


def test_save_with_cast(
    tet_mesh: TensorMesh[EitherTensor], tmp_path: pathlib.Path
) -> None:
    """
    save(..., cast=True) writes the extracted surface for VTP output.
    """
    output = tmp_path / "nested" / "mesh.vtp"
    tet_mesh.save(output, cast=True)
    assert output.exists()
    saved = pv.read(output)
    assert isinstance(saved, pv.PolyData)
    assert saved.n_cells == 4


def test_save_removes_time_field(
    tet_mesh: TensorMesh[EitherTensor], tmp_path: pathlib.Path
) -> None:
    """
    save writes VTU output, drops TimeValue by default.
    """
    backend = tet_mesh.backend
    output = tmp_path / "out" / "mesh.vtu"
    expected = np.arange(tet_mesh.n_points, dtype=np.float64)[:, None]
    tet_mesh.point_data["temperature"] = backend.as_tensor(
        expected, dimension={"Theta": 1}
    )
    tet_mesh.pvmesh.point_data["temperature"] = np.full(
        (tet_mesh.n_points, 1), -1.0, dtype=np.float64
    )
    tet_mesh.pvmesh.field_data[FeatureName.TIME_VALUE] = np.array([1.0])

    tet_mesh.save(output, overwrite_features=True)

    saved = pv.read(output)
    assert output.exists()
    np.testing.assert_allclose(saved.point_data["temperature"], expected[:, 0])
    assert FeatureName.TIME_VALUE not in tet_mesh.pvmesh.field_data


def test_save_raises_on_unexpected_extension(
    tet_mesh: TensorMesh[EitherTensor], tmp_path: pathlib.Path
) -> None:
    """save rejects file extensions that are not explicitly supported."""
    output = tmp_path / "mesh.invalid"

    with pytest.raises(ValueError, match="Unexpected extension: invalid"):
        tet_mesh.save(output)


# =============================================================================
# Test extract_surface parent-point mapping
# =============================================================================
def test_parent_point_ids_maps_correctly(
    volume_mesh: TensorMesh[EitherTensor],
) -> None:
    """parent_point_ids maps correctly."""
    backend = volume_mesh.backend
    original_points = volume_mesh.backend.to_numpy(volume_mesh.points)
    original_T = volume_mesh.backend.to_numpy(volume_mesh.point_data["T"])
    surface = volume_mesh.extract_surface()
    actual_ids = volume_mesh.backend.to_numpy(surface.parent_point_ids)
    # fmt: off
    expected_ids = np.array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
        # 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
    ])
    # fmt: on
    expected_points = original_points[expected_ids]
    expected_T = original_T[expected_ids]
    np.testing.assert_array_equal(actual_ids, expected_ids)
    np.testing.assert_array_almost_equal(
        backend.to_numpy(surface.points), expected_points
    )
    np.testing.assert_array_almost_equal(
        backend.to_numpy(surface.point_data["T"]), expected_T
    )


def test_parent_point_ids_raises_on_no_parent_point_map(
    volume_mesh: TensorMesh[EitherTensor],
) -> None:
    """parent_point_ids raises on no parent point map."""
    with pytest.raises(
        ValueError, match="This mesh has no parent point mapping."
    ):
        _ = volume_mesh.parent_point_ids


def test_gather_parent_point_data(
    volume_mesh: TensorMesh[EitherTensor],
) -> None:
    """
    gather_parent_point_data gathers parent point data.
    """
    backend = volume_mesh.backend
    surface = volume_mesh.extract_surface()
    expected = volume_mesh.point_data["T"][surface.parent_point_ids]

    actual = surface.gather_parent_point_data(volume_mesh.point_data["T"])

    np.testing.assert_array_equal(
        backend.to_numpy(actual), backend.to_numpy(expected)
    )


def test_gather_parent_point_data_raises_different_n_points(
    volume_mesh: TensorMesh[EitherTensor],
) -> None:
    """gather_parent_point_data raises on different n_points."""
    surface = volume_mesh.extract_surface()
    with pytest.raises(
        ValueError,
        match="must match the parent mesh's n_points.",
    ):
        _ = surface.gather_parent_point_data(volume_mesh.point_data["T"][1:])


def test_scatter_add_to_parent_point_data(
    volume_mesh: TensorMesh[EitherTensor],
) -> None:
    """scatter_add_to_parent_point_data adds surface data to parent slots."""
    backend = volume_mesh.backend
    surface = volume_mesh.extract_surface()
    point_ids = backend.to_numpy(surface.parent_point_ids)
    gathered = surface.gather_parent_point_data(volume_mesh.point_data["T"])

    actual = surface.scatter_add_to_parent_point_data(gathered)
    np_actual = backend.to_numpy(actual)
    expected = np.zeros(actual.shape)
    expected[point_ids] = np_actual[point_ids]

    np.testing.assert_array_equal(np_actual, expected)


def test_scatter_add_to_parent_point_data_raises_different_n_points(
    volume_mesh: TensorMesh[EitherTensor],
) -> None:
    """scatter_add_to_parent_point_data raises on different n_points."""
    surface = volume_mesh.extract_surface()
    gathered = surface.gather_parent_point_data(volume_mesh.point_data["T"])

    with pytest.raises(
        ValueError,
        match="must match this mesh's n_points.",
    ):
        _ = surface.scatter_add_to_parent_point_data(gathered[1:])


def test_scatter_add_to_parent_point_data_raises_different_shape(
    volume_mesh: TensorMesh[EitherTensor],
) -> None:
    """scatter_add_to_parent_point_data raises on different shape."""
    surface = volume_mesh.extract_surface()
    gathered = surface.gather_parent_point_data(volume_mesh.point_data["T"])
    dst_point_data = volume_mesh.point_data["T"][1:]

    with pytest.raises(
        ValueError,
        match="must match the parent point-space shape.",
    ):
        _ = surface.scatter_add_to_parent_point_data(gathered, dst_point_data)

"""Fixtures for unit tests."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import phlower_tensor as pt
import pytest
import pyvista as pv
import torch

import graphlow
from graphlow.core.backend.base import Backend
from graphlow.core.backend.phlower import PhlowerBackend
from graphlow.core.backend.torch import TorchBackend
from graphlow.core.mesh import TensorMesh

logger = logging.getLogger(__name__)


# =============================================================================
# Backend fixture
# =============================================================================
@dataclass
class BackendParams:
    name: Literal["torch", "phlower"]
    dtype: torch.dtype
    device: torch.device


@pytest.fixture
def test_device(request: pytest.FixtureRequest) -> torch.device:
    """Device to run tests on: cpu or cuda."""
    device_name = request.config.getoption("--device")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("Requested --device=cuda, but CUDA is not available")
        return torch.device("cuda")

    return torch.device("cpu")


@pytest.fixture(
    params=[
        ("torch", torch.float32),
        ("phlower", torch.float64),
    ]
)
def bparam(
    request: pytest.FixtureRequest, test_device: torch.device
) -> BackendParams:
    """BackendParams for (name, dtype) on device."""
    name, dtype = request.param
    if name == "phlower":
        pytest.importorskip("phlower_tensor")
    logger.debug(
        "Using backend %s on %s (%s) for test", name, test_device, dtype
    )
    return BackendParams(name=name, dtype=dtype, device=test_device)


@pytest.fixture
def backend(bparam: BackendParams) -> Backend:
    """Backend instance for (name, dtype) on device."""
    if bparam.name == "torch":
        return TorchBackend(dtype=bparam.dtype, device=bparam.device)
    if bparam.name == "phlower":
        return PhlowerBackend(dtype=bparam.dtype, device=bparam.device)
    raise NotImplementedError(f"Unknown backend: {bparam.name}")


@pytest.fixture
def backend_phlower(test_device: torch.device) -> PhlowerBackend:
    pytest.importorskip("phlower_tensor")
    return PhlowerBackend(dtype=torch.float64, device=test_device)


@pytest.fixture
def backend_torch(test_device: torch.device) -> TorchBackend:
    return TorchBackend(dtype=torch.float32, device=test_device)


# =============================================================================
# Mesh fixture
# =============================================================================


@pytest.fixture
def tet_mesh(bparam: BackendParams) -> TensorMesh[pt.PhlowerTensor]:
    """Single tetrahedron: 4 points, 1 cell."""
    pytest.importorskip("phlower_tensor")
    pts = np.array(
        [[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]],
        dtype=np.float64,
    )
    cells = np.array([4, 0, 1, 2, 3])
    ctypes = np.array([10])  # VTK_TETRA
    grid = pv.UnstructuredGrid(cells, ctypes, pts)
    return graphlow.from_pyvista(
        grid,
        bparam.name,
        dtype=bparam.dtype,
        device=bparam.device,
    )


@pytest.fixture
def mix_poly_grid(data_dir: Path) -> pv.UnstructuredGrid:
    """UnstructuredGrid from mix_poly/mesh.vtu (1 polyhedron, 2 other cells)."""
    path = data_dir / "vtu" / "mix_poly" / "mesh.vtu"
    grid = pv.read(path)
    return grid.cast_to_unstructured_grid()


@pytest.fixture
def mix_poly_mesh(
    mix_poly_grid: pv.UnstructuredGrid,
    backend_phlower: Backend[pt.PhlowerTensor],
) -> TensorMesh[pt.PhlowerTensor]:
    """
    TensorMesh built from the mixed-cell fixture using the phlower backend.
    """
    backend = backend_phlower
    return graphlow.from_pyvista(
        mix_poly_grid,
        backend.name,
        backend.dtype,
        device=backend.device,
    )

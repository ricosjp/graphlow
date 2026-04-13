"""PyVista grid → :class:`graphlow.core.mesh.TensorMesh` conversion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, overload

import torch
from pyvista import DataSet, UnstructuredGrid

from graphlow.core.backend.factory import get_backend
from graphlow.core.cache import BackendCache
from graphlow.core.mesh import TensorMesh
from graphlow.utils.enums import DEFAULT_DIMENSIONS, FeatureName, FloatPrecision

if TYPE_CHECKING:
    import phlower_tensor as pt

    from graphlow.core.backend.base import Backend, TensorLike


logger = logging.getLogger(__name__)


@overload
def from_pyvista(
    grid: DataSet,
    backend: Literal["torch"],
    float_precision: FloatPrecision | int = FloatPrecision.FLOAT32,
    *,
    dimension_collection: dict[str, dict[str, float]] | None = None,
    device: torch.device | str | None = None,
    validate_mesh: bool = False,
) -> TensorMesh[torch.Tensor]: ...


@overload
def from_pyvista(
    grid: DataSet,
    backend: Literal["phlower"],
    float_precision: FloatPrecision | int = FloatPrecision.FLOAT32,
    *,
    dimension_collection: dict[str, dict[str, float]] | None = None,
    device: torch.device | str | None = None,
    validate_mesh: bool = False,
) -> TensorMesh[pt.PhlowerTensor]: ...


def from_pyvista(
    grid: DataSet,
    backend: Literal["torch", "phlower"] = "torch",
    float_precision: FloatPrecision | int = FloatPrecision.FLOAT32,
    *,
    dimension_collection: dict[str, dict[str, float]] | None = None,
    device: torch.device | str | None = None,
    validate_mesh: bool = False,
) -> TensorMesh[torch.Tensor] | TensorMesh[pt.PhlowerTensor]:
    """
    Build a :class:`~graphlow.core.mesh.TensorMesh` from a PyVista mesh.

    Parameters
    ----------
    grid : pyvista.DataSet
        Input mesh. Will be converted to an ``UnstructuredGrid``.
    backend : {"torch", "phlower"}, default="torch"
        Backend used for tensors in the returned mesh.
    float_precision : FloatPrecision or int, default=FloatPrecision.FLOAT32
        Floating point precision used by the backend.
    dimension_collection : dict[str, dict[str, float]] or None, optional
        Optional per-array dimension metadata (for phlower_tensor).
        Keys correspond to ``grid.point_data`` / ``grid.cell_data`` names.
    device : torch.device or str or None, optional
        Target device. Interpretation depends on the backend.
    validate_mesh: bool, default=False
        If True, validate the mesh using PyVista's ``validate_mesh`` method.

    Returns
    -------
    TensorMesh[torch.Tensor] or TensorMesh[pt.PhlowerTensor]
        Mesh with backend tensors for points and data arrays, and connectivity
        stored in ``pvmesh``.

    Examples
    --------
    >>> mesh = from_pyvista(grid, backend="torch")

    Notes
    -----
    The returned mesh eagerly attaches \
        :class:`~graphlow.core.topology.MeshTopology`
    and :class:`~graphlow.core.geometry.MeshGeometry` instances.
    """
    grid = grid.cast_to_unstructured_grid()

    # Validate input mesh
    # non_planar_faces / non_convex / intersecting_edges / inverted_faces
    # are allowed
    if validate_mesh:
        report = grid.validate_mesh()
        if not report.is_valid:
            disallowed = set(report.invalid_fields) - {
                "non_planar_faces",
                "non_convex",
                "intersecting_edges",
                "inverted_faces",
            }
            if disallowed:
                raise ValueError(f"{report.message}")
            logger.warning(
                "Mesh validation reported issues (allowed): %s",
                report.message or report.invalid_fields,
            )

    backend_instance = get_backend(
        backend, float_precision=float_precision, device=device
    )
    return _from_pyvista_impl(
        grid, backend_instance, dimension_collection=dimension_collection
    )


def _from_pyvista_impl[T: TensorLike](
    grid: UnstructuredGrid,
    backend_instance: Backend[T],
    *,
    dimension_collection: dict[str, dict[str, float]] | None = None,
) -> TensorMesh[T]:
    # Convert to TensorMesh
    points = backend_instance.as_tensor(
        grid.points, dimension=DEFAULT_DIMENSIONS[FeatureName.POINTS]
    )
    dims = dimension_collection or {}
    point_data = {
        name: backend_instance.as_tensor(data, dimension=dims.get(name, {}))
        for name, data in grid.point_data.items()
    }
    cell_data = {
        name: backend_instance.as_tensor(data, dimension=dims.get(name, {}))
        for name, data in grid.cell_data.items()
    }

    bcache = BackendCache(backend_instance)
    return TensorMesh(
        points=points,
        pvmesh=grid,
        backend=backend_instance,
        bcache=bcache,
        point_data=point_data,
        cell_data=cell_data,
    )

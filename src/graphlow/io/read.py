"""Public IO helpers (file → :class:`graphlow.core.mesh.TensorMesh`)."""

from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Literal, overload

import pyvista as pv
import torch

from graphlow.core.mesh import TensorMesh
from graphlow.io.pyvista import from_pyvista
from graphlow.utils.enums import FloatPrecision

if TYPE_CHECKING:
    import phlower_tensor as pt

logger = logging.getLogger(__name__)


@overload
def read(
    file: str | pathlib.Path,
    backend: Literal["torch"],
    float_precision: FloatPrecision | int = FloatPrecision.FLOAT32,
    *,
    dimension_collection: dict[str, dict[str, float]] | None = None,
    device: torch.device | str | None = None,
    validate_mesh: bool = False,
) -> TensorMesh[torch.Tensor]: ...


@overload
def read(
    file: str | pathlib.Path,
    backend: Literal["phlower"],
    float_precision: FloatPrecision | int = FloatPrecision.FLOAT32,
    *,
    dimension_collection: dict[str, dict[str, float]] | None = None,
    device: torch.device | str | None = None,
    validate_mesh: bool = False,
) -> TensorMesh[pt.PhlowerTensor]: ...


def read(
    file: str | pathlib.Path,
    backend: Literal["torch", "phlower"] = "torch",
    float_precision: FloatPrecision | int = FloatPrecision.FLOAT32,
    *,
    dimension_collection: dict[str, dict[str, float]] | None = None,
    device: torch.device | str | None = None,
    validate_mesh: bool = False,
) -> TensorMesh[torch.Tensor] | TensorMesh[pt.PhlowerTensor]:
    """
    Read a mesh file into a :class:`~graphlow.core.mesh.TensorMesh`.

    This is a thin wrapper around :func:`graphlow.io.pyvista.from_pyvista`.

    Parameters
    ----------
    file : str or pathlib.Path
        Path to a mesh readable by PyVista/VTK
        (e.g. ``.vtu``, ``.vtp``, ``.vtk``).
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
        Mesh with backend tensors for points and data arrays.

    Notes
    -----
    To use the ``phlower`` backend, ``phlower_tensor`` must be installed.
    """
    grid: pv.DataSet = pv.read(file)
    mesh = from_pyvista(
        grid,
        backend,
        float_precision=float_precision,
        dimension_collection=dimension_collection,
        device=device,
        validate_mesh=validate_mesh,
    )
    logger.info(
        "Read mesh from %s: %d points, %d cells",
        pathlib.Path(file).resolve(),
        mesh.n_points,
        mesh.n_cells,
    )
    return mesh

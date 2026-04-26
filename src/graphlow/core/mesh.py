from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyvista as pv
import torch

from graphlow.core.cache import BackendCache
from graphlow.core.geometry import MeshGeometry
from graphlow.core.topology import MeshTopology
from graphlow.utils.dimension import get_dimension
from graphlow.utils.enums import (
    POLYDATA_EXTENSIONS,
    UNSTRUCTURED_GRID_EXTENSIONS,
    FeatureName,
)

if TYPE_CHECKING:
    from graphlow.core.backend.base import Backend, TensorLike

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParentPointMap:
    """Point mapping from a derived mesh back to its parent mesh."""

    parent_kind: Literal["volume"]
    parent_n_points: int
    point_ids: torch.Tensor  # (n_points_of_child,)

    def to(
        self,
        device: torch.device | str | None = None,
        non_blocking: bool = False,
    ) -> ParentPointMap:
        """Move the mapping to a different device."""

        # NOTE: So far, dtype is not allowed to change
        # See `as_index_tensor` method in Backend class.
        point_ids = self.point_ids.to(
            device=device,
            non_blocking=non_blocking,
        )
        return ParentPointMap(
            parent_kind=self.parent_kind,
            parent_n_points=self.parent_n_points,
            point_ids=point_ids,
        )


@dataclass
class TensorMesh[T: TensorLike]:
    """
    Differentiable mesh container backed by a PyVista grid.

    This class separates the mesh into:

    - Differentiable parts (geometry): ``points`` and ``point_data`` /
      ``cell_data`` as backend tensors.
    - Non-differentiable parts (topology): ``pvmesh`` and cached
      topology structures derived from it.

    Notes
    -----
    - Topology is assumed to be fixed (connectivity does not change). Updating
      ``points`` does not invalidate topology caches.
    - ``pvmesh`` is used as the source of truth for connectivity.
    """

    points: T
    """(n_points, 3) backend tensor; only differentiable part."""

    pvmesh: pv.UnstructuredGrid
    """PyVista UnstructuredGrid."""

    backend: Backend[T]
    """Torch or phlower_tensor wrapper."""

    bcache: BackendCache[T]
    """Backend-dependent sparse/compiled cache."""

    point_data: dict[str, T] = field(default_factory=dict)
    """(n_points, ...) backend tensor; data associated with points."""

    cell_data: dict[str, T] = field(default_factory=dict)
    """(n_cells, ...) backend tensor; data associated with cells."""

    _parent_point_map: ParentPointMap | None = None
    """Internal parent-point mapping metadata for extracted surface meshes."""

    topology: MeshTopology[T] = field(init=False)
    """Topology layer (cached)."""

    geometry: MeshGeometry[T] = field(init=False)
    """Geometry layer (cached)."""

    def __post_init__(self) -> None:
        self.topology = MeshTopology(self)
        self.geometry = MeshGeometry(self)
        logger.info(
            "Built TensorMesh: backend=%s, n_points=%d, n_cells=%d",
            self.backend,
            self.n_points,
            self.n_cells,
        )

    @property
    def n_points(self) -> int:
        """Number of points."""
        return int(self.points.shape[0])

    @property
    def n_cells(self) -> int:
        """Number of cells."""
        return self.pvmesh.n_cells

    @property
    def parent_point_map(self) -> ParentPointMap:
        """
        Advanced API:
        Return the parent point map instance for this derived mesh.
        Use this instance to transfer data from the parent mesh
        to the derived mesh and vice versa.
        This property works only for extracted surface meshes.

        Returns
        -------
        ParentPointMap[T]
            Parent point map for this derived mesh.

        Raises
        ------
        ValueError
            If this mesh is not an extracted surface mesh.
        """
        if self._parent_point_map is None:
            raise ValueError("This mesh has no parent point mapping.")
        return self._parent_point_map

    @property
    def parent_point_ids(self) -> torch.Tensor:
        """
        Return the parent point index for each point of this derived mesh.
        This property works only for extracted surface meshes.

        Returns
        -------
        torch.Tensor
            Torch index tensor of shape ``(n_points,)``.

        Raises
        ------
        ValueError
            If this mesh is not an extracted surface mesh.
        """
        return self.parent_point_map.point_ids

    def requires_grad(
        self,
        requires_grad: bool = True,
        point_data: bool = False,
        cell_data: bool = False,
    ) -> None:
        """
        Set ``requires_grad`` flags for backend tensors.

        Parameters
        ----------
        requires_grad : bool
            If True, gradients are tracked. If False, gradients are not tracked.
        point_data : bool, default=False
            If True, also applies to floating-point arrays in ``point_data``.
            (in addition to ``points``).
        cell_data : bool, default=False
            If True, also applies to floating-point arrays in ``cell_data``.
            (in addition to ``points``).
        """
        self.backend.to_torch(self.points).requires_grad_(requires_grad)

        if point_data:
            for data in self.point_data.values():
                torch_data = self.backend.to_torch(data)
                if torch_data.dtype.is_floating_point:
                    torch_data.requires_grad_(requires_grad)
        if cell_data:
            for data in self.cell_data.values():
                torch_data = self.backend.to_torch(data)
                if torch_data.dtype.is_floating_point:
                    torch_data.requires_grad_(requires_grad)

    def extract_surface(self, *, keep_graph: bool = True) -> TensorMesh[T]:
        """
        Extract the boundary surface as a new 2D mesh.

        The returned surface mesh reuses this mesh's point tensor,
        so autograd remains connected to the original volume points.

        The mapping from surface points to parent volume points is available via
        ``surface.parent_point_ids`` and related gather/scatter helpers.

        Parameters
        ----------
        keep_graph : bool, default=True
            If True, surface points (and transferred ``point_data``) are views
            into the original backend tensors, preserving the computation
            graph. If False, the returned tensors are detached, so gradients do
            not flow back to this mesh.

        Returns
        -------
        TensorMesh[T]
            Surface mesh (2D boundary cells) with:

            - ``points``: subset of the volume ``points``,
            - ``point_data``: subset of the volume ``point_data``,
            - ``parent_point_map``: metadata mapping surface points
              back to parent volume points.
        """
        grid_copy = self.pvmesh.copy()
        n_points = self.n_points
        grid_copy.point_data[FeatureName.ORIGINAL_INDEX] = np.arange(
            n_points, dtype=np.int64
        )
        surface_pv = grid_copy.extract_surface(algorithm=None)
        surface_pv = surface_pv.cast_to_unstructured_grid()

        backend = self.backend
        original_ids = backend.as_index_tensor(
            surface_pv.point_data.pop(FeatureName.ORIGINAL_INDEX)
        )

        surf_points = self.points[original_ids]
        surf_point_data = {
            k: v[original_ids] for k, v in self.point_data.items()
        }
        parent_point_map = ParentPointMap(
            parent_kind="volume",
            parent_n_points=self.n_points,
            point_ids=original_ids,
        )

        if not keep_graph:
            surf_points = backend.as_tensor(
                backend.to_torch(surf_points).detach(),
                dimension=get_dimension(surf_points),
            )
            surf_point_data = {
                k: backend.as_tensor(
                    backend.to_torch(v).detach(),
                    dimension=get_dimension(v),
                )
                for k, v in surf_point_data.items()
            }

        bcache = BackendCache(backend)
        surf_mesh = TensorMesh(
            points=surf_points,
            pvmesh=surface_pv,
            backend=backend,
            bcache=bcache,
            point_data=surf_point_data,
            cell_data={},
            _parent_point_map=parent_point_map,
        )
        return surf_mesh

    def save(
        self,
        file_name: pathlib.Path | str,
        *,
        binary: bool = True,
        remove_time: bool = True,
        overwrite_features: bool = False,
        overwrite_file: bool = False,
        cast: bool = True,
    ):
        """
        Save mesh data. On writing, point_data and cell_data
        will be copied to pyvista mesh.

        Parameters
        ----------
        file_name: pathlib.Path | str
            File name to be written. If the parent directory does not exist,
            it will be created.
        binary: bool
            If True, write binary file. The default is True.
        remove_time: bool
            If True, remove TimeValue field data.
        overwrite_features: bool
            If True, allow overwriting features. The default is False.
        overwrite_file: bool
            If True, allow overwriting the file. The default is False.
        cast: bool
            If True, cast mesh if needed. The default is True.
        """
        file_path = pathlib.Path(file_name)
        if not overwrite_file and file_path.exists():
            raise ValueError(f"{file_path} already exists.")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        self.copy_features_to_pyvista(overwrite=overwrite_features)

        if remove_time:
            self.pvmesh.field_data.pop(FeatureName.TIME_VALUE, None)

        if not cast:
            self.pvmesh.save(file_name, binary=binary)
            logger.info(f"File writtein in: {file_name}")
            return

        ext = file_path.suffix.lstrip(".")
        if ext in UNSTRUCTURED_GRID_EXTENSIONS:
            unstructured_grid = self.pvmesh.cast_to_unstructured_grid()
            unstructured_grid.save(file_name, binary=binary)
            logger.info(f"File writtein in: {file_name}")
            return

        if ext in POLYDATA_EXTENSIONS:
            if isinstance(self.pvmesh, pv.PolyData):
                self.pvmesh.save(file_name, binary=binary)
                return
            poly_data = self.pvmesh.extract_surface(algorithm=None)
            poly_data.save(file_name, binary=binary)
            logger.info(f"File writtein in: {file_name}")
            return

        raise ValueError(f"Unexpected extension: {ext}")

    def copy_features_to_pyvista(self, overwrite: bool = False) -> None:
        """
        Copy point and cell tensor data to pyvista mesh.

        Parameters
        ----------
        overwrite: bool
            If True, allow overwriting existing features. The default is False.

        Returns
        -------
        None
        """
        if not overwrite:
            conflicting = set(self.point_data.keys()) & set(
                self.pvmesh.point_data.keys()
            )
            if conflicting:
                raise ValueError(f"Keys already exist: {sorted(conflicting)}")
            conflicting = set(self.cell_data.keys()) & set(
                self.pvmesh.cell_data.keys()
            )
            if conflicting:
                raise ValueError(f"Keys already exist: {sorted(conflicting)}")

        backend = self.backend

        def to_numpy(x: T) -> np.ndarray:
            return backend.to_torch(x).detach().cpu().numpy()

        self.pvmesh.point_data.update(
            {k: to_numpy(v) for k, v in self.point_data.items()}
        )
        self.pvmesh.cell_data.update(
            {k: to_numpy(v) for k, v in self.cell_data.items()}
        )

    def copy_features_from_pyvista(
        self,
        overwrite: bool = False,
        dimension_collection: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """
        Copy point and cell tensor data from pyvista mesh.

        Parameters
        ----------
        overwrite: bool
            If True, allow overwriting existing features. The default is False.
        dimension_collection: dict[str, dict[str, float]] | None
            Optional per-array dimension metadata (for phlower_tensor).
            Keys correspond to ``grid.point_data`` / ``grid.cell_data`` names.

        Returns
        -------
        None
        """
        if not overwrite:
            conflicting = set(self.point_data.keys()) & set(
                self.pvmesh.point_data.keys()
            )
            if conflicting:
                raise ValueError(f"Keys already exist: {sorted(conflicting)}")
            conflicting = set(self.cell_data.keys()) & set(
                self.pvmesh.cell_data.keys()
            )
            if conflicting:
                raise ValueError(f"Keys already exist: {sorted(conflicting)}")

        backend = self.backend
        dims = dimension_collection or {}

        self.point_data.update(
            {
                k: backend.as_tensor(v, dimension=dims.get(k, {}))
                for k, v in self.pvmesh.point_data.items()
            }
        )
        self.cell_data.update(
            {
                k: backend.as_tensor(v, dimension=dims.get(k, {}))
                for k, v in self.pvmesh.cell_data.items()
            }
        )

    def gather_parent_point_data(self, src_point_data: T) -> T:
        """
        Gather parent point data onto this derived mesh.
        This method works only for extracted surface meshes.

        Parameters
        ----------
        src_point_data : T
            Backend tensor of shape ``(n_points_of_parent, ...)``.

        Returns
        -------
        T
            Backend tensor of shape ``(n_points, ...)`` on this derived mesh.

        Raises
        ------
        ValueError
            If ``src_point_data.shape[0]`` does not match \
                the parent mesh's n_points.
        """
        point_ids = self.parent_point_ids
        n_points_of_parent = self.parent_point_map.parent_n_points
        if src_point_data.shape[0] != n_points_of_parent:
            raise ValueError(
                "src_point_data.shape[0] must match the parent mesh's n_points."
            )
        return src_point_data[point_ids]

    def scatter_add_to_parent_point_data(
        self,
        src_point_data: T,
        dst_point_data: T | None = None,
    ) -> T:
        """
        Scatter this mesh's point data to the parent point space and add to it.
        This method works only for extracted surface meshes.

        Parameters
        ----------
        src_point_data : T
            Backend tensor of shape ``(n_points_of_this_mesh, ...)``.
        dst_point_data : T | None, optional
            Optional destination tensor of shape ``(n_points_of_parent, ...)``.
            If omitted, a zero-initialized tensor is created.

        Returns
        -------
        T
            Backend tensor of shape ``(n_points_of_parent, ...)``.

        Raises
        ------
        ValueError
            If ``src_point_data.shape[0]`` does not match \
                this mesh's n_points.
            If ``dst_point_data.shape`` does not match \
                the parent point-space shape.
        """
        point_ids = self.parent_point_ids
        n_points_of_parent = self.parent_point_map.parent_n_points
        if src_point_data.shape[0] != self.n_points:
            raise ValueError(
                "src_point_data.shape[0] must match this mesh's n_points."
            )

        target_shape = (
            n_points_of_parent,
            *src_point_data.shape[1:],
        )
        if dst_point_data is None:
            dst_point_data = self.backend.zeros(
                target_shape,
                dimension=get_dimension(src_point_data),
            )
        if tuple(dst_point_data.shape) != target_shape:
            raise ValueError(
                "dst_point_data.shape must match the parent point-space shape."
            )

        return dst_point_data.index_add_(0, point_ids, src_point_data)

    def to(
        self,
        device: torch.device | str | None = None,
        non_blocking: bool = False,
        dtype: torch.dtype | None = None,
    ) -> TensorMesh[T]:
        """
        Move the mesh data to a different device and/or dtype.

        Parameters
        ----------
        device: torch.device | str | None
            The device to move the mesh to. The default is None.
        non_blocking: bool
            If True, the transfer happens asynchronously. The default is False.
        dtype: torch.dtype | None
            The floating-point dtype to move the mesh to. The default is None.

        Returns
        -------
        TensorMesh[T]
            The moved mesh.
        """
        backend = self.backend.to(
            device=device,
            dtype=dtype,
        )
        bcache = self.bcache.to(
            device=device,
            non_blocking=non_blocking,
            dtype=dtype,
        )
        points = self.points.to(
            device=device,
            non_blocking=non_blocking,
            dtype=dtype,
        )
        point_data = {}
        for k, data in self.point_data.items():
            point_data[k] = data.to(
                device=device,
                non_blocking=non_blocking,
                dtype=(dtype if data.dtype.is_floating_point else None),
            )
        cell_data = {}
        for k, data in self.cell_data.items():
            cell_data[k] = data.to(
                device=device,
                non_blocking=non_blocking,
                dtype=(dtype if data.dtype.is_floating_point else None),
            )

        parent_point_map = (
            None
            if self._parent_point_map is None
            else self._parent_point_map.to(
                device=device, non_blocking=non_blocking
            )
        )

        return TensorMesh[T](
            points=points,
            pvmesh=self.pvmesh,
            backend=backend,
            bcache=bcache,
            point_data=point_data,
            cell_data=cell_data,
            _parent_point_map=parent_point_map,
        )

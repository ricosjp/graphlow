from __future__ import annotations

import functools
import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pyvista as pv
import torch
from scipy import sparse as sp

from graphlow.base.dict_tensor import GraphlowDictTensor
from graphlow.base.mesh_interface import IReadOnlyGraphlowMesh
from graphlow.base.tensor_property import GraphlowTensorProperty
from graphlow.processors.geometry_processor import GeometryProcessor
from graphlow.processors.graph_processor import GraphProcessor
from graphlow.util import constants
from graphlow.util.enums import FeatureName
from graphlow.util.logger import get_logger

logger = get_logger(__name__)


def use_cache_decorator(method: Callable) -> Callable:
    @functools.wraps(method)
    def wrapper(self: GraphlowMesh, *args: Any, **kwargs: Any) -> Any:
        logger.debug(method.__name__)

        cached = self.get_cached_results(method.__name__)
        if cached is None:
            result = method(self, *args, **kwargs)
            self.set_cached_results(method.__name__, result)
            return result
        else:
            logger.debug("use cached value")
            return cached

    return wrapper


class GraphlowMesh(IReadOnlyGraphlowMesh):
    def __init__(
        self,
        pvmesh: pv.UnstructuredGrid,
        *,
        dict_point_tensor: GraphlowDictTensor | None = None,
        dict_cell_tensor: GraphlowDictTensor | None = None,
        dict_sparse_tensor: GraphlowDictTensor | None = None,
        device: torch.device | int = -1,
        dtype: torch.dtype | type | None = None,
    ):
        """Initialize GraphlowMesh object.

        Parameters
        ----------
        mesh: pyvista.PointGrid
            Mesh data.
        dict_point_tensor: GraphlowDictTensor | None
        dict_tensor: dict[str, graphlow.ArrayDataType]
            Dict of tensor data.
        device: torch.device | int
            Device ID. int < 0 implies CPU.
        dtype: torch.dtype | type | None
            Data type.
        """
        self._geometry_processor = GeometryProcessor()
        self._graph_processor = GraphProcessor()
        self._cached = {}

        self._tensor_property = GraphlowTensorProperty(
            device=device, dtype=dtype
        )

        self._pvmesh = pvmesh.cast_to_unstructured_grid()
        self._dict_point_tensor = dict_point_tensor or GraphlowDictTensor(
            {}, length=self.n_points
        )
        if FeatureName.POINTS not in self._dict_point_tensor:
            self._dict_point_tensor.update(
                {FeatureName.POINTS: self.pvmesh.points}
            )
        self._dict_cell_tensor = dict_cell_tensor or GraphlowDictTensor(
            {}, length=self.n_cells
        )
        self._dict_sparse_tensor = dict_sparse_tensor or GraphlowDictTensor(
            {}, length=None
        )

        self.send()
        return

    @property
    def pvmesh(self) -> pv.PointGrid:
        return self._pvmesh

    @property
    def points(self) -> torch.Tensor:
        return self.dict_point_tensor[FeatureName.POINTS]

    @property
    def n_points(self) -> int:
        return self._pvmesh.n_points

    @property
    def n_cells(self) -> int:
        return self._pvmesh.n_cells

    @property
    def dict_point_tensor(self) -> GraphlowDictTensor:
        return self._dict_point_tensor

    @property
    def dict_cell_tensor(self) -> GraphlowDictTensor:
        return self._dict_cell_tensor

    @property
    def dict_sparse_tensor(self) -> GraphlowDictTensor:
        return self._dict_sparse_tensor

    @property
    def device(self) -> torch.Tensor:
        return self._tensor_property.device

    @property
    def dtype(self) -> torch.Tensor:
        return self._tensor_property.dtype

    def get_cached_results(self, method_name: str) -> None:
        if method_name not in self._cached:
            return None
        return self._cached[method_name]

    def set_cached_results(self, method_name: str, value: Any) -> None:
        # HACK: if necessary, check method name
        self._cached[method_name] = value

    def save(
        self,
        file_name: pathlib.Path | str,
        *,
        binary: bool = True,
        cast: bool = True,
        remove_time: bool = True,
        overwrite_features: bool = False,
        overwrite_file: bool = False,
    ):
        """Save mesh data. On writing, dict_point_tensor and dict_cell_tensor
        will be copied to pyvista mesh.

        Parameters
        ----------
        file_name: pathlib.Path | str
            File name to be written. If the parent directory does not exist,
            it will be created.
        binary: bool
            If True, write binary file. The default is True.
        cast: bool
            If True, cast mesh if needed. The default is True.
        remove_time: bool
            If True, remove TimeValue field data.
        overwrite_features: bool
            If True, allow overwriting features. The default is False.
        overwrite_file: bool
            If True, allow overwriting the file. The default is False.
        """
        file_path = pathlib.Path(file_name)
        if not overwrite_file and file_path.exists():
            raise ValueError(f"{file_path} already exists.")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        self.copy_features_to_pyvista(overwrite=overwrite_features)

        if not cast:
            self.pvmesh.save(file_name, binary=binary)
            logger.info(f"File writtein in: {file_name}")
            return

        if remove_time:
            self.pvmesh.field_data.pop(FeatureName.TIME_VALUE, None)

        ext = file_path.suffix.lstrip(".")
        if ext in constants.UNSTRUCTURED_GRID_EXTENSIONS:
            unstructured_grid = self.pvmesh.cast_to_unstructured_grid()
            unstructured_grid.save(file_name, binary=binary)
            logger.info(f"File writtein in: {file_name}")
            return

        if ext in constants.POLYDATA_EXTENSIONS:
            if isinstance(self.pvmesh, pv.PolyData):
                self.pvmesh.save(file_name, binary=binary)
                return
            poly_data = self.pvmesh.extract_surface()
            poly_data.save(file_name, binary=binary)
            logger.info(f"File writtein in: {file_name}")
            return

        raise ValueError(f"Unexpected extension: {ext}")

    def send(
        self,
        *,
        device: torch.device | int | None = None,
        dtype: torch.dtype | type | None = None,
    ):
        """Convert features to the specified device and dtype. It does not
        modify pyvista mesh.

        Parameters
        ----------
        device: torch.device | int | None
        dtype: torch.dtype | type | None
        """
        self._tensor_property.device = device or self.device
        self._tensor_property.dtype = dtype or self.dtype

        self._dict_point_tensor.send(device=self.device, dtype=self.dtype)
        self._dict_cell_tensor.send(device=self.device, dtype=self.dtype)
        self._dict_sparse_tensor.send(device=self.device, dtype=self.dtype)
        return

    def copy_features_from_pyvista(self, *, overwrite: bool = False):
        """Copy point and cell data from pyvista mesh.

        overwrite: bool
            If True, allow overwriting exsiting items. The default is False.
        """
        self.dict_point_tensor.update(
            self.pvmesh.point_data, overwrite=overwrite
        )
        self.dict_cell_tensor.update(self.pvmesh.cell_data, overwrite=overwrite)
        return

    def copy_features_to_pyvista(self, *, overwrite: bool = False):
        """Copy point and cell tensor data to pyvista mesh.

        overwrite: bool
            If True, allow overwriting exsiting items. The default is False.
        """
        self.update_pyvista_data(
            self.dict_point_tensor, self.pvmesh.point_data, overwrite=overwrite
        )
        self.update_pyvista_data(
            self.dict_cell_tensor, self.pvmesh.cell_data, overwrite=overwrite
        )
        return

    def update_pyvista_data(
        self,
        dict_tensor: GraphlowDictTensor,
        pyvista_dataset: pv.DataSetAttributes,
        *,
        overwrite: bool = False,
    ):
        """Update PyVista dataset with the specified GraphlowDictTensor.

        Parameters
        ----------
        dict_tensor: graphlow.GraphlowDictTensor
            DataSet to update. Typically dict_point_tensor or dict_cell_tensor.
        pyvista_dataset: pyvista.DataSetAttributes
            DataSet to be updated. Typically point_data or cell_data.
        overwrite: bool
            If True, allow overwriting exsiting items. The default is False.
        """
        if not overwrite:
            for key in dict_tensor.keys():
                if key in pyvista_dataset:
                    keys = list(pyvista_dataset.keys())
                    raise ValueError(f"{key} already exists in {keys}")
        pyvista_dataset.update(dict_tensor.convert_to_numpy_scipy())
        return

    def add_original_index(self):
        """Set original indices to points and cells. We do not use
        vtkOriginalPointIds and vtkOriginalCellIds because they are hidden.
        """
        self.pvmesh.point_data[FeatureName.ORIGINAL_INDEX] = np.arange(
            self.pvmesh.n_points
        )
        self.pvmesh.cell_data[FeatureName.ORIGINAL_INDEX] = np.arange(
            self.pvmesh.n_cells
        )
        return

    def extract_surface(self, add_original_index: bool = True) -> GraphlowMesh:
        """Extract surface.

        Parameters
        ----------
        add_original_index: bool
            If True, add original index feature to enable relative incidence
            matrix computation. The default is True.
        """
        if add_original_index:
            self.add_original_index()

        surface_mesh = self.pvmesh.extract_surface(
            pass_pointid=False, pass_cellid=False
        ).cast_to_unstructured_grid()
        return GraphlowMesh(surface_mesh, device=self.device, dtype=self.dtype)

    def extract_facets(self) -> tuple[GraphlowMesh, sp.csr_array]:
        """Extract all internal/external facets of the volume mesh
        with (n_faces, n_cells)-shaped sparse signed incidence matrix
        """
        poly, scipy_fc_inc = self._extract_facets_impl()
        return GraphlowMesh(
            poly, device=self.device, dtype=self.dtype
        ), scipy_fc_inc

    def _extract_facets_impl(self) -> tuple[pv.PolyData, sp.csr_array]:
        """Implementation of `extract_facets`

        Returns
        -------
        pyvista.PolyData
            PolyData with all internal/external faces registered as cells

        scipy.sparse.csr_array
            (n_faces, n_cells)-shaped sparse signed incidence matrix
        """
        vol = self.pvmesh

        polygon_cells = []
        sign_values = []
        row_indices = []
        col_indices = []

        n_facets = 0
        n_cells = vol.n_cells
        cell_centers = torch.from_numpy(
            self.pvmesh.cell_centers().points.astype(np.float32)
        ).clone()

        facet_idmap = {}

        for cell_id in range(n_cells):
            cell = vol.get_cell(cell_id)
            cell_center = cell_centers[cell_id]
            for j in range(cell.n_faces):
                face = cell.get_face(j).point_ids
                vtk_polygon_cell = [len(face), *face]

                face_points = self.points[face]
                face_center = torch.mean(face_points, dim=0)
                side_vec = face_points - face_center
                cc2fc = face_center - cell_center
                cross = torch.linalg.cross(
                    side_vec, torch.roll(side_vec, shifts=-1, dims=0)
                )
                normal = torch.mean(cross, dim=0)
                dot = torch.dot(cc2fc, normal)
                sign_value = 0
                if dot < 0:
                    sign_value = -1
                else:
                    sign_value = 1

                # check duplicated face
                tri = tuple(sorted(face)[0:3])
                if tri in facet_idmap:
                    facet_id, sign = facet_idmap[tri]
                    sign_value = -sign
                else:
                    facet_id = n_facets
                    facet_idmap[tri] = (facet_id, sign_value)
                    n_facets += 1
                    polygon_cells.extend(vtk_polygon_cell)

                sign_values.append(sign_value)
                row_indices.append(facet_id)
                col_indices.append(cell_id)

        poly = pv.PolyData(self.points.numpy(), polygon_cells)
        scipy_fc_inc = sp.csr_array(
            (sign_values, (row_indices, col_indices)),
            shape=(n_facets, n_cells),
        )
        return poly, scipy_fc_inc

    @functools.wraps(GeometryProcessor.compute_areas)
    def compute_areas(self, raise_negative_area: bool = True) -> torch.Tensor:
        val = self._geometry_processor.compute_areas(self, raise_negative_area)
        return val

    @functools.wraps(GeometryProcessor.compute_volumes)
    def compute_volumes(
        self, raise_negative_volume: bool = True
    ) -> torch.Tensor:
        val = self._geometry_processor.compute_volumes(
            self, raise_negative_volume
        )
        return val

    @functools.wraps(GeometryProcessor.compute_normals)
    def compute_normals(self) -> torch.Tensor:
        val = self._geometry_processor.compute_normals(self)
        return val

    @functools.wraps(GeometryProcessor.compute_IsoAM)
    def compute_IsoAM(
        self, with_moment_matrix: bool = True, consider_volume: bool = False
    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        val = self._geometry_processor.compute_IsoAM(
            self, with_moment_matrix, consider_volume
        )
        return val

    @functools.wraps(GraphProcessor.compute_cell_point_incidence)
    @use_cache_decorator
    def compute_cell_point_incidence(self) -> torch.Tensor:
        val = self._graph_processor.compute_cell_point_incidence(self)
        return val

    @functools.wraps(GraphProcessor.compute_cell_adjacency)
    @use_cache_decorator
    def compute_cell_adjacency(self) -> torch.Tensor:
        val = self._graph_processor.compute_cell_adjacency(self)
        return val

    @functools.wraps(GraphProcessor.compute_point_adjacency)
    @use_cache_decorator
    def compute_point_adjacency(self) -> torch.Tensor:
        val = self._graph_processor.compute_point_adjacency(self)
        return val

    @functools.wraps(GraphProcessor.compute_point_relative_incidence)
    @use_cache_decorator
    def compute_point_relative_incidence(
        self, other_mesh: GraphlowMesh
    ) -> torch.Tensor:
        val = self._graph_processor.compute_point_relative_incidence(
            self, other_mesh
        )
        return val

    @functools.wraps(GraphProcessor.compute_cell_relative_incidence)
    @use_cache_decorator
    def compute_cell_relative_incidence(
        self, other_mesh: GraphlowMesh
    ) -> torch.Tensor:
        val = self._graph_processor.compute_cell_relative_incidence(
            self, other_mesh
        )
        return val

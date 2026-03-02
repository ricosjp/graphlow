from __future__ import annotations

import pathlib
from enum import StrEnum
from typing import Literal

import numpy as np
import phlower_tensor as pt
import pyvista as pv
import torch
from phlower_tensor._tensor._dimension import PhysicDimensionLikeObject
from phlower_tensor.collections import (
    IPhlowerTensorCollections,
    phlower_tensor_collection,
)
from pyvista.core._typing_core import VectorLike
from scipy import sparse as sp

from graphlow.base.mesh_interface import IReadOnlyGraphlowMesh
from graphlow.processors.geometry_processor import GeometryProcessor
from graphlow.processors.graph_processor import GraphProcessor
from graphlow.processors.isoAM_processor import IsoAMProcessor
from graphlow.util import constants
from graphlow.util.enums import (
    DEFAULT_DIMENSIONS,
    PRECISION_TO_DTYPE,
    FeatureName,
    FloatPrecision,
    SparseMatrixName,
)
from graphlow.util.logger import get_logger

logger = get_logger(__name__)


class GraphlowMesh(IReadOnlyGraphlowMesh):
    def __init__(
        self,
        pvmesh: pv.UnstructuredGrid,
        dict_dimensions: dict[StrEnum, PhysicDimensionLikeObject] | None = None,
        dict_is_time_series: dict[StrEnum, bool] | None = None,
        float_precision: FloatPrecision | int = FloatPrecision.FLOAT32,
        device: torch.device | str | None = None,
        *,
        dict_point_tensor: IPhlowerTensorCollections | None = None,
        dict_cell_tensor: IPhlowerTensorCollections | None = None,
        dict_sparse_tensor: IPhlowerTensorCollections | None = None,
    ):
        """Initialize GraphlowMesh object.

        Parameters
        ----------
        mesh: pyvista.PointGrid
            Mesh data.
        dict_dimensions: DictDimensions | None
            Dimensions for each key in
            dict_point_tensor, dict_cell_tensor, and dict_sparse_tensor.
        dict_is_time_series: DictIsTimeSeries | None
            Specifies if the data is time series or not. Can be specified
            for each value by inputting dict[Key, bool].
        float_precision: FloatPrecision | int | None
            Float precision. 32 or 64. Default is 32.
        device: torch.device | str | None
            Device.
        dict_point_tensor: DictTensors | None
            Tensor dictionary for points.
        dict_cell_tensor: DictTensors | None
            Tensor dictionary for cells.
        dict_sparse_tensor: DictSparseTensors | None
            Sparse tensor dictionary. Keys are SparseMatrixName.
        """
        self._geometry_processor = GeometryProcessor()
        self._graph_processor = GraphProcessor()
        self._isoAM_processor = IsoAMProcessor()

        self._dict_dimensions = dict_dimensions or {}
        self._dict_is_time_series = dict_is_time_series or {}
        self._float_precision = FloatPrecision(float_precision)
        self._dtype = PRECISION_TO_DTYPE[self._float_precision]

        self._pvmesh = pvmesh.cast_to_unstructured_grid()

        self._dict_point_tensor = (
            dict_point_tensor or phlower_tensor_collection(values={})
        )
        self._dict_cell_tensor = dict_cell_tensor or phlower_tensor_collection(
            values={}
        )
        self._dict_sparse_tensor = (
            dict_sparse_tensor or phlower_tensor_collection(values={})
        )

        if FeatureName.POINTS not in self._dict_point_tensor:
            self._dict_point_tensor.update(
                {
                    FeatureName.POINTS: pt.phlower_tensor(
                        self.pvmesh.points,
                        dimension=DEFAULT_DIMENSIONS[FeatureName.POINTS],
                        dtype=self._dtype,
                        device=device,
                    )
                }
            )
        self._device = self._dict_point_tensor[FeatureName.POINTS].device
        self.copy_features_from_pyvista(overwrite=True)

    @property
    def pvmesh(self) -> pv.UnstructuredGrid:
        return self._pvmesh

    @property
    def points(self) -> pt.PhlowerTensor:
        return self._dict_point_tensor[FeatureName.POINTS]

    @property
    def n_points(self) -> int:
        return self._pvmesh.n_points

    @property
    def n_cells(self) -> int:
        return self._pvmesh.n_cells

    @property
    def dict_point_tensor(self) -> IPhlowerTensorCollections:
        return self._dict_point_tensor

    @property
    def dict_cell_tensor(self) -> IPhlowerTensorCollections:
        return self._dict_cell_tensor

    @property
    def dict_sparse_tensor(self) -> IPhlowerTensorCollections:
        return self._dict_sparse_tensor

    @property
    def float_precision(self) -> FloatPrecision:
        return self._float_precision

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

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
            poly_data = self.pvmesh.extract_surface(algorithm="dataset_surface")
            poly_data.save(file_name, binary=binary)
            logger.info(f"File writtein in: {file_name}")
            return

        raise ValueError(f"Unexpected extension: {ext}")

    def send(self, device: str | torch.device, non_blocking: bool = False):
        """Send features to the specified device.

        Parameters
        ----------
        device: str | torch.device
            Device to send the features to.
        non_blocking: bool
            If True, the copy will be done asynchronously with respect to the
            host. For more details, see
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.Tensor.to.
            The default is False.
        """
        self._dict_point_tensor.to(device=device, non_blocking=non_blocking)
        self._dict_cell_tensor.to(device=device, non_blocking=non_blocking)
        self._device = self.points.device

    def copy_features_from_pyvista(self, overwrite: bool = False):
        """Copy point and cell data from pyvista mesh.

        Parameters
        ----------
        overwrite: bool
            If True, allow overwriting exsiting items. The default is False.
        """
        point_tensors = self._convert_pvdataset_to_phlower_tensors(
            self.pvmesh.point_data
        )
        cell_tensors = self._convert_pvdataset_to_phlower_tensors(
            self.pvmesh.cell_data
        )
        self.dict_point_tensor.update(point_tensors, overwrite=overwrite)
        self.dict_cell_tensor.update(cell_tensors, overwrite=overwrite)

    def copy_features_to_pyvista(self, overwrite: bool = False):
        """Copy point and cell tensor data to pyvista mesh.

        Parameters
        ----------
        overwrite: bool
            If True, allow overwriting exsiting items. The default is False.
        """
        self._update_pyvista_data(
            self.dict_point_tensor, self.pvmesh.point_data, overwrite=overwrite
        )
        self._update_pyvista_data(
            self.dict_cell_tensor, self.pvmesh.cell_data, overwrite=overwrite
        )

    def _convert_pvdataset_to_phlower_tensors(
        self, dataset_attributes: pv.DataSetAttributes
    ) -> dict[str, pt.PhlowerTensor]:
        """Convert PyVista dataset attributes to a dict of PhlowerTensor.

        Parameters
        ----------
        dataset_attributes : pyvista.DataSetAttributes
            Attributes such as ``pvmesh.point_data`` or ``pvmesh.cell_data``.

        Returns
        -------
        dict[str, phlower_tensor.PhlowerTensor]
            Mapping from attribute key to tensorized value.
        """
        out: dict[str, pt.PhlowerTensor] = {}
        for key, value in dataset_attributes.items():
            out[key] = pt.phlower_tensor(
                value,
                dimension=self._dict_dimensions.get(key, {}),
                is_time_series=self._dict_is_time_series.get(key, False),
                dtype=self.dtype,
                device=self.device,
            )
        return out

    def _update_pyvista_data(
        self,
        dict_tensor: IPhlowerTensorCollections,
        pyvista_dataset: pv.DataSetAttributes,
        *,
        overwrite: bool = False,
    ):
        """Update PyVista dataset with the specified GraphlowDictTensor.

        Parameters
        ----------
        dict_tensor: IPhlowerTensorCollections
            DataSet to update. Typically dict_point_tensor or dict_cell_tensor.
        pyvista_dataset: pyvista.DataSetAttributes
            DataSet to be updated. Typically point_data or cell_data.
        overwrite: bool
            If True, allow overwriting exsiting items. The default is False.
        """
        if not overwrite:
            conflicting = set(dict_tensor.keys()) & set(pyvista_dataset.keys())
            if conflicting:
                raise ValueError(f"Keys already exist: {sorted(conflicting)}")
        pyvista_dataset.update(dict_tensor.to_numpy())

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

    def extract_surface(
        self,
        add_original_index: bool = True,
        pass_point_data: bool = False,
    ) -> GraphlowMesh:
        """Extract surface.

        Parameters
        ----------
        add_original_index: bool, optional [True]
            If True, add original index feature to enable relative incidence
            matrix computation.
        pass_point_data: bool, optional [False]
            If True, the extracted mesh will inherit
            the dict_point_tensor from this mesh.
            This parameter is used, for example,
            when you want to differentiate the metrics of the extracted mesh
            based on the point information of this mesh.

        Returns
        -------
        graphlow.GraphlowMesh
            Extracted surface mesh.
        """
        if add_original_index or pass_point_data:
            self.add_original_index()

        pv_surface = self.pvmesh.extract_surface(
            algorithm="dataset_surface", pass_pointid=False, pass_cellid=False
        )

        surface = GraphlowMesh(
            pv_surface,
            dict_dimensions=self._dict_dimensions,
            dict_is_time_series=self._dict_is_time_series,
            float_precision=self.float_precision,
            device=self.device,
        )
        if not pass_point_data:
            return surface

        point_rel_inc = self.compute_point_relative_incidence(surface)
        new_point_tensors = self.dict_point_tensor.apply(
            lambda x: point_rel_inc @ x
        )
        surface.dict_point_tensor.update(new_point_tensors, overwrite=True)
        return surface

    def extract_cells(
        self,
        ind: VectorLike[int],
        invert: bool = False,
        add_original_index: bool = True,
        pass_point_data: bool = False,
        pass_cell_data: bool = False,
    ) -> GraphlowMesh:
        """Extract cells by indices.

        Parameters
        ----------
        ind : sequence[int]
            Numpy array of cell indices to be extracted.
        invert : bool, optional [False]
            Invert the selection.
        add_original_index: bool, optional [True]
            If True, add original index feature to enable relative incidence
            matrix computation.
        pass_point_data: bool, optional [False]
            If True, the extracted mesh will inherit
            the dict_point_tensor from this mesh.
            This parameter is used, for example,
            when you want to differentiate the metrics of the extracted mesh
            based on the point information of this mesh.
        pass_cell_data: bool, optional [False]
            If True, the extracted mesh will inherit
            the dict_cell_data from this mesh.
            This parameter is used, for example,
            when you want to differentiate the metrics of the extracted mesh
            based on the cell information of this mesh.

        Returns
        -------
        graphlow.GraphlowMesh
            Extracted cells mesh.
        """
        if add_original_index or pass_point_data or pass_cell_data:
            self.add_original_index()

        pv_extracted = self.pvmesh.extract_cells(ind, invert=invert)

        extracted = GraphlowMesh(
            pv_extracted,
            dict_dimensions=self._dict_dimensions,
            dict_is_time_series=self._dict_is_time_series,
            float_precision=self.float_precision,
            device=self.device,
        )

        if pass_point_data:
            point_rel_inc = self.compute_point_relative_incidence(extracted)
            new_point_tensors = self.dict_point_tensor.apply(
                lambda x: point_rel_inc @ x
            )
            extracted.dict_point_tensor.update(
                new_point_tensors, overwrite=True
            )

        if pass_cell_data:
            cell_rel_inc = self.compute_cell_relative_incidence(extracted)
            new_cell_tensors = self.dict_cell_tensor.apply(
                lambda x: cell_rel_inc @ x
            )
            extracted.dict_cell_tensor.update(new_cell_tensors, overwrite=True)

        return extracted

    def extract_facets(
        self,
        add_original_index: bool = True,
        pass_point_data: bool = False,
    ) -> GraphlowMesh:
        """Extract all internal/external facets of the volume mesh
        with (n_faces, n_cells)-shaped sparse signed incidence matrix

        Parameters
        ----------
        add_original_index: bool, optional [True]
            If True, add original index feature to enable relative incidence
            matrix computation.
        pass_point_data: bool, optional [False]
            If True, the extracted mesh will inherit
            the dict_point_tensor from this mesh.
            This parameter is used, for example,
            when you want to differentiate the metrics of the extracted mesh
            based on the point information of this mesh.

        Returns
        -------
        facets: graphlow.GraphlowMesh
            Extracted facets mesh.
        """
        if add_original_index or pass_point_data:
            self.add_original_index()
        poly, scipy_fc_inc = self._extract_facets_impl()
        fc_inc = pt.phlower_array(scipy_fc_inc).to_tensor().to(self.dtype)
        fc_inc = pt.phlower_tensor(fc_inc.coalesce(), dimension={}).to(
            device=self.device
        )

        self.dict_sparse_tensor.update(
            {SparseMatrixName.FACET_CELL_INCIDENCE: fc_inc}, overwrite=True
        )

        extracted = GraphlowMesh(
            poly.cast_to_unstructured_grid(),
            dict_dimensions=self._dict_dimensions,
            dict_is_time_series=self._dict_is_time_series,
            float_precision=self.float_precision,
            device=self.device,
            dict_sparse_tensor=self.dict_sparse_tensor,
        )

        if not pass_point_data:
            return extracted

        point_rel_inc = self.compute_point_relative_incidence(extracted)
        new_point_tensors = self.dict_point_tensor.apply(
            lambda x: point_rel_inc @ x
        )
        extracted.dict_point_tensor.update(new_point_tensors, overwrite=True)
        return extracted

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
        cell_centers = pt.phlower_tensor(
            self.pvmesh.cell_centers().points,
            dimension=DEFAULT_DIMENSIONS[FeatureName.POINTS],
            dtype=self.dtype,
            device=self.device,
        )

        facet_idmap = {}

        for cell_id in range(n_cells):
            cell = vol.get_cell(cell_id)
            cell_center = cell_centers[cell_id]
            for j in range(cell.n_faces):
                face = cell.get_face(j).point_ids
                vtk_polygon_cell = [len(face), *face]

                # check orientation
                face_points = self.points[face]
                face_center = torch.mean(face_points, dim=0)
                side_vec = face_points - face_center
                cc2fc = face_center - cell_center
                cross = torch.linalg.cross(
                    side_vec, torch.roll(side_vec, shifts=-1, dims=0)
                )
                normal = torch.mean(cross, dim=0)
                dot = torch.dot(cc2fc.to_tensor(), normal.to_tensor())
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

        poly = pv.PolyData(self.points.to_numpy(), polygon_cells)
        for k, v in vol.point_data.items():
            poly.point_data[k] = v
        scipy_fc_inc = sp.csr_array(
            (sign_values, (row_indices, col_indices)),
            shape=(n_facets, n_cells),
        )
        return poly, scipy_fc_inc

    def convert_elemental2nodal(
        self,
        elemental_data: pt.PhlowerTensor,
        mode: Literal["mean", "conservative"] = "mean",
    ) -> pt.PhlowerTensor:
        return self._geometry_processor.convert_elemental2nodal(
            self, elemental_data, mode
        )

    def convert_nodal2elemental(
        self,
        nodal_data: pt.PhlowerTensor,
        mode: Literal["mean", "conservative"] = "mean",
    ) -> pt.PhlowerTensor:
        return self._geometry_processor.convert_nodal2elemental(
            self, nodal_data, mode
        )

    def compute_median(
        self,
        data: pt.PhlowerTensor,
        mode: Literal["elemental", "nodal"] = "elemental",
        n_hop: int = 1,
    ) -> pt.PhlowerTensor:
        return self._geometry_processor.compute_median(self, data, mode, n_hop)

    def compute_area_vecs(self) -> pt.PhlowerTensor:
        return self._geometry_processor.compute_area_vecs(self)

    def compute_areas(
        self, allow_negative_area: bool = False
    ) -> pt.PhlowerTensor:
        return self._geometry_processor.compute_areas(self, allow_negative_area)

    def compute_volumes(
        self, allow_negative_volume: bool = True
    ) -> pt.PhlowerTensor:
        return self._geometry_processor.compute_volumes(
            self, allow_negative_volume
        )

    def compute_normals(self) -> pt.PhlowerTensor:
        return self._geometry_processor.compute_normals(self)

    def compute_surface_volume(self) -> pt.PhlowerTensor:
        return self._geometry_processor.compute_surface_volume(self)

    def compute_isoAM(
        self,
        with_moment_matrix: bool = True,
        consider_volume: bool = False,
        normal_interp_mode: Literal["mean", "conservative"] = "conservative",
        eps: float | None = None,
    ) -> tuple[pt.PhlowerTensor, pt.PhlowerTensor | None]:
        return self._isoAM_processor.compute_isoAM(
            self, with_moment_matrix, consider_volume, normal_interp_mode, eps
        )

    def compute_isoAM_with_neumann(
        self,
        normal_weight: float = 10.0,
        with_moment_matrix: bool = True,
        consider_volume: bool = False,
        normal_interp_mode: Literal["mean", "conservative"] = "conservative",
        eps: float | None = None,
    ) -> tuple[pt.PhlowerTensor, pt.PhlowerTensor, pt.PhlowerTensor | None]:
        return self._isoAM_processor.compute_isoAM_with_neumann(
            self,
            normal_weight,
            with_moment_matrix,
            consider_volume,
            normal_interp_mode,
            eps,
        )

    def compute_cell_point_incidence(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        return self._graph_processor.compute_cell_point_incidence(
            self, refresh_cache
        )

    def compute_cell_adjacency(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        return self._graph_processor.compute_cell_adjacency(self, refresh_cache)

    def compute_point_adjacency(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        return self._graph_processor.compute_point_adjacency(
            self, refresh_cache
        )

    def compute_point_degree(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        return self._graph_processor.compute_point_degree(self, refresh_cache)

    def compute_cell_degree(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        return self._graph_processor.compute_cell_degree(self, refresh_cache)

    def compute_normalized_point_adjacency(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        return self._graph_processor.compute_normalized_point_adjacency(
            self, refresh_cache
        )

    def compute_normalized_cell_adjacency(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        return self._graph_processor.compute_normalized_cell_adjacency(
            self, refresh_cache
        )

    def compute_point_relative_incidence(
        self, other_mesh: IReadOnlyGraphlowMesh
    ) -> pt.PhlowerTensor:
        return self._graph_processor.compute_point_relative_incidence(
            self, other_mesh
        )

    def compute_cell_relative_incidence(
        self,
        other_mesh: IReadOnlyGraphlowMesh,
        minimum_n_sharing: int | None = None,
    ) -> pt.PhlowerTensor:
        return self._graph_processor.compute_cell_relative_incidence(
            self,
            other_mesh,
            minimum_n_sharing=minimum_n_sharing,
        )

    def compute_facet_cell_incidence(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        if (
            not refresh_cache
            and SparseMatrixName.FACET_CELL_INCIDENCE in self.dict_sparse_tensor
        ):
            return self.dict_sparse_tensor[
                SparseMatrixName.FACET_CELL_INCIDENCE
            ]
        _ = self.extract_facets(add_original_index=True)
        return self.dict_sparse_tensor[SparseMatrixName.FACET_CELL_INCIDENCE]

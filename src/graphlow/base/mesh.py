from __future__ import annotations

import pathlib
from typing import Any, Literal

import numpy as np
import pyvista as pv
import torch
from scipy import sparse as sp

from graphlow.base.dict_tensor import GraphlowDictTensor
from graphlow.base.mesh_interface import IReadOnlyGraphlowMesh
from graphlow.base.tensor_property import GraphlowTensorProperty
from graphlow.processors.geometry_processor import GeometryProcessor
from graphlow.processors.graph_processor import GraphProcessor
from graphlow.processors.isoAM_processor import IsoAMProcessor
from graphlow.util import array_handler, constants
from graphlow.util.enums import FeatureName
from graphlow.util.logger import get_logger

logger = get_logger(__name__)


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
        self._isoAM_processor = IsoAMProcessor()

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

        Parameters
        ----------
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

        Parameters
        ----------
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

    def extract_surface(
        self, add_original_index: bool = True, pass_points: bool = False
    ) -> GraphlowMesh:
        """Extract surface.

        Parameters
        ----------
        add_original_index: bool, optional [True]
            If True, add original index feature to enable relative incidence
            matrix computation.
        pass_points: bool, optional [False]
            If True, the extracted mesh will inherit the point tensor
            with gradients from this mesh. This parameter is used,
            for example, when you want to differentiate the metrics of
            the extracted mesh based on the point data of this mesh.

        Returns
        -------
        graphlow.GraphlowMesh
            Extracted surface mesh.
        """
        if add_original_index or pass_points:
            self.add_original_index()

        surface_mesh = self.pvmesh.extract_surface(
            pass_pointid=False, pass_cellid=False
        ).cast_to_unstructured_grid()

        if pass_points:
            dict_point_tensor = self._get_extracted_dict_point_tensor(
                surface_mesh
            )
            return GraphlowMesh(
                surface_mesh,
                dict_point_tensor=dict_point_tensor,
                device=self.device,
                dtype=self.dtype,
            )
        return GraphlowMesh(surface_mesh, device=self.device, dtype=self.dtype)

    def extract_cells(
        self,
        ind: Any,
        invert: bool = False,
        add_original_index: bool = True,
        pass_points: bool = False,
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
        pass_points: bool, optional [False]
            If True, the extracted mesh will inherit the point tensor
            with gradients from this mesh. This parameter is used,
            for example, when you want to differentiate the metrics of
            the extracted mesh based on the point data of this mesh.

        Returns
        -------
        graphlow.GraphlowMesh
            Extracted cells mesh.
        """
        if add_original_index or pass_points:
            self.add_original_index()

        extracted = self.pvmesh.extract_cells(
            ind, invert=invert
        ).cast_to_unstructured_grid()

        if pass_points:
            dict_point_tensor = self._get_extracted_dict_point_tensor(extracted)
            return GraphlowMesh(
                extracted,
                dict_point_tensor=dict_point_tensor,
                device=self.device,
                dtype=self.dtype,
            )
        return GraphlowMesh(extracted, device=self.device, dtype=self.dtype)

    def extract_facets(
        self, add_original_index: bool = True, pass_points: bool = False
    ) -> tuple[GraphlowMesh, torch.Tensor]:
        """Extract all internal/external facets of the volume mesh
        with (n_faces, n_cells)-shaped sparse signed incidence matrix

        Parameters
        ----------
        add_original_index: bool, optional [True]
            If True, add original index feature to enable relative incidence
            matrix computation.
        pass_points: bool, optional [False]
            If True, the extracted mesh will inherit the point tensor
            with gradients from this mesh. This parameter is used,
            for example, when you want to differentiate the metrics of
            the extracted mesh based on the point data of this mesh.

        Returns
        -------
        facets: graphlow.GraphlowMesh
            Extracted facets mesh.
        fc_inc: torch.Tensor
            (n_faces, n_cells)-shaped sparse signed incidence matrix.
        """
        if add_original_index or pass_points:
            self.add_original_index()
        poly, scipy_fc_inc = self._extract_facets_impl()
        fc_inc = array_handler.convert_to_torch_sparse_csr(
            scipy_fc_inc.astype(float),
            device=self.device,
            dtype=self.dtype,
        )

        if pass_points:
            return GraphlowMesh(
                poly,
                dict_point_tensor=self.dict_point_tensor,
                device=self.device,
                dtype=self.dtype,
            ), fc_inc

        return GraphlowMesh(poly, device=self.device, dtype=self.dtype), fc_inc

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

        poly = pv.PolyData(self.points.detach().numpy(), polygon_cells)
        scipy_fc_inc = sp.csr_array(
            (sign_values, (row_indices, col_indices)),
            shape=(n_facets, n_cells),
        )
        return poly, scipy_fc_inc

    def _get_extracted_dict_point_tensor(
        self, pvmesh: pv.UnstructuredGrid
    ) -> GraphlowDictTensor:
        if pvmesh.n_points > self.n_points:
            raise ValueError(
                "The provided mesh was not extracted from this object."
            )
        if FeatureName.ORIGINAL_INDEX not in pvmesh.point_data:
            raise ValueError(
                f"{FeatureName.ORIGINAL_INDEX} not found in "
                "Run mesh operation with add_original_index=True option."
            )

        col = torch.from_numpy(pvmesh.point_data[FeatureName.ORIGINAL_INDEX])
        row = torch.arange(len(col))
        value = torch.ones(len(col))
        indices = torch.stack([row, col], dim=0)
        size = (pvmesh.n_points, self.n_points)
        point_relative_incidence = array_handler.convert_to_torch_sparse_csr(
            torch.sparse_coo_tensor(indices, value, size=size),
            device=self.device,
            dtype=self.dtype,
        )
        extracted_points = point_relative_incidence @ self.points
        return GraphlowDictTensor({FeatureName.POINTS: extracted_points})

    def convert_elemental2nodal(
        self,
        elemental_data: torch.Tensor,
        mode: Literal["mean", "effective"] = "mean",
    ) -> torch.Tensor:
        return self._geometry_processor.convert_elemental2nodal(
            self, elemental_data, mode
        )

    def convert_nodal2elemental(
        self,
        nodal_data: torch.Tensor,
        mode: Literal["mean", "effective"] = "mean",
    ) -> torch.Tensor:
        return self._geometry_processor.convert_nodal2elemental(
            self, nodal_data, mode
        )

    def compute_median(
        self,
        data: torch.Tensor,
        mode: Literal["elemental", "nodal"] = "elemental",
        n_hop: int = 1,
    ) -> torch.Tensor:
        return self._geometry_processor.compute_median(self, data, mode, n_hop)

    def compute_area_vecs(self) -> torch.Tensor:
        return self._geometry_processor.compute_area_vecs(self)

    def compute_areas(self, allow_negative_area: bool = False) -> torch.Tensor:
        return self._geometry_processor.compute_areas(self, allow_negative_area)

    def compute_volumes(
        self, allow_negative_volume: bool = True
    ) -> torch.Tensor:
        return self._geometry_processor.compute_volumes(
            self, allow_negative_volume
        )

    def compute_normals(self) -> torch.Tensor:
        return self._geometry_processor.compute_normals(self)

    def compute_isoAM(
        self, with_moment_matrix: bool = True, consider_volume: bool = False
    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        return self._isoAM_processor.compute_isoAM(
            self, with_moment_matrix, consider_volume
        )

    def compute_isoAM_with_neumann(
        self,
        normal_weight: float = 10.0,
        with_moment_matrix: bool = True,
        consider_volume: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, None | torch.Tensor]:
        return self._isoAM_processor.compute_isoAM_with_neumann(
            self, normal_weight, with_moment_matrix, consider_volume
        )

    def compute_cell_point_incidence(self, cache: bool = True) -> torch.Tensor:
        return self._graph_processor.compute_cell_point_incidence(self, cache)

    def compute_cell_adjacency(self, cache: bool = True) -> torch.Tensor:
        return self._graph_processor.compute_cell_adjacency(self, cache)

    def compute_point_adjacency(self, cache: bool = True) -> torch.Tensor:
        return self._graph_processor.compute_point_adjacency(self, cache)

    def compute_point_degree(self, cache: bool = True) -> torch.Tensor:
        return self._graph_processor.compute_point_degree(self, cache)

    def compute_cell_degree(self, cache: bool = True) -> torch.Tensor:
        return self._graph_processor.compute_cell_degree(self, cache)

    def compute_normalized_point_adjacency(
        self, cache: bool = True
    ) -> torch.Tensor:
        return self._graph_processor.compute_normalized_point_adjacency(
            self, cache
        )

    def compute_normalized_cell_adjacency(
        self, cache: bool = True
    ) -> torch.Tensor:
        return self._graph_processor.compute_normalized_cell_adjacency(
            self, cache
        )

    def compute_point_relative_incidence(
        self, other_mesh: GraphlowMesh
    ) -> torch.Tensor:
        return self._graph_processor.compute_point_relative_incidence(
            self, other_mesh
        )

    def compute_cell_relative_incidence(
        self,
        other_mesh: GraphlowMesh,
        minimum_n_sharing: int | None = None,
    ) -> torch.Tensor:
        return self._graph_processor.compute_cell_relative_incidence(
            self,
            other_mesh,
            minimum_n_sharing=minimum_n_sharing,
        )

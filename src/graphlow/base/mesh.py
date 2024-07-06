
import pathlib

import numpy as np
import pyvista as pv
import torch
from typing_extensions import Self

from graphlow.base.dict_tensor import GraphlowDictTensor
from graphlow.base.tensor_property import GraphlowTensorProperty
from graphlow.processors.graph_processor import GraphProcessorMixin
from graphlow.util import constants
from graphlow.util.enums import FeatureName


class GraphlowMesh(GraphProcessorMixin):

    def __init__(
            self, mesh: pv.UnstructuredGrid,
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
        self._tensor_property = GraphlowTensorProperty(
            device=device, dtype=dtype)

        self._mesh = mesh.cast_to_unstructured_grid()
        self._dict_point_tensor = dict_point_tensor or GraphlowDictTensor(
            {}, length=self.n_points)
        if FeatureName.POINTS not in self._dict_point_tensor:
            self._dict_point_tensor.update(
                {FeatureName.POINTS: self.mesh.points})
        self._dict_cell_tensor = dict_cell_tensor or GraphlowDictTensor(
            {}, length=self.n_cells)
        self._dict_sparse_tensor = dict_sparse_tensor or GraphlowDictTensor(
            {}, length=None)

        self.send()
        return

    @property
    def mesh(self) -> pv.PointGrid:
        return self._mesh

    @property
    def points(self) -> torch.Tensor:
        return self.dict_point_tensor[FeatureName.POINTS]

    @property
    def n_points(self) -> int:
        return self._mesh.n_points

    @property
    def n_cells(self) -> int:
        return self._mesh.n_cells

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
            self, file_name: pathlib.Path | str, *,
            binary: bool = True,
            cast: bool = True,
            overwrite_features: bool = False,
            overwrite_file: bool = False):
        file_path = pathlib.Path(file_name)
        if not overwrite_file and file_path.exists():
            raise ValueError(f"{file_path} already exists.")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        self.copy_features_to_pyvista(overwrite=overwrite_features)

        if not cast:
            self.mesh.save(file_name, binary=binary)
            return

        ext = file_path.suffix.lstrip('.')
        if ext in constants.UNSTRUCTURED_GRID_EXTENSIONS:
            unstructured_grid = self.mesh.cast_to_unstructured_grid()
            unstructured_grid.save(file_name, binary=binary)
            return

        if ext in constants.POLYDATA_EXTENSIONS:
            if isinstance(self.mesh, pv.PolyData):
                self.mesh.save(file_name, binary=binary)
                return
            poly_data = self.mesh.extract_surface()
            poly_data.save(file_name, binary=binary)
            return

        raise ValueError(f"Unexpected extension: {ext}")

    def send(
            self, *,
            device: torch.device | int | None = None,
            dtype: torch.dtype | type | None = None):
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
            self.mesh.point_data, overwrite=overwrite)
        self.dict_cell_tensor.update(
            self.mesh.cell_data, overwrite=overwrite)
        return

    def copy_features_to_pyvista(self, *, overwrite: bool = False):
        """Copy point and cell tensor data to pyvista mesh.

        overwrite: bool
            If True, allow overwriting exsiting items. The default is False.
        """
        self.update_pyvista_data(
            self.dict_point_tensor, self.mesh.point_data, overwrite=overwrite)
        self.update_pyvista_data(
            self.dict_cell_tensor, self.mesh.cell_data, overwrite=overwrite)
        return

    def update_pyvista_data(
            self,
            dict_tensor: GraphlowDictTensor,
            pyvista_dataset: pv.DataSetAttributes, *,
            overwrite: bool = False):
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
        self.mesh.point_data[FeatureName.ORIGINAL_INDEX] = np.arange(
            self.mesh.n_points)
        self.mesh.cell_data[FeatureName.ORIGINAL_INDEX] = np.arange(
            self.mesh.n_cells)
        return

    def extract_surface(self, add_original_index: bool = True) -> Self:
        """Extract surface.

        Parameters
        ----------
        add_original_index: bool
            If True, add original index feature to enable relative incidence
            matrix computation. The default is True.
        """
        if add_original_index:
            self.add_original_index()

        surface_mesh = self.mesh.extract_surface(
            pass_pointid=False, pass_cellid=False).cast_to_unstructured_grid()
        return GraphlowMesh(surface_mesh, device=self.device, dtype=self.dtype)

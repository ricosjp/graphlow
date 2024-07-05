
import numpy as np
import pyvista as pv
import torch
from typing_extensions import Self

from graphlow.base.dict_tensor import GraphlowDictTensor
from graphlow.base.tensor_property import GraphlowTensorProperty
from graphlow.processors.graph_processor import GraphProcessorMixin


class GraphlowMesh(GraphProcessorMixin):

    def __init__(
            self, mesh: pv.PointGrid,
            *,
            dict_point_data: GraphlowDictTensor | None = None,
            dict_cell_data: GraphlowDictTensor | None = None,
            dict_sparse_data: GraphlowDictTensor | None = None,
            device: torch.device | int = -1,
            dtype: torch.dtype | type | None = None,
    ):
        """Initialize GraphlowMesh object.

        Parameters
        ----------
        mesh: pyvista.PointGrid
            Mesh data.
        dict_point_data: GraphlowDictTensor | None
        dict_tensor: dict[str, graphlow.ArrayDataType]
            Dict of tensor data.
        device: torch.device | int
            Device ID. int < 0 implies CPU.
        dtype: torch.dtype | type | None
            Data type.
        """
        self.tensor_property = GraphlowTensorProperty(
            device=device, dtype=dtype)

        self._mesh = mesh
        self._dict_point_data = dict_point_data or GraphlowDictTensor(
            {'points': self.mesh.points}, length=self.mesh.n_points)
        self._dict_cell_data = dict_cell_data or GraphlowDictTensor(
            {}, length=self.mesh.n_cells)
        self._dict_sparse_data = dict_sparse_data or GraphlowDictTensor(
            {}, length=None)

        self.to(self.tensor_property.device)
        return

    @property
    def mesh(self) -> pv.PointGrid:
        return self._mesh

    @property
    def dict_point_data(self) -> GraphlowDictTensor:
        return self._dict_point_data

    @property
    def dict_cell_data(self) -> GraphlowDictTensor:
        return self._dict_cell_data

    @property
    def dict_sparse_data(self) -> GraphlowDictTensor:
        return self._dict_sparse_data

    def to(
            self, device: torch.device | int,
            dtype: torch.dtype | type | None = None):
        """Convert features to the specified device and dtype. It does not
        modify pyvista mesh.

        Parameters
        ----------
        device: torch.device | int
        dtype: torch.dtype | type | None
        """
        self.tensor_property.device = device
        self.tensor_property.dtype = dtype or self.tensor_property.dtype

        self._dict_point_data.to(
            device=self.tensor_property.device,
            dtype=self.tensor_property.dtype)
        self._dict_cell_data.to(
            device=self.tensor_property.device,
            dtype=self.tensor_property.dtype)
        self._dict_sparse_data.to(
            device=self.tensor_property.device,
            dtype=self.tensor_property.dtype)
        return

    def add_original_indices(self):
        self.mesh.point_data['original_index'] = np.arange(self.mesh.n_points)
        self.mesh.cell_data['original_index'] = np.arange(self.mesh.n_cells)
        return

    def convert_to_surface(self) -> Self:
        self.add_original_indices()
        surface_mesh = self.mesh.extract_surface(
            pass_pointid=True, pass_cellid=True).cast_to_unstructured_grid()
        return GraphlowMesh(
            surface_mesh,
            device=self.tensor_property.device,
            dtype=self.tensor_property.dtype)

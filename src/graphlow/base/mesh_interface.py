from __future__ import annotations

import abc

import pyvista as pv
import torch

from graphlow.base.dict_tensor import GraphlowDictTensor


class IReadOnlyGraphlowMesh(metaclass=abc.ABCMeta):
    # Write methods you want to share with processors.
    @property
    @abc.abstractmethod
    def pvmesh(self) -> pv.PointGrid:
        pass

    @property
    @abc.abstractmethod
    def points(self) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def n_points(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def n_cells(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def dict_point_tensor(self) -> GraphlowDictTensor:
        pass

    @property
    @abc.abstractmethod
    def dict_cell_tensor(self) -> GraphlowDictTensor:
        pass

    @property
    @abc.abstractmethod
    def dict_sparse_tensor(self) -> GraphlowDictTensor:
        pass

    @property
    @abc.abstractmethod
    def device(self) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def dtype(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def compute_areas(self, raise_negative_area: bool = True) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def compute_volumes(
        self, raise_negative_volume: bool = True
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def compute_normals(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def compute_IsoAM(
        self, with_moment_matrix: bool, consider_volume: bool
    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        pass

    @abc.abstractmethod
    def compute_cell_point_incidence(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def compute_cell_adjacency(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def compute_point_adjacency(self) -> torch.Tensor:
        pass

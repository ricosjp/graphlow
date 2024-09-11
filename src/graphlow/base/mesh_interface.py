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

from __future__ import annotations

import abc

import pyvista as pv
import torch

from graphlow.base.dict_tensor import GraphlowDictTensor


class IGraphlowMesh(metaclass=abc.ABCMeta):
    # Write methods you want to share with processors.
    @property
    @abc.abstractmethod
    def pvmesh(self) -> pv.PointGrid:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def points(self) -> torch.Tensor:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def n_points(self) -> int:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def n_cells(self) -> int:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def dict_point_tensor(self) -> GraphlowDictTensor:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def dict_cell_tensor(self) -> GraphlowDictTensor:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def dict_sparse_tensor(self) -> GraphlowDictTensor:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def device(self) -> torch.Tensor:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def dtype(self) -> torch.Tensor:
        raise NotImplementedError()

from __future__ import annotations

import abc
from typing import Literal

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
    def extract_surface(
        self, add_original_index: bool
    ) -> IReadOnlyGraphlowMesh:
        pass

    @abc.abstractmethod
    def extract_facets(
        self, add_original_index: bool
    ) -> tuple[IReadOnlyGraphlowMesh, torch.Tensor]:
        pass

    @abc.abstractmethod
    def convert_elemental2nodal(
        self,
        elemental_data: torch.Tensor,
        mode: Literal["mean", "effective"],
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def convert_nodal2elemental(
        self, nodal_data: torch.Tensor, mode: Literal["mean", "effective"]
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def compute_areas(self, raise_negative_area: bool) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def compute_volumes(self, raise_negative_volume: bool) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def compute_normals(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def compute_isoAM(
        self, with_moment_matrix: bool, consider_volume: bool
    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        pass

    @abc.abstractmethod
    def compute_isoAM_with_neumann(
        self,
        mesh: IReadOnlyGraphlowMesh,
        normal_weight: float,
        with_moment_matrix: bool,
        consider_volume: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, None | torch.Tensor]:
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

    @abc.abstractmethod
    def compute_point_relative_incidence(
        self, other_mesh: IReadOnlyGraphlowMesh
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def compute_cell_relative_incidence(
        self, other_mesh: IReadOnlyGraphlowMesh, minimum_n_sharing: int | None
    ) -> torch.Tensor:
        pass

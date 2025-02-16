from __future__ import annotations

import abc
from typing import Any, Literal

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
        self, add_original_index: bool = True, pass_points: bool = False
    ) -> IReadOnlyGraphlowMesh:
        pass

    @abc.abstractmethod
    def extract_cells(
        self,
        ind: Any,
        invert: bool = False,
        add_original_index: bool = True,
        pass_points: bool = False,
    ) -> IReadOnlyGraphlowMesh:
        pass

    @abc.abstractmethod
    def extract_facets(
        self, add_original_index: bool = True, pass_points: bool = False
    ) -> tuple[IReadOnlyGraphlowMesh, torch.Tensor]:
        pass

    @abc.abstractmethod
    def convert_elemental2nodal(
        self,
        elemental_data: torch.Tensor,
        mode: Literal["mean", "effective"] = "mean",
    ) -> torch.Tensor:
        """Convert elemental data to nodal data.

        Parameters
        ----------
        elemental_data: torch.Tensor
            elemental data to convert.
        mode: "mean", or "effective", default: "mean"
            The way to convert.
            - "mean": averages the values of \
                elements connected to each node. (default)
            - "effective": distributes element information \
                to the connected nodes, ensuring consistent volume.

        Returns
        -------
        torch.Tensor
        """
        pass

    @abc.abstractmethod
    def convert_nodal2elemental(
        self,
        nodal_data: torch.Tensor,
        mode: Literal["mean", "effective"] = "mean",
    ) -> torch.Tensor:
        """Convert nodal data to elemental data.

        Parameters
        ----------
        nodal_data: torch.Tensor
            nodal data to convert.
        mode: "mean", or "effective", default: "mean"
            The way to convert.
            - "mean": averages the values of \
                nodes connected to each element. (default)
            - "effective": distributes node information \
                to the connected elements, ensuring consistent volume.

        Returns
        -------
        torch.Tensor
        """
        pass

    @abc.abstractmethod
    def compute_median(
        self,
        data: torch.Tensor,
        mode: Literal["elemental", "nodal"] = "elemental",
        n_hop: int = 1,
    ) -> torch.Tensor:
        """Perform median filter according with adjacency of the mesh.

        Parameters
        ----------
        data: torch.Tensor
            data to be filtered.
        mode: str, "elemental", or "nodal", default: "elemental"
            specify the mode of the data.
        n_hop: int, optional [1]
            The number of hops to make filtering.

        Returns
        -------
        torch.Tensor
        """
        pass

    @abc.abstractmethod
    def compute_area_vecs(self) -> torch.Tensor:
        """Compute (n_elements, dims)-shaped area vectors.

        Available celltypes are:
        VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON

        Returns
        -------
        torch.Tensor[float]
        """
        pass

    @abc.abstractmethod
    def compute_areas(self, allow_negative_area: bool = False) -> torch.Tensor:
        """Compute (n_elements,)-shaped areas.

        Available celltypes are:
        VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON

        Parameters
        ----------
        allow_negative_area : bool, optional [False]

        Returns
        -------
        torch.Tensor[float]
        """
        pass

    @abc.abstractmethod
    def compute_volumes(
        self, allow_negative_volume: bool = True
    ) -> torch.Tensor:
        """Compute (n_elements,)-shaped volumes.

        Available celltypes are:
        VTK_TETRA, VTK_PYRAMID, VTK_WEDGE, VTK_VOXEL,
        VTK_HEXAHEDRON, VTK_POLYHEDRON

        Parameters
        ----------
        allow_negative_volume: bool, optional [True]
            If True, compute the signed volume.

        Returns
        -------
        torch.Tensor[float]
        """
        pass

    @abc.abstractmethod
    def compute_normals(self) -> torch.Tensor:
        """Compute (n_elements, dims)-shaped normals.

        Available celltypes are:
        VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON

        Returns
        -------
        torch.Tensor[float]
        """
        pass

    @abc.abstractmethod
    def compute_isoAM(
        self, with_moment_matrix: bool = True, consider_volume: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute (dims, n_points, n_points)-shaped isoAM.

        Parameters
        ----------
        with_moment_matrix: bool, optional [True]
            If True, scale the matrix with moment matrices, which are
            tensor products of relative position tensors.
        consider_volume: bool, optional [False]
            If True, consider effective volume of each vertex.

        Returns
        -------
        isoAM: torch.Tensor | None
            (dims, n_points, n_points)-shaped sparse csr tensor
        Minv: torch.Tensor | None
            if `with_moment_matrix` is True,
                return (n_points, dims, dims)-shaped tensor
            if `with_moment_matrix` is False,
                return None
        """
        pass

    @abc.abstractmethod
    def compute_isoAM_with_neumann(
        self,
        mesh: IReadOnlyGraphlowMesh,
        normal_weight: float = 10.0,
        with_moment_matrix: bool = True,
        consider_volume: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Compute (dims, n_points, n_points)-shaped
        Neumann boundary model IsoAM.

        Parameters
        ----------
        normal_weight: float, optional [10.0]
            Weight of the normal vector.
        with_moment_matrix: bool, optional [True]
            If True, scale the matrix with moment matrices, which are
            tensor products of relative position tensors.
        consider_volume: bool, optional [False]
            If True, consider effective volume of each vertex.

        Returns
        -------
        NIsoAM: torch.Tensor
            (dims, n_points, n_points)-shaped sparse csr tensor
        weighted_normals: torch.Tensor
            (n_points, dims)-shaped tensor
        Minv: torch.Tensor | None
            if `with_moment_matrix` is True,
                return (n_points, dims, dims)-shaped tensor
            if `with_moment_matrix` is False,
                return None
        """
        pass

    @abc.abstractmethod
    def compute_cell_point_incidence(self, cache: bool = True) -> torch.Tensor:
        """Compute (n_cells, n_points)-shaped sparse incidence matrix.
        The method is cached.

        Parameters
        ----------
        cache: bool, optional [True]
            If True, the result is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_cells, n_points)-shaped sparse csr tensor.
        """
        pass

    @abc.abstractmethod
    def compute_cell_adjacency(self, cache: bool = True) -> torch.Tensor:
        """Compute (n_cells, n_cells)-shaped sparse adjacency matrix including
        self-loops. The method is cached.

        Parameters
        ----------
        cache: bool, optional [True]
            If True, the result is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_cells, n_cells)-shaped sparse csr tensor.
        """
        pass

    @abc.abstractmethod
    def compute_point_adjacency(self, cache: bool = True) -> torch.Tensor:
        """Compute (n_points, n_points)-shaped sparse adjacency matrix
        including self-loops. The method is cached.

        Parameters
        ----------
        cache: bool, optional [True]
            If True, the result is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_points, n_points)-shaped sparse csr tensor.
        """
        pass

    @abc.abstractmethod
    def compute_point_degree(self, cache: bool = True) -> torch.Tensor:
        """Compute (n_points, n_points)-shaped degree matrix.

        Parameters
        ----------
        cache: bool, optional [True]
            If True, the result is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_points, n_points)-shaped sparse csr tensor.
        """
        pass

    @abc.abstractmethod
    def compute_cell_degree(self, cache: bool = True) -> torch.Tensor:
        """Compute (n_cells, n_cells)-shaped degree matrix.

        Parameters
        ----------
        cache: bool, optional [True]
            If True, the result is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_cells, n_cells)-shaped sparse csr tensor.
        """
        pass

    @abc.abstractmethod
    def compute_normalized_point_adjacency(
        self, cache: bool = True
    ) -> torch.Tensor:
        """Compute (n_points, n_points)-shaped normalized adjacency matrix.

        Parameters
        ----------
        cache: bool, optional [True]
            If True, the result is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_points, n_points)-shaped sparse csr tensor.
        """
        pass

    @abc.abstractmethod
    def compute_normalized_cell_adjacency(
        self, cache: bool = True
    ) -> torch.Tensor:
        """Compute (n_cells, n_cells)-shaped normalized adjacency matrix.

        Parameters
        ----------
        cache: bool, optional [True]
            If True, the result is cached.

        Returns
        -------
        torch.Tensor[float]
            (n_cells, n_cells)-shaped sparse csr tensor.
        """
        pass

    @abc.abstractmethod
    def compute_point_relative_incidence(
        self, other_mesh: IReadOnlyGraphlowMesh
    ) -> torch.Tensor:
        """Compute (n_points_other, n_points_self)-shaped sparse incidence
        matrix based on points.

        Parameters
        ----------
        other_mesh: graphlow.GraphlowMesh
            Other mesh object to be

        Returns
        -------
        torch.Tensor[float]
            (n_points_other, n_points_self)-shaped sparse csr tensor.
        """
        pass

    @abc.abstractmethod
    def compute_cell_relative_incidence(
        self,
        other_mesh: IReadOnlyGraphlowMesh,
        minimum_n_sharing: int | None = None,
    ) -> torch.Tensor:
        """Compute (n_cells_other, n_cells_self)-shaped sparse incidence
        matrix based on cells.

        Parameters
        ----------
        other_mesh: graphlow.GraphlowMesh
            Other mesh object to be
        minimum_n_sharing: int | None
            Minimum number of sharing points to define connectivity. If not
            set, it will be the number of points for each cell.

        Returns
        -------
        torch.Tensor[float]
            (n_cells_other, n_cells_self)-shaped sparse csr tensor.
        """
        pass

from __future__ import annotations

import abc
from typing import Literal

import phlower_tensor as pt
import pyvista as pv
import torch
from phlower_tensor.collections import IPhlowerTensorCollections
from pyvista.core._typing_core import VectorLike

from graphlow.util.enums import FloatPrecision


class IReadOnlyGraphlowMesh(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def pvmesh(self) -> pv.UnstructuredGrid:
        pass

    @property
    @abc.abstractmethod
    def points(self) -> pt.PhlowerTensor:
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
    def dict_point_tensor(self) -> IPhlowerTensorCollections:
        pass

    @property
    @abc.abstractmethod
    def dict_cell_tensor(self) -> IPhlowerTensorCollections:
        pass

    @property
    @abc.abstractmethod
    def dict_sparse_tensor(self) -> IPhlowerTensorCollections:
        pass

    @property
    @abc.abstractmethod
    def float_precision(self) -> FloatPrecision:
        pass

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        pass

    @property
    @abc.abstractmethod
    def dtype(self) -> torch.dtype:
        pass

    @abc.abstractmethod
    def extract_surface(
        self, add_original_index: bool = True, pass_point_data: bool = False
    ) -> IReadOnlyGraphlowMesh:
        pass

    @abc.abstractmethod
    def extract_cells(
        self,
        ind: VectorLike[int],
        invert: bool = False,
        add_original_index: bool = True,
        pass_point_data: bool = False,
        pass_cell_data: bool = False,
    ) -> IReadOnlyGraphlowMesh:
        pass

    @abc.abstractmethod
    def extract_facets(
        self,
        add_original_index: bool = True,
        pass_point_data: bool = False,
    ) -> IReadOnlyGraphlowMesh:
        pass

    @abc.abstractmethod
    def convert_elemental2nodal(
        self,
        elemental_data: pt.PhlowerTensor,
        mode: Literal["mean", "conservative"] = "mean",
    ) -> pt.PhlowerTensor:
        """Convert elemental data to nodal data.

        Parameters
        ----------
        elemental_data: pt.PhlowerTensor
            elemental data to convert.
        mode: "mean", or "conservative", default: "mean"
            The way to convert.
            - "mean": For each node, \
                we consider all the elements that share this node \
                and compute the average of their values. \
                This approach provides \
                a smoothed representation at each node.
            - "conservative": For each element, \
                we consider all the nodes that share this element \
                and distribute the element value to them equally. \
                The values are then summed at each node. \
                This approach ensures that the total quantity \
                (such as mass or volume) is conserved.

        Returns
        -------
        pt.PhlowerTensor
        """
        pass

    @abc.abstractmethod
    def convert_nodal2elemental(
        self,
        nodal_data: pt.PhlowerTensor,
        mode: Literal["mean", "conservative"] = "mean",
    ) -> pt.PhlowerTensor:
        """Convert nodal data to elemental data.

        Parameters
        ----------
        nodal_data: pt.PhlowerTensor
            nodal data to convert.
        mode: "mean", or "conservative", default: "mean"
            The way to convert.
            - "mean": For each element, \
                we consider all the nodes that share this element \
                and compute the average of their values. \
                This approach provides \
                a smoothed representation at each element.
            - "conservative": For each node, \
                we consider all the elements that share this node \
                and distribute the node value to them equally. \
                The values are then summed at each element. \
                This approach ensures that the total quantity \
                (such as mass or volume) is conserved.

        Returns
        -------
        pt.PhlowerTensor
        """
        pass

    @abc.abstractmethod
    def compute_median(
        self,
        data: pt.PhlowerTensor,
        mode: Literal["elemental", "nodal"] = "elemental",
        n_hop: int = 1,
    ) -> pt.PhlowerTensor:
        """Perform median filter according with adjacency of the mesh.

        Parameters
        ----------
        data: pt.PhlowerTensor
            data to be filtered.
        mode: str, "elemental", or "nodal", default: "elemental"
            specify the mode of the data.
        n_hop: int, optional [1]
            The number of hops to make filtering.

        Returns
        -------
        pt.PhlowerTensor
        """
        pass

    @abc.abstractmethod
    def compute_area_vecs(self) -> pt.PhlowerTensor:
        """Compute (n_elements, dims)-shaped area vectors.

        Available celltypes are:
        VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON

        Returns
        -------
        pt.PhlowerTensor
        """
        pass

    @abc.abstractmethod
    def compute_areas(
        self, allow_negative_area: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_elements,)-shaped areas.

        Available celltypes are:
        VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON

        Parameters
        ----------
        allow_negative_area : bool, optional [False]

        Returns
        -------
        pt.PhlowerTensor
        """
        pass

    @abc.abstractmethod
    def compute_volumes(
        self, allow_negative_volume: bool = True
    ) -> pt.PhlowerTensor:
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
        pt.PhlowerTensor
        """
        pass

    @abc.abstractmethod
    def compute_normals(self) -> pt.PhlowerTensor:
        """Compute (n_elements, dims)-shaped normals.

        Available celltypes are:
        VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON

        Returns
        -------
        pt.PhlowerTensor
        """
        pass

    @abc.abstractmethod
    def compute_surface_volume(self) -> pt.PhlowerTensor:
        """Compute (1,)-shaped surface volume.

        Available celltypes are:
        VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON

        Returns
        -------
        pt.PhlowerTensor
        """
        pass

    @abc.abstractmethod
    def compute_isoAM(
        self,
        with_moment_matrix: bool = True,
        consider_volume: bool = False,
        normal_interp_mode: Literal["mean", "conservative"] = "conservative",
    ) -> tuple[pt.PhlowerTensor, pt.PhlowerTensor | None]:
        """Compute (dims, n_points, n_points)-shaped isoAM.

        Parameters
        ----------
        with_moment_matrix: bool, optional [True]
            If True, scale the matrix with moment matrices, which are
            tensor products of relative position tensors.
        consider_volume: bool, optional [False]
            If True, consider effective volume of each vertex.
        normal_interp_mode: Literal["mean", "conservative"], \
            default: "conservative"
            The way to interpolate normals. cf. convert_elemental2nodal.
            - "mean": averages the values of \
                nodes connected to each element.
            - "conservative": distributes node information \
                to the connected elements, ensuring consistent volume.

        Returns
        -------
        isoAM: pt.PhlowerTensor | None
            (dims, n_points, n_points)-shaped sparse coo tensor
        Minv: pt.PhlowerTensor | None
            if `with_moment_matrix` is True,
                return (n_points, dims, dims)-shaped tensor
            if `with_moment_matrix` is False,
                return None
        """
        pass

    @abc.abstractmethod
    def compute_isoAM_with_neumann(
        self,
        normal_weight: float = 10.0,
        with_moment_matrix: bool = True,
        consider_volume: bool = False,
        normal_interp_mode: Literal["mean", "conservative"] = "conservative",
    ) -> tuple[pt.PhlowerTensor, pt.PhlowerTensor, pt.PhlowerTensor | None]:
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
        normal_interp_mode: Literal["mean", "conservative"], \
            default: "conservative"
            The way to interpolate normals. cf. convert_elemental2nodal.
            - "mean": averages the values of \
                nodes connected to each element.
            - "conservative": distributes node information \
                to the connected elements, ensuring consistent volume.

        Returns
        -------
        NIsoAM: pt.PhlowerTensor
            (dims, n_points, n_points)-shaped sparse coo tensor
        weighted_normals: pt.PhlowerTensor
            (n_points, dims)-shaped tensor
        Minv: pt.PhlowerTensor | None
            if `with_moment_matrix` is True,
                return (n_points, dims, dims)-shaped tensor
            if `with_moment_matrix` is False,
                return None
        """
        pass

    @abc.abstractmethod
    def compute_cell_point_incidence(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_cells, n_points)-shaped sparse incidence matrix.
        The result is cached in the mesh object.

        Parameters
        ----------
        refresh_cache: bool, optional [False]
            If True, recompute the incidence matrix.
            Otherwise, return the cached result if available.

        Returns
        -------
        pt.PhlowerTensor
            (n_cells, n_points)-shaped sparse coo tensor.
        """
        pass

    @abc.abstractmethod
    def compute_cell_adjacency(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_cells, n_cells)-shaped sparse adjacency matrix including
        self-loops. The result is cached in the mesh object.

        Parameters
        ----------
        refresh_cache: bool, optional [False]
            If True, recompute the adjacency matrix.
            Otherwise, return the cached result if available.

        Returns
        -------
        pt.PhlowerTensor
            (n_cells, n_cells)-shaped sparse coo tensor.
        """
        pass

    @abc.abstractmethod
    def compute_point_adjacency(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_points, n_points)-shaped sparse adjacency matrix
        including self-loops. The result is cached in the mesh object.

        Parameters
        ----------
        refresh_cache: bool, optional [False]
            If True, recompute the adjacency matrix.
            Otherwise, return the cached result if available.

        Returns
        -------
        pt.PhlowerTensor
            (n_points, n_points)-shaped sparse coo tensor.
        """
        pass

    @abc.abstractmethod
    def compute_point_degree(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_points, n_points)-shaped degree matrix.
        The result is cached in the mesh object.

        Parameters
        ----------
        refresh_cache: bool, optional [False]
            If True, recompute the degree matrix.
            Otherwise, return the cached result if available.

        Returns
        -------
        pt.PhlowerTensor
            (n_points, n_points)-shaped sparse coo tensor.
        """
        pass

    @abc.abstractmethod
    def compute_cell_degree(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_cells, n_cells)-shaped degree matrix.
        The result is cached in the mesh object.

        Parameters
        ----------
        refresh_cache: bool, optional [False]
            If True, recompute the degree matrix.
            Otherwise, return the cached result if available.

        Returns
        -------
        pt.PhlowerTensor
            (n_cells, n_cells)-shaped sparse coo tensor.
        """
        pass

    @abc.abstractmethod
    def compute_normalized_point_adjacency(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_points, n_points)-shaped normalized adjacency matrix.
        The result is cached in the mesh object.

        Parameters
        ----------
        refresh_cache: bool, optional [False]
            If True, recompute the normalized adjacency matrix.
            Otherwise, return the cached result if available.

        Returns
        -------
        pt.PhlowerTensor
            (n_points, n_points)-shaped sparse coo tensor.
        """
        pass

    @abc.abstractmethod
    def compute_normalized_cell_adjacency(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_cells, n_cells)-shaped normalized adjacency matrix.
        The result is cached in the mesh object.

        Parameters
        ----------
        refresh_cache: bool, optional [False]
            If True, recompute the normalized adjacency matrix.
            Otherwise, return the cached result if available.

        Returns
        -------
        pt.PhlowerTensor
            (n_cells, n_cells)-shaped sparse coo tensor.
        """
        pass

    @abc.abstractmethod
    def compute_point_relative_incidence(
        self, other_mesh: IReadOnlyGraphlowMesh
    ) -> pt.PhlowerTensor:
        """Compute (n_points_other, n_points_self)-shaped sparse incidence
        matrix based on points.

        Parameters
        ----------
        other_mesh: graphlow.GraphlowMesh
            Other mesh object to be

        Returns
        -------
        pt.PhlowerTensor
            (n_points_other, n_points_self)-shaped sparse coo tensor.
        """
        pass

    @abc.abstractmethod
    def compute_cell_relative_incidence(
        self,
        other_mesh: IReadOnlyGraphlowMesh,
        minimum_n_sharing: int | None = None,
    ) -> pt.PhlowerTensor:
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
        pt.PhlowerTensor
            (n_cells_other, n_cells_self)-shaped sparse coo tensor.
        """
        pass

    @abc.abstractmethod
    def compute_facet_cell_incidence(
        self, refresh_cache: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_facets, n_cells)-shaped sparse incidence matrix.

        Parameters
        ----------
        refresh_cache: bool, optional [False]
            If True, recompute the incidence matrix.
            Otherwise, return the cached result if available.

        Returns
        -------
        pt.PhlowerTensor
            (n_facets, n_cells)-shaped sparse coo tensor.
        """
        pass

from typing import Literal

import numpy as np
import phlower_tensor as pt
import pyvista as pv
import torch

from graphlow.base.mesh_interface import IReadOnlyGraphlowMesh
from graphlow.util.logger import get_logger

logger = get_logger(__name__)


class GeometryProcessor:
    """A class for geometry processing."""

    def __init__(self) -> None:
        pass

    def convert_elemental2nodal(
        self,
        mesh: IReadOnlyGraphlowMesh,
        elemental_data: pt.PhlowerTensor,
        mode: Literal["mean", "conservative"] = "mean",
    ) -> pt.PhlowerTensor:
        """Convert elemental data to nodal data.

        Parameters
        ----------
        mesh: GraphlowMesh
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
        pc_inc = mesh.compute_cell_point_incidence().transpose(
            0, 1
        )  # (n_points, n_cells)
        if mode == "mean":
            n_cells_per_point = (
                pc_inc.to_tensor().sum(dim=1).to_dense()
            )  # (n_points,)
            mean_pc_inc = pc_inc * n_cells_per_point.reciprocal().unsqueeze(
                1
            )  # (n_points, n_cells)
            nodal_data = mean_pc_inc @ elemental_data
            return nodal_data
        if mode == "conservative":
            n_points_per_cell = (
                pc_inc.to_tensor().sum(dim=0).to_dense()
            )  # (n_cells,)
            conservative_pc_inc = (
                pc_inc * n_points_per_cell.reciprocal().unsqueeze(0)
            )  # (n_points, n_cells)
            nodal_data = conservative_pc_inc @ elemental_data
            return nodal_data
        raise ValueError(f"Invalid mode: {mode}")

    def convert_nodal2elemental(
        self,
        mesh: IReadOnlyGraphlowMesh,
        nodal_data: pt.PhlowerTensor,
        mode: Literal["mean", "conservative"] = "mean",
    ) -> pt.PhlowerTensor:
        """Convert nodal data to elemental data.

        Parameters
        ----------
        mesh: GraphlowMesh
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
        cp_inc = mesh.compute_cell_point_incidence()  # (n_cells, n_points)
        if mode == "mean":
            n_points_per_cell = (
                cp_inc.to_tensor().sum(dim=1).to_dense()
            )  # (n_cells,)
            mean_cp_inc = cp_inc * n_points_per_cell.reciprocal().unsqueeze(
                1
            )  # (n_cells, n_points)
            nodal_data = mean_cp_inc @ nodal_data
            return nodal_data
        if mode == "conservative":
            n_cells_per_point = (
                cp_inc.to_tensor().sum(dim=0).to_dense()
            )  # (n_points,)
            conservative_cp_inc = (
                cp_inc * n_cells_per_point.reciprocal().unsqueeze(0)
            )  # (n_cells, n_points)
            nodal_data = conservative_cp_inc @ nodal_data
            return nodal_data
        raise ValueError(f"Invalid mode: {mode}")

    def compute_median(
        self,
        mesh: IReadOnlyGraphlowMesh,
        data: pt.PhlowerTensor,
        mode: Literal["nodal", "elemental"] = "elemental",
        n_hop: int = 1,
    ) -> pt.PhlowerTensor:
        """Perform median filter according with adjacency of the mesh.

        Parameters
        ----------
        mesh: GraphlowMesh
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
        if n_hop < 0:
            raise ValueError(f"Invalid n_hop: {n_hop}")
        if n_hop == 0:
            return data

        if mode == "elemental":
            adj = mesh.compute_cell_adjacency()
        elif mode == "nodal":
            adj = mesh.compute_point_adjacency()
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if adj.shape[0] != data.shape[0]:
            raise ValueError(
                f"Input data shape does not match \
                the specified mode: {data.shape} != {adj.shape}"
            )

        N = data.shape[0]

        n_hop_adj = adj
        for _ in range(n_hop - 1):
            n_hop_adj = n_hop_adj @ adj

        out = data.detach().clone()
        rows, cols = n_hop_adj.indices()
        # To avoid huge memory consumption, we iterate over the rows
        for i in range(N):
            nhop_neighbors = cols[rows == i]
            median_value = torch.median(data[nhop_neighbors])
            out[i] = median_value
        return out

    def compute_area_vecs(
        self, mesh: IReadOnlyGraphlowMesh
    ) -> pt.PhlowerTensor:
        """Compute (n_elements, dims)-shaped area vectors.

        Available celltypes are:
        VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON

        Parameters
        ----------
        mesh: GraphlowMesh

        Returns
        -------
        pt.PhlowerTensor
        """
        area_vecs_by_celltype = {
            pv.CellType.TRIANGLE: self._tri_area_vecs,
            pv.CellType.QUAD: self._quad_area_vecs,
            pv.CellType.POLYGON: self._poly_area_vecs,
        }
        area_vecs = pt.phlower_tensor(
            torch.zeros((mesh.n_cells, mesh.points.shape[1]), dtype=mesh.dtype),
            dimension={"L": 2},
        ).to(device=mesh.device)
        celltypes = mesh.pvmesh.celltypes

        # non-polygon cells
        nonpoly_mask = celltypes != pv.CellType.POLYGON
        if np.any(nonpoly_mask):
            nonpolys = mesh.extract_cells(nonpoly_mask, pass_point_data=True)
            nonpolys_dict = nonpolys.pvmesh.cells_dict
            for celltype, cells in nonpolys_dict.items():
                if celltype not in area_vecs_by_celltype:
                    raise KeyError(
                        f"Unavailable celltype: {pv.CellType(celltype).name}"
                    )
                mask = celltypes == celltype
                cell_points = nonpolys.points[cells]
                area_vecs[mask] = area_vecs_by_celltype[celltype](cell_points)

        # polygon cells
        poly_mask = celltypes == pv.CellType.POLYGON
        if np.any(poly_mask):
            polys = mesh.extract_cells(poly_mask, pass_point_data=True)
            area_vecs[poly_mask] = self._poly_area_vecs(polys.points, polys)
        return area_vecs

    def compute_areas(
        self, mesh: IReadOnlyGraphlowMesh, allow_negative_area: bool = False
    ) -> pt.PhlowerTensor:
        """Compute (n_elements,)-shaped areas.

        Available celltypes are:
        VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON

        Parameters
        ----------
        mesh: GraphlowMesh
        allow_negative_area : bool, optional [False]

        Returns
        -------
        pt.PhlowerTensor
        """
        area_vecs = mesh.compute_area_vecs()
        areas: pt.PhlowerTensor = torch.linalg.vector_norm(area_vecs, dim=1)
        if not allow_negative_area and torch.any(areas.to_tensor() < 0.0):
            indices = (areas.to_tensor() < 0).nonzero(as_tuple=True)
            raise ValueError(f"Negative area found: cell indices: {indices}")
        return areas

    def compute_volumes(
        self, mesh: IReadOnlyGraphlowMesh, allow_negative_volume: bool = True
    ) -> pt.PhlowerTensor:
        """Compute (n_elements,)-shaped volumes.

        Available celltypes are:
        VTK_TETRA, VTK_PYRAMID, VTK_WEDGE, VTK_VOXEL,
        VTK_HEXAHEDRON, VTK_POLYHEDRON

        Parameters
        ----------
        mesh: GraphlowMesh
        allow_negative_volume: bool, optional [True]
            If True, compute the signed volume.

        Returns
        -------
        pt.PhlowerTensor
        """
        volumes_by_celltype = {
            pv.CellType.TETRA: self._tet_volumes,
            pv.CellType.PYRAMID: self._pyramid_volumes,
            pv.CellType.WEDGE: self._wedge_volumes,
            pv.CellType.VOXEL: self._voxel_volumes,
            pv.CellType.HEXAHEDRON: self._hex_volumes,
            pv.CellType.POLYHEDRON: self._poly_volumes,
        }
        volumes = pt.phlower_tensor(
            torch.zeros(mesh.n_cells, dtype=mesh.dtype), dimension={"L": 3}
        ).to(device=mesh.device)
        celltypes = mesh.pvmesh.celltypes

        # non-polyhedron cells
        nonpoly_mask = celltypes != pv.CellType.POLYHEDRON
        if np.any(nonpoly_mask):
            nonpolys = mesh.extract_cells(nonpoly_mask, pass_point_data=True)
            nonpolys_dict = nonpolys.pvmesh.cells_dict
            for celltype, cells in nonpolys_dict.items():
                if celltype not in volumes_by_celltype:
                    raise KeyError(
                        f"Unavailable celltype: {pv.CellType(celltype).name}"
                    )
                mask = celltypes == celltype
                cell_points = nonpolys.points[cells]
                volumes[mask] = volumes_by_celltype[celltype](cell_points)

        # polyhedron cells
        poly_mask = celltypes == pv.CellType.POLYHEDRON
        if np.any(poly_mask):
            polys: IReadOnlyGraphlowMesh = mesh.extract_cells(
                poly_mask, pass_point_data=True
            )
            volumes[poly_mask] = self._poly_volumes(polys)

        if not allow_negative_volume and torch.any(volumes.to_tensor() < 0.0):
            indices = (volumes.to_tensor() < 0).nonzero(as_tuple=True)
            raise ValueError(f"Negative volume found: cell indices: {indices}")
        return volumes

    def compute_normals(self, mesh: IReadOnlyGraphlowMesh) -> pt.PhlowerTensor:
        """Compute (n_elements, dims)-shaped normals.

        Available celltypes are:
        VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON

        Parameters
        ----------
        mesh: GraphlowMesh

        Returns
        -------
        pt.PhlowerTensor
        """
        area_vecs = mesh.compute_area_vecs()
        areas = torch.linalg.vector_norm(area_vecs, dim=1, keepdim=True)
        normals = area_vecs / areas
        return normals

    def compute_surface_volume(
        self, mesh: IReadOnlyGraphlowMesh
    ) -> pt.PhlowerTensor:
        """Compute (1,)-shaped surface volume.

        Available celltypes are:
        VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON

        Parameters
        ----------
        mesh: GraphlowMesh

        Returns
        -------
        pt.PhlowerTensor
        """
        surface_mesh = mesh.extract_surface(pass_point_data=True)
        boundary = surface_mesh.pvmesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        if boundary.n_cells != 0:
            raise ValueError("Surface mesh is not watertight")
        cone_volumes_by_celltype = {
            pv.CellType.TRIANGLE: self._tri_cone_volumes,
            pv.CellType.QUAD: self._quad_cone_volumes,
            pv.CellType.POLYGON: self._poly_cone_volumes,
        }
        cone_volumes = pt.phlower_tensor(
            torch.zeros(surface_mesh.n_cells, dtype=surface_mesh.dtype),
            dimension={"L": 3},
        ).to(device=surface_mesh.device)
        celltypes = surface_mesh.pvmesh.celltypes

        # non-polygon cells
        nonpoly_mask = celltypes != pv.CellType.POLYGON
        if np.any(nonpoly_mask):
            nonpolys = surface_mesh.extract_cells(
                nonpoly_mask, pass_point_data=True
            )
            nonpolys_dict = nonpolys.pvmesh.cells_dict
            for celltype, cells in nonpolys_dict.items():
                if celltype not in cone_volumes_by_celltype:
                    raise KeyError(
                        f"Unavailable celltype: {pv.CellType(celltype).name}"
                    )
                mask = celltypes == celltype
                cell_points = nonpolys.points[cells]
                cone_volumes[mask] = cone_volumes_by_celltype[celltype](
                    cell_points
                )

        # polygon cells
        poly_mask = celltypes == pv.CellType.POLYGON
        if np.any(poly_mask):
            polys = surface_mesh.extract_cells(poly_mask, pass_point_data=True)
            cone_volumes[poly_mask] = self._poly_cone_volumes(
                polys.points, polys
            )
        return torch.abs(torch.sum(cone_volumes))

    #
    # Area function
    #
    def _tri_area_vecs(self, cell_points: pt.PhlowerTensor) -> pt.PhlowerTensor:
        v01 = cell_points[:, 1] - cell_points[:, 0]  # n_cell, dim
        v02 = cell_points[:, 2] - cell_points[:, 0]
        cross = torch.linalg.cross(v01, v02)  # n_cell, dim
        return 0.5 * cross

    def _quad_area_vecs(
        self, cell_points: pt.PhlowerTensor
    ) -> pt.PhlowerTensor:
        v1 = cell_points
        v2 = torch.roll(v1, shifts=-1, dims=1)
        cross = torch.linalg.cross(v1, v2)
        return 0.5 * torch.sum(cross, dim=1)

    def _poly_area_vecs(
        self, points: pt.PhlowerTensor, polys: IReadOnlyGraphlowMesh
    ) -> pt.PhlowerTensor:
        area_vecs = pt.phlower_tensor(
            torch.zeros(polys.n_cells, points.shape[1], dtype=points.dtype),
            dimension={"L": 2},
        ).to(device=points.device)
        for i in range(polys.n_cells):
            cell = polys.pvmesh.get_cell(i)
            face = torch.tensor(cell.point_ids, dtype=torch.int)
            v1 = points[face]
            v2 = torch.roll(v1, shifts=-1, dims=0)
            cross = torch.linalg.cross(v1, v2)
            area_vecs[i] = 0.5 * torch.sum(cross, dim=0)
        return area_vecs

    #
    # Cone volume function
    #
    def _tri_cone_volumes(
        self, cell_points: pt.PhlowerTensor
    ) -> pt.PhlowerTensor:
        v01 = cell_points[:, 1] - cell_points[:, 0]  # n_cell, dim
        v02 = cell_points[:, 2] - cell_points[:, 0]
        cross = torch.linalg.cross(v01, v02)  # n_cell, dim
        v0 = cell_points[:, 0]
        return torch.sum(cross * v0, dim=1) / 6.0

    def _quad_cone_volumes(
        self, cell_points: pt.PhlowerTensor
    ) -> pt.PhlowerTensor:
        v1 = cell_points  # n_cell, n_point, dim
        v2 = torch.roll(v1, shifts=-1, dims=1)
        cross = torch.sum(torch.linalg.cross(v1, v2), dim=1)  # n_cell, dim
        v0 = cell_points[:, 0]
        return torch.sum(cross * v0, dim=1) / 6.0

    def _poly_cone_volumes(
        self, points: pt.PhlowerTensor, polys: IReadOnlyGraphlowMesh
    ) -> pt.PhlowerTensor:
        cone_volumes = pt.phlower_tensor(
            torch.zeros(polys.n_cells, dtype=points.dtype),
            dimension={"L": 3},
        ).to(device=points.device)
        for i in range(polys.n_cells):
            cell = polys.pvmesh.get_cell(i)
            face = torch.tensor(cell.point_ids, dtype=torch.int)
            v1 = points[face]  # n_point, dim
            v2 = torch.roll(v1, shifts=-1, dims=0)
            cross = torch.sum(torch.linalg.cross(v1, v2), dim=0)  # dim
            v0 = v1[0]
            cone_volumes[i] = torch.sum(cross * v0, dim=0) / 6.0
        return cone_volumes

    #
    # Volume function
    #
    def _tet_volumes(self, cell_points: pt.PhlowerTensor) -> pt.PhlowerTensor:
        v01 = cell_points[:, 1] - cell_points[:, 0]  # n_cell, dim
        v02 = cell_points[:, 2] - cell_points[:, 0]
        v03 = cell_points[:, 3] - cell_points[:, 0]
        cross = torch.linalg.cross(v01, v02)  # n_cell, dim
        return torch.sum(cross * v03, dim=1) / 6.0

    def _pyramid_volumes(
        self, cell_points: pt.PhlowerTensor
    ) -> pt.PhlowerTensor:
        quad_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int)
        quads = cell_points[:, quad_idx]  # n_cell, n_point, dim
        quad_centers = torch.mean(quads, dim=1)  # n_cell, dim
        tops = cell_points[:, 4]  # n_cell, dim
        center2top = tops - quad_centers  # n_cell, dim
        v1 = quads - torch.unsqueeze(tops, dim=1)
        v2 = torch.roll(v1, shifts=-1, dims=1)
        cross = torch.linalg.cross(v1, v2)  # n_cell, n_point, dim
        return (
            torch.sum(cross * torch.unsqueeze(center2top, dim=1), dim=(1, 2))
            / 6.0
        )

    def _wedge_volumes(self, cell_points: pt.PhlowerTensor) -> pt.PhlowerTensor:
        # divide the wedge into 2 tets + 3 pyramids
        # This is a better solution than 3 tets because
        # if the wedge is twisted then the 3 quads will be twisted.
        tops = torch.mean(cell_points, dim=1, keepdim=True)  # n_cell, 1, dim
        quad_tops = tops.repeat((1, 3, 1))
        tet_tops = tops.repeat((1, 2, 1))

        # pyramid
        quad_idx = torch.tensor(
            [[0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]], dtype=torch.int
        )
        quads = cell_points[:, quad_idx]  # n_cell, n_face, n_point, dim
        quad_centers = torch.mean(quads, dim=2)  # n_cell, n_face, dim

        center2top_quads = quad_tops - quad_centers  # n_cell, n_face, dim
        v1_quads = quads - torch.unsqueeze(tops, dim=1)
        v2_quads = torch.roll(v1_quads, shifts=-1, dims=2)
        cross_quads = torch.linalg.cross(
            v1_quads, v2_quads
        )  # n_cell, n_face, n_point, dim
        pyramid_volumes = (
            torch.sum(
                cross_quads * torch.unsqueeze(center2top_quads, dim=2),
                dim=(1, 2, 3),
            )
            / 6.0
        )

        # tetra
        tri_idx = torch.tensor([[0, 2, 1], [3, 4, 5]], dtype=torch.int)
        tris = cell_points[:, tri_idx]  # n_cell, n_face, n_point, dim
        tri_centers = torch.mean(tris, dim=2)  # n_cell, n_face, dim

        center2top_tris = tet_tops - tri_centers  # n_cell, n_face, dim
        v1_tris = tris - torch.unsqueeze(tops, dim=1)
        v2_tris = torch.roll(v1_tris, shifts=-1, dims=2)
        cross_tris = torch.linalg.cross(
            v1_tris, v2_tris
        )  # n_cell, n_face, n_point, dim
        tet_volumes = (
            torch.sum(
                cross_tris * torch.unsqueeze(center2top_tris, dim=2),
                dim=(1, 2, 3),
            )
            / 6.0
        )

        return pyramid_volumes + tet_volumes

    def _voxel_volumes(self, cell_points: pt.PhlowerTensor) -> pt.PhlowerTensor:
        # divide the voxel into 6 pyramids
        tops = torch.mean(cell_points, dim=1, keepdim=True)  # n_cell, 1, dim
        quad_tops = tops.repeat((1, 6, 1))

        quad_idx = torch.tensor(
            [
                [0, 4, 5, 1],
                [2, 3, 7, 6],
                [0, 2, 6, 4],
                [4, 6, 7, 5],
                [5, 7, 3, 1],
                [1, 3, 2, 0],
            ],
            dtype=torch.int,
        )
        quads = cell_points[:, quad_idx]  # n_cell, n_face, n_point, dim
        quad_centers = torch.mean(quads, dim=2)  # n_cell, n_face, dim
        center2top_quads = quad_tops - quad_centers  # n_cell, n_face, dim
        v1_quads = quads - torch.unsqueeze(tops, dim=1)
        v2_quads = torch.roll(v1_quads, shifts=-1, dims=2)
        cross_quads = torch.linalg.cross(
            v1_quads, v2_quads
        )  # n_cell, n_face, n_point, dim
        volumes = (
            torch.sum(
                cross_quads * torch.unsqueeze(center2top_quads, dim=2),
                dim=(1, 2, 3),
            )
            / 6.0
        )
        return volumes

    def _hex_volumes(self, cell_points: pt.PhlowerTensor) -> pt.PhlowerTensor:
        # divide the hex into 6 pyramids
        tops = torch.mean(cell_points, dim=1, keepdim=True)  # n_cell, 1, dim
        quad_tops = tops.repeat((1, 6, 1))

        quad_idx = torch.tensor(
            [
                [0, 4, 5, 1],
                [3, 2, 6, 7],
                [0, 3, 7, 4],
                [4, 7, 6, 5],
                [5, 6, 2, 1],
                [1, 2, 3, 0],
            ],
            dtype=torch.int,
        )
        quads = cell_points[:, quad_idx]  # n_cell, n_face, n_point, dim
        quad_centers = torch.mean(quads, dim=2)  # n_cell, n_face, dim
        center2top_quads = quad_tops - quad_centers  # n_cell, n_face, dim
        v1_quads = quads - torch.unsqueeze(tops, dim=1)
        v2_quads = torch.roll(v1_quads, shifts=-1, dims=2)
        cross_quads = torch.linalg.cross(
            v1_quads, v2_quads
        )  # n_cell, n_face, n_point, dim
        volumes = (
            torch.sum(
                cross_quads * torch.unsqueeze(center2top_quads, dim=2),
                dim=(1, 2, 3),
            )
            / 6.0
        )
        return volumes

    def _poly_volumes(self, polys: IReadOnlyGraphlowMesh) -> pt.PhlowerTensor:
        facets = polys.extract_facets(pass_point_data=True)
        facet_centers = facets.convert_nodal2elemental(facets.points)
        area_vecs = facets.compute_area_vecs()
        cone_volumes = torch.sum(area_vecs * facet_centers, dim=1) / 3.0
        cf_inc = facets.compute_facet_cell_incidence().transpose(0, 1)
        cell_volumes = cf_inc @ cone_volumes
        return cell_volumes

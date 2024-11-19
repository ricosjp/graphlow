from typing import Literal

import numpy as np
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
        elemental_data: torch.Tensor,
        mode: Literal["mean", "effective"] = "mean",
    ) -> torch.Tensor:
        pc_inc = mesh.compute_cell_point_incidence().to_sparse_coo().T
        if mode == "mean":
            n_connected_cells = pc_inc.sum(dim=1).to_dense()
            mean_pc_inc = pc_inc.multiply(n_connected_cells.pow(-1).view(-1, 1))
            nodal_data = mean_pc_inc @ elemental_data
            return nodal_data
        if mode == "effective":
            n_points_in_cells = pc_inc.sum(dim=0).to_dense()
            effective_pc_inc = pc_inc.multiply(n_points_in_cells.pow(-1))
            nodal_data = effective_pc_inc @ elemental_data
            return nodal_data
        raise ValueError(f"Invalid mode: {mode}")

    def convert_nodal2elemental(
        self,
        mesh: IReadOnlyGraphlowMesh,
        nodal_data: torch.Tensor,
        mode: Literal["mean", "effective"] = "mean",
    ) -> torch.Tensor:
        cp_inc = mesh.compute_cell_point_incidence().to_sparse_coo()
        if mode == "mean":
            n_points_in_cells = cp_inc.sum(dim=1).to_dense()
            mean_cp_inc = cp_inc.multiply(n_points_in_cells.pow(-1).view(-1, 1))
            nodal_data = mean_cp_inc @ nodal_data
            return nodal_data
        if mode == "effective":
            n_connected_cells = cp_inc.sum(dim=0).to_dense()
            effective_cp_inc = cp_inc.multiply(n_connected_cells.pow(-1))
            nodal_data = effective_cp_inc @ nodal_data
            return nodal_data
        raise ValueError(f"Invalid mode: {mode}")

    def compute_area_vecs(self, mesh: IReadOnlyGraphlowMesh) -> torch.Tensor:
        """Compute (n_elements, dims)-shaped area vectors.
        Available celltypes are:
        VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON

        Returns:
        --------
        torch.Tensor[float]
        """
        available_celltypes = {
            pv.CellType.TRIANGLE,
            pv.CellType.QUAD,
            pv.CellType.POLYGON,
        }
        area_vecs = torch.empty((mesh.n_cells, mesh.points.shape[1]))
        celltypes = mesh.pvmesh.celltypes

        # non-polygon cells
        nonpoly_mask = celltypes != pv.CellType.POLYGON
        if np.any(nonpoly_mask):
            nonpolys = mesh.extract_cells(nonpoly_mask, pass_points=True)
            nonpolys_dict = nonpolys.pvmesh.cells_dict
            for celltype, cells in nonpolys_dict.items():
                if celltype not in available_celltypes:
                    raise KeyError(
                        f"Unavailable celltype: {pv.CellType(celltype).name}"
                    )
                mask = celltypes == celltype
                area_vecs[mask] = self._non_poly_area_vecs(
                    nonpolys.points[cells]
                )

        # polygon cells
        poly_mask = celltypes == pv.CellType.POLYGON
        if np.any(poly_mask):
            polys = mesh.extract_cells(poly_mask, pass_points=True)
            area_vecs[poly_mask] = self._poly_area_vecs(polys.points, polys)
        return area_vecs

    def compute_areas(
        self, mesh: IReadOnlyGraphlowMesh, allow_negative_area: bool = False
    ) -> torch.Tensor:
        """Compute (n_elements,)-shaped areas.
        Available celltypes are:
        VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON

        Parameters
        ----------
        allow_negative_area: bool, optional [False]

        Returns:
        --------
        torch.Tensor[float]
        """
        area_vecs = mesh.compute_area_vecs()
        areas = torch.norm(area_vecs, dim=1)
        if not allow_negative_area and torch.any(areas < 0.0):
            indices = (areas < 0).nonzero(as_tuple=True)
            raise ValueError(f"Negative area found: cell indices: {indices}")
        return areas

    def compute_volumes(
        self, mesh: IReadOnlyGraphlowMesh, allow_negative_volume: bool = True
    ) -> torch.Tensor:
        """Compute (n_elements,)-shaped volumes.
        Available celltypes are:
        VTK_TETRA, VTK_PYRAMID, VTK_WEDGE, VTK_HEXAHEDRON, VTK_POLYHEDRON

        Parameters
        ----------
        allow_negative_area: bool, optional [True]
            If True, compute the signed volume.

        Returns:
        --------
        torch.Tensor[float]
        """
        volumes_by_celltype = {
            pv.CellType.TETRA: self._tet_volumes,
            pv.CellType.PYRAMID: self._pyramid_volumes,
            pv.CellType.WEDGE: self._wedge_volumes,
            pv.CellType.VOXEL: self._voxel_volumes,
            pv.CellType.HEXAHEDRON: self._hex_volumes,
            pv.CellType.POLYHEDRON: self._poly_volumes,
        }
        volumes = torch.empty(mesh.n_cells)
        celltypes = mesh.pvmesh.celltypes

        # non-polyhedron cells
        nonpoly_mask = celltypes != pv.CellType.POLYHEDRON
        if np.any(nonpoly_mask):
            nonpolys = mesh.extract_cells(nonpoly_mask, pass_points=True)
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
                poly_mask, pass_points=True
            )
            volumes[poly_mask] = self._poly_volumes(polys)

        if not allow_negative_volume and torch.any(volumes < 0.0):
            indices = (volumes < 0).nonzero(as_tuple=True)
            raise ValueError(f"Negative volume found: cell indices: {indices}")
        return volumes

    def compute_normals(self, mesh: IReadOnlyGraphlowMesh) -> torch.Tensor:
        """Compute (n_elements, dims)-shaped normals
        Available celltypes are:
        VTK_TRIANGLE, VTK_QUAD, VTK_POLYGON

        Returns
        -------
        torch.Tensor[float]
        """
        area_vecs = mesh.compute_area_vecs()
        areas = torch.norm(area_vecs, dim=1, keepdim=True)
        normals = area_vecs / areas
        return normals

    #
    # Area function
    #
    def _non_poly_area_vecs(self, cell_points: torch.Tensor) -> torch.Tensor:
        v1 = cell_points
        v2 = torch.roll(v1, shifts=-1, dims=1)
        cross = torch.linalg.cross(v1, v2)
        return 0.5 * torch.sum(cross, dim=1)

    def _poly_area_vecs(
        self, points: torch.Tensor, polys: IReadOnlyGraphlowMesh
    ) -> torch.Tensor:
        area_vecs = torch.empty((polys.n_cells, points.shape[1]))
        for i in range(polys.n_cells):
            cell = polys.pvmesh.get_cell(i)
            face = torch.tensor(cell.point_ids, dtype=torch.int)
            v1 = points[face]
            v2 = torch.roll(v1, shifts=-1, dims=0)
            cross = torch.linalg.cross(v1, v2)
            area_vecs[i] = 0.5 * torch.sum(cross, dim=0)
        return area_vecs

    #
    # Volume function
    #
    def _tet_volumes(self, cell_points: torch.Tensor) -> torch.Tensor:
        v01 = cell_points[:, 1] - cell_points[:, 0]  # n_cell, dim
        v02 = cell_points[:, 2] - cell_points[:, 0]
        v03 = cell_points[:, 3] - cell_points[:, 0]
        cross = torch.linalg.cross(v01, v02)  # n_cell, dim
        return torch.sum(cross * v03, dim=1) / 6.0

    def _pyramid_volumes(self, cell_points: torch.Tensor) -> torch.Tensor:
        quad_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int)
        quads = cell_points[:, quad_idx]  # n_cell, n_point, dim
        quad_centers = torch.mean(quads, dim=1)  # n_cell, dim
        tops = cell_points[:, 4]  # n_cell, dim
        center2top = tops - quad_centers  # n_cell, dim
        v1 = quads - tops.unsqueeze(1)
        v2 = torch.roll(v1, shifts=-1, dims=1)
        cross = torch.linalg.cross(v1, v2)  # n_cell, n_point, dim
        return torch.sum(cross * center2top.unsqueeze(1), dim=(1, 2)) / 6.0

    def _wedge_volumes(self, cell_points: torch.Tensor) -> torch.Tensor:
        # divide the wedge into 2 tets + 3 pyramids
        # This is a better solution than 3 tets because
        # if the wedge is twisted then the 3 quads will be twisted.
        tops = torch.mean(cell_points, dim=1, keepdim=True)  # n_cell, 1, dim
        quad_tops = tops.repeat(1, 3, 1)
        tet_tops = tops.repeat(1, 2, 1)

        # pyramid
        quad_idx = torch.tensor(
            [[0, 3, 4, 1], [1, 4, 5, 2], [0, 2, 5, 3]], dtype=torch.int
        )
        quads = cell_points[:, quad_idx]  # n_cell, n_face, n_point, dim
        quad_centers = torch.mean(quads, dim=2)  # n_cell, n_face, dim

        center2top_quads = quad_tops - quad_centers  # n_cell, n_face, dim
        v1_quads = quads - tops.unsqueeze(1)
        v2_quads = torch.roll(v1_quads, shifts=-1, dims=2)
        cross_quads = torch.linalg.cross(
            v1_quads, v2_quads
        )  # n_cell, n_face, n_point, dim
        pyramid_volumes = (
            torch.sum(
                cross_quads * center2top_quads.unsqueeze(2), dim=(1, 2, 3)
            )
            / 6.0
        )

        # tetra
        tri_idx = torch.tensor([[0, 1, 2], [3, 5, 4]], dtype=torch.int)
        tris = cell_points[:, tri_idx]  # n_cell, n_face, n_point, dim
        tri_centers = torch.mean(tris, dim=2)  # n_cell, n_face, dim

        center2top_tris = tet_tops - tri_centers  # n_cell, n_face, dim
        v1_tris = tris - tops.unsqueeze(1)
        v2_tris = torch.roll(v1_tris, shifts=-1, dims=2)
        cross_tris = torch.linalg.cross(
            v1_tris, v2_tris
        )  # n_cell, n_face, n_point, dim
        tet_volumes = (
            torch.sum(cross_tris * center2top_tris.unsqueeze(2), dim=(1, 2, 3))
            / 6.0
        )

        return pyramid_volumes + tet_volumes

    def _voxel_volumes(self, cell_points: torch.Tensor) -> torch.Tensor:
        # divide the voxel into 6 pyramids
        tops = torch.mean(cell_points, dim=1, keepdim=True)  # n_cell, 1, dim
        quad_tops = tops.repeat(1, 6, 1)

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
        v1_quads = quads - tops.unsqueeze(1)
        v2_quads = torch.roll(v1_quads, shifts=-1, dims=2)
        cross_quads = torch.linalg.cross(
            v1_quads, v2_quads
        )  # n_cell, n_face, n_point, dim
        volumes = (
            torch.sum(
                cross_quads * center2top_quads.unsqueeze(2), dim=(1, 2, 3)
            )
            / 6.0
        )
        return volumes

    def _hex_volumes(self, cell_points: torch.Tensor) -> torch.Tensor:
        # divide the hex into 6 pyramids
        tops = torch.mean(cell_points, dim=1, keepdim=True)  # n_cell, 1, dim
        quad_tops = tops.repeat(1, 6, 1)

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
        v1_quads = quads - tops.unsqueeze(1)
        v2_quads = torch.roll(v1_quads, shifts=-1, dims=2)
        cross_quads = torch.linalg.cross(
            v1_quads, v2_quads
        )  # n_cell, n_face, n_point, dim
        volumes = (
            torch.sum(
                cross_quads * center2top_quads.unsqueeze(2), dim=(1, 2, 3)
            )
            / 6.0
        )
        return volumes

    def _poly_volumes(self, polys: IReadOnlyGraphlowMesh) -> torch.Tensor:
        facets, fc_inc = polys.extract_facets(pass_points=True)
        facet_centers = facets.convert_nodal2elemental(facets.points)
        area_vecs = facets.compute_area_vecs()
        cone_volumes = torch.sum(area_vecs * facet_centers, dim=1) / 3.0
        cf_inc = fc_inc.to_sparse_coo().T
        cell_volumes = cf_inc @ cone_volumes
        return cell_volumes

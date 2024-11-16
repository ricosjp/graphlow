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
        self, mesh: IReadOnlyGraphlowMesh, raise_negative_area: bool = True
    ) -> torch.Tensor:
        area_vecs = mesh.compute_area_vecs()
        areas = torch.norm(area_vecs, dim=1)
        if raise_negative_area and torch.any(areas < 0.0):
            indices = (areas < 0).nonzero(as_tuple=True)
            raise ValueError(f"Negative area found: cell indices: {indices}")
        return areas

    def compute_volumes(
        self, mesh: IReadOnlyGraphlowMesh, raise_negative_volume: bool = True
    ) -> torch.Tensor:
        volumes = torch.empty(mesh.n_cells)
        cell_type_to_function = {
            pv.CellType.TETRA: self._tet_volume,
            pv.CellType.PYRAMID: self._pyramid_volume,
            pv.CellType.WEDGE: self._wedge_volume,
            pv.CellType.HEXAHEDRON: self._hex_volume,
            pv.CellType.POLYHEDRON: self._poly_volume,
        }
        points = mesh.points
        for i in range(mesh.n_cells):
            cell = mesh.pvmesh.get_cell(i)
            celltype = cell.type
            if celltype not in cell_type_to_function:
                raise KeyError(
                    f"Unavailable cell type for area computation: cell[{i}]"
                )

            pids = torch.tensor(cell.point_ids, dtype=torch.int)
            func = cell_type_to_function[celltype]
            if celltype == pv.CellType.POLYHEDRON:
                volumes[i] = func(pids, points, cell.faces)
            else:
                volumes[i] = func(pids, points)

        if raise_negative_volume and torch.any(volumes < 0.0):
            indices = (volumes < 0).nonzero(as_tuple=True)
            raise ValueError(f"Negative volume found: cell indices: {indices}")
        return volumes

    def compute_normals(self, mesh: IReadOnlyGraphlowMesh) -> torch.Tensor:
        """Compute the normals of PolyData

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
    def _tet_volume(
        self, pids: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        tet_points = points[pids]
        v10 = tet_points[1] - tet_points[0]
        v20 = tet_points[2] - tet_points[0]
        v30 = tet_points[3] - tet_points[0]
        return torch.abs(torch.dot(torch.linalg.cross(v10, v20), v30)) / 6.0

    def _pyramid_volume(
        self, pids: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        quad_idx = torch.tensor([0, 1, 2, 3], dtype=torch.int)
        quad_center = torch.mean(points[pids[quad_idx]], dim=0)
        top = points[pids[4]]
        axis = quad_center - top
        side_vec = points[pids[quad_idx]] - top
        cross = torch.linalg.cross(
            side_vec, torch.roll(side_vec, shifts=-1, dims=0)
        )
        tet_volumes = torch.abs(torch.sum(cross * axis, dim=1)) / 6.0
        return torch.sum(tet_volumes)

    def _wedge_volume(
        self, pids: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        # divide the wedge into 11 tets
        # This is a better solution than 3 tets because
        # if the wedge is twisted then the 3 quads will be twisted.
        quad_idx = torch.tensor(
            [[0, 3, 4, 1], [1, 4, 5, 2], [0, 2, 5, 3]], dtype=torch.int
        )
        quad_centers = torch.mean(points[pids[quad_idx]], dim=1)

        sub_tet_points = torch.empty(11, 4, 3)
        sub_tet_points[0][0] = points[pids[0]]
        sub_tet_points[0][1] = quad_centers[0]
        sub_tet_points[0][2] = points[pids[3]]
        sub_tet_points[0][3] = quad_centers[2]

        sub_tet_points[1][0] = points[pids[1]]
        sub_tet_points[1][1] = quad_centers[1]
        sub_tet_points[1][2] = points[pids[4]]
        sub_tet_points[1][3] = quad_centers[0]

        sub_tet_points[2][0] = points[pids[2]]
        sub_tet_points[2][1] = quad_centers[2]
        sub_tet_points[2][2] = points[pids[5]]
        sub_tet_points[2][3] = quad_centers[1]

        sub_tet_points[3][0] = quad_centers[0]
        sub_tet_points[3][1] = quad_centers[1]
        sub_tet_points[3][2] = quad_centers[2]
        sub_tet_points[3][3] = points[pids[0]]

        sub_tet_points[4][0] = points[pids[1]]
        sub_tet_points[4][1] = quad_centers[1]
        sub_tet_points[4][2] = quad_centers[0]
        sub_tet_points[4][3] = points[pids[0]]

        sub_tet_points[5][0] = points[pids[2]]
        sub_tet_points[5][1] = quad_centers[1]
        sub_tet_points[5][2] = points[pids[1]]
        sub_tet_points[5][3] = points[pids[0]]

        sub_tet_points[6][0] = points[pids[2]]
        sub_tet_points[6][1] = quad_centers[2]
        sub_tet_points[6][2] = quad_centers[1]
        sub_tet_points[6][3] = points[pids[0]]

        sub_tet_points[7][0] = quad_centers[0]
        sub_tet_points[7][1] = quad_centers[2]
        sub_tet_points[7][2] = quad_centers[1]
        sub_tet_points[7][3] = points[pids[3]]

        sub_tet_points[8][0] = points[pids[5]]
        sub_tet_points[8][1] = quad_centers[1]
        sub_tet_points[8][2] = quad_centers[2]
        sub_tet_points[8][3] = points[pids[3]]

        sub_tet_points[9][0] = points[pids[4]]
        sub_tet_points[9][1] = quad_centers[1]
        sub_tet_points[9][2] = points[pids[5]]
        sub_tet_points[9][3] = points[pids[3]]

        sub_tet_points[10][0] = points[pids[4]]
        sub_tet_points[10][1] = quad_centers[0]
        sub_tet_points[10][2] = quad_centers[1]
        sub_tet_points[10][3] = points[pids[3]]

        sub_tet_vec = sub_tet_points[:, 1:] - sub_tet_points[:, 0].unsqueeze(1)
        cross = torch.linalg.cross(sub_tet_vec[:, 0], sub_tet_vec[:, 1])
        tet_volumes = (
            torch.abs(torch.sum(cross * sub_tet_vec[:, 2], dim=1)) / 6.0
        )
        return torch.sum(tet_volumes)

    def _hex_volume(
        self, pids: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        # divide the hex into 24 (=4*6) tets for the same reason as a wedge
        face_idx = torch.tensor(
            [
                [0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7],
                [3, 2, 1, 0],
                [4, 5, 6, 7],
            ],
            dtype=torch.int,
        )
        face_centers = torch.mean(points[pids[face_idx]], dim=1)
        cell_center = torch.mean(points[pids], dim=0)
        cc2fc = face_centers - cell_center
        side_vec = points[pids[face_idx]] - cell_center
        cross = torch.linalg.cross(
            side_vec, torch.roll(side_vec, shifts=-1, dims=1)
        )
        tet_volumes = (
            torch.abs(torch.sum(cross * cc2fc.unsqueeze(1), dim=2)) / 6.0
        )
        return torch.sum(tet_volumes)

    def _poly_volume(
        self, pids: torch.Tensor, points: torch.Tensor, faces: list[pv.Cell]
    ) -> torch.Tensor:
        # Assume cell is convex
        volume = 0.0
        cell_center = torch.mean(points[pids], dim=0)
        for face in faces:
            face_pids = face.point_ids
            face_centers = torch.mean(points[face_pids], dim=0)
            cc2fc = face_centers - cell_center
            side_vec = points[face_pids] - cell_center
            cross = torch.linalg.cross(
                side_vec, torch.roll(side_vec, shifts=-1, dims=0)
            )
            tet_volumes = torch.abs(torch.sum(cross * cc2fc, dim=1)) / 6.0
            volume += torch.sum(tet_volumes)
        return volume

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
        elif mode == "effective":
            n_points_in_cells = pc_inc.sum(dim=0).to_dense()
            effective_pc_inc = pc_inc.multiply(n_points_in_cells.pow(-1))
            nodal_data = effective_pc_inc @ elemental_data
            return nodal_data
        else:
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
        elif mode == "effective":
            n_connected_cells = cp_inc.sum(dim=0).to_dense()
            effective_cp_inc = cp_inc.multiply(n_connected_cells.pow(-1))
            nodal_data = effective_cp_inc @ nodal_data
            return nodal_data
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def compute_areas(
        self, mesh: IReadOnlyGraphlowMesh, raise_negative_area: bool = True
    ) -> torch.Tensor:
        areas = torch.empty(mesh.n_cells)
        cell_type_to_function = {
            pv.CellType.TRIANGLE: self._tri_area,
            pv.CellType.QUAD: self._poly_area,
            pv.CellType.POLYGON: self._poly_area,
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
            areas[i] = cell_type_to_function[celltype](pids, points)

        if raise_negative_area and torch.any(areas < 0.0):
            indices = (areas < 0).nonzero(as_tuple=True)
            raise ValueError(f"Negative volume found: cell indices: {indices}")
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
        points = mesh.points
        n_faces = mesh.n_cells
        normals = torch.empty(size=(n_faces, 3))
        for fid in range(n_faces):
            face = mesh.pvmesh.get_cell(fid).point_ids
            face_points = points[face]
            face_center = torch.mean(face_points, dim=0)
            side_vec = face_points - face_center
            cross = torch.linalg.cross(
                side_vec, torch.roll(side_vec, shifts=-1, dims=0)
            )
            normal = torch.mean(cross, dim=0)
            normals[fid] = normal / torch.norm(normal)
        return normals

    #
    # Area function
    #
    def _tri_area(
        self, pids: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        tri_points = points[pids]
        v10 = tri_points[1] - tri_points[0]
        v20 = tri_points[2] - tri_points[0]
        cross = torch.linalg.cross(v10, v20)
        return 0.5 * torch.linalg.vector_norm(cross)

    def _poly_area(
        self, pids: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        v1 = points[pids]
        v2 = torch.roll(v1, shifts=-1, dims=0)
        signed_area = torch.sum(torch.linalg.cross(v1, v2), dim=0)
        return 0.5 * torch.linalg.vector_norm(signed_area)

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

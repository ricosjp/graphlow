import pyvista as pv
import torch


class GeometryProcessorMixin:
    """A mix-in class for geometry processing."""

    def compute_area(self, raise_negative_area=True) -> torch.Tensor:
        areas = torch.empty(self.n_cells)
        cell_type_to_function = {
            pv.CellType.TRIANGLE: self._tri_area,
            pv.CellType.QUAD: self._poly_area,
            pv.CellType.POLYGON: self._poly_area,
        }
        for i in range(self.n_cells):
            cell = self.mesh.get_cell(i)
            celltype = cell.type
            if celltype not in cell_type_to_function:
                raise KeyError(f"Unavailable cell type for area computation: cell[{i}]")

            pids = torch.tensor(cell.point_ids, dtype=torch.int)
            areas[i] = cell_type_to_function[celltype](pids)

        if raise_negative_area and torch.any(areas < 0.0):
            indices = (areas < 0).nonzero(as_tuple=True)
            raise ValueError(f"Negative volume found: cell indices: {indices}")
        return areas

    def compute_volume(self, raise_negative_volume=True) -> torch.Tensor:
        volumes = torch.empty(self.n_cells)
        cell_type_to_function = {
            pv.CellType.TETRA: self._tet_volume,
            pv.CellType.PYRAMID: self._pyramid_volume,
            pv.CellType.WEDGE: self._wedge_volume,
            pv.CellType.HEXAHEDRON: self._hex_volume,
            pv.CellType.POLYHEDRON: self._poly_volume,
        }
        for i in range(self.n_cells):
            cell = self.mesh.get_cell(i)
            celltype = cell.type
            if celltype not in cell_type_to_function:
                raise KeyError(f"Unavailable cell type for area computation: cell[{i}]")

            pids = torch.tensor(cell.point_ids, dtype=torch.int)
            func = cell_type_to_function[celltype]
            if celltype == pv.CellType.POLYHEDRON:
                volumes[i] = func(pids, cell.faces)
            else:
                volumes[i] = func(pids)

        if raise_negative_volume and torch.any(volumes < 0.0):
            indices = (volumes < 0).nonzero(as_tuple=True)
            raise ValueError(f"Negative volume found: cell indices: {indices}")
        return volumes


    #
    # Area function
    #
    def _tri_area(self, pids):
        points = self.points[pids]
        v10 = points[1] - points[0]
        v20 = points[2] - points[0]
        cross = torch.linalg.cross(v10, v20)
        return 0.5 * torch.linalg.vector_norm(cross)

    def _poly_area(self, pids):
        v1 = self.points[pids]
        v2 = torch.roll(v1, shifts=-1, dims=0)
        signed_area = torch.sum(torch.linalg.cross(v1, v2), dim=0)
        return 0.5 * torch.linalg.vector_norm(signed_area)

    #
    # Volume function
    #
    def _tet_volume(self, pids):
        tet_points = self.points[pids]
        v10 = tet_points[1] - tet_points[0]
        v20 = tet_points[2] - tet_points[0]
        v30 = tet_points[3] - tet_points[0]
        return torch.abs(torch.dot(torch.linalg.cross(v10, v20), v30)) / 6.0

    def _pyramid_volume(self, pids):
        quad_idx =  torch.tensor([0, 1, 2, 3], dtype=torch.int)
        quad_center = torch.mean(self.points[pids[quad_idx]], dim=0)
        top = self.points[pids[4]]
        axis = quad_center - top
        side_vec = self.points[pids[quad_idx]] - top
        cross = torch.linalg.cross(side_vec, torch.roll(side_vec, shifts=-1, dims=0))
        tet_volumes = torch.abs(torch.sum(cross * axis, dim=1)) / 6.0
        return torch.sum(tet_volumes)

    def _wedge_volume(self, pids):
        # divide the wedge into 11 tets
        # This is a better solution than 3 tets because
        # if the wedge is twisted then the 3 quads will be twisted.
        quad_idx = torch.tensor([[0, 3, 4, 1], [1, 4, 5, 2], [0, 2, 5, 3]], dtype=torch.int)
        quad_centers = torch.mean(self.points[pids[quad_idx]], dim=1)

        sub_tet_points = torch.empty(11, 4, 3)
        sub_tet_points[0][0] = self.points[pids[0]]
        sub_tet_points[0][1] = quad_centers[0]
        sub_tet_points[0][2] = self.points[pids[3]]
        sub_tet_points[0][3] = quad_centers[2]

        sub_tet_points[1][0] = self.points[pids[1]]
        sub_tet_points[1][1] = quad_centers[1]
        sub_tet_points[1][2] = self.points[pids[4]]
        sub_tet_points[1][3] = quad_centers[0]

        sub_tet_points[2][0] = self.points[pids[2]]
        sub_tet_points[2][1] = quad_centers[2]
        sub_tet_points[2][2] = self.points[pids[5]]
        sub_tet_points[2][3] = quad_centers[1]

        sub_tet_points[3][0] = quad_centers[0]
        sub_tet_points[3][1] = quad_centers[1]
        sub_tet_points[3][2] = quad_centers[2]
        sub_tet_points[3][3] = self.points[pids[0]]

        sub_tet_points[4][0] = self.points[pids[1]]
        sub_tet_points[4][1] = quad_centers[1]
        sub_tet_points[4][2] = quad_centers[0]
        sub_tet_points[4][3] = self.points[pids[0]]

        sub_tet_points[5][0] = self.points[pids[2]]
        sub_tet_points[5][1] = quad_centers[1]
        sub_tet_points[5][2] = self.points[pids[1]]
        sub_tet_points[5][3] = self.points[pids[0]]

        sub_tet_points[6][0] = self.points[pids[2]]
        sub_tet_points[6][1] = quad_centers[2]
        sub_tet_points[6][2] = quad_centers[1]
        sub_tet_points[6][3] = self.points[pids[0]]

        sub_tet_points[7][0] = quad_centers[0]
        sub_tet_points[7][1] = quad_centers[2]
        sub_tet_points[7][2] = quad_centers[1]
        sub_tet_points[7][3] = self.points[pids[3]]

        sub_tet_points[8][0] = self.points[pids[5]]
        sub_tet_points[8][1] = quad_centers[1]
        sub_tet_points[8][2] = quad_centers[2]
        sub_tet_points[8][3] = self.points[pids[3]]

        sub_tet_points[9][0] = self.points[pids[4]]
        sub_tet_points[9][1] = quad_centers[1]
        sub_tet_points[9][2] = self.points[pids[5]]
        sub_tet_points[9][3] = self.points[pids[3]]

        sub_tet_points[10][0] = self.points[pids[4]]
        sub_tet_points[10][1] = quad_centers[0]
        sub_tet_points[10][2] = quad_centers[1]
        sub_tet_points[10][3] = self.points[pids[3]]

        sub_tet_vec = sub_tet_points[:, 1:] - sub_tet_points[:, 0].unsqueeze(1)
        cross = torch.linalg.cross(sub_tet_vec[:, 0], sub_tet_vec[:, 1])
        tet_volumes = torch.abs(torch.sum(cross * sub_tet_vec[:, 2], dim=1)) / 6.0
        return torch.sum(tet_volumes)

    def _hex_volume(self, pids):
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
        face_centers = torch.mean(self.points[pids[face_idx]], dim=1)
        cell_center = torch.mean(self.points[pids], dim=0)
        cc2fc = face_centers - cell_center
        side_vec = self.points[pids[face_idx]] - cell_center
        cross = torch.linalg.cross(side_vec, torch.roll(side_vec, shifts=-1, dims=1))
        tet_volumes = torch.abs(torch.sum(cross * cc2fc.unsqueeze(1), dim=2)) / 6.0
        return torch.sum(tet_volumes)

    def _poly_volume(self, pids, faces):
        # Assume cell is convex
        volume = 0.0
        cell_center = torch.mean(self.points[pids], dim=0)
        for face in faces:
            face_pids = face.point_ids
            face_centers = torch.mean(self.points[face_pids], dim=0)
            cc2fc = face_centers - cell_center
            side_vec = self.points[face_pids] - cell_center
            cross = torch.linalg.cross(side_vec, torch.roll(side_vec, shifts=-1, dims=0))
            tet_volumes = torch.abs(torch.sum(cross * cc2fc, dim=1)) / 6.0
            volume += torch.sum(tet_volumes)
        return volume

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pyvista as pv
from pydantic import (
    BaseModel,
    PlainValidator,
)

from graphlow.core.blocks import FaceBlock, FixedFaceBlock, JaggedFaceBlock
from graphlow.utils.topology_helper import (
    TopologyDim,
    get_cell_dimension,
)

# =============================================================================
# Cell <-> Face Relationship
# =============================================================================

_FIXED_FACE_PATTERNS = {
    pv.CellType.TETRA: {
        # https://github.com/Kitware/VTK/blob/df1b46c3f5fe315d588016191607401fccfa80fd/Common/DataModel/vtkTetra.cxx#L301
        pv.CellType.TRIANGLE: np.array(
            [
                [0, 1, 3],
                [1, 2, 3],
                [2, 0, 3],
                [0, 2, 1],
            ],
            dtype=np.int64,
        ),
    },
    pv.CellType.VOXEL: {
        # https://github.com/Kitware/VTK/blob/df1b46c3f5fe315d588016191607401fccfa80fd/Common/DataModel/vtkVoxel.cxx#L391
        pv.CellType.PIXEL: np.array(
            [
                [2, 0, 6, 4],
                [1, 3, 5, 7],
                [0, 1, 4, 5],
                [3, 2, 7, 6],
                [1, 0, 3, 2],
                [4, 5, 6, 7],
            ],
            dtype=np.int64,
        ),
    },
    pv.CellType.HEXAHEDRON: {
        # https://github.com/Kitware/VTK/blob/df1b46c3f5fe315d588016191607401fccfa80fd/Common/DataModel/vtkHexahedron.cxx#L406
        pv.CellType.QUAD: np.array(
            [
                [0, 4, 7, 3],
                [1, 2, 6, 5],
                [0, 1, 5, 4],
                [3, 7, 6, 2],
                [0, 3, 2, 1],
                [4, 5, 6, 7],
            ],
            dtype=np.int64,
        ),
    },
    pv.CellType.WEDGE: {
        # https://github.com/Kitware/VTK/blob/df1b46c3f5fe315d588016191607401fccfa80fd/Common/DataModel/vtkWedge.cxx#L57
        # https://github.com/Kitware/VTK/blob/df1b46c3f5fe315d588016191607401fccfa80fd/Common/DataModel/vtkWedge.cxx#L688
        pv.CellType.TRIANGLE: np.array(
            [
                [0, 1, 2],
                [3, 5, 4],
            ],
            dtype=np.int64,
        ),
        pv.CellType.QUAD: np.array(
            [
                [0, 3, 4, 1],
                [1, 4, 5, 2],
                [2, 5, 3, 0],
            ],
            dtype=np.int64,
        ),
    },
    pv.CellType.PYRAMID: {
        # https://github.com/Kitware/VTK/blob/df1b46c3f5fe315d588016191607401fccfa80fd/Common/DataModel/vtkPyramid.cxx#L57
        # https://github.com/Kitware/VTK/blob/df1b46c3f5fe315d588016191607401fccfa80fd/Common/DataModel/vtkPyramid.cxx#L674
        pv.CellType.QUAD: np.array(
            [
                [0, 3, 2, 1],
            ],
            dtype=np.int64,
        ),
        pv.CellType.TRIANGLE: np.array(
            [
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
                [3, 0, 4],
            ],
            dtype=np.int64,
        ),
    },
}


def get_fixed_face_patterns(
    cell_type: pv.CellType,
) -> dict[pv.CellType, np.ndarray] | None:
    """
    Get the fixed face patterns for a given cell type.

    Parameters
    ----------
    cell_type : pv.CellType
        VTK cell type to query.

    Returns
    -------
    dict[pv.CellType, np.ndarray] or None
        Face patterns keyed by face type, or None if the cell type does not use
        a fixed face layout in this module.
    """
    return _FIXED_FACE_PATTERNS.get(cell_type)


# =============================================================================
# Face Registry
# =============================================================================
# Datastructures to collect unique faces (tri/quad/polygon) from 3D cells,
# deduplicate by sorted vertex key, and record owner/neighbor. Finalizes to
# arrays suitable for building FaceBlock instances.


def _validate_immutable_int64_ndarray(v: list[int] | np.ndarray) -> np.ndarray:
    """Validate that the input is an immutable ndarray of int64."""
    arr = np.asarray(v, dtype=np.int64)
    arr.setflags(write=False)
    return arr


ReadonlyLongArray = Annotated[
    np.ndarray,
    PlainValidator(_validate_immutable_int64_ndarray),
]


class FaceRegistry(BaseModel, frozen=True):
    """
    Immutable registry of unique faces with owner/neighbor and per-type
    connectivity.
    """

    owner: ReadonlyLongArray  # (Ntotal_f,)
    neighbor: ReadonlyLongArray  # (Ntotal_f,)
    face_type: ReadonlyLongArray  # (Ntotal_f,)

    tri_gids: ReadonlyLongArray  # (Ntri_f,)
    tri_conn: ReadonlyLongArray  # (Ntri_f, 3)

    quad_gids: ReadonlyLongArray  # (Nquad_f,)
    quad_conn: ReadonlyLongArray  # (Nquad_f, 4)

    pixel_gids: ReadonlyLongArray  # (Npixel_f,)
    pixel_conn: ReadonlyLongArray  # (Npixel_f, 4)

    poly_gids: ReadonlyLongArray  # (Npoly_f,)
    poly_conn: ReadonlyLongArray  # (Npoly_total_pts,)
    poly_offsets: ReadonlyLongArray  # (Npoly_f + 1,)

    def n_faces(self) -> int:
        """Return the number of registered faces."""
        return len(self.owner)

    def boundary_mask(self) -> np.ndarray:
        """
        Return a mask selecting boundary faces.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(n_faces,)``.
        """
        return self.neighbor == -1


@dataclass
class FaceRegistryBuilder:
    """
    Mutable builder for FaceRegistry: register faces, resolve neighbor,
    then finalize.

    Faces are keyed by (face_type, sorted(point_ids)); second registration
    of the same key sets neighbor (manifold: at most two cells per face).
    """

    key_to_gid: dict[tuple, int]

    owner: list[int]
    neighbor: list[int]
    face_type: list[pv.CellType]

    tri_gids: list[int]
    tri_conn: list[np.ndarray]

    quad_gids: list[int]
    quad_conn: list[np.ndarray]

    pixel_gids: list[int]
    pixel_conn: list[np.ndarray]

    poly_gids: list[int]
    poly_faces: list[np.ndarray]

    @classmethod
    def create(cls) -> FaceRegistryBuilder:
        """
        Create an empty face registry builder.

        Returns
        -------
        FaceRegistryBuilder
            Empty builder instance.
        """
        return cls(
            key_to_gid={},
            owner=[],
            neighbor=[],
            face_type=[],
            tri_gids=[],
            tri_conn=[],
            quad_gids=[],
            quad_conn=[],
            pixel_gids=[],
            pixel_conn=[],
            poly_gids=[],
            poly_faces=[],
        )

    def register_face(
        self,
        face_type: pv.CellType,
        owner: int,
        point_ids: list[int] | np.ndarray,
    ) -> None:
        """
        Register a face (triangle, quad, or polygon) for a cell.

        If the face (key = sorted point_ids) is new, append it and
        set owner. If it already exists, set neighbor to owner (fails if
        already set, i.e. non-manifold).
        """
        point_ids = np.asarray(point_ids, dtype=np.int64)
        key = tuple(np.sort(point_ids))
        gid = self.key_to_gid.get(key)

        if gid is None:
            gid = len(self.owner)
            self.key_to_gid[key] = gid
            self.owner.append(owner)
            self.neighbor.append(-1)
            self.face_type.append(face_type)

            # classify face type
            match face_type:
                case pv.CellType.TRIANGLE:
                    self.tri_gids.append(gid)
                    self.tri_conn.append(point_ids)
                case pv.CellType.QUAD:
                    self.quad_gids.append(gid)
                    self.quad_conn.append(point_ids)
                case pv.CellType.PIXEL:
                    self.pixel_gids.append(gid)
                    self.pixel_conn.append(point_ids)
                case pv.CellType.POLYGON:
                    self.poly_gids.append(gid)
                    self.poly_faces.append(point_ids)
                case _:
                    raise ValueError(f"Unsupported face type: {face_type}")
        else:
            if self.neighbor[gid] != -1:
                raise ValueError(
                    f"Non-manifold face detected for key={key}: "
                    "a face is shared by more than two cells."
                )
            self.neighbor[gid] = owner

    def finalize(self) -> FaceRegistry:
        """
        Build immutable FaceRegistry from current state; numpy arrays only.
        """
        owner = np.asarray(self.owner, dtype=np.int64)
        neighbor = np.asarray(self.neighbor, dtype=np.int64)
        face_type = np.asarray(self.face_type, dtype=np.int64)

        tri_gids = np.asarray(self.tri_gids, dtype=np.int64)
        tri_conn = np.asarray(self.tri_conn, dtype=np.int64)

        quad_gids = np.asarray(self.quad_gids, dtype=np.int64)
        quad_conn = np.asarray(self.quad_conn, dtype=np.int64)

        pixel_gids = np.asarray(self.pixel_gids, dtype=np.int64)
        pixel_conn = np.asarray(self.pixel_conn, dtype=np.int64)

        poly_gids = np.asarray(self.poly_gids, dtype=np.int64)
        if self.poly_faces:
            lengths = np.array(
                [len(f) for f in self.poly_faces], dtype=np.int64
            )
            poly_offsets = np.zeros(len(self.poly_faces) + 1, dtype=np.int64)
            poly_offsets[1:] = np.cumsum(lengths)
            poly_conn = np.concatenate(self.poly_faces)
        else:
            poly_conn = np.asarray([], dtype=np.int64)
            poly_offsets = np.array([0], dtype=np.int64)

        return FaceRegistry(
            owner=owner,
            neighbor=neighbor,
            face_type=face_type,
            tri_gids=tri_gids,
            tri_conn=tri_conn,
            quad_gids=quad_gids,
            quad_conn=quad_conn,
            pixel_gids=pixel_gids,
            pixel_conn=pixel_conn,
            poly_gids=poly_gids,
            poly_conn=poly_conn,
            poly_offsets=poly_offsets,
        )


# =============================================================================
# Face Registry Build (Public API)
# =============================================================================
# Build a FaceRegistry from mesh arrays, then convert to FaceBlock dict.


def build_face_registry(
    grid: pv.UnstructuredGrid,
) -> FaceRegistry:
    """
    Build a FaceRegistry from mesh cell types, offsets, connectivity and grid.

    Iterates over 3D cells, expands fixed-face cells from built-in patterns,
    falls back to ``grid.get_cell()`` when necessary, and finalizes the
    registry.

    Parameters
    ----------
    grid : pv.UnstructuredGrid
        Input volume mesh.

    Returns
    -------
    FaceRegistry
        Immutable registry of unique faces.
    """
    n_cells = grid.n_cells
    celltypes = grid.celltypes
    offsets = grid.offset
    cell_conn = grid.cell_connectivity

    reg = FaceRegistryBuilder.create()

    for cell_id in range(n_cells):
        ctype = pv.CellType(celltypes[cell_id])
        if get_cell_dimension(ctype) < TopologyDim.VOLUME:
            continue

        face_patterns = get_fixed_face_patterns(ctype)

        # fallback to get_cell if face patterns are not supported
        if face_patterns is None:
            cell = grid.get_cell(cell_id)
            for j in range(cell.n_faces):
                face = cell.get_face(j)
                reg.register_face(face.type, cell_id, face.point_ids)
            continue

        start, end = offsets[cell_id], offsets[cell_id + 1]
        pts = cell_conn[start:end]

        for ftype, local_faces in face_patterns.items():
            for lf in local_faces:
                reg.register_face(ftype, cell_id, pts[lf])

    return reg.finalize()


def build_face_blocks_from_registry(
    registry: FaceRegistry,
) -> dict[pv.CellType, FaceBlock]:
    """
    Convert a FaceRegistry into a dict of FaceBlock by face type.

    Returns FixedFaceBlock for TRIANGLE, QUAD, and PIXEL, and JaggedFaceBlock
    for POLYGON. Only includes face types that have at least one face.

    Parameters
    ----------
    registry : FaceRegistry
        Registry produced by :func:`build_face_registry`.

    Returns
    -------
    dict[pv.CellType, FaceBlock]
        Face blocks keyed by face type.
    """
    face_blocks: dict[pv.CellType, FaceBlock] = {}

    if len(registry.tri_gids) > 0:
        gids = registry.tri_gids
        face_blocks[pv.CellType.TRIANGLE] = FixedFaceBlock(
            cell_type=pv.CellType.TRIANGLE,
            global_indices=gids,
            owner=registry.owner[gids],
            neighbor=registry.neighbor[gids],
            conn=registry.tri_conn,
        )

    if len(registry.quad_gids) > 0:
        gids = registry.quad_gids
        face_blocks[pv.CellType.QUAD] = FixedFaceBlock(
            cell_type=pv.CellType.QUAD,
            global_indices=gids,
            owner=registry.owner[gids],
            neighbor=registry.neighbor[gids],
            conn=registry.quad_conn,
        )

    if len(registry.pixel_gids) > 0:
        gids = registry.pixel_gids
        face_blocks[pv.CellType.PIXEL] = FixedFaceBlock(
            cell_type=pv.CellType.PIXEL,
            global_indices=gids,
            owner=registry.owner[gids],
            neighbor=registry.neighbor[gids],
            conn=registry.pixel_conn,
        )

    if len(registry.poly_gids) > 0:
        gids = registry.poly_gids
        face_blocks[pv.CellType.POLYGON] = JaggedFaceBlock(
            cell_type=pv.CellType.POLYGON,
            global_indices=gids,
            owner=registry.owner[gids],
            neighbor=registry.neighbor[gids],
            conn=registry.poly_conn,
            offsets=registry.poly_offsets,
        )

    return face_blocks

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import pyvista as pv

from graphlow.utils.topology_helper import (
    TopologyDim,
    gather_jagged_elements,
    get_cell_dimension,
)


@dataclass(frozen=True)
class CellBlock:
    """Base class for cell blocks."""

    cell_type: pv.CellType
    global_indices: np.ndarray  # (Nc,)    # Nc: thenumber of cells

    @property
    def n_cells(self) -> int:
        return self.global_indices.shape[0]

    @property
    def topological_dimension(self) -> TopologyDim:
        return get_cell_dimension(self.cell_type)


@dataclass(frozen=True)
class FixedCellBlock(CellBlock):
    """
    Fixed-length cell block (ex. Tri, Quad, Tet, Hex).
    """

    conn: np.ndarray  # (Nc, k)


@dataclass(frozen=True)
class JaggedCellBlock(CellBlock):
    """
    Jagged cell block (ex. Polygon, Polyhedron).
    """

    conn: np.ndarray  # (N_total_points,) flat point indices
    offsets: np.ndarray  # (Nc+1,) start index per cell in conn


# =============================================================================
# Face Blocks (Interface Blocks)
# =============================================================================


@dataclass(frozen=True)
class FaceBlock:
    """Base class for face blocks."""

    cell_type: pv.CellType
    global_indices: np.ndarray  # (Nf,) global face indices
    owner: np.ndarray  # (Nf,) global cell indices owning each face
    neighbor: (
        np.ndarray
    )  # (Nf,) global index of neighboring cells (boundary faces are -1)

    @property
    def n_faces(self) -> int:
        return self.global_indices.shape[0]

    @property
    def topological_dimension(self) -> TopologyDim:
        return get_cell_dimension(self.cell_type)

    @abstractmethod
    def boundary_faces(self) -> FaceBlock:
        """
        Return the boundary faces of the face block.
        """
        pass


@dataclass(frozen=True)
class FixedFaceBlock(FaceBlock):
    """
    Fixed-length face block (ex. Tri, Quad).
    """

    conn: np.ndarray  # (Nf, k)

    def boundary_faces(self) -> FixedFaceBlock:
        """
        Return the boundary faces of the face block.
        """
        mask = self.neighbor == -1
        owner = self.owner[mask]
        neighbor = self.neighbor[mask]
        conn = self.conn[mask]
        return FixedFaceBlock(
            cell_type=self.cell_type,
            global_indices=self.global_indices[mask],
            owner=owner,
            neighbor=neighbor,
            conn=conn,
        )


@dataclass(frozen=True)
class JaggedFaceBlock(FaceBlock):
    """
    Jagged face block (ex. Polygon).
    """

    conn: np.ndarray  # (N_total_points,) flat point indices
    offsets: np.ndarray  # (Nf+1,) start index per face in conn

    def boundary_faces(self) -> JaggedFaceBlock:
        """
        Return the boundary faces of the face block.
        """
        mask = self.neighbor == -1
        face_ids = np.flatnonzero(mask)
        start_offsets = self.offsets[face_ids]
        end_offsets = self.offsets[face_ids + 1]

        new_conn, new_offsets = gather_jagged_elements(
            self.conn, start_offsets, end_offsets
        )

        return JaggedFaceBlock(
            cell_type=self.cell_type,
            global_indices=self.global_indices[mask],
            owner=self.owner[mask],
            neighbor=self.neighbor[mask],
            conn=new_conn,
            offsets=new_offsets,
        )

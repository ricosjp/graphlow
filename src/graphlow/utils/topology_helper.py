from __future__ import annotations

from enum import IntEnum

import numpy as np
import pyvista as pv

# =============================================================================
# Cell Type and Dimension
# =============================================================================
# Enums and lookups: topological dimension of cell types (POINT/LINE/SURFACE/
# VOLUME) and reverse lookup (cell types for a given dimension).


class TopologyDim(IntEnum):
    """Topological mesh dimension."""

    POINT = 0
    LINE = 1
    SURFACE = 2
    VOLUME = 3


_CELL_TYPE_TO_DIM: dict[pv.CellType, TopologyDim] = {
    # 1D cells
    pv.CellType.LINE: TopologyDim.LINE,
    # 2D cells
    pv.CellType.TRIANGLE: TopologyDim.SURFACE,
    pv.CellType.QUAD: TopologyDim.SURFACE,
    pv.CellType.PIXEL: TopologyDim.SURFACE,
    pv.CellType.POLYGON: TopologyDim.SURFACE,
    # 3D cells
    pv.CellType.TETRA: TopologyDim.VOLUME,
    pv.CellType.HEXAHEDRON: TopologyDim.VOLUME,
    pv.CellType.WEDGE: TopologyDim.VOLUME,
    pv.CellType.PYRAMID: TopologyDim.VOLUME,
    pv.CellType.VOXEL: TopologyDim.VOLUME,
    pv.CellType.POLYHEDRON: TopologyDim.VOLUME,
}


def get_cell_dimension(cell_type: pv.CellType) -> TopologyDim:
    """
    Return the topological dimension of a VTK cell type.

    Parameters
    ----------
    cell_type : pv.CellType
        VTK cell type (e.g. TRIANGLE, TETRA, HEXAHEDRON).

    Returns
    -------
    TopologyDim
        POINT(0), LINE(1), SURFACE(2), or VOLUME(3).

    Raises
    ------
    ValueError
        If cell_type is not in the supported set.
    """
    if cell_type not in _CELL_TYPE_TO_DIM:
        raise ValueError(f"Unknown or unsupported cell type: {cell_type}")
    return _CELL_TYPE_TO_DIM[cell_type]


# =============================================================================
# Surface mesh: watertight (closed) check
# =============================================================================
def is_surface_mesh_watertight(
    grid: pv.UnstructuredGrid | pv.PolyData,
) -> bool:
    """
    Return True if the surface mesh is watertight (closed).

    Uses PyVista's extract_feature_edges with boundary_edges=True only.
    Watertight if no boundary edges exist (edges.n_cells == 0).

    Parameters
    ----------
    grid : pv.UnstructuredGrid or pv.PolyData
        Surface mesh (2D cells in 3D space).

    Returns
    -------
    bool
        True if there are no boundary edges (closed surface).
    """
    pvmesh = grid.cast_to_unstructured_grid()
    edges = pvmesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    return edges.n_cells == 0


# =============================================================================
# Cell Selection
# =============================================================================
# Select cells of a given type from mesh-wide cell type and offset arrays.


def select_cells_by_type(
    cell_types: np.ndarray,
    cell_offsets: np.ndarray,
    cell_type: pv.CellType,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return global indices and offset ranges for cells of the given type.

    Parameters
    ----------
    cell_types : np.ndarray
        Per-cell type IDs (e.g. from pvmesh.celltypes).
    cell_offsets : np.ndarray
        Per-cell start indices into connectivity (length n_cells + 1).
    cell_type : pv.CellType
        The cell type to select.

    Returns
    -------
    global_indices : np.ndarray
        Global cell indices of the selected cells.
    start_offsets : np.ndarray
        Start index into connectivity for each selected cell.
    end_offsets : np.ndarray
        End index into connectivity for each selected cell.

    Raises
    ------
    ValueError
        If the mesh does not contain any cell of the given type.
    """
    mask = cell_types == cell_type.value
    global_indices = np.where(mask)[0]
    if len(global_indices) == 0:
        raise ValueError(f"The mesh does not contain {cell_type.name}.")

    start_offsets = cell_offsets[:-1][mask]
    end_offsets = cell_offsets[1:][mask]
    return global_indices, start_offsets, end_offsets


# =============================================================================
# Element Gathering
# =============================================================================
# Extract fixed-length or jagged (variable-length) elements from a flat
# connectivity array using start/end offsets.


def gather_fixed_elements(
    conn: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """
    Gather fixed-length elements from a flat connectivity array.

    All elements must have the same length (starts[i+1]-starts[i] constant).
    Returns a 2D array of shape ``(n_elements, k)`` where k is that length.

    Parameters
    ----------
    conn : np.ndarray
        Flat connectivity array (point indices).
    starts : np.ndarray
        Start index of each element in conn.
    ends : np.ndarray
        End index (exclusive) of each element in conn.

    Returns
    -------
    np.ndarray
        Shape ``(n_elements, k)``. conn[indices] with indices built from
        starts/ends.

    Raises
    ------
    ValueError
        If element lengths are not all equal.
    """
    lengths = ends - starts
    if not np.all(lengths == lengths[0]):
        raise ValueError(
            "gather_fixed_elements requires fixed-length elements."
        )
    k = lengths[0]
    indices = starts[:, None] + np.arange(k)
    return conn[indices]


def gather_jagged_elements(
    conn: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gather jagged (variable-length) elements and compute compact offsets.

    Extracts conn[starts[i]:ends[i]] for each i and concatenates into a single
    array, with new_offsets such that element j is
    new_conn[new_offsets[j]:new_offsets[j+1]].

    Parameters
    ----------
    conn : np.ndarray
        Flat connectivity array.
    starts : np.ndarray
        Start index of each element in conn.
    ends : np.ndarray
        End index (exclusive) of each element in conn.

    Returns
    -------
    gathered : np.ndarray
        Concatenation of the selected segments.
    new_offsets : np.ndarray
        Shape ``(n_elements + 1,)``. Segment boundaries into gathered.
    """
    if len(starts) == 0:
        return np.zeros(0, dtype=conn.dtype), np.array([0], dtype=np.int64)

    length = ends - starts

    # calculate new offsets: [0, len0, len0+len1, ...]
    new_offsets = np.zeros(len(starts) + 1, dtype=starts.dtype)
    new_offsets[1:] = np.cumsum(length)

    total_length = length.sum()
    ones = np.ones(total_length, dtype=np.int64)
    ones[0] = starts[0]
    if len(starts) > 1:
        jump_idx = new_offsets[1:-1]
        # the start of the next cell - the end of the previous cell + 1
        jumps = starts[1:] - ends[:-1] + 1
        ones[jump_idx] = jumps

    idx = np.cumsum(ones)
    return conn[idx], new_offsets

"""Skeleton builders: CP, PC, CC, PP via scipy.sparse.

Advanced / low-level API: this module works directly with scipy.sparse and
is intended for users who need host-side sparse matrices or custom
pipeline control. Prefer mesh.topology.incidence/adjacency for
backend tensors.
"""

from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING, Protocol

import numpy as np
import scipy.sparse as sps

if TYPE_CHECKING:
    from graphlow.core.face_registry import FaceRegistry


class IncidenceName(StrEnum):
    """
    Incidence names.

    - CP : Cell-Point incidence ``(n_cells, n_points)``
    - PC : Point-Cell incidence ``(n_points, n_cells)``
    - FC : Face-Cell incidence ``(n_faces, n_cells)``
    - CF : Cell-Face incidence ``(n_cells, n_faces)``
    - FP : Face-Point incidence ``(n_faces, n_points)``
    - PF : Point-Face incidence ``(n_points, n_faces)``
    """

    #: Cell-point incidence ``(n_cells, n_points)``.
    CP = auto()
    #: Point-cell incidence ``(n_points, n_cells)``.
    PC = auto()
    #: Face-cell incidence ``(n_faces, n_cells)``.
    FC = auto()
    #: Cell-face incidence ``(n_cells, n_faces)``.
    CF = auto()
    #: Face-point incidence ``(n_faces, n_points)``.
    FP = auto()
    #: Point-face incidence ``(n_points, n_faces)``.
    PF = auto()


class AdjacencyName(StrEnum):
    """
    Adjacency names.

    - CC : Cell adjacency ``(n_cells, n_cells)``
    - PP : Point adjacency ``(n_points, n_points)``
    """

    #: Cell adjacency ``(n_cells, n_cells)``.
    CC = auto()
    #: Point adjacency ``(n_points, n_points)``.
    PP = auto()


class TopologyLike(Protocol):
    """Builder required topology interface."""

    @property
    def n_cells(self) -> int: ...
    @property
    def n_points(self) -> int: ...
    def cell_conn(self) -> np.ndarray: ...
    def cell_offsets(self) -> np.ndarray: ...
    def face_registry(self) -> FaceRegistry: ...


def _build_cp(
    topo: TopologyLike,
) -> sps.csr_array:
    """
    Cell -> Point incidence ``(n_cells, n_points)`` as scipy CSR.

    Parameters
    ----------
    topo: TopologyLike

    Returns
    -------
    scipy.sparse.csr_array
        CSR array of shape ``(n_cells, n_points)``.
        dtype is np.int64.
    """
    cell_conn = topo.cell_conn()
    cell_offsets = topo.cell_offsets()
    n_points = topo.n_points
    n_cells = topo.n_cells
    data = np.ones_like(cell_conn)
    return sps.csr_array(
        (data, cell_conn, cell_offsets),
        shape=(n_cells, n_points),
    )


def _build_pc(
    topo: TopologyLike,
) -> sps.csr_array:
    """
    Point -> Cell incidence ``(n_points, n_cells)`` as scipy CSR.

    Parameters
    ----------
    topo: TopologyLike

    Returns
    -------
    scipy.sparse.csr_array
        CSR array of shape ``(n_points, n_cells)``.
        dtype is np.int64.
    """
    cp = _build_cp(topo)
    return cp.T.tocsr()


def _build_fc(
    topo: TopologyLike,
) -> sps.csr_array:
    """
    Face -> Cell incidence ``(n_faces, n_cells)`` as scipy CSR.

    FC[f, owner[f]] = +1, FC[f, neighbor[f]] = -1 for internal faces.
    Boundary faces have only owner (+1). Oriented so that FC^T @ flux
    gives cell-wise divergence when flux is owner-to-neighbor.

    Parameters
    ----------
    topo : TopologyLike

    Returns
    -------
    scipy.sparse.csr_array
        CSR array of shape ``(n_faces, n_cells)``.
        dtype is np.int64.
    """
    face_registry = topo.face_registry()
    owner = face_registry.owner
    neighbor = face_registry.neighbor
    n_faces = face_registry.n_faces()
    n_cells = topo.n_cells

    # Owner: every face contributes (face_id, owner[f], +1)
    rows_o = np.arange(n_faces, dtype=np.int64)
    cols_o = owner
    vals_o = np.ones(n_faces, dtype=np.int64)

    # Neighbor: internal faces only, (face_id, neighbor[f], -1)
    internal = neighbor != -1
    rows_n = np.flatnonzero(internal)
    cols_n = neighbor[internal]
    vals_n = -np.ones(len(rows_n), dtype=np.int64)

    rows = np.concatenate([rows_o, rows_n])
    cols = np.concatenate([cols_o, cols_n])
    vals = np.concatenate([vals_o, vals_n])

    return sps.coo_array((vals, (rows, cols)), shape=(n_faces, n_cells)).tocsr()


def _build_cf(
    topo: TopologyLike,
) -> sps.csr_array:
    """
    Cell -> Face incidence (n_cells, n_faces) as scipy CSR.

    CF = FC.T, so CF[c, f] is +1 if c is owner of f, -1 if c is neighbor,
    0 otherwise.

    Parameters
    ----------
    topo : TopologyLike

    Returns
    -------
    scipy.sparse.csr_array
        CSR array of shape ``(n_cells, n_faces)``.
        dtype is np.int64.
    """
    fc = _build_fc(topo)
    return fc.T.tocsr()


def _build_fp(
    topo: TopologyLike,
) -> sps.csr_array:
    """
    Face -> Point incidence ``(n_faces, n_points)`` as scipy CSR.

    FP[f, conn[f]] = +1 for each point in face f.

    Parameters
    ----------
    topo : TopologyLike

    Returns
    -------
    scipy.sparse.csr_array
        CSR array of shape ``(n_faces, n_points)``.
        dtype is np.int64.
    """
    registry = topo.face_registry()
    n_faces = registry.n_faces()
    n_points = topo.n_points
    rows_list = []
    cols_list = []

    # TRIANGLE
    if len(registry.tri_gids) > 0:
        rows = np.repeat(registry.tri_gids, registry.tri_conn.shape[1])
        cols = registry.tri_conn.reshape(-1)
        rows_list.append(rows)
        cols_list.append(cols)

    # QUAD
    if len(registry.quad_gids) > 0:
        rows = np.repeat(registry.quad_gids, registry.quad_conn.shape[1])
        cols = registry.quad_conn.reshape(-1)
        rows_list.append(rows)
        cols_list.append(cols)

    # PIXEL
    if len(registry.pixel_gids) > 0:
        rows = np.repeat(registry.pixel_gids, registry.pixel_conn.shape[1])
        cols = registry.pixel_conn.reshape(-1)
        rows_list.append(rows)
        cols_list.append(cols)

    # POLYGON
    if len(registry.poly_gids) > 0:
        lengths = np.diff(registry.poly_offsets)
        rows = np.repeat(registry.poly_gids, lengths)
        cols = registry.poly_conn  # already flat
        rows_list.append(rows)
        cols_list.append(cols)

    if not rows_list:
        return sps.coo_array((0, n_points), dtype=np.int64).tocsr()

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    vals = np.ones(len(rows), dtype=np.int64)

    return sps.coo_array(
        (vals, (rows, cols)), shape=(n_faces, n_points)
    ).tocsr()


def _build_pf(
    topo: TopologyLike,
) -> sps.csr_array:
    """
    Point -> Face incidence ``(n_points, n_faces)`` as scipy CSR.

    Parameters
    ----------
    topo : TopologyLike

    Returns
    -------
    scipy.sparse.csr_array
        CSR array of shape ``(n_points, n_faces)``.
        dtype is np.int64.
    """
    fp = _build_fp(topo)
    return fp.T.tocsr()


def _build_cc(
    topo: TopologyLike,
) -> sps.csr_array:
    """Cell adjacency ``(n_cells, n_cells)`` as scipy CSR.

    Parameters
    ----------
    topo: TopologyLike

    Returns
    -------
    scipy.sparse.csr_array
        CSR array of shape ``(n_cells, n_cells)``.
        dtype is np.int64.
    """
    cp = _build_cp(topo).astype(bool)
    pc = _build_pc(topo).astype(bool)
    pp = cp @ pc
    return pp.astype(np.int64)


def _build_pp(
    topo: TopologyLike,
) -> sps.csr_array:
    """Point adjacency ``(n_points, n_points)`` as scipy CSR.

    Parameters
    ----------
    topo: TopologyLike

    Returns
    -------
    scipy.sparse.csr_array
        CSR array of shape ``(n_points, n_points)``.
        dtype is np.int64.
    """
    cp = _build_cp(topo).astype(bool)
    pc = _build_pc(topo).astype(bool)
    pp = pc @ cp
    return pp.astype(np.int64)


_BASIC_SPARSES = {
    IncidenceName.CP: _build_cp,
    IncidenceName.PC: _build_pc,
    IncidenceName.FC: _build_fc,
    IncidenceName.CF: _build_cf,
    IncidenceName.FP: _build_fp,
    IncidenceName.PF: _build_pf,
    AdjacencyName.CC: _build_cc,
    AdjacencyName.PP: _build_pp,
}


def build_skeleton(
    topo: TopologyLike, name: IncidenceName | AdjacencyName
) -> sps.csr_array:
    """
    Build basic skeleton matrix by name.

    Advanced API: returns scipy.sparse on the host. Use when you need
    scipy directly (e.g. for scipy solvers or custom algorithms) rather
    than backend tensors from mesh.topology.incidence/adjacency.

    Parameters
    ----------
    topo : TopologyLike
        Topology object.
    name : IncidenceName | AdjacencyName
        Incidence or adjacency name (CP, PC, FC, CF, FP, PF, CC, PP).

    Returns
    -------
    scipy.sparse.csr_array
        CSR array. Shape depends on name (e.g. ``(n_cells, n_points)`` for CP).
        dtype is np.int64.
    """
    if name not in _BASIC_SPARSES:
        raise ValueError(f"Unknown incidence or adjacency name: {name}")

    return _BASIC_SPARSES[name](topo)

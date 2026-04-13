"""Point <-> Cell reduction (segment_sum vs sparse matmul)."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.sparse as sps
import torch

from graphlow.graph.skeleton_builder import AdjacencyName
from graphlow.utils.dimension import get_dimension

if TYPE_CHECKING:
    from graphlow.core.backend.base import Backend, TensorLike
    from graphlow.core.mesh import TensorMesh
    from graphlow.core.topology import MeshTopology


def _dispatch_map[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x: T,
    mode: Literal["sum", "mean", "conservative", "diff", "div"],
    method: Literal["segment", "sparse"],
    segment_fns: Mapping[str, Callable[[Backend[T], MeshTopology[T], T], T]],
    sparse_fns: Mapping[str, Callable[[MeshTopology[T], T], T]],
) -> T:
    """Dispatch to segment or sparse implementation by method and mode."""
    match method:
        case "segment":
            segment_fn = segment_fns.get(mode)
            if segment_fn is None:
                raise ValueError(
                    f"Invalid mode; got {mode!r}",
                )
            return segment_fn(backend, topo, x)
        case "sparse":
            sparse_fn = sparse_fns.get(mode)
            if sparse_fn is None:
                raise ValueError(
                    f"Invalid mode; got {mode!r}",
                )
            return sparse_fn(topo, x)
        case _:
            raise ValueError(
                "method must be 'segment', 'sparse'", f"got {method!r}"
            )


# =============================================================================
# map point to cell
# =============================================================================
def map_point_to_cell[T: TensorLike](
    mesh: TensorMesh[T],
    x_point: T,
    mode: Literal["sum", "mean", "conservative"],
    method: Literal["segment", "sparse"] = "segment",
) -> T:
    r"""
    Map point field to cell.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Input mesh.
    x_point : T
        Point field of shape ``(n_points, ...)``.
    mode : "sum" | "mean" | "conservative"
        Mapping mode.

        ``"sum"``
            Sum of point values in each cell.

        ``"mean"``
            Row-normalized CP operator.

            The cell value is computed as the simple average of the
            values at the points connected to the cell:

            .. math::

               x_{\text{cell}} =
               \frac{1}{\operatorname{deg}(C)}
               \sum_{p \in C} x_{\text{point}}

            where :math:`\operatorname{deg}(C)` is the number of points
            connected to cell :math:`C`.

            This corresponds to a local arithmetic averaging over the
            cell stencil. It treats each point in the cell equally,
            regardless of how many cells that point belongs to.
            In general, this method does not guarantee global conservation
            of the total quantity.

        ``"conservative"``
            Column-normalized CP operator.

            The cell value is obtained by distributing each point value
            equally among the cells connected to that point:

            .. math::

               x_{\text{cell}} =
               \sum_{p \in C}
               \left(\frac{1}{\operatorname{deg}(p)}\right)
               x_{\text{point}}

            where :math:`\operatorname{deg}(p)` is the number of cells
            connected to point :math:`p`.

            Each point contributes its value conservatively to its
            neighboring cells, and the total sum over cells equals
            the total sum over points,
            making this formulation suitable for conservative transfers.
    method : "segment" | "sparse"
        Mapping method. Default is "segment".

        ``"segment"``
            Computes the reduction
            without explicitly constructing the CP matrix.
            It is typically more memory efficient, especially on GPU.
            In most practical settings, this method is recommended.

        ``"sparse"``
            Computes the reduction by multiplying the CP matrix
            with the point values.
            It depends strongly on backend implementation quality but
            it can be beneficial when the CP matrix is used repeatedly.


    Returns
    -------
    T
        Tensor of shape ``(n_cells, ...)``.
    """
    backend = mesh.backend
    topo = mesh.topology
    segment_fns = {
        "sum": _segment_sum_map_point_to_cell,
        "mean": _segment_mean_map_point_to_cell,
        "conservative": _segment_conservative_map_point_to_cell,
    }
    sparse_fns = {
        "sum": _sparse_sum_map_point_to_cell,
        "mean": _sparse_mean_map_point_to_cell,
        "conservative": _sparse_conservative_map_point_to_cell,
    }
    return _dispatch_map(
        backend, topo, x_point, mode, method, segment_fns, sparse_fns
    )


# --- segment ---
def _segment_sum_map_point_to_cell[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_point: T,
) -> T:
    """Segment sum map point to cell."""
    n_cells = topo.n_cells
    cell_conn = backend.as_index_tensor(topo.cell_conn())
    cell_offsets = topo.cell_offsets()

    points_per_cell = np.diff(cell_offsets)
    segment_ids = backend.as_index_tensor(
        np.repeat(np.arange(n_cells), points_per_cell)
    )

    source = x_point[cell_conn]
    out_shape = (n_cells,) + tuple(source.shape[1:])
    y_cell = backend.zeros(out_shape, dimension=get_dimension(x_point))
    return y_cell.index_add_(0, segment_ids, source)


def _segment_mean_map_point_to_cell[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_point: T,
) -> T:
    """Segment mean map point to cell."""
    cell_sums = _segment_sum_map_point_to_cell(backend, topo, x_point)
    deg_cp = topo.degree_cp().reshape((-1, *([1] * (cell_sums.ndim - 1))))
    return cell_sums / deg_cp


def _segment_conservative_map_point_to_cell[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_point: T,
) -> T:
    """Segment conservative map point to cell."""
    n_cells = topo.n_cells
    cell_conn = backend.as_index_tensor(topo.cell_conn())
    cell_offsets = topo.cell_offsets()
    deg_pc = topo.degree_pc().reshape((-1, *([1] * (x_point.ndim - 1))))
    w_point = x_point / deg_pc

    points_per_cell = np.diff(cell_offsets)
    segment_ids = backend.as_index_tensor(
        np.repeat(np.arange(n_cells), points_per_cell)
    )

    src = w_point[cell_conn]
    out_shape = (n_cells,) + tuple(src.shape[1:])
    y_cell = backend.zeros(out_shape, dimension=get_dimension(x_point))
    return y_cell.index_add_(0, segment_ids, src)


# --- sparse ---
def _sparse_sum_map_point_to_cell[T: TensorLike](
    topo: MeshTopology[T],
    x_point: T,
) -> T:
    """CP @ x_point"""
    cp_matrix = topo.cell_point_incidence()
    return cp_matrix @ x_point


def _sparse_mean_map_point_to_cell[T: TensorLike](
    topo: MeshTopology[T],
    x_point: T,
) -> T:
    """D_cell^-1 * CP @ x_point"""
    cell_sums = _sparse_sum_map_point_to_cell(topo, x_point)
    deg_cp = topo.degree_cp().reshape((-1, *([1] * (cell_sums.ndim - 1))))
    return cell_sums / deg_cp


def _sparse_conservative_map_point_to_cell[T: TensorLike](
    topo: MeshTopology[T],
    x_point: T,
) -> T:
    """CP @ (D_point^-1 * x_point)"""
    cp_matrix = topo.cell_point_incidence()
    deg_pc = topo.degree_pc().reshape((-1, *([1] * (x_point.ndim - 1))))
    w_point = x_point / deg_pc
    return cp_matrix @ w_point


# =============================================================================
# map cell to point
# =============================================================================
def map_cell_to_point[T: TensorLike](
    mesh: TensorMesh[T],
    x_cell: T,
    mode: Literal["sum", "mean", "conservative"],
    method: Literal["segment", "sparse"] = "segment",
) -> T:
    r"""
    Map cell field to point.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Input mesh.
    x_cell : T
        Cell field of shape ``(n_cells, ...)``.
    mode : "sum" | "mean" | "conservative"
        Mapping mode.

        ``"sum"``
            Sum of cell values that are connected to the point.

        ``"mean"``
            Row-normalized PC operator.

            The point value is computed as the arithmetic average of
            the values of the cells connected to the point:

            .. math::

               x_{\text{point}} =
               \frac{1}{\operatorname{deg}(p)}
               \sum_{C \ni p} x_{\text{cell}}

            where :math:`\operatorname{deg}(p)` is the number of cells
            connected to point :math:`p`.

            This corresponds to a local averaging over the cell stencil
            around each point. Each neighboring cell contributes equally,
            regardless of the number of points in that cell.
            In general, this method does not guarantee global conservation
            of the total quantity.

        ``"conservative"``
            Column-normalized PC operator.

            The point value is obtained by distributing each cell value
            equally among the points connected to that cell:

            .. math::

               x_{\text{point}} =
               \sum_{C \ni p}
               \left(\frac{1}{\operatorname{deg}(C)}\right)
               x_{\text{cell}}

            where :math:`\operatorname{deg}(C)` is the number of points
            in cell :math:`C`.

            Each cell contributes its value conservatively to its
            vertices, and the total sum over points equals
            the total sum over cells,
            making this formulation suitable for conservative transfers.
    method : "segment" | "sparse"
        Mapping method. Default is "segment".

        ``"segment"``
            Computes the reduction
            without explicitly constructing the PC matrix.
            It is typically more memory efficient, especially on GPU.
            In most practical settings, this method is recommended.

        ``"sparse"``
            Computes the reduction by multiplying the PC matrix
            with the cell values.
            It depends strongly on backend implementation quality but
            it can be beneficial when the PC matrix is used repeatedly.


    Returns
    -------
    T
        Tensor of shape ``(n_points, ...)``.
    """
    backend = mesh.backend
    topo = mesh.topology
    segment_fns = {
        "sum": _segment_sum_map_cell_to_point,
        "mean": _segment_mean_map_cell_to_point,
        "conservative": _segment_conservative_map_cell_to_point,
    }
    sparse_fns = {
        "sum": _sparse_sum_map_cell_to_point,
        "mean": _sparse_mean_map_cell_to_point,
        "conservative": _sparse_conservative_map_cell_to_point,
    }
    return _dispatch_map(
        backend, topo, x_cell, mode, method, segment_fns, sparse_fns
    )


# --- segment ---
def _segment_sum_map_cell_to_point[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_cell: T,
) -> T:
    """Segment sum map cell to point."""
    n_cells = topo.n_cells
    n_points = topo.n_points
    cell_conn = backend.as_index_tensor(topo.cell_conn())
    cell_offsets = topo.cell_offsets()

    points_per_cell = np.diff(cell_offsets)
    cell_ids = backend.as_index_tensor(
        np.repeat(np.arange(n_cells), points_per_cell)
    )

    source = x_cell[cell_ids]
    out_shape = (n_points,) + tuple(source.shape[1:])
    y_point = backend.zeros(out_shape, dimension=get_dimension(x_cell))
    return y_point.index_add_(0, cell_conn, source)


def _segment_mean_map_cell_to_point[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_cell: T,
) -> T:
    """Segment mean map cell to point."""
    point_sums = _segment_sum_map_cell_to_point(backend, topo, x_cell)
    deg_pc = topo.degree_pc().reshape((-1, *([1] * (point_sums.ndim - 1))))
    return point_sums / deg_pc


def _segment_conservative_map_cell_to_point[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_cell: T,
) -> T:
    """Segment conservative map cell to point."""
    n_cells = topo.n_cells
    n_points = topo.n_points
    cell_conn = backend.as_index_tensor(topo.cell_conn())
    cell_offsets = topo.cell_offsets()
    deg_cp = topo.degree_cp().reshape((-1, *([1] * (x_cell.ndim - 1))))
    w_cell = x_cell / deg_cp

    points_per_cell = np.diff(cell_offsets)
    cell_ids = backend.as_index_tensor(
        np.repeat(np.arange(n_cells), points_per_cell)
    )

    source = w_cell[cell_ids]
    out_shape = (n_points,) + tuple(source.shape[1:])
    y_point = backend.zeros(out_shape, dimension=get_dimension(x_cell))
    return y_point.index_add_(0, cell_conn, source)


# --- sparse ---
def _sparse_sum_map_cell_to_point[T: TensorLike](
    topo: MeshTopology[T],
    x_cell: T,
) -> T:
    """PC @ x_cell"""
    pc_matrix = topo.point_cell_incidence()
    return pc_matrix @ x_cell


def _sparse_mean_map_cell_to_point[T: TensorLike](
    topo: MeshTopology[T],
    x_cell: T,
) -> T:
    """D_point^-1 * PC @ x_cell"""
    point_sums = _sparse_sum_map_cell_to_point(topo, x_cell)
    deg_pc = topo.degree_pc().reshape((-1, *([1] * (point_sums.ndim - 1))))
    return point_sums / deg_pc


def _sparse_conservative_map_cell_to_point[T: TensorLike](
    topo: MeshTopology[T],
    x_cell: T,
) -> T:
    """PC @ (D_cell^-1 * x_cell)"""
    pc_matrix = topo.point_cell_incidence()
    deg_cp = topo.degree_cp().reshape((-1, *([1] * (x_cell.ndim - 1))))
    w_cell = x_cell / deg_cp
    return pc_matrix @ w_cell


# =============================================================================
# map cell to face
# =============================================================================
def map_cell_to_face[T: TensorLike](
    mesh: TensorMesh[T],
    x_cell: T,
    mode: Literal["sum", "mean", "conservative", "diff"],
    method: Literal["segment", "sparse"] = "segment",
) -> T:
    r"""
    Map cell field to face.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Input mesh.
    x_cell : T
        Cell field of shape ``(n_cells, ...)``.
    mode : "sum" | "mean" | "conservative" | "diff"
        Mapping mode.

        ``"sum"``
            Sum of cell values that are connected to the face.
            For manifold meshes, this corresponds to:

            .. math::

               x_{\text{face}} =
               x_{\text{cell}}[\text{owner}] +
               x_{\text{cell}}[\text{neighbor}]

        ``"mean"``
            The face value is computed as the arithmetic average of
            the values of the cells connected to the face.
            For manifold meshes, this corresponds to:

            .. math::

               x_{\text{face}} =
               \frac{1}{2}
               \left(
               x_{\text{cell}}[\text{owner}] +
               x_{\text{cell}}[\text{neighbor}]
               \right)

            In general, this method does not guarantee global conservation
            of the total quantity.

        ``"conservative"``
            The face value is obtained by distributing each cell value
            equally among the faces connected to that cell:

            .. math::

               x_{\text{face}} =
               \sum_{C \ni f}
               \left(\frac{1}{\operatorname{deg}(C)}\right)
               x_{\text{cell}}

            where :math:`\operatorname{deg}(C)` is the number of faces
            in cell :math:`C`.

            Each cell contributes its value conservatively to its
            faces, and the total sum over faces equals
            the total sum over cells,
            making this formulation suitable for conservative transfers.

        ``"diff"``
            Difference of the cell field.
            The face value is computed as the difference of the cell field:

            .. math::

               x_{\text{face}} =
               x_{\text{cell}}[\text{owner}] -
               x_{\text{cell}}[\text{neighbor}]

            This corresponds to a local difference over the cell stencil.
    method : "segment" | "sparse"
        Mapping method. Default is "segment".

        ``"segment"``
            Computes the reduction
            without explicitly constructing the FC matrix.
            It is typically more memory efficient, especially on GPU.
            In most practical settings, this method is recommended.

        ``"sparse"``
            Computes the reduction by multiplying the FC matrix
            with the cell values.
            It depends strongly on backend implementation quality but
            it can be beneficial when the FC matrix is used repeatedly.


    Returns
    -------
    T
        Tensor of shape ``(n_faces, ...)``.
    """
    backend = mesh.backend
    topo = mesh.topology
    segment_fns = {
        "sum": _segment_sum_map_cell_to_face,
        "mean": _segment_mean_map_cell_to_face,
        "conservative": _segment_conservative_map_cell_to_face,
        "diff": _segment_diff_map_cell_to_face,
    }
    sparse_fns = {
        "sum": _sparse_sum_map_cell_to_face,
        "mean": _sparse_mean_map_cell_to_face,
        "conservative": _sparse_conservative_map_cell_to_face,
        "diff": _sparse_diff_map_cell_to_face,
    }
    return _dispatch_map(
        backend, topo, x_cell, mode, method, segment_fns, sparse_fns
    )


# --- segment ---
def _segment_sum_map_cell_to_face[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_cell: T,
) -> T:
    """Segment sum map cell to face."""
    registry = topo.face_registry()
    owner = backend.as_index_tensor(registry.owner)
    neighbor = backend.as_index_tensor(registry.neighbor)
    mask_bnd = neighbor == -1
    y_face = x_cell[owner].clone()
    y_face[~mask_bnd] += x_cell[neighbor[~mask_bnd]]
    return y_face


def _segment_mean_map_cell_to_face[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_cell: T,
) -> T:
    """Segment mean map cell to face."""
    face_sums = _segment_sum_map_cell_to_face(backend, topo, x_cell)
    deg_fc = topo.degree_fc().reshape((-1, *([1] * (face_sums.ndim - 1))))
    return face_sums / deg_fc


def _segment_conservative_map_cell_to_face[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_cell: T,
) -> T:
    """Segment conservative map cell to face."""
    registry = topo.face_registry()
    owner = backend.as_index_tensor(registry.owner)
    neighbor = backend.as_index_tensor(registry.neighbor)
    mask_bnd = neighbor == -1
    deg_cf = topo.degree_cf().reshape((-1, *([1] * (x_cell.ndim - 1))))
    w_cell = x_cell / deg_cf
    y_face = w_cell[owner].clone()
    y_face[~mask_bnd] += w_cell[neighbor[~mask_bnd]]
    return y_face


def _segment_diff_map_cell_to_face[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_cell: T,
) -> T:
    """Segment diff map cell to face."""
    registry = topo.face_registry()
    owner = backend.as_index_tensor(registry.owner)
    neighbor = backend.as_index_tensor(registry.neighbor)
    mask_bnd = neighbor == -1
    y_face = x_cell[owner].clone()
    y_face[~mask_bnd] -= x_cell[neighbor[~mask_bnd]]
    return y_face


# --- sparse ---
def _sparse_sum_map_cell_to_face[T: TensorLike](
    topo: MeshTopology[T],
    x_cell: T,
) -> T:
    """abs(FC) @ x_cell"""
    fc_matrix = topo.face_cell_incidence()
    abs_fc_matrix = torch.abs(fc_matrix)
    return abs_fc_matrix @ x_cell


def _sparse_mean_map_cell_to_face[T: TensorLike](
    topo: MeshTopology[T],
    x_cell: T,
) -> T:
    """D_face^-1 * FC @ x_cell"""
    face_sums = _sparse_sum_map_cell_to_face(topo, x_cell)
    deg_fc = topo.degree_fc().reshape((-1, *([1] * (face_sums.ndim - 1))))
    return face_sums / deg_fc


def _sparse_conservative_map_cell_to_face[T: TensorLike](
    topo: MeshTopology[T],
    x_cell: T,
) -> T:
    """FC @ (D_cell^-1 * x_cell)"""
    fc_matrix = topo.face_cell_incidence()
    abs_fc_matrix = torch.abs(fc_matrix)
    deg_cf = topo.degree_cf().reshape((-1, *([1] * (x_cell.ndim - 1))))
    w_cell = x_cell / deg_cf
    return abs_fc_matrix @ w_cell


def _sparse_diff_map_cell_to_face[T: TensorLike](
    topo: MeshTopology[T],
    x_cell: T,
) -> T:
    """FC @ x_cell"""
    fc_matrix = topo.face_cell_incidence()
    return fc_matrix @ x_cell


# =============================================================================
# map face to cell
# =============================================================================
def map_face_to_cell[T: TensorLike](
    mesh: TensorMesh[T],
    x_face: T,
    mode: Literal["sum", "mean", "conservative", "div"],
    method: Literal["segment", "sparse"] = "segment",
) -> T:
    r"""
    Map face field to cell.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Input mesh.
    x_face : T
        Face field of shape ``(n_faces, ...)``.
    mode : "sum" | "mean" | "conservative" | "div"
        Mapping mode.

        ``"sum"``
            Sum of face values in cell.

        ``"mean"``
            Row-normalized CF operator.

            The cell value is computed as the arithmetic average of
            the values of the faces connected to the cell:

            .. math::

               x_{\text{cell}} =
               \frac{1}{\operatorname{deg}(C)}
               \sum_{f \ni C} x_{\text{face}}

            where :math:`\operatorname{deg}(C)` is the number of faces
            connected to cell :math:`C`.

            This corresponds to a local arithmetic averaging over the
            cell stencil.
            In general, this method does not guarantee global conservation
            of the total quantity.

        ``"conservative"``
            Column-normalized CF operator.

            The cell value is obtained by distributing each face value
            equally among the cells connected to that face:

            .. math::

               x_{\text{cell}} =
               \sum_{f \ni C}
               \left(\frac{1}{\operatorname{deg}(f)}\right)
               x_{\text{face}}

            where :math:`\operatorname{deg}(f)` is the number of cells
            connected to face :math:`f`.

            Each face contributes its value conservatively to its
            neighboring cells, and the total sum over cells equals
            the total sum over faces,
            making this formulation suitable for conservative transfers.

        ``"div"``
            Divergence of the face field.
            The cell value is computed as the divergence of the face field:

            .. math::

               x_{\text{cell}} =
               \sum_{f \ni C} x_{\text{face}} \cdot n_f

            where :math:`n_f` is the normal vector of face :math:`f`.

            This corresponds to a local divergence over the cell stencil.
    method : "segment" | "sparse"
        Mapping method. Default is "segment".

        ``"segment"``
            Computes the reduction
            without explicitly constructing the CF matrix.
            It is typically more memory efficient, especially on GPU.
            In most practical settings, this method is recommended.

        ``"sparse"``
            Computes the reduction by multiplying the CF matrix
            with the face values.
            It depends strongly on backend implementation quality but
            it can be beneficial when the CF matrix is used repeatedly.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, ...)``.
    """
    backend = mesh.backend
    topo = mesh.topology
    segment_fns = {
        "sum": _segment_sum_map_face_to_cell,
        "mean": _segment_mean_map_face_to_cell,
        "conservative": _segment_conservative_map_face_to_cell,
        "div": _segment_div_map_face_to_cell,
    }
    sparse_fns = {
        "sum": _sparse_sum_map_face_to_cell,
        "mean": _sparse_mean_map_face_to_cell,
        "conservative": _sparse_conservative_map_face_to_cell,
        "div": _sparse_div_map_face_to_cell,
    }
    return _dispatch_map(
        backend, topo, x_face, mode, method, segment_fns, sparse_fns
    )


# --- segment ---
def _segment_sum_map_face_to_cell[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_face: T,
) -> T:
    """Segment sum map face to cell."""
    n_cells = topo.n_cells
    registry = topo.face_registry()
    owner = backend.as_index_tensor(registry.owner)
    neighbor = backend.as_index_tensor(registry.neighbor)
    mask_bnd = neighbor != -1

    out_shape = (n_cells,) + tuple(x_face.shape[1:])
    y_cell = backend.zeros(out_shape, dimension=get_dimension(x_face))
    y_cell.index_add_(0, owner, x_face)
    y_cell.index_add_(0, neighbor[mask_bnd], x_face[mask_bnd])
    return y_cell


def _segment_mean_map_face_to_cell[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_face: T,
) -> T:
    """Segment mean map face to cell."""
    cell_sums = _segment_sum_map_face_to_cell(backend, topo, x_face)
    deg_cf = topo.degree_cf().reshape((-1, *([1] * (cell_sums.ndim - 1))))
    return cell_sums / deg_cf


def _segment_conservative_map_face_to_cell[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_face: T,
) -> T:
    """Segment conservative map face to cell."""
    n_cells = topo.n_cells
    registry = topo.face_registry()
    owner = backend.as_index_tensor(registry.owner)
    neighbor = backend.as_index_tensor(registry.neighbor)
    mask_bnd = neighbor != -1
    deg_fc = topo.degree_fc().reshape((-1, *([1] * (x_face.ndim - 1))))
    w_face = x_face / deg_fc

    out_shape = (n_cells,) + tuple(w_face.shape[1:])
    y_cell = backend.zeros(out_shape, dimension=get_dimension(x_face))
    y_cell.index_add_(0, owner, w_face)
    y_cell.index_add_(0, neighbor[mask_bnd], w_face[mask_bnd])
    return y_cell


def _segment_div_map_face_to_cell[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_face: T,
) -> T:
    """Segment div map face to cell."""
    n_cells = topo.n_cells
    registry = topo.face_registry()
    owner = backend.as_index_tensor(registry.owner)
    neighbor = backend.as_index_tensor(registry.neighbor)
    mask_bnd = neighbor != -1

    out_shape = (n_cells,) + tuple(x_face.shape[1:])
    y_cell = backend.zeros(out_shape, dimension=get_dimension(x_face))
    y_cell.index_add_(0, owner, x_face)
    y_cell.index_add_(0, neighbor[mask_bnd], -x_face[mask_bnd])
    return y_cell


# --- sparse ---
def _sparse_sum_map_face_to_cell[T: TensorLike](
    topo: MeshTopology[T],
    x_face: T,
) -> T:
    """abs(CF) @ x_face"""
    cf_matrix = topo.cell_face_incidence()
    abs_cf_matrix = torch.abs(cf_matrix)
    return abs_cf_matrix @ x_face


def _sparse_mean_map_face_to_cell[T: TensorLike](
    topo: MeshTopology[T],
    x_face: T,
) -> T:
    """D_cell^-1 * CF @ x_face"""
    cell_sums = _sparse_sum_map_face_to_cell(topo, x_face)
    deg_cf = topo.degree_cf().reshape((-1, *([1] * (cell_sums.ndim - 1))))
    return cell_sums / deg_cf


def _sparse_conservative_map_face_to_cell[T: TensorLike](
    topo: MeshTopology[T],
    x_face: T,
) -> T:
    """CF @ (D_face^-1 * x_face)"""
    cf_matrix = topo.cell_face_incidence()
    abs_cf_matrix = torch.abs(cf_matrix)
    deg_fc = topo.degree_fc().reshape((-1, *([1] * (x_face.ndim - 1))))
    w_face = x_face / deg_fc
    return abs_cf_matrix @ w_face


def _sparse_div_map_face_to_cell[T: TensorLike](
    topo: MeshTopology[T],
    x_face: T,
) -> T:
    """CF @ x_face"""
    cf_matrix = topo.cell_face_incidence()
    return cf_matrix @ x_face


# =============================================================================
# map face to point
# =============================================================================
def map_face_to_point[T: TensorLike](
    mesh: TensorMesh[T],
    x_face: T,
    mode: Literal["sum", "mean", "conservative"],
    method: Literal["segment", "sparse"] = "segment",
) -> T:
    r"""
    Map face field to point.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Input mesh.
    x_face : T
        Face field of shape ``(n_faces, ...)``.
    mode : "sum" | "mean" | "conservative"
        Mapping mode.

        ``"sum"``
            Sum of face values that are connected to the point.

        ``"mean"``
            Row-normalized PF operator.

            The point value is computed as the arithmetic average of
            the values of the faces connected to the point:

            .. math::

               x_{\text{point}} =
               \frac{1}{\operatorname{deg}(p)}
               \sum_{f \ni p} x_{\text{face}}

            where :math:`\operatorname{deg}(p)` is the number of faces
            connected to point :math:`p`.

            This corresponds to a local averaging over the face stencil
            around each point. Each neighboring face contributes equally,
            regardless of the number of points in that face.
            In general, this method does not guarantee global conservation
            of the total quantity.

        ``"conservative"``
            Column-normalized PF operator.

            The point value is obtained by distributing each face value
            equally among the points connected to that face:

            .. math::

               x_{\text{point}} =
               \sum_{f \ni p}
               \left(\frac{1}{\operatorname{deg}(f)}\right)
               x_{\text{face}}

            where :math:`\operatorname{deg}(f)` is the number of points
            in face :math:`f`.

            Each face contributes its value conservatively to its
            vertices, and the total sum over points equals
            the total sum over faces,
            making this formulation suitable for conservative transfers.
    method : "segment" | "sparse"
        Mapping method. Default is "segment".

        ``"segment"``
            Computes the reduction
            without explicitly constructing the PF matrix.
            It is typically more memory efficient, especially on GPU.
            In most practical settings, this method is recommended.

        ``"sparse"``
            Computes the reduction by multiplying the PF matrix
            with the face values.
            It depends strongly on backend implementation quality but
            it can be beneficial when the PF matrix is used repeatedly.


    Returns
    -------
    T
        Tensor of shape ``(n_points, ...)``.
    """
    backend = mesh.backend
    topo = mesh.topology
    segment_fns = {
        "sum": _segment_sum_map_face_to_point,
        "mean": _segment_mean_map_face_to_point,
        "conservative": _segment_conservative_map_face_to_point,
    }
    sparse_fns = {
        "sum": _sparse_sum_map_face_to_point,
        "mean": _sparse_mean_map_face_to_point,
        "conservative": _sparse_conservative_map_face_to_point,
    }
    return _dispatch_map(
        backend, topo, x_face, mode, method, segment_fns, sparse_fns
    )


# --- segment ---
def _segment_sum_map_face_to_point[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_face: T,
) -> T:
    """Segment sum map face to point."""
    n_points = topo.n_points
    registry = topo.face_registry()

    values: list[T] = []
    segment_ids: list[torch.Tensor] = []

    # --- triangle faces ---
    if len(registry.tri_gids) > 0:
        gids = backend.as_index_tensor(
            np.repeat(registry.tri_gids, repeats=3, axis=0)
        )  # (Ntri*3,)
        conn = backend.as_index_tensor(
            registry.tri_conn.reshape(-1)
        )  # (Ntri*3,)

        values.append(x_face[gids])
        segment_ids.append(conn)

    # --- quad faces ---
    if len(registry.quad_gids) > 0:
        gids = backend.as_index_tensor(
            np.repeat(registry.quad_gids, repeats=4, axis=0)
        )  # (Nquad*4,)
        conn = backend.as_index_tensor(
            registry.quad_conn.reshape(-1)
        )  # (Nquad*4,)

        values.append(x_face[gids])
        segment_ids.append(conn)

    # --- pixel faces ---
    if len(registry.pixel_gids) > 0:
        gids = backend.as_index_tensor(
            np.repeat(registry.pixel_gids, repeats=4, axis=0)
        )  # (Npixel*4,)
        conn = backend.as_index_tensor(
            registry.pixel_conn.reshape(-1)
        )  # (Npixel*4,)

        values.append(x_face[gids])
        segment_ids.append(conn)

    # --- polygon faces ---
    if len(registry.poly_gids) > 0:
        offsets = registry.poly_offsets  # (Npoly+1,)
        face_sizes = offsets[1:] - offsets[:-1]  # (Npoly,)

        gids = backend.as_index_tensor(
            np.repeat(registry.poly_gids, repeats=face_sizes, axis=0)
        )  # (nnz,)
        conn = backend.as_index_tensor(registry.poly_conn.reshape(-1))  # (nnz,)

        values.append(x_face[gids])
        segment_ids.append(conn)

    values = torch.cat(values, dim=0)
    segment_ids = torch.cat(segment_ids, dim=0)
    out_shape = (n_points,) + tuple(values.shape[1:])
    y_point = backend.zeros(out_shape, dimension=get_dimension(x_face))
    return y_point.index_add_(0, segment_ids, values)


def _segment_mean_map_face_to_point[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_face: T,
) -> T:
    """Segment mean map face to point."""
    point_sums = _segment_sum_map_face_to_point(backend, topo, x_face)
    deg_pf = topo.degree_pf().reshape((-1, *([1] * (point_sums.ndim - 1))))
    return point_sums / deg_pf


def _segment_conservative_map_face_to_point[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_face: T,
) -> T:
    """Segment conservative map face to point."""

    n_points = topo.n_points
    registry = topo.face_registry()

    values: list[T] = []
    segment_ids: list[torch.Tensor] = []

    # --- triangle faces ---
    if len(registry.tri_gids) > 0:
        gids = backend.as_index_tensor(
            np.repeat(registry.tri_gids, repeats=3, axis=0)
        )  # (Ntri*3,)
        conn = backend.as_index_tensor(
            registry.tri_conn.reshape(-1)
        )  # (Ntri*3,)

        values.append(x_face[gids] / 3.0)
        segment_ids.append(conn)

    # --- quad faces ---
    if len(registry.quad_gids) > 0:
        gids = backend.as_index_tensor(
            np.repeat(registry.quad_gids, repeats=4, axis=0)
        )  # (Nquad*4,)
        conn = backend.as_index_tensor(
            registry.quad_conn.reshape(-1)
        )  # (Nquad*4,)

        values.append(x_face[gids] / 4.0)
        segment_ids.append(conn)

    # --- pixel faces ---
    if len(registry.pixel_gids) > 0:
        gids = backend.as_index_tensor(
            np.repeat(registry.pixel_gids, repeats=4, axis=0)
        )  # (Npixel*4,)
        conn = backend.as_index_tensor(
            registry.pixel_conn.reshape(-1)
        )  # (Npixel*4,)

        values.append(x_face[gids] / 4.0)
        segment_ids.append(conn)

    # --- polygon faces ---
    if len(registry.poly_gids) > 0:
        offsets = registry.poly_offsets  # (Npoly + 1,)
        face_sizes = offsets[1:] - offsets[:-1]  # (Npoly,)

        gids = backend.as_index_tensor(
            np.repeat(registry.poly_gids, repeats=face_sizes, axis=0)
        )  # (nnz,)
        conn = backend.as_index_tensor(registry.poly_conn.reshape(-1))  # (nnz,)
        divisor = backend.as_tensor(
            np.repeat(face_sizes, repeats=face_sizes, axis=0),
            dimension={},
        ).reshape((-1, *([1] * (x_face.ndim - 1))))

        values.append(x_face[gids] / divisor)
        segment_ids.append(conn)

    values = torch.cat(values, dim=0)
    segment_ids = torch.cat(segment_ids, dim=0)
    out_shape = (n_points,) + tuple(values.shape[1:])
    y_point = backend.zeros(out_shape, dimension=get_dimension(x_face))
    return y_point.index_add_(0, segment_ids, values)


# --- sparse ---
def _sparse_sum_map_face_to_point[T: TensorLike](
    topo: MeshTopology[T],
    x_face: T,
) -> T:
    """PF @ x_face"""
    pf_matrix = topo.point_face_incidence()
    return pf_matrix @ x_face


def _sparse_mean_map_face_to_point[T: TensorLike](
    topo: MeshTopology[T],
    x_face: T,
) -> T:
    """D_point^-1 * PF @ x_face"""
    point_sums = _sparse_sum_map_face_to_point(topo, x_face)
    deg_pf = topo.degree_pf().reshape((-1, *([1] * (point_sums.ndim - 1))))
    return point_sums / deg_pf


def _sparse_conservative_map_face_to_point[T: TensorLike](
    topo: MeshTopology[T],
    x_face: T,
) -> T:
    """PF @ (D_face^-1 * x_face)"""
    pf_matrix = topo.point_face_incidence()
    deg_fp = topo.degree_fp().reshape((-1, *([1] * (x_face.ndim - 1))))
    w_face = x_face / deg_fp
    return pf_matrix @ w_face


# =============================================================================
# map point to face
# =============================================================================
def map_point_to_face[T: TensorLike](
    mesh: TensorMesh[T],
    x_point: T,
    mode: Literal["sum", "mean", "conservative"],
    method: Literal["segment", "sparse"] = "segment",
) -> T:
    r"""
    Map point field to face.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Input mesh.
    x_point : T
        Point field of shape ``(n_points, ...)``.
    mode : "sum" | "mean" | "conservative"
        Mapping mode.

        ``"sum"``
            Sum of point values in each face.

        ``"mean"``
            Row-normalized FP operator.

            The face value is computed as the simple average of the
            values at the points connected to the face:

            .. math::

               x_{\text{face}} =
               \frac{1}{\operatorname{deg}(f)}
               \sum_{p \in f} x_{\text{point}}

            where :math:`\operatorname{deg}(f)` is the number of points
            connected to face :math:`f`.

            This corresponds to a local arithmetic averaging over the
            face stencil. It treats each point in the face equally,
            regardless of how many faces that point belongs to.
            In general, this method does not guarantee global conservation
            of the total quantity.

        ``"conservative"``
            Column-normalized FP operator.

            The face value is obtained by distributing each point value
            equally among the faces connected to that point:

            .. math::

               x_{\text{face}} =
               \sum_{p \in f}
               \left(\frac{1}{\operatorname{deg}(p)}\right)
               x_{\text{point}}

            where :math:`\operatorname{deg}(p)` is the number of faces
            connected to point :math:`p`.

            Each point contributes its value conservatively to its
            neighboring faces, and the total sum over faces equals
            the total sum over points,
            making this formulation suitable for conservative transfers.
    method : "segment" | "sparse"
        Mapping method. Default is "segment".

        ``"segment"``
            Computes the reduction
            without explicitly constructing the FP matrix.
            It is typically more memory efficient, especially on GPU.
            In most practical settings, this method is recommended.

        ``"sparse"``
            Computes the reduction by multiplying the FP matrix
            with the point values.
            It depends strongly on backend implementation quality but
            it can be beneficial when the FP matrix is used repeatedly.


    Returns
    -------
    T
        Tensor of shape ``(n_faces, ...)``.
    """
    backend = mesh.backend
    topo = mesh.topology
    segment_fns = {
        "sum": _segment_sum_map_point_to_face,
        "mean": _segment_mean_map_point_to_face,
        "conservative": _segment_conservative_map_point_to_face,
    }
    sparse_fns = {
        "sum": _sparse_sum_map_point_to_face,
        "mean": _sparse_mean_map_point_to_face,
        "conservative": _sparse_conservative_map_point_to_face,
    }
    return _dispatch_map(
        backend, topo, x_point, mode, method, segment_fns, sparse_fns
    )


# --- segment ---
def _segment_sum_map_point_to_face[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_point: T,
) -> T:
    """Segment sum map point to face."""
    registry = topo.face_registry()
    n_faces = registry.n_faces()

    values: list[T] = []
    segment_ids: list[torch.Tensor] = []

    # --- triangle faces ---
    if len(registry.tri_gids) > 0:
        gids = backend.as_index_tensor(
            np.repeat(registry.tri_gids, repeats=3, axis=0)
        )  # (Ntri*3,)
        conn = backend.as_index_tensor(
            registry.tri_conn.reshape(-1)
        )  # (Ntri*3,)

        values.append(x_point[conn])
        segment_ids.append(gids)

    # --- quad faces ---
    if len(registry.quad_gids) > 0:
        gids = backend.as_index_tensor(
            np.repeat(registry.quad_gids, repeats=4, axis=0)
        )  # (Nquad*4,)
        conn = backend.as_index_tensor(
            registry.quad_conn.reshape(-1)
        )  # (Nquad*4,)

        values.append(x_point[conn])
        segment_ids.append(gids)

    # --- pixel faces ---
    if len(registry.pixel_gids) > 0:
        gids = backend.as_index_tensor(
            np.repeat(registry.pixel_gids, repeats=4, axis=0)
        )  # (Npixel*4,)
        conn = backend.as_index_tensor(
            registry.pixel_conn.reshape(-1)
        )  # (Npixel*4,)

        values.append(x_point[conn])
        segment_ids.append(gids)

    # --- polygon faces ---
    if len(registry.poly_gids) > 0:
        offsets = registry.poly_offsets  # (Npoly+1,)
        face_sizes = offsets[1:] - offsets[:-1]  # (Npoly,)

        gids = backend.as_index_tensor(
            np.repeat(registry.poly_gids, repeats=face_sizes, axis=0)
        )  # (nnz,)
        conn = backend.as_index_tensor(registry.poly_conn.reshape(-1))  # (nnz,)

        values.append(x_point[conn])
        segment_ids.append(gids)

    values = torch.cat(values, dim=0)
    segment_ids = torch.cat(segment_ids, dim=0)

    out_shape = (n_faces,) + tuple(values.shape[1:])
    y_face = backend.zeros(out_shape, dimension=get_dimension(x_point))
    return y_face.index_add_(0, segment_ids, values)


def _segment_mean_map_point_to_face[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_point: T,
) -> T:
    """Segment mean map point to face."""
    face_sums = _segment_sum_map_point_to_face(backend, topo, x_point)
    deg_fp = topo.degree_fp().reshape((-1, *([1] * (face_sums.ndim - 1))))
    return face_sums / deg_fp


def _segment_conservative_map_point_to_face[T: TensorLike](
    backend: Backend[T],
    topo: MeshTopology[T],
    x_point: T,
) -> T:
    """Segment conservative map point to face."""
    registry = topo.face_registry()
    n_faces = registry.n_faces()

    values: list[T] = []
    segment_ids: list[torch.Tensor] = []

    deg_pf = topo.degree_pf().reshape((-1, *([1] * (x_point.ndim - 1))))

    # --- triangle faces ---
    if len(registry.tri_gids) > 0:
        gids = backend.as_index_tensor(
            np.repeat(registry.tri_gids, repeats=3, axis=0)
        )  # (Ntri*3,)
        conn = backend.as_index_tensor(
            registry.tri_conn.reshape(-1)
        )  # (Ntri*3,)

        values.append(x_point[conn] / deg_pf[conn])
        segment_ids.append(gids)

    # --- quad faces ---
    if len(registry.quad_gids) > 0:
        gids = backend.as_index_tensor(
            np.repeat(registry.quad_gids, repeats=4, axis=0)
        )  # (Nquad*4,)
        conn = backend.as_index_tensor(
            registry.quad_conn.reshape(-1)
        )  # (Nquad*4,)

        values.append(x_point[conn] / deg_pf[conn])
        segment_ids.append(gids)

    # --- pixel faces ---
    if len(registry.pixel_gids) > 0:
        gids = backend.as_index_tensor(
            np.repeat(registry.pixel_gids, repeats=4, axis=0)
        )  # (Npixel*4,)
        conn = backend.as_index_tensor(
            registry.pixel_conn.reshape(-1)
        )  # (Npixel*4,)

        values.append(x_point[conn] / deg_pf[conn])
        segment_ids.append(conn)

    # --- polygon faces ---
    if len(registry.poly_gids) > 0:
        offsets = registry.poly_offsets  # (Npoly+1,)
        face_sizes = offsets[1:] - offsets[:-1]  # (Npoly,)

        gids = backend.as_index_tensor(
            np.repeat(registry.poly_gids, repeats=face_sizes, axis=0)
        )  # (nnz,)
        conn = backend.as_index_tensor(registry.poly_conn.reshape(-1))  # (nnz,)

        values.append(x_point[conn] / deg_pf[conn])
        segment_ids.append(gids)

    values = torch.cat(values, dim=0)
    segment_ids = torch.cat(segment_ids, dim=0)

    out_shape = (n_faces,) + tuple(values.shape[1:])
    y_face = backend.zeros(out_shape, dimension=get_dimension(x_point))
    return y_face.index_add_(0, segment_ids, values)


# --- sparse ---
def _sparse_sum_map_point_to_face[T: TensorLike](
    topo: MeshTopology[T],
    x_point: T,
) -> T:
    """FP @ x_point"""
    fp_matrix = topo.face_point_incidence()
    return fp_matrix @ x_point


def _sparse_mean_map_point_to_face[T: TensorLike](
    topo: MeshTopology[T],
    x_point: T,
) -> T:
    """D_face^-1 * FP @ x_point"""
    face_sums = _sparse_sum_map_point_to_face(topo, x_point)
    deg_fp = topo.degree_fp().reshape((-1, *([1] * (face_sums.ndim - 1))))
    return face_sums / deg_fp


def _sparse_conservative_map_point_to_face[T: TensorLike](
    topo: MeshTopology[T],
    x_point: T,
) -> T:
    """FP @ (D_point^-1 * x_point)"""
    fp_matrix = topo.face_point_incidence()
    deg_pf = topo.degree_pf().reshape((-1, *([1] * (x_point.ndim - 1))))
    w_point = x_point / deg_pf
    return fp_matrix @ w_point


# =============================================================================
# median
# =============================================================================
def median_points[T: TensorLike](
    mesh: TensorMesh[T],
    x_point: T,
    n_hop: int = 1,
) -> T:
    """
    Compute median among n-hop point neighbors.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Input mesh.
    x_point : T
        Point field of shape ``(n_points, ...)``.
    n_hop : int
        The number of adjacent hops to consider.

    Returns
    -------
    T
        Tensor of shape ``(n_points, ...)``.
    """
    adj = mesh.topology.get_skeleton(AdjacencyName.PP)
    return _median(mesh.backend, adj, x_point, n_hop)


def median_cells[T: TensorLike](
    mesh: TensorMesh[T],
    x_cell: T,
    n_hop: int = 1,
) -> T:
    """
    Compute median among n-hop cell neighbors.

    Parameters
    ----------
    mesh : TensorMesh[T]
        Input mesh.
    x_cell : T
        Cell field of shape ``(n_cells, ...)``.
    n_hop : int
        The number of adjacent hops to consider.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, ...)``.
    """
    adj = mesh.topology.get_skeleton(AdjacencyName.CC)
    return _median(mesh.backend, adj, x_cell, n_hop)


def _median[T: TensorLike](
    backend: Backend[T],
    adj: sps.csr_array,
    x: T,
    n_hop: int = 1,
) -> T:
    """
    Compute median among n-hop neighbors from adjacency and x.

    Parameters
    ----------
    backend : Backend[T]
        Backend used to allocate the result tensor.
    adj : scipy.sparse.csr_array
        Adjacency matrix of shape ``(n_elements, n_elements)``.
    x : T
        Tensor of shape ``(n_elements, ...)``.
    n_hop : int
        The number of adjacent hops to consider.

    Returns
    -------
    T
        Tensor of shape ``(n_elements, ...)``.

    Raises
    ------
    TypeError
        If adj and x have different number of rows.
    """
    if not adj.shape[0] == x.shape[0]:
        raise TypeError("adj and x must have the same number of rows")

    n = adj.shape[0]

    crow = adj.indptr
    col = adj.indices

    out_shape = (n,) + tuple(x.shape[1:])
    y = backend.zeros(out_shape, dimension=get_dimension(x))

    for src in range(n):
        if n_hop == 0:
            y[src] = x[src]
            continue

        visited = [False] * n
        q = deque()

        visited[src] = True
        q.append((src, 0))

        neighborhood = [src]

        while q:
            u, dist = q.popleft()
            if dist == n_hop:
                continue

            start = int(crow[u].item())
            end = int(crow[u + 1].item())
            nbrs = col[start:end]

            for v in nbrs:
                if not visited[v]:
                    visited[v] = True
                    neighborhood.append(v)
                    q.append((v, dist + 1))

        vals = x[backend.as_index_tensor(neighborhood)]
        y[src] = torch.median(vals)

    return y

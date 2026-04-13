"""Analytic geometry methods for linear elements (batched tensor operations)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from graphlow.utils.dimension import get_dimension

if TYPE_CHECKING:
    from graphlow.core.backend.base import Backend, TensorLike


# =============================================================================
# Area Vector Methods
# =============================================================================
def triangle_area_vectors[T: TensorLike](
    points: T, conn: torch.Tensor, backend: Backend[T]
) -> T:
    """
    Compute area vectors for triangles.

    Parameters
    ----------
    points : TensorLike
        Point coordinates of shape ``(n_points, 3)``.
    conn : torch.Tensor
        Triangle vertex indices of shape ``(n_cells, 3)``.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, 3)``.
    """
    p = points[conn]

    v0 = p[:, 0, :]
    v1 = p[:, 1, :]
    v2 = p[:, 2, :]

    # area vectors: 0.5* (e1 x e2)
    return 0.5 * torch.linalg.cross(v1 - v0, v2 - v0)


def quad_area_vectors[T: TensorLike](
    points: T, conn: torch.Tensor, backend: Backend[T]
) -> T:
    """
    Compute area vectors for quadrilaterals.
    Accurately computes the projected normal vector for non-planar quads,
    using the diagonal cross product method to obtain the surface area vector.

    Parameters
    ----------
    points : TensorLike
        Quadrilateral vertex coordinates of shape ``(n_points, 3)``.
    conn : torch.Tensor
        Quadrilateral vertex indices of shape ``(n_cells, 4)``.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, 3)``.
    """
    p = points[conn]

    v0 = p[:, 0, :]
    v1 = p[:, 1, :]
    v2 = p[:, 2, :]
    v3 = p[:, 3, :]

    # diagonal vectors: d1 = v2 - v0, d2 = v3 - v1
    d1 = v2 - v0
    d2 = v3 - v1

    # area vectors: 0.5 * (d1 x d2)
    return 0.5 * torch.linalg.cross(d1, d2)


def pixel_area_vectors[T: TensorLike](
    points: T, conn: torch.Tensor, backend: Backend[T]
) -> T:
    """
    Compute area vectors for pixels.
    Accurately computes the projected normal vector for non-planar pixels,
    using the diagonal cross product method to obtain the surface area vector.

    Parameters
    ----------
    points : TensorLike
        Pixel vertex coordinates of shape ``(n_points, 3)``.
    conn : torch.Tensor
        Pixel vertex indices of shape ``(n_cells, 4)``.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, 3)``.
    """
    p = points[conn]

    v0 = p[:, 0, :]
    v1 = p[:, 1, :]
    v2 = p[:, 2, :]
    v3 = p[:, 3, :]

    # diagonal vectors: d1 = v3 - v0, d2 = v2 - v1
    d1 = v3 - v0
    d2 = v2 - v1

    # area vectors: 0.5 * (d1 x d2)
    return 0.5 * torch.linalg.cross(d1, d2)


def polygon_area_vectors[T: TensorLike](
    points: T, conn: torch.Tensor, offsets: torch.Tensor, backend: Backend[T]
) -> T:
    """
    Compute area vectors for polygons using signed sum.

    Parameters
    ----------
    points : TensorLike
        Polygon vertex coordinates of shape ``(n_points, 3)``.
    conn : torch.Tensor
        Flattened polygon vertex indices of shape ``(n_total_points,)``.
    offsets : torch.Tensor
        Polygon offsets of shape ``(n_cells + 1,)``.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, 3)``.
    """
    n_total = len(conn)
    n_faces = len(offsets) - 1

    # Create the index array for the next vertices (wrap around at the end).
    # For example, suppose there are two faces:
    # a triangle [A, B, C] and a quad [D, E, F, G],
    # and a combined connectivity array is [A, B, C, D, E, F, G].
    # Then, the next_conn array should be [B, C, A, E, F, G, D].
    # so shifting the index by 1: np.arange(n_total) + 1
    # then setting the start index to the previous start index: offsets[1:] - 1
    next_idx = backend.as_index_tensor(np.arange(n_total) + 1)
    next_idx[offsets[1:] - 1] = offsets[:-1]
    next_conn = conn[next_idx]

    # Create the segment IDs for each vertex.
    segment_ids = torch.zeros(n_total, dtype=torch.int64, device=backend.device)
    if n_faces > 1:
        # flag the start of each face
        segment_ids[offsets[1:-1]] = 1
    segment_ids = torch.cumsum(segment_ids, dim=0)

    p_curr = points[conn]
    p_next = points[next_conn]

    # cross product of each edge: p_curr x p_next
    tri_pieces = 0.5 * torch.linalg.cross(p_curr, p_next)

    # sum the cross products of each face
    face_areas = backend.zeros(
        (n_faces, 3), dimension=get_dimension(tri_pieces)
    )
    return face_areas.index_add_(0, segment_ids, tri_pieces)


# =============================================================================
# Face Centroid Methods
# =============================================================================
def triangle_centroids[T: TensorLike](
    points: T, conn: torch.Tensor, backend: Backend[T]
) -> T:
    """
    Compute face centroids for triangles.

    Parameters
    ----------
    points : TensorLike
        Point coordinates of shape ``(n_points, 3)``.
    conn : torch.Tensor
        Triangle vertex indices of shape ``(n_cells, 3)``.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, 3)``.
    """
    p = points[conn]
    v0 = p[:, 0, :]
    v1 = p[:, 1, :]
    v2 = p[:, 2, :]
    return (v0 + v1 + v2) / 3


def quad_centroids[T: TensorLike](
    points: T, conn: torch.Tensor, backend: Backend[T]
) -> T:
    """
    Compute face centroids for quadrilaterals.

    Parameters
    ----------
    points : TensorLike
        Quadrilateral vertex coordinates of shape ``(n_points, 3)``.
    conn : torch.Tensor
        Quadrilateral vertex indices of shape ``(n_cells, 4)``.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, 3)``.
    """
    p = points[conn]
    v0 = p[:, 0, :]
    v1 = p[:, 1, :]
    v2 = p[:, 2, :]
    v3 = p[:, 3, :]
    return (v0 + v1 + v2 + v3) / 4


def pixel_centroids[T: TensorLike](
    points: T, conn: torch.Tensor, backend: Backend[T]
) -> T:
    """
    Compute face centroids for pixels.

    Parameters
    ----------
    points : TensorLike
        Pixel vertex coordinates of shape ``(n_points, 3)``.
    conn : torch.Tensor
        Pixel vertex indices of shape ``(n_cells, 4)``.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, 3)``.
    """
    p = points[conn]
    v0 = p[:, 0, :]
    v1 = p[:, 1, :]
    v2 = p[:, 2, :]
    v3 = p[:, 3, :]
    return (v0 + v1 + v2 + v3) / 4


def polygon_centroids[T: TensorLike](
    points: T, conn: torch.Tensor, offsets: torch.Tensor, backend: Backend[T]
) -> T:
    """
    Compute face centroids for polygons using signed sum.

    Parameters
    ----------
    points : TensorLike
        Polygon vertex coordinates of shape ``(n_points, 3)``.
    conn : torch.Tensor
        Flattened polygon vertex indices of shape ``(n_total_points,)``.
    offsets : torch.Tensor
        Polygon offsets of shape ``(n_cells + 1,)``.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, 3)``.
    """
    n_total = len(conn)
    n_faces = len(offsets) - 1

    # Create the index array for the next vertices (wrap around at the end).
    # For example, suppose there are two faces:
    # a triangle [A, B, C] and a quad [D, E, F, G],
    # and a combined connectivity array is [A, B, C, D, E, F, G].
    # Then, the next_conn array should be [B, C, A, E, F, G, D].
    # so shifting the index by 1: np.arange(n_total) + 1
    # then setting the start index to the previous start index: offsets[1:] - 1
    next_idx = backend.as_index_tensor(np.arange(n_total) + 1)
    next_idx[offsets[1:] - 1] = offsets[:-1]
    next_conn = conn[next_idx]

    lengths = offsets[1:] - offsets[:-1]
    p0_idx = torch.repeat_interleave(offsets[:-1], lengths, dim=0)
    p0 = points[conn[p0_idx]]

    # Create the segment IDs for each vertex.
    segment_ids = torch.zeros(n_total, dtype=torch.int64, device=backend.device)
    if n_faces > 1:
        # flag the start of each face
        segment_ids[offsets[1:-1]] = 1
    segment_ids = torch.cumsum(segment_ids, dim=0)

    p1 = points[conn]
    p2 = points[next_conn]

    # cross product of each edge: p_curr x p_next
    area_vec_c_Ti = 0.5 * torch.linalg.cross(p1 - p0, p2 - p0)
    area_c_Ti = torch.linalg.vector_norm(area_vec_c_Ti, dim=-1, keepdim=True)
    moment_c_Ti = area_c_Ti * (p1 + p2 + p0) / 3

    area_c = backend.zeros((n_faces, 1), dimension=get_dimension(area_vec_c_Ti))
    area_c.index_add_(0, segment_ids, area_c_Ti)

    moment_c = backend.zeros((n_faces, 3), dimension=get_dimension(moment_c_Ti))
    moment_c.index_add_(0, segment_ids, moment_c_Ti)
    centroid_c = moment_c / area_c

    return centroid_c


# =============================================================================
# Volume methods
# =============================================================================
def tetra_volume[T: TensorLike](
    points: T, conn: torch.Tensor, backend: Backend[T]
) -> T:
    """
    Compute volumes of tetrahedrons in a fully batched manner.

    Parameters
    ----------
    points : TensorLike
        Tetrahedron vertex coordinates of shape ``(n_points, 3)``.
    conn : torch.Tensor
        Tetrahedron vertex indices of shape ``(n_cells, 4)``.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, 1)``.
    """
    p = points[conn]

    v0 = p[:, 0, :]
    v1 = p[:, 1, :]
    v2 = p[:, 2, :]
    v3 = p[:, 3, :]

    # Calculate edge vectors
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0

    # Scalar triple product: e1 . (e2 x e3)
    cross_e2_e3 = torch.linalg.cross(e2, e3)
    volume = torch.sum(e1 * cross_e2_e3, dim=-1) / 6.0
    return volume[:, None]


def pyramid_volume[T: TensorLike](
    points: T, conn: torch.Tensor, backend: Backend[T]
) -> T:
    """
    Compute volumes of pyramids using symmetric tetrahedral decomposition.
    Ensures consistency even for pyramids with non-planar faces.

    Parameters
    ----------
    points : TensorLike
        Pyramid vertex coordinates of shape ``(n_points, 3)``.
    conn : torch.Tensor
        Pyramid vertex indices of shape ``(n_cells, 5)``.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, 1)``.
    """
    top = points[conn[:, 4]]  # (Nc, 3)
    pyramid_bottoms_idx = backend.as_index_tensor([0, 3, 2, 1])
    pyramid_bottoms = points[conn[:, pyramid_bottoms_idx]]  # (Nc, 4, 3)
    pyramid_bottom_centroids = torch.mean(pyramid_bottoms, dim=1)  # (Nc, 3)
    top2bottom = pyramid_bottom_centroids - top  # (Nc, 3)
    e1 = pyramid_bottoms - top[:, None, :]  # (Nc, 4, 3)
    e2 = torch.roll(e1, shifts=-1, dims=1)
    cross_e1_e2 = torch.linalg.cross(e1, e2)
    volume = torch.sum(top2bottom[:, None, :] * cross_e1_e2, dim=(1, 2)) / 6.0
    return volume[:, None]


def wedge_volume[T: TensorLike](
    points: T, conn: torch.Tensor, backend: Backend[T]
) -> T:
    """
    Compute volumes of wedges using symmetric tetrahedral decomposition.
    Ensures consistency even for wedges with non-planar faces.

    Parameters
    ----------
    points : TensorLike
        Wedge vertex coordinates of shape ``(n_points, 3)``.
    conn : torch.Tensor
        Wedge vertex indices of shape ``(n_cells, 6)``.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, 1)``.
    """
    pts = points[conn]
    # divide the wedge into 2 tets + 3 pyramids
    # This is a better solution than 3 tets because
    # if the wedge is twisted then the 3 quads will be twisted.
    cell_centroid = torch.mean(pts, dim=1, keepdim=True)  # (Nc, 1, 3)

    # pyramids
    pyramid_bottoms_idx = backend.as_index_tensor(
        [[0, 3, 4, 1], [2, 5, 3, 0], [1, 4, 5, 2]]
    )
    pyramid_bottoms = pts[:, pyramid_bottoms_idx]  # (Nc, 3, 4, 3)
    pyramid_bottom_centroids = torch.mean(pyramid_bottoms, dim=2)  # (Nc, 3, 3)
    top2bottom = pyramid_bottom_centroids - cell_centroid  # (Nc, 3, 3)
    e1 = pyramid_bottoms - cell_centroid[:, :, None, :]  # (Nc, 3, 4, 3)
    e2 = torch.roll(e1, shifts=-1, dims=2)
    cross_e1_e2 = torch.linalg.cross(e1, e2)  # (Nc, 3, 4, 3)
    pyramid_volumes = (
        torch.sum(top2bottom[:, :, None, :] * cross_e1_e2, dim=(1, 2, 3)) / 6.0
    )

    # tets
    tet_bottoms_idx = backend.as_index_tensor([[0, 1, 2], [3, 5, 4]])
    tet_bottoms = pts[:, tet_bottoms_idx]  # (Nc, 2, 3, 3)
    tet_bottom_centroids = torch.mean(tet_bottoms, dim=2)  # (Nc, 2, 3)
    top2bottom = tet_bottom_centroids - cell_centroid  # (Nc, 2, 3)
    e1 = tet_bottoms - cell_centroid[:, :, None, :]  # (Nc, 2, 3, 3)
    e2 = torch.roll(e1, shifts=-1, dims=2)
    cross_e1_e2 = torch.linalg.cross(e1, e2)  # (Nc, 2, 3, 3)
    tet_volumes = (
        torch.sum(top2bottom[:, :, None, :] * cross_e1_e2, dim=(1, 2, 3)) / 6.0
    )

    total_volume = pyramid_volumes + tet_volumes
    return total_volume[:, None]


def voxel_volume[T: TensorLike](
    points: T, conn: torch.Tensor, backend: Backend[T]
) -> T:
    """
    Compute volumes of voxels using symmetric tetrahedral decomposition.
    Ensures consistency even for voxels with non-planar faces.

    Parameters
    ----------
    points : TensorLike
        Voxel vertex coordinates of shape ``(n_points, 3)``.
    conn : torch.Tensor
        Voxel vertex indices of shape ``(n_cells, 8)``.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, 1)``.
    """
    pts = points[conn]
    # divide the hex into 6 pyramids
    cell_centroid = torch.mean(pts, dim=1, keepdim=True)  # (Nc, 1, 3)

    # Define vertex indices for 6 faces of the hexahedron (VTK order)
    # (6, 4) array
    pyramid_bottoms_idx = backend.as_index_tensor(
        [
            [0, 2, 3, 1],
            [4, 5, 7, 6],
            [0, 1, 5, 4],
            [1, 3, 7, 5],
            [3, 2, 6, 7],
            [2, 0, 4, 6],
        ]
    )

    pyramid_bottoms = pts[:, pyramid_bottoms_idx]  # (Nc, 6, 4, 3)
    pyramid_bottom_centroids = torch.mean(pyramid_bottoms, dim=2)  # (Nc, 6, 3)

    # Vectors from cell centroid to the bottom of the pyramids
    top2bottom = pyramid_bottom_centroids - cell_centroid  # (Nc, 6, 3)
    e1 = pyramid_bottoms - cell_centroid[:, :, None, :]  # (Nc, 6, 4, 3)
    e2 = torch.roll(e1, shifts=-1, dims=2)
    cross_e1_e2 = torch.linalg.cross(e1, e2)  # (Nc, 6, 4, 3)
    volume = (
        torch.sum(top2bottom[:, :, None, :] * cross_e1_e2, dim=(1, 2, 3)) / 6.0
    )
    return volume[:, None]


def hexahedron_volume[T: TensorLike](
    points: T, conn: torch.Tensor, backend: Backend[T]
) -> T:
    """
    Compute volumes of hexahedrons using symmetric tetrahedral decomposition.
    Ensures consistency even for hexahedrons with non-planar faces.

    Parameters
    ----------
    points : TensorLike
        Hexahedron vertex coordinates of shape ``(n_points, 3)``.
    conn : torch.Tensor
        Hexahedron vertex indices of shape ``(n_cells, 8)``.
    backend : Backend[T]
        Tensor backend.

    Returns
    -------
    T
        Tensor of shape ``(n_cells, 1)``.
    """
    pts = points[conn]
    # divide the hex into 6 pyramids
    cell_centroid = torch.mean(pts, dim=1, keepdim=True)  # (Nc, 1, 3)

    # Define vertex indices for 6 faces of the hexahedron (VTK order)
    # (6, 4) array
    pyramid_bottoms_idx = backend.as_index_tensor(
        [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ]
    )

    pyramid_bottoms = pts[:, pyramid_bottoms_idx]  # (Nc, 6, 4, 3)
    pyramid_bottom_centroids = torch.mean(pyramid_bottoms, dim=2)  # (Nc, 6, 3)

    # Vectors from cell centroid to the bottom of the pyramids
    top2bottom = pyramid_bottom_centroids - cell_centroid  # (Nc, 6, 3)
    e1 = pyramid_bottoms - cell_centroid[:, :, None, :]  # (Nc, 6, 4, 3)
    e2 = torch.roll(e1, shifts=-1, dims=2)
    cross_e1_e2 = torch.linalg.cross(e1, e2)  # (Nc, 6, 4, 3)
    volume = (
        torch.sum(top2bottom[:, :, None, :] * cross_e1_e2, dim=(1, 2, 3)) / 6.0
    )
    return volume[:, None]

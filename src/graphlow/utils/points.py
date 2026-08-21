"""Helpers for mesh point coordinate tensors."""

from __future__ import annotations

from graphlow.core.backend.base import TensorLike


def as_xyz_points[T: TensorLike](points: T) -> T:
    """
    Return XYZ coordinates of shape ``(n_points, 3)``.

    Phlower rank-1 fields use a trailing feature axis, so point coordinates
    may be stored as ``(n_points, 3, 1)``. Geometry operators that treat
    points as 3-vectors need ``(n_points, 3)``.

    Parameters
    ----------
    points : T
        Point coordinates of shape ``(n_points, 3)`` or ``(n_points, 3, 1)``.

    Returns
    -------
    T
        Coordinates of shape ``(n_points, 3)``.

    Raises
    ------
    ValueError
        If ``points`` does not have shape ``(n_points, 3)`` or
        ``(n_points, 3, 1)``.
    """
    shape = tuple(points.shape)
    if len(shape) == 2 and shape[-1] == 3:
        return points
    if len(shape) == 3 and shape[-2] == 3 and shape[-1] == 1:
        return points[:, :, 0]
    raise ValueError(
        f"points must have shape (n_points, 3) or (n_points, 3, 1), got {shape}"
    )

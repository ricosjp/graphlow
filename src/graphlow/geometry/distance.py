from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from graphlow.core.backend.base import TensorLike

if TYPE_CHECKING:
    from graphlow.core.mesh import TensorMesh


def hausdorff_distance[T: TensorLike](
    mesh: TensorMesh[T],
    target_points: T,
    *,
    softmin_temperature: float | None = None,
) -> T:
    """
    Compute symmetric Hausdorff distance between mesh points and target points.

    With ``softmin_temperature is None`` (default), uses hard min/max: gradient
    flows only to the nearest/farthest pair. With ``softmin_temperature > 0``,
    uses soft min for per-point distances and soft max for the outer max so
    gradient flows to all mesh points and targets.

    Hard definition: hd = max( max_i min_j d(p_i,q_j), max_j min_i d(p_i,q_j) ).

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh with differentiable points.
    target_points : T
        Target points of shape ``(n_target, 3)``.
    softmin_temperature : float or None, optional
        If None, use hard min/max. If positive, use soft min/max with this
        temperature so that gradient flows to all vertices.

    Returns
    -------
    T
        Tensor of shape ``(1,)``.
    """
    p = mesh.points  # (N, 3)
    q = target_points  # (M, 3)
    dists = torch.cdist(p, q)  # (N, M) L2 distances
    n_p, n_q = dists.shape[0], dists.shape[1]

    if softmin_temperature is None or softmin_temperature <= 0:
        d_p2Q = torch.min(dists, dim=1).values  # (N,)
        d_q2P = torch.min(dists, dim=0).values  # (M,)
        h1 = torch.max(d_p2Q)
        h2 = torch.max(d_q2P)
        hd = torch.maximum(h1, h2)
    else:
        tau = softmin_temperature
        # Soft min over targets/sources: -tau * log(mean(exp(-d/tau)))
        d_p2Q = tau * (
            math.log(n_q) - torch.logsumexp(-dists / tau, dim=1)
        )  # (N,)
        d_q2P = tau * (
            math.log(n_p) - torch.logsumexp(-dists / tau, dim=0)
        )  # (M,)
        # Soft max over points: tau * log(mean(exp(d/tau)))
        h1 = tau * (torch.logsumexp(d_p2Q / tau, dim=0) - math.log(n_p))
        h2 = tau * (torch.logsumexp(d_q2P / tau, dim=0) - math.log(n_q))
        hd = torch.maximum(h1, h2)

    return hd[None]


def chamfer_distance[T: TensorLike](
    mesh: TensorMesh[T],
    target_points: T,
    *,
    softmin_temperature: float | None = None,
) -> T:
    """
    Compute symmetric Chamfer distance between mesh points and target points.

    With ``softmin_temperature is None`` (default), uses hard min: gradient
    flows only to the nearest pair per point, so some vertices may receive
    little or no gradient. With ``softmin_temperature > 0``, uses a smooth
    soft-min so that gradient flows to all mesh points and targets,
    giving more uniform fitting.

    Hard-min definition:

        d = (1/|P|) sum_{p in P} min_{q in Q} |p - q|^2
            + (1/|Q|) sum_{q in Q} min_{p in P} |p - q|^2

    Soft-min replaces each min by -tau * log(mean(exp(-d^2/tau))).

    Parameters
    ----------
    mesh : TensorMesh[T]
        Mesh with differentiable points.
    target_points : T
        Target points of shape ``(n_target, 3)``.
    softmin_temperature : float or None, optional
        If None, use hard min. If positive, use soft min with this temperature
        (smaller = closer to hard min; larger = more uniform gradients).

    Returns
    -------
    T
        Tensor of shape ``(1,)``.
    """
    p = mesh.points  # (N, 3)
    q = target_points  # (M, 3)
    dists = torch.cdist(p, q) ** 2  # (N, M)
    n_p, n_q = dists.shape[0], dists.shape[1]

    if softmin_temperature is None or softmin_temperature <= 0:
        d_p2Q = torch.min(dists, dim=1).values  # (N,)
        d_q2P = torch.min(dists, dim=0).values  # (M,)
    else:
        tau = softmin_temperature
        d_p2Q = tau * (
            math.log(n_q) - torch.logsumexp(-dists / tau, dim=1)
        )  # (N,)
        d_q2P = tau * (
            math.log(n_p) - torch.logsumexp(-dists / tau, dim=0)
        )  # (M,)

    cd = torch.mean(d_p2Q) + torch.mean(d_q2P)
    return cd[None]

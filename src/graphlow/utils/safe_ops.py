from __future__ import annotations

import torch

from graphlow.core.backend.base import TensorLike


def safe_vector_norm[T: TensorLike](
    x: T,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-12,
) -> T:
    """
    Compute a clamped vector norm.

    Parameters
    ----------
    x : T
        Input tensor.
    dim : int, default=-1
        Axis along which to compute the norm.
    keepdim : bool, default=False
        Whether to keep the dimension of the input tensor.
    eps : float, default=1e-12
        Lower bound applied to the norm.

    Returns
    -------
    T
        Tensor with the norm computed along ``dim`` and clamped to ``eps``.
    """
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)
    return torch.clamp(norm, min=eps)


def safe_normalize[T: TensorLike](
    x: T,
    dim: int = -1,
    eps: float = 1e-12,
) -> T:
    """
    Normalize a tensor with a clamped norm.

    Parameters
    ----------
    x : T
        Input tensor.
    dim : int, default=-1
        Axis along which to normalize.
    eps : float, default=1e-12
        Lower bound applied to the norm before division.

    Returns
    -------
    T
        Normalized tensor with the same shape as ``x``.
    """
    norm = safe_vector_norm(x, dim=dim, keepdim=True, eps=eps)
    return x / norm

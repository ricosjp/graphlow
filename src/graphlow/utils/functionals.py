from __future__ import annotations

import torch
from phlower_tensor import PhlowerDimensionTensor, PhlowerTensor, functionals

from graphlow.core.backend.base import TensorLike


def einsum[T: TensorLike](
    equation: str,
    *args: T,
    dimension: PhlowerDimensionTensor | None = None,
    is_time_series: bool | None = None,
    is_voxel: bool | None = None,
) -> T:
    if isinstance(args[0], torch.Tensor):
        return torch.einsum(equation, *args)

    elif isinstance(args[0], PhlowerTensor):
        return functionals.einsum(
            equation,
            *args,
            dimension=dimension,
            is_time_series=is_time_series,
            is_voxel=is_voxel,
        )
    else:
        raise ValueError(f"Unexpected tensor: {args[0].__class__}")


def rearrange[T: TensorLike](
    tensor: T,
    pattern: str,
    **axes_length: int,
) -> T:
    """
    Rearrange the tensor.

    Returns:
        PhlowerTensor: Rearranged tensor.
    """
    if isinstance(tensor, torch.Tensor):
        return torch.einsum(tensor, pattern, **axes_length)
    elif isinstance(tensor, PhlowerTensor):
        return tensor.rearrange(pattern, **axes_length)
    else:
        raise ValueError(f"Unexpected tensor: {tensor.__class__}")

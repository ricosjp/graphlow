import phlower_tensor as pt
import torch
from phlower_tensor._tensor._dimension import PhysicDimensionLikeObject


def phlower_zeros(
    shape: tuple[int, ...],
    dimension: PhysicDimensionLikeObject,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> pt.PhlowerTensor:
    """Create a PhlowerTensor of zeros with the given shape and dimension.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the tensor.
    dimension : pt.PhlowerTensor | dict
        Physical dimension (use .dimension from a reference tensor or a dict).
    dtype : torch.dtype | None
        Optional dtype.
    device : torch.device | None
        Optional device.

    Returns
    -------
    pt.PhlowerTensor
    """
    out = pt.phlower_tensor(
        torch.zeros(shape, dtype=dtype), dimension=dimension
    ).to(device=device)
    return out


def phlower_ones(
    shape: tuple[int, ...],
    dimension: PhysicDimensionLikeObject,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> pt.PhlowerTensor:
    return pt.phlower_tensor(
        torch.ones(shape, dtype=dtype), dimension=dimension
    ).to(device=device)

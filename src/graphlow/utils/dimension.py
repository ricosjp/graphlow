from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phlower_tensor import PhlowerDimensionTensor

    from graphlow.core.backend.base import TensorLike


def get_dimension(tensor: TensorLike) -> PhlowerDimensionTensor | None:
    """
    Return dimension metadata attached to a tensor.

    Parameters
    ----------
    tensor : TensorLike
        Input tensor.

    Returns
    -------
    PhlowerDimensionTensor or None
        ``tensor.dimension`` if present (for example on ``PhlowerTensor``);
        otherwise None.
    """
    return getattr(tensor, "dimension", None)

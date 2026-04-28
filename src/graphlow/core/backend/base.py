from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Protocol, Self

import numpy as np
import scipy.sparse as sps
import torch

if TYPE_CHECKING:
    from phlower_tensor import PhlowerDimensionTensor


class TensorLike(Protocol):
    """
    Structural type for backend tensors from third-party libraries.

    This protocol is aimed at types such as ``torch.Tensor`` and
    ``phlower_tensor.PhlowerTensor`` that graphlow does not own. In graphlow it
    is used only as a bound for generic type parameters (e.g. ``T`` in
    ``Backend[T]``), not as a base class for new tensor implementations.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def clone(self) -> Self:
        """Create a copy of the tensor."""
        ...

    def detach(self) -> Self:
        """Detach the tensor from the computation graph."""
        ...

    def index_add_(self, dim: int, index: torch.Tensor, source: Self) -> Self:
        """Add source to self at the indices in index."""
        ...

    def indices(self) -> torch.Tensor:
        """Return the indices of the tensor."""
        ...

    def values(self) -> torch.Tensor:
        """Return the values of the tensor."""
        ...

    def reshape(self, shape: tuple[int, ...]) -> Self:
        """Reshape the tensor."""
        ...

    def to(
        self,
        device: torch.device | str | None = None,
        non_blocking: bool = False,
        dtype: torch.dtype | None = None,
    ) -> TensorLike:
        """Move the tensor to a different device and/or dtype."""
        ...


class Backend[T: TensorLike](ABC):
    """
    Minimal backend API for ops.

    Only tensor creation and sparse materialization.
    For ops, use torch functions directly.
    """

    @property
    @abstractmethod
    def name(self) -> Literal["phlower", "torch"]:
        """Backend name."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device | None:
        """Device for tensor creation. None means default."""
        ...

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Float dtype for tensor."""
        ...

    @abstractmethod
    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Backend[T]:
        """Move the backend to a different device and/or dtype."""
        ...

    @abstractmethod
    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dimension: dict[str, float] | PhlowerDimensionTensor | None = None,
    ) -> T:
        """
        Create a tensor of zeros.
        dimension is for phlower only.
        """
        ...

    @abstractmethod
    def zeros_like(self, x: T) -> T:
        """Create a tensor of zeros with the same shape as x."""
        ...

    @abstractmethod
    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dimension: dict[str, float] | PhlowerDimensionTensor | None = None,
    ) -> T:
        """
        Create a tensor of ones.
        dimension is for phlower only.
        """
        ...

    @abstractmethod
    def ones_like(self, x: T) -> T:
        """Create a tensor of ones with the same shape as x."""
        ...

    @abstractmethod
    def as_tensor(
        self,
        arr: np.ndarray | list | tuple | torch.Tensor,
        *,
        dimension: dict[str, float] | PhlowerDimensionTensor | None = None,
    ) -> T:
        """
        Convert numpy array or list to backend tensor.
        dimension is for phlower only.
        """
        ...

    @abstractmethod
    def as_index_tensor(
        self,
        arr: np.ndarray | list | tuple[int, ...] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert index-like array to a torch.Tensor on this backend's device.

        This is intended only for indexing (e.g. points[indices]) and always
        returns a plain torch.Tensor, not the backend tensor type T.
        """
        ...

    @abstractmethod
    def einsum(
        self,
        equation: str,
        *args: T,
        dimension: PhlowerDimensionTensor | None = None,
        is_time_series: bool | None = None,
        is_voxel: bool | None = None,
    ) -> T:
        """Compute einsum for the backend tensor."""
        ...

    @abstractmethod
    def rearrange(self, x: T, pattern: str, **axes_length: int) -> T:
        """Rearrange the backend tensor."""
        ...

    @abstractmethod
    def to_torch(self, x: T) -> torch.Tensor:
        """Convert backend tensor to torch tensor."""
        ...

    @abstractmethod
    def to_numpy(self, x: T) -> np.ndarray:
        """Convert backend tensor to numpy array."""
        ...

    @abstractmethod
    def make_sparse_from_skeleton(
        self,
        skeleton: sps.csr_array,
        layout: Literal["csr", "coo"] = "coo",
    ) -> T:
        """Materialize backend sparse from scipy sparse array."""
        ...

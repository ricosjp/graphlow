"""Torch backend implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import scipy.sparse as sps
import torch

from graphlow.core.backend.base import Backend
from graphlow.utils.validate_dtype import validate_floating_point_dtype

if TYPE_CHECKING:
    from phlower_tensor import PhlowerDimensionTensor


class TorchBackend(Backend[torch.Tensor]):
    """
    Torch backend.
    For ops, use torch.* functions directly.
    This backend only does creation/sparse.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        *,
        device: torch.device | str | None = None,
    ):
        self._dtype = validate_floating_point_dtype(dtype)
        self._device = torch.device(device) if device is not None else None

    @property
    def name(self) -> Literal["torch"]:
        """Backend name."""
        return "torch"

    @property
    def device(self) -> torch.device | None:
        """Device for tensor creation. None means default."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Float dtype for tensor."""
        return self._dtype

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> TorchBackend:
        """Move the backend to a different device and/or float precision."""
        if dtype is not None:
            dtype = validate_floating_point_dtype(dtype)
        dtype = dtype or self.dtype

        if device is not None:
            device = torch.device(device)
        device = device or self.device

        return TorchBackend(dtype=dtype, device=device)

    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dimension: dict[str, float] | PhlowerDimensionTensor | None = None,
    ) -> torch.Tensor:
        """Create a tensor of zeros. dimension is for phlower only."""
        return torch.zeros(shape, dtype=self.dtype, device=self.device)

    def zeros_like(self, x: torch.Tensor) -> torch.Tensor:
        """Create a tensor of zeros with the same shape as x."""
        return torch.zeros_like(x)

    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dimension: dict[str, float] | PhlowerDimensionTensor | None = None,
    ) -> torch.Tensor:
        """Create a tensor of ones. dimension is for phlower only."""
        return torch.ones(shape, dtype=self.dtype, device=self.device)

    def ones_like(self, x: torch.Tensor) -> torch.Tensor:
        """Create a tensor of ones with the same shape as x."""
        return torch.ones_like(x)

    def as_tensor(
        self,
        arr: np.ndarray | list | tuple | torch.Tensor,
        *,
        dimension: dict[str, float] | PhlowerDimensionTensor | None = None,
    ) -> torch.Tensor:
        """Convert numpy/list/torch.Tensor to backend tensor."""
        if isinstance(arr, torch.Tensor):
            if arr.dtype.is_floating_point:
                return arr.to(dtype=self.dtype, device=self.device)
            return arr.to(device=self.device)

        if isinstance(arr, np.ndarray | list | tuple):
            arr = np.array(arr)
            arr = np.ascontiguousarray(arr)
            dtype = (
                self.dtype if np.issubdtype(arr.dtype, np.floating) else None
            )
            return torch.from_numpy(arr).to(dtype=dtype, device=self.device)
        raise NotImplementedError(
            f"{type(arr)} cannot be converted to backend tensor"
        )

    def to_numpy(self, x: torch.Tensor) -> np.ndarray:
        """Convert backend tensor to numpy array."""
        return x.cpu().detach().numpy()

    def as_index_tensor(
        self,
        arr: np.ndarray | list | tuple[int, ...] | torch.Tensor,
    ) -> torch.Tensor:
        """Convert array to torch.Tensor on this backend's device."""
        if isinstance(arr, torch.Tensor):
            if arr.dtype in (torch.int64, torch.int32):
                return arr.to(dtype=torch.int64, device=self.device)
            raise ValueError("arr must be int64 or int32")

        if isinstance(arr, np.ndarray | list | tuple):
            arr = np.asarray(arr)
            if np.can_cast(arr.dtype, np.int64, casting="safe"):
                arr = np.asarray(arr, dtype=np.int64)
                arr = np.ascontiguousarray(arr)
                if not arr.flags.writeable:
                    arr = arr.copy()
                return torch.from_numpy(arr).to(device=self.device)
            raise ValueError("arr must be int64 or int32")
        raise NotImplementedError(
            f"{type(arr)} cannot be converted to index tensor"
        )

    def to_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Convert backend tensor to torch tensor."""
        return x

    def make_sparse_from_skeleton(
        self,
        skeleton: sps.csr_array,
        layout: Literal["csr", "coo"] = "coo",
    ) -> torch.Tensor:
        """Materialize torch sparse from scipy skeleton."""
        if layout == "csr":
            csr = skeleton.tocsr()
            crow = torch.from_numpy(np.asarray(csr.indptr, dtype=np.int64))
            col = torch.from_numpy(np.asarray(csr.indices, dtype=np.int64))
            values = torch.from_numpy(np.asarray(csr.data)).to(dtype=self.dtype)
            out = torch.sparse_csr_tensor(crow, col, values, size=csr.shape)
        else:
            coo = skeleton.tocoo()
            indices = torch.from_numpy(
                np.stack([coo.row, coo.col], axis=0).astype(np.int64)
            )
            values = torch.from_numpy(np.asarray(coo.data)).to(dtype=self.dtype)
            out = torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()
        return out.to(device=self.device)

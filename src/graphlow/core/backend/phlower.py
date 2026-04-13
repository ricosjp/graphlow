"""Phlower backend implementation."""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sps
import torch

try:
    import phlower_tensor as pt
    from phlower_tensor import PhlowerDimensionTensor
except ImportError as e:
    raise RuntimeError(
        "PhlowerBackend requires phlower_tensor. "
        "Install with `pip install graphlow[phlower]`."
    ) from e

from graphlow.core.backend.base import Backend
from graphlow.utils.enums import PRECISION_TO_DTYPE, FloatPrecision


class PhlowerBackend(Backend[pt.PhlowerTensor]):
    """
    Phlower backend.
    For ops, use torch.* functions directly.
    This backend only does creation/sparse.
    """

    def __init__(
        self,
        float_precision: FloatPrecision | int = FloatPrecision.FLOAT32,
        *,
        device: torch.device | str | None = None,
    ):
        self._float_precision = FloatPrecision(float_precision)
        self._dtype = PRECISION_TO_DTYPE[self._float_precision]
        self._device = device

    @property
    def name(self) -> Literal["phlower"]:
        """Backend name."""
        return "phlower"

    @property
    def device(self) -> torch.device | str | None:
        """Device for tensor creation. None means default."""
        return self._device

    @property
    def float_precision(self) -> FloatPrecision:
        """Float precision to use."""
        return self._float_precision

    @property
    def dtype(self) -> torch.dtype:
        """Float dtype for tensor."""
        return self._dtype

    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dimension: dict[str, float] | PhlowerDimensionTensor | None = None,
    ) -> pt.PhlowerTensor:
        """Create a tensor of zeros."""
        arr = np.zeros(shape)
        return pt.phlower_tensor(
            arr, dimension=dimension, dtype=self.dtype, device=self.device
        )

    def zeros_like(self, x: pt.PhlowerTensor) -> pt.PhlowerTensor:
        """Create a tensor of zeros with the same shape as x."""
        return torch.zeros_like(x)

    def ones(
        self,
        shape: int | tuple[int, ...],
        *,
        dimension: dict[str, float] | PhlowerDimensionTensor | None = None,
    ) -> pt.PhlowerTensor:
        """Create a tensor of ones."""
        arr = np.ones(shape)
        return pt.phlower_tensor(
            arr, dimension=dimension, dtype=self.dtype, device=self.device
        )

    def ones_like(self, x: pt.PhlowerTensor) -> pt.PhlowerTensor:
        """Create a tensor of ones with the same shape as x."""
        return torch.ones_like(x)

    def as_tensor(
        self,
        arr: np.ndarray | list | tuple | torch.Tensor,
        *,
        dimension: dict[str, float] | PhlowerDimensionTensor | None = None,
    ) -> pt.PhlowerTensor:
        """Convert numpy to backend tensor. dimension is optional (phlower)."""
        if isinstance(arr, torch.Tensor):
            t = pt.phlower_tensor(arr.to(dtype=self.dtype), dimension=dimension)
            return t.to(device=self.device)

        if isinstance(arr, np.ndarray | list | tuple):
            arr = np.array(arr)
            arr = np.ascontiguousarray(arr)
            return pt.phlower_tensor(
                arr, dimension=dimension, dtype=self.dtype, device=self.device
            )
        raise NotImplementedError(
            f"{type(arr)} cannot be converted to backend tensor"
        )

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

    def to_torch(self, x: pt.PhlowerTensor) -> torch.Tensor:
        """Convert backend tensor to torch tensor."""
        return x.to_tensor()

    def to_numpy(self, x: pt.PhlowerTensor | torch.Tensor) -> np.ndarray:
        """Convert backend tensor to numpy array."""
        if isinstance(x, pt.PhlowerTensor):
            return x.to_numpy()
        return x.cpu().detach().numpy()

    def make_sparse_from_skeleton(
        self,
        skeleton: sps.csr_array,
        layout: Literal["csr", "coo"] = "coo",
    ) -> pt.PhlowerTensor:
        """Materialize torch sparse from scipy skeleton."""
        if layout == "csr":
            csr: sps.csr_array = skeleton.tocsr()
            crow = torch.from_numpy(np.asarray(csr.indptr, dtype=np.int64))
            col = torch.from_numpy(np.asarray(csr.indices, dtype=np.int64))
            values = torch.from_numpy(np.asarray(csr.data)).to(dtype=self.dtype)
            out = torch.sparse_csr_tensor(crow, col, values, size=csr.shape)
        else:
            coo: sps.coo_array = skeleton.tocoo()
            indices = torch.from_numpy(
                np.stack([coo.row, coo.col], axis=0).astype(np.int64)
            )
            values = torch.from_numpy(np.asarray(coo.data)).to(dtype=self.dtype)
            out = torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()

        return pt.phlower_tensor(out, dimension={}).to(device=self.device)

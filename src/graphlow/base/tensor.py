from __future__ import annotations

import torch

from graphlow.base.tensor_property import GraphlowTensorProperty
from graphlow.util import array_handler, typing


class GraphlowTensor:
    def __init__(
        self,
        tensor: typing.ArrayDataType,
        *,
        time_series: bool = False,
        device: torch.device | int = -1,
        dtype: torch.dtype | type | None = None,
    ):
        self._tensor_property = GraphlowTensorProperty(
            device=device, dtype=dtype
        )
        self._tensor: torch.Tensor = array_handler.convert_to_torch_tensor(
            tensor, device=self.device, dtype=self.dtype
        )
        self._time_series = time_series
        return

    def __len__(self) -> int:
        if self.time_series:
            return self.shape[1]
        else:
            return self.shape[0]

    @property
    def device(self) -> torch.device:
        return self._tensor_property.device

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor_property.dtype

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def time_series(self) -> bool:
        return self._time_series

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    def send(
        self,
        *,
        device: torch.device | int | None = None,
        dtype: torch.dtype | type | None = None,
    ):
        """Convert tensor to the specified device and dtype.

        Parameters
        ----------
        device: torch.device | int
        dtype: torch.dtype | type | None
        """
        self._tensor_property.device = device or self.device
        self._tensor_property.dtype = dtype or self.dtype
        self._tensor = array_handler.convert_to_torch_tensor(
            self._tensor, device=self.device, dtype=self.dtype
        )
        return

    def convert_to_numpy_scipy(self) -> typing.NumpyScipyArray:
        return array_handler.convert_to_numpy_scipy(self.tensor)

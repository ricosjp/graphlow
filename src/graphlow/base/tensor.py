
import torch
from typing_extensions import Self

from graphlow.base.tensor_property import GraphlowTensorProperty
from graphlow.util import array_handler, typing


class GraphlowTensor:

    def __init__(
            self, tensor: typing.ArrayDataType,
            *,
            time_series: bool = False,
            device: torch.device | int = -1,
            dtype: torch.dtype | type | None = None,
    ):
        self.tensor_property = GraphlowTensorProperty(
            device=device, dtype=dtype)
        self._tensor: torch.Tensor = self.convert_to_torch_tensor(tensor)
        self._time_series = time_series
        return

    def __len__(self) -> int:
        if self.time_series:
            return self.shape[1]
        else:
            return self.shape[0]

    @property
    def tensor(self) -> torch.Tensor:
        return self._tensor

    @property
    def time_series(self) -> bool:
        return self._time_series

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    def to(
            self, device: torch.device | int,
            dtype: torch.dtype | type | None = None):
        """Convert tensor to the specified device and dtype.

        Parameters
        ----------
        device: torch.device | int
        dtype: torch.dtype | type | None
        """
        self.tensor_property.device = device
        self.tensor_property.dtype = dtype or self.tensor_property.dtype
        self._tensor.to(
            device=self.tensor_property.device,
            dtype=self.tensor_property.dtype)
        return

    def convert_to_torch_tensor(
            self, tensor: Self | typing.ArrayDataType) -> torch.Tensor:
        if isinstance(tensor, GraphlowTensor):
            return self._tensor
        else:
            return array_handler.convert_to_torch_tensor(tensor)


from collections import abc

import numpy as np
import torch
from typing_extensions import Self

from graphlow.base.tensor import GraphlowTensor
from graphlow.base.tensor_property import GraphlowTensorProperty
from graphlow.util import typing


class GraphlowDictTensor:

    def __init__(
            self,
            dict_tensor: Self | dict[str, typing.ArrayDataType],
            length: int | None = None,
            *,
            time_series: bool | list[bool] = False,
            device: torch.device | int = -1,
            dtype: torch.dtype | type | None = None,
    ):
        """Initialize GraphlowDictTensor object.

        Parameters
        ----------
        dict_tensor: GraphlowDictTensor | dict[str, graphlow.ArrayDataType]
            Dict of tensor data.
        length: int | None
            Length of the data. Typically, n_points or n_cells.
            If None is fed, no shape check will run.
        time_series: bool | list[bool]
            Specifies if the data is time series or not. Can be specified
            for each value by inputting list[bool].
        device: torch.device | int
            Device ID. int < 0 implies CPU.
        dtype: torch.dtype | type | None
            Data type.
        """
        self.tensor_property = GraphlowTensorProperty(
            device=device, dtype=dtype)

        if dict_tensor is None:
            self._dict_tensor: dict[str, GraphlowTensor] = {}
        else:
            if isinstance(time_series, bool):
                time_series = [time_series] * len(dict_tensor)

            self._dict_tensor: dict[str, GraphlowTensor] = {
                k: GraphlowTensor(
                    v, time_series=ts,
                    device=self.tensor_property.device,
                    dtype=self.tensor_property.dtype)
                for ts, (k, v) in zip(
                    time_series, dict_tensor.items(), strict=True)}
        self.length = length
        self.validate_length_if_needed()
        return

    def __contains__(self, key: str) -> bool:
        return key in self.dict_tensor

    def __getitem__(self, key: str) -> torch.Tensor:
        if key not in self:
            keys = list(self.keys())
            raise KeyError(f"{key} not in {keys}")
        return self.dict_tensor[key]

    @property
    def dict_tensor(self) -> dict[str, GraphlowTensor]:
        return self._dict_tensor

    def keys(self) -> abc.KeysView:
        return self.dict_tensor.keys()

    def values(self) -> abc.ValuesView:
        return self.dict_tensor.values()

    def items(self) -> abc.ItemsView:
        return self.dict_tensor.items()

    def to(
            self, device: torch.device | int,
            dtype: torch.dtype | type | None = None):
        """Convert features to the specified device and dtype.

        Parameters
        ----------
        device: torch.device | int
        dtype: torch.dtype | type | None
        """
        self.tensor_property.device = device
        self.tensor_property.dtype = dtype or self.tensor_property.dtype

        self._dict_tensor = {
            k: v.to(
                device=self.tensor_property.device,
                dtype=self.tensor_property.dtype)
            for k, v in self._dict_tensor.items()}
        return

    def has_time_series(self) -> bool:
        """Test if it has time series data.

        Returns
        -------
        bool
        """
        return np.any([
            v.time_series for v in self.values()])

    def update(
            self, dict_tensor: dict[str, torch.Tensor] | Self, *,
            overwrite: bool = False):
        """Update GraphlowDictTensor with input dict.

        Parameters
        ----------
        dict_data: dict | graphlow.GraphlowDictTensor
        """
        for key, value in dict_tensor.items():
            if key in self.dict_tensor:
                if not overwrite:
                    keys = list(self.keys())
                    raise ValueError(f"{key} already exists in {keys}")
                self._dict_tensor[keys] = value
        self.validate_length_if_needed()
        return

    def validate_length_if_needed(self):
        """Validate graphlow tensors' lengths."""
        if self.length is None:
            return
        for key, value in self.dict_tensor.items():
            if len(value) != self.length:
                raise ValueError(
                    f"Invalid length for: {key} "
                    f"(expected: {self.length}, given: {len(value)})")
        return

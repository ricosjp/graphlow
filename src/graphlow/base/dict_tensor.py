
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
            dict_tensor: Self | dict[typing.KeyType, typing.ArrayDataType],
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
        self._tensor_property = GraphlowTensorProperty(
            device=device, dtype=dtype)

        if dict_tensor is None:
            self._dict_tensor: dict[typing.KeyType, GraphlowTensor] = {}
        else:
            if isinstance(time_series, bool):
                time_series = [time_series] * len(dict_tensor)

            self._dict_tensor: dict[typing.KeyType, GraphlowTensor] = {
                k: GraphlowTensor(
                    v, time_series=ts,
                    device=self.device, dtype=self.dtype)
                for ts, (k, v) in zip(
                    time_series, dict_tensor.items(), strict=True)}
        self.length = length
        self.validate_length_if_needed()
        return

    def __contains__(self, key: typing.KeyType) -> bool:
        return key in self.dict_tensor

    def __getitem__(self, key: typing.KeyType) -> torch.Tensor:
        if key not in self:
            keys = list(self.keys())
            raise KeyError(f"{key} not in {keys}")
        return self.dict_tensor[key].tensor

    @property
    def device(self) -> torch.Tensor:
        return self._tensor_property.device

    @property
    def dtype(self) -> torch.Tensor:
        return self._tensor_property.dtype

    @property
    def dict_tensor(self) -> dict[str, GraphlowTensor]:
        return self._dict_tensor

    def keys(self) -> abc.KeysView:
        return self.dict_tensor.keys()

    def values(self) -> abc.ValuesView:
        return self.dict_tensor.values()

    def items(self) -> abc.ItemsView:
        return self.dict_tensor.items()

    def pop(self, key: typing.KeyType) -> torch.Tensor:
        return self._dict_tensor.pop(key)

    def send(
            self, *,
            device: torch.device | int | None = None,
            dtype: torch.dtype | type | None = None):
        """Convert features to the specified device and dtype.

        Parameters
        ----------
        device: torch.device | int | None
        dtype: torch.dtype | type | None
        """
        self._tensor_property.device = device or self.device
        self._tensor_property.dtype = dtype or self.dtype

        for v in self._dict_tensor.values():
            v.send(device=self.device, dtype=self.dtype)
        # self._dict_tensor = {
        #     k: v.send(device=self.device, dtype=self.dtype)
        #     for k, v in self._dict_tensor.items()}
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
            self, dict_tensor: dict[str, typing.ArrayDataType] | Self, *,
            time_series: bool | list[bool] = False,
            overwrite: bool = False):
        """Update GraphlowDictTensor with input dict.

        Parameters
        ----------
        dict_data: dict | graphlow.GraphlowDictTensor
        overwrite: bool
            If True, allow overwriting exsiting items. The default is False.
        """
        if isinstance(time_series, bool):
            time_series = [time_series] * len(dict_tensor)

        for ts, (key, value) in zip(
                time_series, dict_tensor.items(), strict=True):
            if key in self.dict_tensor:
                if not overwrite:
                    keys = list(self.keys())
                    raise ValueError(f"{key} already exists in {keys}")

            self._dict_tensor[key] = GraphlowTensor(
                value, device=self.device, dtype=self.dtype,
                time_series=ts)

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

    def convert_to_numpy_scipy(self) -> dict[
            typing.KeyType, typing.NumpyScipyArray]:
        return {
            k: v.convert_to_numpy_scipy()
            for k, v in self.items()}

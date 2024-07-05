
import torch


class GraphlowTensorProperty:

    DEFAULT_FLOAT_TYPE = torch.float32

    def __init__(
            self,
            device: torch.device | int = -1,
            dtype: torch.dtype | type | None = None):
        self.device = device
        self.dtype: torch.dtype = dtype
        return

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device | int):
        if isinstance(device, int):
            if device < 0:
                self._device = torch.device('cpu')
                return
            self._device = torch.device(device)
            return
        self._device = device
        return

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype | type | None):
        if dtype is None:
            self._dtype = self.DEFAULT_FLOAT_TYPE
            return
        if isinstance(dtype, torch.dtype):
            self._dtype = dtype
            return

        str_type = dtype.__name__
        if str_type == 'float':
            self._dtype = torch.float
            return
        elif str_type == 'int':
            self._dtype = torch.int
            return
        elif str_type == 'bool':
            self._dtype = torch.bool
            return
        else:
            raise ValueError(f"Unexpected dtype: {dtype}")

import numpy as np
import torch
from scipy import sparse as sp

from graphlow.util import typing


def convert_to_numpy_scipy(
    array: typing.ArrayDataType, torch_dtype: torch.dtype = torch.float32
) -> np.ndarray | sp.sparray:
    """Convert input array to numpy array or scipy sparse array.

    Parameters
    ----------
    array: graphlow.util.typing.ArrayDataType
    torch_dtype: torch.dtype
        Torch float dtype. The default is torch.float32.

    Returns
    -------
    numpy.ndarray | sp.sparray
    """
    if isinstance(array, torch.Tensor):
        return array.to("cpu", torch_dtype).detach().to_dense().numpy()
    elif isinstance(array, sp.sparray):
        return array
    elif isinstance(array, np.ndarray):
        return array
    elif hasattr(array, "data"):
        return convert_to_numpy_scipy(array.data)
    else:
        raise TypeError(f"Unexpected input type: {array.__class__}")


def convert_to_dense_numpy(array: typing.ArrayDataType) -> np.ndarray:
    """Convert input array to numpy array. If input is sparse, converted to
    dense.

    Parameters
    ----------
    array: graphlow.util.typing.ArrayDataType

    Returns
    -------
    numpy.ndarray
    """
    converted = convert_to_numpy_scipy(array)
    if isinstance(converted, sp.sparray):
        return converted.toarray()
    else:
        return converted


def convert_to_scipy_sparse_csr(array: typing.ArrayDataType) -> sp.csr_array:
    """Convert input array to scipy CSR array.

    Parameters
    ----------
    array: graphlow.util.typing.ArrayDataType

    Returns
    -------
    scipy.sparse.csr_array
    """
    if isinstance(array, torch.Tensor):
        if array.is_sparse_csr:
            indptr = array.crow_indices().numpy()
            indices = array.col_indices().numpy()
            values = array.values().numpy()
            shape = tuple(array.size())
            return sp.csr_array((values, indices, indptr), shape=shape)
        else:
            return convert_to_scipy_sparse_csr(array.to_sparse_csr())
    elif isinstance(array, np.ndarray):
        return sp.csr_array(array)
    elif isinstance(array, sp.sparray):
        return array.tocsr()
    else:
        raise TypeError(f"Unexpected input type: {array.__class__}")


def send(
    tensor: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert input torch tensor to desired device and dtype.

    Parameters
    ----------
    tensor: torch.Tensor
    device: torch.device | None
    dtype: torch.dtype | None
        Data type of the converted tensor. Note that it is effective only when
        tensor is float-like, i.e., bool or int will be preserved.

    Returns
    -------
    torch.Tensor
    """
    if "float" in str(tensor.dtype):
        return tensor.to(device=device, dtype=dtype)
    else:
        # NOTE: Preserve type if int or bool
        return tensor.to(device=device)


def convert_to_torch_tensor(
    array: typing.ArrayDataType,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert input array to torch tensor. If the input is sparse, output
    tensor will be also sparse.

    Parameters
    ----------
    array: graphlow.util.typing.ArrayDataType
    device: torch.device | None
    dtype: torch.dtype | None
        Data type of the converted tensor. Note that it is effective only when
        tensor is float-like, i.e., bool or int will be preserved.

    Returns
    -------
    torch.Tensor
    """
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
    elif isinstance(array, torch.Tensor):
        tensor = array
    elif isinstance(array, sp.sparray):
        tensor = convert_to_torch_sparse_csr(array)
    elif hasattr(array, "data"):
        tensor = convert_to_torch_tensor(array.data)
    else:
        raise TypeError(f"Unexpected input type: {array.__class__}")

    return send(tensor, device=device, dtype=dtype)


def convert_to_torch_sparse_csr(
    array: typing.ArrayDataType,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert input array to torch CSR tensor.

    Parameters
    ----------
    array: graphlow.util.typing.ArrayDataType
    device: torch.device | None
    dtype: torch.dtype | None
        Data type of the converted tensor. Note that it is effective only when
        tensor is float-like, i.e., bool or int will be preserved.

    Returns
    -------
    torch.Tensor
    """
    if isinstance(array, sp.sparray):
        scipy_csr = array.tocsr()
        indptr = torch.from_numpy(scipy_csr.indptr)
        indices = torch.from_numpy(scipy_csr.indices)
        data = torch.from_numpy(scipy_csr.data)
        size = scipy_csr.shape
        tensor = torch.sparse_csr_tensor(indptr, indices, data, size=size)
    elif isinstance(array, torch.Tensor):
        tensor = array.to_sparse_csr()
    elif isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array).to_sparse_csr()
    else:
        raise TypeError(f"Unexpected input type: {array.__class__}")
    return send(tensor, device=device, dtype=dtype)

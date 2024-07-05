import numpy as np
import torch
from scipy import sparse as sp

from graphlow.util import typing


def convert_to_numpy(array: typing.ArrayDataType) -> np.ndarray:
    """Convert input array to numpy array.

    Parameters
    ----------
    array: graphlow.util.typing.ArrayDataType

    Returns
    -------
    numpy.ndarray
    """
    if isinstance(array, torch.Tensor):
        return array.to('cpu').detach().to_dense().numpy()
    elif isinstance(array, sp.sparray):
        return array.todense()
    elif isinstance(array, np.ndarray):
        return array
    elif hasattr(array, 'data'):
        return convert_to_numpy(array.data)
    else:
        raise TypeError(f"Unexpected input type: {array.__class__}")


def convert_to_torch_tensor(array: typing.ArrayDataType) -> torch.Tensor:
    """Convert input array to torch tensor. If the input is sparse, output
    tensor will be also sparse.

    Parameters
    ----------
    array: graphlow.util.typing.ArrayDataType

    Returns
    -------
    torch.Tensor
    """
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    elif isinstance(array, torch.Tensor):
        return array
    elif isinstance(array, sp.sparray):
        return convert_to_torch_sparse_csr(array)
    elif hasattr(array, 'data'):
        return convert_to_torch_tensor(array.data)
    else:
        raise TypeError(f"Unexpected input type: {array.__class__}")


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


def convert_to_torch_sparse_csr(array: typing.ArrayDataType) -> torch.Tensor:
    """Convert input array to torch CSR tensor.

    Parameters
    ----------
    array: graphlow.util.typing.ArrayDataType

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
        return torch.sparse_csr_tensor(indptr, indices, data, size=size)
    elif isinstance(array, torch.Tensor):
        return array.to_sparse_csr()
    elif isinstance(array, np.ndarray):
        return torch.from_numpy(array).to_sparse_csr()
    else:
        raise TypeError(f"Unexpected input type: {array.__class__}")

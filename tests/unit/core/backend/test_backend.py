"""Unit tests for Backend (TorchBackend, PhlowerBackend).

Run for both torch and phlower.
"""

import logging

import numpy as np
import phlower_tensor as pt
import pytest
import scipy.sparse as sps
import torch
from tests.unit.conftest import BackendParams

from graphlow.core.backend.base import Backend
from graphlow.core.backend.factory import get_backend
from graphlow.core.backend.phlower import PhlowerBackend
from graphlow.core.backend.torch import TorchBackend

logger = logging.getLogger(__name__)


# =============================================================================
# Backend properties
# =============================================================================
class TestBackendProperties:
    def test_properties(self, bparam: BackendParams) -> None:
        """name returns the backend name."""
        backend = get_backend(bparam.name, bparam.dtype, device=bparam.device)
        assert backend.name == bparam.name
        assert backend.dtype == bparam.dtype
        assert backend.device == bparam.device

    def test_to_updates_dtype(self, bparam: BackendParams) -> None:
        """to updates float precision in-place."""
        backend = get_backend(bparam.name, bparam.dtype, device=bparam.device)

        # switch the dtype
        target_dtype = (
            torch.float64 if backend.dtype == torch.float32 else torch.float32
        )
        returned = backend.to(dtype=target_dtype)

        assert returned is not backend
        assert returned.dtype == target_dtype
        assert returned.device == bparam.device

    def test_to_updates_device_cycle(self, bparam: BackendParams) -> None:
        """to updates device in-place, cuda -> cpu -> cuda."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")

        backend = get_backend(bparam.name, bparam.dtype, device=bparam.device)

        # switch the device
        target_device = (
            torch.device("cuda")
            if bparam.device.type == "cpu"
            else torch.device("cpu")
        )
        returned = backend.to(device=target_device)
        assert returned is not backend
        assert returned.dtype == bparam.dtype
        assert returned.device == target_device

        # switch the device back
        returned = returned.to(device=bparam.device)
        assert returned is not backend
        assert returned.dtype == bparam.dtype
        assert returned.device == bparam.device

    @pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
    def test_to_invalid_dtype(
        self, bparam: BackendParams, dtype: torch.dtype
    ) -> None:
        """to raises ValueError for invalid dtype."""
        backend = get_backend(bparam.name, bparam.dtype, device=bparam.device)
        with pytest.raises(ValueError):
            backend.to(dtype=dtype)


# =============================================================================
# as_tensor (both backends)
# =============================================================================
class TestBackendAsTensor:
    def test_numpy_float_shape_dtype(self, backend: Backend) -> None:
        """as_tensor(numpy float) returns tensor with expected shape/dtype."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        t = backend.as_tensor(arr)
        assert t.shape == (2, 2)
        assert t.dtype == backend.dtype
        expected = torch.from_numpy(arr).to(
            dtype=backend.dtype, device=backend.device
        )
        torch.testing.assert_close(backend.to_torch(t), expected)

    def test_numpy_int_preserves_integer(self, backend: Backend) -> None:
        """as_tensor(numpy int) keeps integer dtype without float cast."""
        arr = np.array([1, 2, 3], dtype=np.int32)
        t = backend.as_tensor(arr)
        assert t.dtype == torch.int32
        expected = torch.tensor(
            [1, 2, 3], dtype=torch.int32, device=backend.device
        )
        torch.testing.assert_close(backend.to_torch(t), expected)

    def test_list_converted_to_tensor(self, backend: Backend) -> None:
        """as_tensor(list) returns tensor with expected values."""
        data = [[1.0, 0.0], [0.0, 1.0]]
        t = backend.as_tensor(data)
        assert t.shape == (2, 2)
        expected = torch.tensor(
            data, dtype=backend.dtype, device=backend.device
        )
        torch.testing.assert_close(backend.to_torch(t), expected)

    def test_tuple_converted_to_tensor(self, backend: Backend) -> None:
        """as_tensor(tuple) returns tensor."""
        data = (1.0, 2.0, 3.0)
        t = backend.as_tensor(data)
        assert t.shape == (3,)
        expected = torch.tensor(
            list(data), dtype=backend.dtype, device=backend.device
        )
        torch.testing.assert_close(backend.to_torch(t), expected)

    def test_torch_tensor_returned_with_dtype(self, backend: Backend) -> None:
        """as_tensor(torch.Tensor) returns tensor with backend dtype."""
        x = torch.tensor([1.0, 2.0], dtype=torch.float64)
        t = backend.as_tensor(x)
        assert t.dtype == backend.dtype
        expected = x.to(dtype=backend.dtype, device=backend.device)
        torch.testing.assert_close(backend.to_torch(t), expected)

    def test_torch_integer_tensor_preserves_dtype(
        self, backend: Backend
    ) -> None:
        """as_tensor(torch int tensor) keeps integer dtype without cast."""
        x = torch.tensor([1, 2, 3], dtype=torch.int32)
        t = backend.as_tensor(x)
        assert t.dtype == torch.int32
        expected = x.to(device=backend.device)
        torch.testing.assert_close(backend.to_torch(t), expected)


# =============================================================================
# as_index_tensor (both backends)
# =============================================================================
class TestBackendAsIndexTensor:
    def test_numpy_int64_long_tensor(self, backend: Backend) -> None:
        """as_index_tensor(numpy int) returns torch int64 tensor."""
        arr = np.array([0, 1, 2], dtype=np.int64)
        t = backend.as_index_tensor(arr)
        assert t.dtype == torch.int64
        assert t.shape == (3,)
        expected = torch.from_numpy(arr).to(device=backend.device)
        torch.testing.assert_close(t, expected)

    def test_list_becomes_long_tensor(self, backend: Backend) -> None:
        """as_index_tensor(list) returns int64 tensor."""
        t = backend.as_index_tensor([3, 1, 4])
        assert t.dtype == torch.int64
        expected = torch.tensor(
            [3, 1, 4], dtype=torch.int64, device=backend.device
        )
        torch.testing.assert_close(t, expected)

    def test_tuple_becomes_long_tensor(self, backend: Backend) -> None:
        """as_index_tensor(tuple) returns int64 tensor."""
        t = backend.as_index_tensor((0, 1))
        assert t.dtype == torch.int64
        expected = torch.tensor(
            [0, 1], dtype=torch.int64, device=backend.device
        )
        torch.testing.assert_close(t, expected)

    def test_torch_tensor_to_int64(self, backend: Backend) -> None:
        """as_index_tensor(torch.Tensor) returns int64 tensor."""
        x = torch.tensor([1, 2], dtype=torch.int32)
        t = backend.as_index_tensor(x)
        assert t.dtype == torch.int64
        expected = torch.tensor(
            [1, 2], dtype=torch.int64, device=backend.device
        )
        torch.testing.assert_close(t, expected)

    def test_rejects_numpy_float(self, backend: Backend) -> None:
        """as_index_tensor rejects float numpy arrays (no silent truncate)."""
        arr = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        with pytest.raises(ValueError, match="int64 or int32"):
            backend.as_index_tensor(arr)

    def test_rejects_float_list(self, backend: Backend) -> None:
        """as_index_tensor rejects lists that become float dtype."""
        with pytest.raises(ValueError, match="int64 or int32"):
            backend.as_index_tensor([0.0, 1.0, 2.0])

    def test_rejects_float_torch_tensor(self, backend: Backend) -> None:
        """as_index_tensor rejects floating torch tensors."""
        x = torch.tensor([0.0, 1.0], dtype=torch.float32)
        with pytest.raises(ValueError, match="int64 or int32"):
            backend.as_index_tensor(x)

    def test_rejects_bool_torch_tensor(self, backend: Backend) -> None:
        """as_index_tensor rejects boolean tensors."""
        x = torch.tensor([True, False])
        with pytest.raises(ValueError, match="int64 or int32"):
            backend.as_index_tensor(x)


# =============================================================================
# zeros, ones, zeros_like, ones_like (both backends)
# =============================================================================
class TestBackendZerosOnes:
    def test_zeros_shape_dtype(self, backend: Backend) -> None:
        """zeros(shape) returns tensor of zeros with backend dtype."""
        t = backend.zeros((2, 3))
        assert t.shape == (2, 3)
        assert t.dtype == backend.dtype
        out = backend.to_torch(t)
        assert out.eq(0).all().item()

    def test_ones_shape_dtype(self, backend: Backend) -> None:
        """ones(shape) returns tensor of ones with backend dtype."""
        t = backend.ones((2, 3))
        assert t.shape == (2, 3)
        assert t.dtype == backend.dtype
        out = backend.to_torch(t)
        assert out.eq(1).all().item()

    def test_zeros_like_same_shape(self, backend: Backend) -> None:
        """zeros_like(x) has same shape as x."""
        x = backend.as_tensor(np.ones((3, 4)))
        t = backend.zeros_like(x)
        assert t.shape == x.shape
        assert backend.to_torch(t).eq(0).all().item()

    def test_ones_like_same_shape(self, backend: Backend) -> None:
        """ones_like(x) has same shape as x."""
        x = backend.as_tensor(np.zeros((2, 2)))
        t = backend.ones_like(x)
        assert t.shape == x.shape
        assert backend.to_torch(t).eq(1).all().item()


# =============================================================================
# make_sparse_from_skeleton (both backends)
# =============================================================================
class TestBackendMakeSparse:
    def test_csr_shape_and_nnz(self, backend: Backend) -> None:
        """make_sparse_from_skeleton(csr) returns sparse tensor, same shape."""
        skeleton = sps.csr_matrix(
            (np.array([1.0, 2.0]), (np.array([0, 1]), np.array([1, 0]))),
            shape=(3, 3),
        )
        t = backend.make_sparse_from_skeleton(skeleton, layout="csr")
        assert t.shape == (3, 3)
        core = backend.to_torch(t)
        assert core.layout == torch.sparse_csr
        assert core._nnz() == 2

    def test_coo_shape_and_nnz(self, backend: Backend) -> None:
        """make_sparse_from_skeleton(coo) returns sparse tensor, same shape."""
        skeleton = sps.coo_matrix(
            (np.array([1.0, 2.0]), (np.array([0, 1]), np.array([1, 0]))),
            shape=(3, 3),
        )
        t = backend.make_sparse_from_skeleton(skeleton, layout="coo")
        assert t.shape == (3, 3)
        core = backend.to_torch(t)
        assert core.layout == torch.sparse_coo
        assert core._nnz() == 2


# =============================================================================
# TorchBackend-only:
# =============================================================================
def test_torch_backend_to_torch_is_identity(
    backend_torch: TorchBackend,
) -> None:
    """to_torch(x) returns x for TorchBackend."""
    x = backend_torch.as_tensor(np.array([1.0, 2.0]))
    out = backend_torch.to_torch(x)
    assert out is x


def test_torch_backend_to_numpy(backend_torch: TorchBackend) -> None:
    """to_numpy(x) returns x for TorchBackend."""
    expected = np.array([1.0, 2.0])
    x = backend_torch.as_tensor(expected)
    out = backend_torch.to_numpy(x)
    np.testing.assert_array_equal(out, expected)


# =============================================================================
# PhlowerBackend-only: dimension, to_torch, to_numpy converts correctly
# =============================================================================
class TestPhlowerBackendOnly:
    def test_as_tensor_preserves_dimension(
        self, backend_phlower: PhlowerBackend
    ) -> None:
        """as_tensor(..., dimension=...) preserves dimension (phlower)."""

        arr = np.array([1.0, 2.0, 3.0])
        dim = {"L": 1}
        t = backend_phlower.as_tensor(arr, dimension=dim)
        assert hasattr(t, "dimension")
        assert t.dimension == pt.phlower_dimension_tensor(
            dim, device=backend_phlower.device
        )

    def test_to_torch_returns_plain_tensor(
        self, backend_phlower: PhlowerBackend
    ) -> None:
        """to_torch(x) returns plain torch.Tensor for PhlowerBackend."""
        x = backend_phlower.as_tensor(np.array([1.0, 2.0]))
        out = backend_phlower.to_torch(x)
        assert isinstance(out, torch.Tensor)
        expected = torch.tensor([1.0, 2.0], dtype=out.dtype, device=out.device)
        torch.testing.assert_close(out, expected)

    def test_to_numpy_returns_numpy_array(
        self, backend_phlower: PhlowerBackend
    ) -> None:
        """to_numpy(x) returns numpy array for PhlowerBackend."""
        expected = np.array([1.0, 2.0])
        x = backend_phlower.as_tensor(expected)
        out = backend_phlower.to_numpy(x)
        np.testing.assert_array_equal(out, expected)

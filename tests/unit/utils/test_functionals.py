"""Unit tests for utils.functionals."""

from __future__ import annotations

import einops
import numpy as np
import pytest
from phlower_tensor._tensor import phlower_dimension_tensor

import graphlow.utils.functionals as F
from graphlow.core.backend.base import Backend
from graphlow.core.backend.phlower import PhlowerBackend


@pytest.mark.parametrize(
    "pattern, shapes, dimensions, desired_dimension",
    (
        (
            "cf,cf->f",
            ((10, 1), (10, 1)),
            [None, None],
            None,
        ),
        (
            "cf,cf->f",
            ((10, 1), (10, 1)),
            [{"L": 2, "M": 1}, {"M": 2, "T": -1}],
            {"L": 2, "M": 3, "T": -1},
        ),
        (
            "cf,cpf->cpf",
            ((10, 1), (10, 3, 1)),
            [None, None],
            None,
        ),
        (
            "cf,cpf->cpf",
            ((10, 1), (10, 3, 1)),
            [{"L": 2, "M": 1}, {"M": 2, "T": -1}],
            {"L": 2, "M": 3, "T": -1},
        ),
        (
            "cpf,cpf->cf",
            ((10, 3, 1), (10, 3, 1)),
            [None, None],
            None,
        ),
        (
            "cpf,cpf->cf",
            ((10, 3, 1), (10, 3, 1)),
            [{"L": 2, "M": 1}, {"M": 2, "T": -1}],
            {"L": 2, "M": 3, "T": -1},
        ),
    ),
)
def test_einsum(
    pattern: str,
    shapes: tuple[tuple[int]],
    dimensions: list[dict[str, int] | None],
    desired_dimension: dict[str, int] | None,
    backend: Backend,
):
    np_arrays = [np.random.randn(*shape) for shape in shapes]
    tensors = [
        backend.as_tensor(a, dimension=d)
        for a, d in zip(np_arrays, dimensions, strict=True)
    ]
    actual = F.einsum(pattern, *tensors, dimension="auto")
    desired = np.einsum(pattern, *np_arrays)

    np.testing.assert_almost_equal(actual.to("cpu").numpy(), desired, decimal=5)
    assert actual.device.type == backend.device.type
    if isinstance(backend, PhlowerBackend):
        if desired_dimension is None:
            assert actual.dimension is None
        else:
            assert actual.dimension.to("cpu") == phlower_dimension_tensor(
                desired_dimension
            )


@pytest.mark.parametrize(
    "pattern, shapes",
    (
        ("cf,cf->f", ((10, 1), (10, 1))),
        ("cf,cpf->cpf", ((10, 1), (10, 3, 1))),
    ),
)
def test_einsum_raises_unexpected_input_type(
    pattern: str, shapes: tuple[tuple[int]]
):
    np_arrays = [np.random.randn(*shape) for shape in shapes]
    with pytest.raises(ValueError, match="Unexpected tensor"):
        F.einsum(pattern, np_arrays)


@pytest.mark.parametrize(
    "pattern, shape",
    (
        ("c f -> (f c)", (10, 1)),
        ("c i j f -> c j i f", (10, 3, 3, 1)),
    ),
)
@pytest.mark.parametrize(
    "dimension",
    (
        None,
        {"L": 1, "T": -1},
    ),
)
def test_rearrange(
    pattern: str,
    shape: tuple[int],
    dimension: dict[str, int] | None,
    backend: Backend,
):
    np_array = np.random.randn(*shape)
    tensor = backend.as_tensor(np_array, dimension=dimension)
    actual = F.rearrange(tensor, pattern=pattern)
    desired = einops.rearrange(np_array, pattern)

    np.testing.assert_almost_equal(actual.to("cpu").numpy(), desired, decimal=5)
    assert actual.device.type == backend.device.type
    if isinstance(backend, PhlowerBackend):
        assert actual.dimension == tensor.dimension


@pytest.mark.parametrize(
    "pattern, shape",
    (
        ("c f -> (f c)", (10, 1)),
        ("c i j f -> c j i f", (10, 3, 3, 1)),
    ),
)
def test_rearrange_raises_unexpected_input_type(
    pattern: str, shape: tuple[int]
):
    np_array = np.random.randn(*shape)
    with pytest.raises(ValueError, match="Unexpected tensor"):
        F.rearrange(np_array, pattern)

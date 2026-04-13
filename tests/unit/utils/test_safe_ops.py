"""Unit tests for utils.safe_ops (safe_vector_norm, safe_normalize)."""

from __future__ import annotations

import pytest
import torch

from graphlow.utils.safe_ops import (
    safe_normalize,
    safe_vector_norm,
)


@pytest.mark.parametrize(
    "input, keepdim, eps, expected",
    (
        (torch.tensor([0.0, 0.0, 0.0]), False, 1e-6, torch.tensor(1e-6)),
        (torch.tensor([0.0, 0.0, 0.0]), True, 1e-6, torch.tensor([1e-6])),
        (
            torch.tensor([[3.0, 4.0], [0.0, 0.0]]),
            False,
            1e-6,
            torch.tensor([5.0, 1e-6]),
        ),
        (
            torch.tensor([[3.0, 4.0], [0.0, 0.0]]),
            True,
            1e-6,
            torch.tensor([[5.0], [1e-6]]),
        ),
    ),
)
def test_safe_vector_norm(
    input: torch.Tensor, keepdim: bool, eps: float, expected: torch.Tensor
):
    """
    safe_vector_norm clamps norm to at least eps to avoid NaNs.
    """
    out = safe_vector_norm(input, dim=-1, keepdim=keepdim, eps=eps)
    assert out.shape == expected.shape
    assert not torch.isnan(out).any()
    assert torch.allclose(out, expected)


@pytest.mark.parametrize(
    "input, eps, expected",
    (
        (torch.tensor([3.0, 4.0]), 1e-6, torch.tensor([0.6, 0.8])),
        (
            torch.tensor([[3.0, 4.0], [0.0, 0.0]]),
            1e-6,
            torch.tensor([[0.6, 0.8], [0.0, 0.0]]),
        ),
    ),
)
def test_safe_normalize(
    input: torch.Tensor, eps: float, expected: torch.Tensor
):
    """
    safe_normalize produces vectors with norm >= eps (unit norm when non-zero).
    """
    out = safe_normalize(input, dim=-1, eps=eps)
    assert out.shape == expected.shape
    assert not torch.isnan(out).any()
    assert torch.allclose(out, expected)

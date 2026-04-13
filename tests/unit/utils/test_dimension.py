"""Unit tests for utils.dimension (get_dimension)."""

from __future__ import annotations

import phlower_tensor as pt
import pytest
import torch

from graphlow.utils.dimension import get_dimension


def test_get_dimension_torch():
    """
    get_dimension returns None for torch.Tensor
    """
    input = torch.rand((3, 4, 5))
    assert get_dimension(input) is None


@pytest.mark.parametrize(
    "input, expected",
    [
        (
            pt.phlower_tensor([1.0, 2.0, 3.0], dimension={"L": 3}),
            pt.phlower_dimension_tensor({"L": 3}),
        ),
        (pt.phlower_tensor([1.0, 2.0, 3.0], dimension=None), None),
        (
            pt.phlower_tensor([1.0, 2.0, 3.0], dimension={}),
            pt.phlower_dimension_tensor({}),
        ),
    ],
)
def test_get_dimension_phlower_tensor(
    input: pt.PhlowerTensor, expected: pt.PhlowerDimensionTensor
):
    result = get_dimension(input)
    assert result == expected

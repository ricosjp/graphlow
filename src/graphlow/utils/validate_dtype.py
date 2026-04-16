import torch


def validate_floating_point_dtype(dtype: torch.dtype) -> torch.dtype:
    """Validate that a dtype is a floating-point dtype."""
    if not dtype.is_floating_point:
        raise ValueError("dtype must be a floating-point dtype")
    return dtype

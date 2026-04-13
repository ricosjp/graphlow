from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import torch

if TYPE_CHECKING:
    from graphlow.core.backend.phlower import PhlowerBackend
    from graphlow.core.backend.torch import TorchBackend

from graphlow.utils.enums import FloatPrecision


@overload
def get_backend(
    name: Literal["torch"],
    float_precision: FloatPrecision | int = FloatPrecision.FLOAT32,
    *,
    device: torch.device | str | None = None,
) -> TorchBackend: ...


@overload
def get_backend(
    name: Literal["phlower"],
    float_precision: FloatPrecision | int = FloatPrecision.FLOAT32,
    *,
    device: torch.device | str | None = None,
) -> PhlowerBackend: ...


def get_backend(
    name: Literal["phlower", "torch"],
    float_precision: FloatPrecision | int = FloatPrecision.FLOAT32,
    *,
    device: torch.device | str | None = None,
) -> TorchBackend | PhlowerBackend:
    """
    Create a backend instance by name.

    Parameters
    ----------
    name : {"phlower", "torch"}
        Backend to use.
    float_precision : FloatPrecision or int, default=FloatPrecision.FLOAT32
        Float precision to use.
    device : torch.device | str | None
        Device (e.g. "cuda", "cpu"). Backend-specific.

    Returns
    -------
    PhlowerBackend or TorchBackend
        Backend instance matching ``name``.
    """
    if name == "phlower":
        from graphlow.core.backend.phlower import PhlowerBackend

        return PhlowerBackend(float_precision=float_precision, device=device)
    if name == "torch":
        from graphlow.core.backend.torch import TorchBackend

        return TorchBackend(float_precision=float_precision, device=device)
    raise ValueError(f"Unknown backend: {name!r}")

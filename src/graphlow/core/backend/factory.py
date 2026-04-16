from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import torch

if TYPE_CHECKING:
    from graphlow.core.backend.phlower import PhlowerBackend
    from graphlow.core.backend.torch import TorchBackend


@overload
def get_backend(
    name: Literal["torch"],
    dtype: torch.dtype = torch.float32,
    *,
    device: torch.device | str | None = None,
) -> TorchBackend: ...


@overload
def get_backend(
    name: Literal["phlower"],
    dtype: torch.dtype = torch.float32,
    *,
    device: torch.device | str | None = None,
) -> PhlowerBackend: ...


def get_backend(
    name: Literal["phlower", "torch"],
    dtype: torch.dtype = torch.float32,
    *,
    device: torch.device | str | None = None,
) -> TorchBackend | PhlowerBackend:
    """
    Create a backend instance by name.

    Parameters
    ----------
    name : {"phlower", "torch"}
        Backend to use.
    dtype : torch.dtype, default=torch.float32
        Floating-point dtype to use.
    device : torch.device | str | None
        Device (e.g. "cuda", "cpu"). Backend-specific.

    Returns
    -------
    PhlowerBackend or TorchBackend
        Backend instance matching ``name``.
    """
    if name == "phlower":
        from graphlow.core.backend.phlower import PhlowerBackend

        return PhlowerBackend(dtype=dtype, device=device)
    if name == "torch":
        from graphlow.core.backend.torch import TorchBackend

        return TorchBackend(dtype=dtype, device=device)
    raise ValueError(f"Unknown backend: {name!r}")

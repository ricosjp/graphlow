import pathlib
from enum import StrEnum

import pyvista as pv
import torch
from phlower_tensor._tensor._dimension import PhysicDimensionLikeObject

from graphlow.base.mesh import GraphlowMesh
from graphlow.util.enums import FloatPrecision


def read(
    file: str | pathlib.Path,
    dict_dimensions: dict[StrEnum, PhysicDimensionLikeObject] | None = None,
    dict_is_time_series: dict[StrEnum, bool] | None = None,
    float_precision: FloatPrecision | int = FloatPrecision.FLOAT32,
    device: torch.device | str | None = None,
) -> GraphlowMesh:
    """Read a mesh file and return a GraphlowMesh object.

    Parameters
    ----------
    file: str | pathlib.Path
        The path to the mesh file.
    dict_dimensions: typing.DictDimensions | None
        Dimensions for each key in
        dict_point_tensor, dict_cell_tensor, and dict_sparse_tensor.
    dict_is_time_series: typing.DictIsTimeSeries | None
        Specifies if the data is time series or not. Can be specified
        for each value by inputting dict[Key, bool].
    float_precision: FloatPrecision | int | None
        Float precision. 32 or 64. Default is 32.
    device: torch.device | str | None
        Device.
    """
    return GraphlowMesh(
        pv.read(file),
        dict_dimensions=dict_dimensions,
        dict_is_time_series=dict_is_time_series,
        float_precision=float_precision,
        device=device,
    )

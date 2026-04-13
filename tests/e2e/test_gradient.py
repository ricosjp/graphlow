"""End-to-end test for the volume gradient computed from cell volumes."""

import pathlib

import numpy as np
import pytest
import torch

import graphlow


@pytest.mark.parametrize(
    "filename",
    [
        # primitives
        pathlib.Path("tests/data/vtu/primitive_cell/cuboid.vtu"),
    ],
)
def test_volume_gradient(filename: pathlib.Path):
    """Volume backward pass matches per-face area accumulation."""
    volmesh = graphlow.read(filename, "phlower")
    volmesh.requires_grad(True)
    cell_volumes = volmesh.geometry.cell_volumes()
    total_volume = torch.sum(cell_volumes)
    total_volume.backward()

    vol_grad = volmesh.points.to_tensor().grad

    for i in range(volmesh.pvmesh.n_cells):
        cell = volmesh.pvmesh.get_cell(i)
        for face in cell.faces:
            pids = face.point_ids
            dV = torch.abs(torch.sum(vol_grad[pids]))
            area = (
                face.cast_to_unstructured_grid()
                .cell_quality(quality_measure="area")
                .cell_data["area"]
            )
            np.testing.assert_equal(dV.detach().numpy(), area)

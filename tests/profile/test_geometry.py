"""Profiling smoke tests for geometry routines."""

import logging
import pathlib

import pytest

import graphlow
from graphlow.utils.enums import FloatPrecision

logger = logging.getLogger(__name__)


@pytest.mark.profile
def test_centroids_profile():
    """Profile cell_centroids on a large cube mesh."""
    filename = pathlib.Path("tests/data/vtu/cube/large.vtu")
    logger.info(
        "Running profile: cell_centroids on %s (phlower, float64)", filename
    )
    mesh = graphlow.read(filename, "phlower", FloatPrecision.FLOAT64)
    mesh.geometry.cell_centroids()

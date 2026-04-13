"""Benchmarks for common geometry routines."""

import logging
import pathlib
from collections.abc import Callable, Generator

import pytest

import graphlow

logger = logging.getLogger(__name__)


def benchmark_with_group(func: Callable) -> Callable:
    """Attach a pytest-benchmark group matching the function name."""
    return pytest.mark.benchmark(group=func.__name__)(func)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "poly_ratio",
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)
@benchmark_with_group
def test_compute_areas_100x100_benchmark(benchmark: Generator, poly_ratio: int):
    """Benchmark ``face_areas`` on benchmark surface meshes."""
    file_name = pathlib.Path(
        f"tests/data/benchmark_mesh/surfaces/100x100_{poly_ratio:03d}.vtu"
    )
    surfmesh = graphlow.read(file_name, "phlower")

    def compute_areas():
        _ = surfmesh.geometry.face_areas()

    benchmark(compute_areas)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "poly_ratio",
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)
@benchmark_with_group
def test_compute_volumes_100x100_benchmark(
    benchmark: Generator, poly_ratio: int
):
    """Benchmark ``cell_volumes`` on benchmark volume meshes."""
    file_name = pathlib.Path(
        f"tests/data/benchmark_mesh/volumes/100x100_{poly_ratio:03d}.vtu"
    )
    volmesh = graphlow.read(file_name, "phlower")

    def compute_volumes():
        _ = volmesh.geometry.cell_volumes()

    benchmark(compute_volumes)

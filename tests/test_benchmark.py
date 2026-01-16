import pathlib
from collections.abc import Callable, Generator

import pytest

import graphlow
from graphlow.util.logger import get_logger

logger = get_logger(__name__)


def benchmark_with_group(func: Callable) -> Callable:
    return pytest.mark.benchmark(group=func.__name__)(func)


@pytest.mark.with_benchmark
@pytest.mark.parametrize(
    "poly_ratio",
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)
@benchmark_with_group
def test_compute_areas_100x100_benchmark(benchmark: Generator, poly_ratio: int):
    file_name = pathlib.Path(
        f"tests/data/benchmark_mesh/surfaces/100x100_{poly_ratio:03d}.vtu"
    )
    surfmesh = graphlow.read(file_name)

    def compute_areas():
        _ = surfmesh.compute_areas()

    benchmark(compute_areas)


@pytest.mark.with_benchmark
@pytest.mark.parametrize(
    "poly_ratio",
    [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
)
@benchmark_with_group
def test_compute_volumes_100x100_benchmark(
    benchmark: Generator, poly_ratio: int
):
    file_name = pathlib.Path(
        f"tests/data/benchmark_mesh/volumes/100x100_{poly_ratio:03d}.vtu"
    )
    volmesh = graphlow.read(file_name)

    def compute_volumes():
        _ = volmesh.compute_volumes()

    benchmark(compute_volumes)

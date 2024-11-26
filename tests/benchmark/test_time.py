import pathlib
from collections.abc import Generator

import pytest

import graphlow
from graphlow.util.logger import get_logger

logger = get_logger(__name__)


@pytest.mark.with_benchmark
@pytest.mark.parametrize(
    "file_name",
    sorted(
        pathlib.Path("tests/data/benchmark_mesh/surfaces").glob("100x100_*.vtu")
    ),
)
def test_compute_areas_100x100_benchmark(
    benchmark: Generator, file_name: pathlib.Path
):
    surfmesh = graphlow.read(file_name)

    def compute_areas():
        _ = surfmesh.compute_areas()

    benchmark(compute_areas)


@pytest.mark.with_benchmark
@pytest.mark.parametrize(
    "file_name",
    sorted(
        pathlib.Path("tests/data/benchmark_mesh/volumes").glob("100x100_*.vtu")
    ),
)
def test_compute_volumes_100x100_benchmark(
    benchmark: Generator, file_name: pathlib.Path
):
    volmesh = graphlow.read(file_name)

    def compute_volumes():
        _ = volmesh.compute_volumes()

    benchmark(compute_volumes)

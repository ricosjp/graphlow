import logging
from pathlib import Path

import pytest

import graphlow

logger = logging.getLogger(__name__)

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data"

# Use library logging config (works from repo and when package is installed).
graphlow.configure_logging(level="DEBUG")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom options for tests (e.g. slow-test --save)."""
    parser.addoption(
        "--save",
        action="store_true",
        default=False,
        help="Save output artifacts from slow tests.",
    )
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run tests on: cpu or cuda",
    )


@pytest.fixture
def data_dir() -> Path:
    """Root of tests/data."""
    return DATA_DIR

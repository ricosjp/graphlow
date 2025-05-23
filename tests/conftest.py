import logging.config

import pytest

fmt = (
    "%(asctime)s | %(levelname)s | %(name)s |"
    "[%(filename)s:%(lineno)d] %(message)s"
)

LOGGING_CONFIG = {
    "version": 1,
    "loggers": {
        "": {  # root
            "level": "INFO",
            "handlers": ["console_handler"],
        },
    },
    "handlers": {
        "console_handler": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simpleFormatter",
            "stream": "ext://sys.stdout",
        },
    },
    "formatters": {
        "simpleFormatter": {"format": fmt},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)


def pytest_addoption(parser):
    parser.addoption("--save", action="store_true")
    parser.addoption("--gpu", action="store_true")


@pytest.fixture
def device(pytestconfig) -> str:
    if pytestconfig.getoption("--gpu"):
        return "cuda"
    return "cpu"

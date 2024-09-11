import logging.config

from .logging_config import LOGGING_CONFIG

logging.config.dictConfig(LOGGING_CONFIG)

def pytest_addoption(parser):
    parser.addoption('--save', action='store_true')

import logging.config

import yaml

logging.config.dictConfig(yaml.load(open("src/graphlow/util/log_config.yaml").read(), yaml.FullLoader))

def pytest_addoption(parser):
    parser.addoption('--save', action='store_true')

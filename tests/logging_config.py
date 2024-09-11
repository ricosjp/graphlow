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
        "simpleFormatter": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | [%(filename)s:%(lineno)d] %(message)s"
        },
    },
}

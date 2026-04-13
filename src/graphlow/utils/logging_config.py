"""Optional logging helpers for graphlow."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

_LevelType = int | str


def configure_logging(
    level: _LevelType | None = None,
    config_file: str | Path | None = None,
) -> None:
    """
    Load bundled logging settings so ``graphlow`` log messages are easy to read.

    Call this from a small script when you want console
    output without writing a ``logging`` configuration yourself. It is not
    required for using the library, and production code should usually set up
    :mod:`logging` (or your framework's logging) instead of
    relying on this helper.

    Does not run on import so you call explicitly when you want this bundled
    behavior. If neither argument is given, loads the bundled default config
    (INFO for the ``graphlow`` logger).

    Parameters
    ----------
    level : int or str, optional
        Override level for the ``graphlow`` logger after loading config
        (e.g. ``logging.DEBUG`` or ``"DEBUG"``).
    config_file : str or Path, optional
        Path to a custom logging config file (e.g. for fileConfig).
        If None, the bundled ``utils/logging.conf`` is used.

    Returns
    -------
    None

    Examples
    --------
    >>> import graphlow
    >>> graphlow.configure_logging(level="DEBUG")
    """
    if config_file is None:
        config_file = Path(__file__).parent / "logging.conf"
    logging.config.fileConfig(
        str(config_file),
        disable_existing_loggers=False,
    )
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
        logging.getLogger("graphlow").setLevel(level)

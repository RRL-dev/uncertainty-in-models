"""Main module for configuration and logging setup."""

from __future__ import annotations

import logging
from os import getenv
from pathlib import Path
from sys import stdout
from types import SimpleNamespace
from typing import Any, ClassVar

from .io import read_file
from .param import has_methods
from .seed import set_global_seed
from .serialization import load_yaml

__all__: list[str] = ["read_file", "has_methods", "set_global_seed"]

FILE: Path = Path(__file__).resolve()
ROOT: Path = FILE.parents[1]

ConfigType = dict[str, Any] | list[Any]

MODEL_CFG_PATH: Path = ROOT / "cfg/model/base.yaml"
MODEL_CFG_DATA: ConfigType = load_yaml(file_path=MODEL_CFG_PATH.as_posix())
if not isinstance(MODEL_CFG_DATA, dict):
    msg: str = f"Expected a dictionary for model config, got {type(MODEL_CFG_DATA).__name__}"
    raise ValueError(msg)  # noqa: TRY004
MODEL_CFG = SimpleNamespace(**MODEL_CFG_DATA)

DATASET_CFG_PATH: Path = ROOT / "cfg/dataset/base.yaml"
DATASET_CFG_DATA: ConfigType = load_yaml(file_path=DATASET_CFG_PATH.as_posix())
if not isinstance(DATASET_CFG_DATA, dict):
    msg = f"Expected a dictionary for dataset config, got {type(DATASET_CFG_DATA).__name__}"
    raise ValueError(msg)  # noqa: TRY004
DATASET_CFG = SimpleNamespace(**DATASET_CFG_DATA)


class CustomFormatter(logging.Formatter):
    """Custom formatter to add colors and timestamps to logs."""

    grey: str = "\x1b[38;21m"
    green: str = "\x1b[32;21m"
    yellow: str = "\x1b[33;21m"
    red: str = "\x1b[31;21m"
    bold_red: str = "\x1b[31;1m"
    reset: str = "\x1b[0m"
    formats: ClassVar[dict[int, str]] = {
        logging.DEBUG: grey + "%(asctime)s - %(message)s" + reset,
        logging.INFO: green + "%(asctime)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(message)s" + reset,
    }

    def format(self: CustomFormatter, record: logging.LogRecord) -> str:
        """Format the log record with colors and timestamps.

        Args:
        ----
            record (logging.LogRecord): The log record to format.

        Returns:
        -------
            str: The formatted log record.

        """
        log_fmt: str | None = self.formats.get(record.levelno)
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(fmt=log_fmt, datefmt=datefmt)
        return formatter.format(record=record)


def set_logging(name="LOGGING_NAME", verbose=True) -> logging.Logger:  # noqa: FBT002, ANN001
    """Set up logging with color and timestamps, and UTF-8 encoding.

    Args:
    ----
        name (str): Name of the logger.
        verbose (bool): If True, set logging level to INFO, otherwise ERROR.

    Returns:
    -------
        logging.Logger: Configured logger instance.

    """
    level: int = logging.INFO if verbose else logging.ERROR

    # Configure the console (stdout) encoding to UTF-8, with checks for compatibility
    formatter = CustomFormatter()  # Use custom formatter with color and timestamps

    # Create and configure the StreamHandler with the appropriate formatter and level
    stream_handler = logging.StreamHandler(stream=stdout)
    stream_handler.setFormatter(fmt=formatter)
    stream_handler.setLevel(level=level)

    # Set up the logger
    logger: logging.Logger = logging.getLogger(name=name)
    logger.setLevel(level=level)
    logger.addHandler(hdlr=stream_handler)
    logger.propagate = False
    return logger


VERBOSE: bool = getenv(key="VERBOSE", default="true").lower() == "true"
LOGGER: logging.Logger = set_logging(name="Logger", verbose=VERBOSE)

"""Data reader."""

from __future__ import annotations

from logging import getLogger
from os.path import exists
from typing import TYPE_CHECKING

from .format import read_csv_format, read_xlsx_format

if TYPE_CHECKING:
    from collections.abc import Callable
    from logging import Logger

    from pandas.core.frame import DataFrame

logger: Logger = getLogger()


def read_file(file_path: str | None, file_format: str | None = "csv") -> DataFrame:
    """Read tabular data.

    Args:
    ----
        file_path (str): Path of csv or xlsx file.
        file_format (str): The file format, csv or xlsx. Defaults to 'csv'.

    Returns:
    -------
        DataFrame: Two-dimensional, size-mutable, potentially heterogeneous tabular data.

    Raises:
    ------
        ValueError: If file path is not defined or the file format is not csv or xlsx.

    """
    if file_path is None or not exists(path=file_path):  # noqa: PTH110
        msg: str = "file path not defined"
        raise ValueError(msg)

    if file_format is None:
        file_format = "csv"
        logger.warning(msg="file format is None, set as default to csv")

    file_reader: dict[str, Callable[[str], DataFrame]] = {
        "csv": read_csv_format,
        "xlsx": read_xlsx_format,
    }

    if file_format not in file_reader:
        msg = f"file format {file_format} not defined as csv or xlsx"
        raise ValueError(msg)

    if file_path.endswith("csv"):
        return file_reader[file_format](file_path)

    if file_path.endswith("xlsx"):
        return file_reader[file_format](file_path)

    msg = f"file path: {file_path} not type of csv or xlsx"
    raise ValueError(msg)

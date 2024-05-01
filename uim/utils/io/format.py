"""Data pandas reader."""

from __future__ import annotations

from typing import Any, final

from pandas.core.frame import DataFrame
from pandas.io.excel import read_excel
from pandas.io.parsers import read_csv


@final
class DataFormat:
    """Data format for pandas reader."""

    def __init__(self, func) -> None:
        super().__init__()
        self._func: Any = func

    def __call__(self, file_path: str) -> Any:
        """Read file decorator."""
        if not isinstance(file_path, str):
            raise TypeError(f"file path need to be str, got {type(file_path)}")

        if file_path.endswith(".csv"):
            if "csv" in self._func.__name__:
                return self._func(file_path)

        if file_path.endswith(".xlsx"):
            if "xlsx" in self._func.__name__:
                return self._func(file_path)

        raise NotImplementedError(
            f"pandas can not handle with file type {file_path} "
            f"with the function: {self._func.__name__}"
        )

    def __repr__(self) -> str:
        return "DataFormat"


@DataFormat
def read_csv_format(file_path: str) -> DataFrame:
    """Read csv with format checking.

    Args:
        file_path (str): path of csv file
    """
    return read_csv(filepath_or_buffer=file_path)


@DataFormat
def read_xlsx_format(file_path: str) -> DataFrame:
    """Read xlsx with format checking.

    Args:
        file_path (str): path of xlsx file
    """
    return read_excel(io=file_path)


if __name__ == "__main__":
    data: DataFrame = read_csv_format(file_path=".../data.csv")

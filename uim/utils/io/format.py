"""Data pandas reader."""

from __future__ import annotations

from typing import TYPE_CHECKING, final

from pandas.io.excel import read_excel
from pandas.io.parsers import read_csv

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandas.core.frame import DataFrame


@final
class DataFormat:
    """Data format for pandas reader."""

    def __init__(self: DataFormat, func: Callable[[str], DataFrame]) -> None:
        """Initialize the DataFormat decorator with a function.

        Args:
        ----
            func (Callable[[str], DataFrame]): The function to be decorated, which reads a file and returns a DataFrame.

        """
        super().__init__()
        self._func: Callable[[str], DataFrame] = func

    def __call__(self: DataFormat, file_path: str) -> DataFrame:
        """Call the decorated function to read the file and return a DataFrame.

        Args:
        ----
            file_path (str): Path to the file to be read.

        Returns:
        -------
            DataFrame: The data read from the file as a DataFrame.

        Raises:
        ------
            TypeError: If the file_path is not a string.
            NotImplementedError: If the file type is not supported by the decorated function.

        """
        if not isinstance(file_path, str):
            msg: str = f"file path need to be str, got {type(file_path)}"
            raise TypeError(msg)

        if file_path.endswith(".csv") and "csv" in self._func.__name__:
            return self._func(file_path)

        if file_path.endswith(".xlsx") and "xlsx" in self._func.__name__:
            return self._func(file_path)

        msg = f"pandas can not handle with file type {file_path} with the function: {self._func.__name__}"
        raise NotImplementedError(
            msg,
        )

    def __repr__(self: DataFormat) -> str:
        """Return the string representation of the DataFormat decorator.

        Returns
        -------
            str: The string representation of the DataFormat decorator.

        """
        return "DataFormat"


@DataFormat
def read_csv_format(file_path: str) -> DataFrame:
    """Read csv with format checking.

    Args:
    ----
        file_path (str): path of csv file

    """
    return read_csv(filepath_or_buffer=file_path)


@DataFormat
def read_xlsx_format(file_path: str) -> DataFrame:
    """Read xlsx with format checking.

    Args:
    ----
        file_path (str): path of xlsx file

    """
    return read_excel(io=file_path)


if __name__ == "__main__":
    data: DataFrame = read_csv_format(file_path=".../data.csv")

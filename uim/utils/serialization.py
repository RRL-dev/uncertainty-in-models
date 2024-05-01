"""Basic functionality for serialization."""

from typing import Any

from yaml import safe_load

from uim.utils.path import suffix


def load_yaml(file_path: str) -> Any | dict[Any, Any]:
    """load_yaml as simple method to load a yaml file.

    Args:
        file (str): file path of yaml file.

    Returns:
        Any | dict[Any, Any]: dictionary of yaml file structure.
    """
    assert suffix(name=file_path) == ".yaml", f"file {file_path} is not a yaml file"

    with open(file=file_path, encoding="utf-8") as obj:
        file: str = obj.read()

    data: Any | dict[Any, Any] = safe_load(stream=file) or {}
    return data

"""Basic functionality for serialization."""

from pathlib import Path

from yaml import safe_load

from uim.utils.path import suffix


def load_yaml(file_path: str) -> dict | list:
    """Load a YAML file.

    Args:
    ----
        file_path (str): File path of the YAML file.

    Returns:
    -------
        Union[dict, list]: Parsed content of the YAML file.

    Raises:
    ------
        AssertionError: If the file is not a YAML file.

    """
    if suffix(name=file_path) != ".yaml":
        msg: str = f"file {file_path} is not a yaml file"
        raise ValueError(msg)

    path = Path(file_path)
    with path.open(encoding="utf-8") as obj:
        file_content: str = obj.read()

    data: dict | list = safe_load(stream=file_content) or {}
    return data

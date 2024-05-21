"""Path utils."""


def suffix(name: str) -> str:
    """Get the suffix of a file name based on pathlib functionality.

    Args:
    ----
        name (str): The name of the file.

    Returns:
    -------
        str: The suffix of the file name, or an empty string if no suffix is found.

    Raises:
    ------
        TypeError: If the provided name is not a string.

    """
    if not isinstance(name, str):
        msg = "name is not type of string"
        raise TypeError(msg)

    value: int = name.rfind(".")
    if 0 < value < len(name) - 1:
        return name[value:]
    return ""

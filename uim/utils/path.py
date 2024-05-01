"""Path utils."""


def suffix(name: str) -> str:
    """Based on pathlib functionality."""
    if not isinstance(name, str):
        raise TypeError("name is not type of string")

    value: int = name.rfind(".")
    if 0 < value < len(name) - 1:
        return name[value:]
    else:
        return ""

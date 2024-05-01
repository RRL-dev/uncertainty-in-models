"""_Setup_"""

from re import M, Match, search
from sys import version_info

from setuptools import find_packages, setup

if version_info < (3, 10):
    raise RuntimeError("vi repository requires Python 3.10")


def find_version(file_path: str) -> str:
    """Find version.

    Args:
        file_path (str): python file.

    Raises:
        RuntimeError: not founded python file with version.

    Returns:
        str: version of data.
    """
    with open(file=file_path, encoding="utf-8") as file:
        version_file: str = file.read()
        version_match: Match[str] | None = search(
            pattern=r"^__version__ = ['\"]([^'\"]*)['\"]", string=version_file, flags=M
        )

        if not version_match:
            raise RuntimeError(f"Unable to find a version string in {file_path}")
        return version_match.group(1)


VERSION: str = find_version(file_path="_version.py")

requirements: list[str] = [
    "matplotlib==3.8.4",
    "pandas",
    "pyyaml",
    "scikit-learn>=1.4.1.post1",
]


if __name__ == "__main__":
    setup(
        name="vi",
        version=VERSION,
        description="",
        author="Roni Reznik",
        author_email="reznik.roni@gmail.com",
        packages=find_packages(),
        install_requires=requirements,
    )

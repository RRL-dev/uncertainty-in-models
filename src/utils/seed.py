"""Reproducibility utils for Deep learning."""

from __future__ import annotations

from logging import getLogger
from os import environ
from random import seed as rnd_seed
from typing import TYPE_CHECKING

from numpy import random

if TYPE_CHECKING:
    from logging import Logger

logger: Logger = getLogger(name=__name__)


def set_global_seed(seed: int) -> None:
    """Set seed to enable reproducible result.

    Args:
        seed (int): number of randomness block.
    """
    rnd_seed(a=seed)
    environ["PYTHONHASHSEED"] = str(object=seed)
    random.seed(seed=seed)

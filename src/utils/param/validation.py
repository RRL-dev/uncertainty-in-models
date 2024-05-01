from typing import Any


def has_methods(estimator: Any, methods: list[str]) -> bool:
    """
    Check if the estimator has all the provided methods.

    Args:
        estimator (BaseEstimator): The scikit-learn estimator to check.
        methods (list[str]): A list of method names as strings to check for.

    Returns:
        bool: True if all methods are present, False otherwise.
    """
    return all(hasattr(estimator, method) for method in methods)

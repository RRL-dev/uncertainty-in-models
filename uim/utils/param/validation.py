"""The module defines the has_methods function."""


def has_methods(estimator: object, methods: list[str]) -> bool:
    """Check if the estimator has all the provided methods.

    Args:
    ----
        estimator (object): The scikit-learn estimator to check.
        methods (list[str]): A list of method names as strings to check for.

    Returns:
    -------
        bool: True if all methods are present, False otherwise.

    """
    return all(hasattr(estimator, method) for method in methods)

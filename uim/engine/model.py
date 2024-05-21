"""The module defines the FittableClassifier protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from numpy import ndarray
    from pandas import DataFrame, Series


@runtime_checkable
class FittableClassifier(Protocol):
    """A protocol that requires fit, predict, and predict_proba methods."""

    def fit(self: FittableClassifier, x: ndarray | DataFrame, y: ndarray | Series[Any]) -> FittableClassifier:
        """Fit the classifier to the training data.

        Args:
        ----
            x (ndarray | DataFrame): Training data features.
            y (ndarray | Series[Any]): Training data labels.

        Returns:
        -------
            FittableClassifier: The instance itself, to allow for method chaining.

        """
        ...

    def predict(self: FittableClassifier, x: ndarray | DataFrame) -> ndarray:
        """Predict class labels for samples in X.

        Args:
        ----
            x (ndarray | DataFrame): Input data for which predictions are to be made.

        Returns:
        -------
            ndarray: Array of predicted class labels.

        """
        ...

    def predict_proba(self: FittableClassifier, x: ndarray | DataFrame) -> ndarray:
        """Predict class probabilities for samples in X.

        Args:
        ----
            x (ndarray | DataFrame): Input data for which probability predictions are to be made.

        Returns:
        -------
            ndarray: Array of predicted class probabilities.

        """
        ...

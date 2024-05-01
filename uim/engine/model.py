from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from numpy import ndarray
    from pandas import Series
    from pandas.core.frame import DataFrame


@runtime_checkable
class FittableClassifier(Protocol):
    def fit(
        self, X: ndarray | DataFrame, y: ndarray | Series[Any]
    ) -> "FittableClassifier":
        """
        Fit the classifier to the training data.

        Args:
            X (ndarray): Training data features.
            y (ndarray | DataFrame | Series[Any]): Training data labels.

        Returns:
            FittableClassifier: The instance itself, to allow for method chaining.
        """
        ...

    def predict(self, X: ndarray | DataFrame) -> ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X (ndarray | DataFrame): Input data for which predictions are to be made.

        Returns:
            ndarray: Array of predicted class labels.
        """
        ...

    def predict_proba(self, X: ndarray | DataFrame) -> ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X (ndarray | DataFrame): Input data for which probability predictions are to be made.

        Returns:
            ndarray: Array of predicted class probabilities.
        """
        ...

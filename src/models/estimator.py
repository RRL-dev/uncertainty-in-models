from __future__ import annotations
from typing import Any

from numpy import array, ndarray
from pandas import Series
from pandas.core.frame import DataFrame
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.engine import FittableClassifier
from src.utils import LOGGER, has_methods


class EstimatorWithCalibration(BaseEstimator):
    """
    Integrates a classifier with a calibration mechanism using logistic regression
    to provide calibrated probability estimates.

    Attributes:
        _clf (FittableClassifier): The primary classifier used for initial predictions and training.
    """


    def __init__(self) -> None:
        """
        Initializes the EstimatorWithCalibration with a specific classifier.
        """
        self.clf: FittableClassifier

    def fit(
        self,
        samples_train: ndarray | DataFrame,
        samples_calib: ndarray | DataFrame,
        targets_train: ndarray | Series[Any],
        targets_calib: ndarray | Series[Any],
    ) -> "EstimatorWithCalibration":
        """
        Fit the primary classifier and calibrate it using the provided training and calibration datasets.

        Args:
            samples_train (ndarray | DataFrame): The input features for training the primary classifier.
            samples_calib (ndarray | DataFrame): The input features for calibrating the model.
            targets_train (ndarray | Series[Any]): The target labels for training the primary classifier.
            targets_calib (ndarray | Series[Any]): The target labels for calibration.

        Returns:
            EstimatorWithCalibration: Returns itself to allow for method chaining.

        Raises:
            ValueError: If no classifier has been set.
            AttributeError: If the classifier does not implement necessary methods (predict_proba or predict).
        """
        self.clf = RandomForestClassifier()

        if self.clf is None:
            raise ValueError("No classifier has been set.")

        if not has_methods(estimator=self.clf, methods=["predict_proba", "predict"]):
            raise AttributeError(
                f"{self.clf.__class__.__name__} do not contain predict_proba or predict attribute"
            )

        self.clf.fit(X=samples_train, y=targets_train)
        self.calibrate(samples_calib=samples_calib, targets_calib=targets_calib)
        return self

    def calibrate(
        self, samples_calib: ndarray | DataFrame, targets_calib: ndarray | Series[Any]
    ) -> None:
        """
        Calibrates the classifier using logistic regression based on the probabilities of the initial classifier.

        Args:
            samples_calib (ndarray | DataFrame): Calibration samples.
            targets_calib (ndarray | DataFrame): Calibration targets.
        """
        unclib_proba: ndarray[Any, Any] = self.clf.predict_proba(X=samples_calib)

        self.calibrator = LogisticRegression(penalty=None)
        self.calibrator.fit(X=unclib_proba[:, 1].reshape(-1, 1), y=targets_calib)

    def predict(self, samples: ndarray | DataFrame) -> ndarray:
        """
        Predict class labels for the given samples using the calibrated model.

        Args:
            samples (ndarray | DataFrame): Samples for which to predict labels.

        Returns:
            ndarray: Array of predicted class labels.
        """
        if self.calibrator:
            unclib_proba: ndarray[Any, Any] = self.clf.predict_proba(X=samples)
            return self.calibrator.predict(X=unclib_proba)
        LOGGER.warning(msg="Calibrator not set. Returning raw predictions.")
        return self.clf.predict(X=samples)

    def predict_proba(self, samples: ndarray | DataFrame) -> ndarray[Any, Any]:
        """
        Predict class probabilities for the given samples using the calibrated model.

        Args:
            samples (ndarray | DataFrame): Samples for which to predict probabilities.

        Returns:
            ndarray: Array of predicted class probabilities.
        """
        if self.calibrator:
            unclib_proba: ndarray[Any, Any] = self.clf.predict_proba(X=samples)
            return self.calibrator.predict_proba(X=unclib_proba[:, -1].reshape(-1, 1))
        LOGGER.warning(
            msg="Calibrator not set. Cannot provide calibrated probabilities."
        )
        return array(object=[], dtype=float)

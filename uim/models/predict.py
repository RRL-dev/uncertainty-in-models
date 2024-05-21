"""The module defines the BasePredictor class."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np

from uim.modules import BaseRiskScore, CalibrationVisualizer
from uim.utils import LOGGER, MODEL_CFG

if TYPE_CHECKING:
    from types import SimpleNamespace

    from numpy import dtype, ndarray
    from pandas import DataFrame, Series

    from uim.engine import FittableClassifier


class BasePredictor(BaseRiskScore, CalibrationVisualizer):
    """A class to encapsulate the prediction process using a trained model loaded from a pickle file."""

    def __init__(self: BasePredictor, model_cfg: SimpleNamespace, show_calibration: bool = True) -> None:  # noqa: FBT001, FBT002
        """BasePredictor with the path to the trained model pickle file.

        Args:
        ----
            model_cfg (SimpleNamespace): Configuration namespace for the model.
            show_calibration (bool): If True, show the calibration curve.

        """
        super().__init__()
        self._model_cfg: SimpleNamespace = model_cfg
        self.show_calibration: bool = show_calibration
        self.model_path: str = self._model_cfg.model["path"]

        self.samples: ndarray | DataFrame
        self.targets: ndarray | Series[Any]
        self.estimator: FittableClassifier
        self.load_model()

    def load_model(self: BasePredictor) -> None:
        """Trained model from the pickle file."""
        model_full_path: str = os.path.join(  # noqa: PTH118
            self.model_path, self._model_cfg.model["type"], "trained_model.pkl"
        )

        if os.path.exists(path=self.model_path):  # noqa: PTH110
            self.estimator, self.samples, self.targets = joblib.load(filename=model_full_path)
            LOGGER.info(msg=f"Model loaded from {model_full_path}")
        else:
            LOGGER.error(msg=f"No model found at {model_full_path}. Please check the file path.")

    def predict_proba(self: BasePredictor, X: np.ndarray | DataFrame) -> list[int] | ndarray[Any, dtype[Any]]:  # noqa: N803
        """Predictions using the loaded model.

        Args:
        ----
            X (np.ndarray | DataFrame): Input features for prediction.

        Returns:
        -------
            list[int] | ndarray[Any, dtype[Any]]: Predicted labels.

        """
        if self.estimator is not None:
            probabilities: ndarray[Any, Any] = self.estimator.predict_proba(X)
            if self.show_calibration:
                self.evaluate_calibration(true_labels=self.targets, predicted_probabilities=probabilities)
            return self.fit(probabilities=probabilities)

        LOGGER.error(msg="Model not loaded. Unable to make predictions.")
        return np.array(object=[])


# Usage
if __name__ == "__main__":
    predictor = BasePredictor(model_cfg=MODEL_CFG)
    score: list[int] | ndarray[Any, dtype[Any]] = predictor.predict_proba(
        X=predictor.samples,
    )

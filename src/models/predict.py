from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import joblib
from numpy import dtype, ndarray
from pandas import Series

from src.modules import BaseRiskScore, CalibrationVisualizer
from src.utils import LOGGER, MODEL_CFG

if TYPE_CHECKING:
    import numpy as np
    from pandas import DataFrame

    from src.engine import FittableClassifier


class BasePredictor(BaseRiskScore, CalibrationVisualizer):
    """
    A class to encapsulate the prediction process using a trained model loaded from a pickle file.
    """

    def __init__(
        self, model_cfg: SimpleNamespace, show_calibration: bool = True
    ) -> None:
        """
        Initializes the BasePredictor with the path to the trained model pickle file.

        Args:
            model_path (str): Path to the trained model pickle file.
            show_calibration (bool): If to show calibration curve.
        """
        self._model_cfg: SimpleNamespace = model_cfg
        self.show_calibration: bool = show_calibration
        self.model_path: str = self._model_cfg.model["path"]

        self.samples: ndarray | DataFrame
        self.targets: ndarray | Series[Any]
        self.estimator: FittableClassifier
        self.load_model()

    def load_model(self) -> None:
        """
        Loads the trained model from the pickle file.
        """
        model_full_path: str = os.path.join(
            self.model_path, self._model_cfg.model["type"], "trained_model.pkl"
        )

        if os.path.exists(path=self.model_path):
            self.estimator, self.samples, self.targets = joblib.load(
                filename=model_full_path
            )
            LOGGER.info(msg=f"Model loaded from {model_full_path}")
        else:
            LOGGER.error(
                msg=f"No model found at {model_full_path}. Please check the file path."
            )

    def predict_proba(
        self, X: np.ndarray | DataFrame
    ) -> list[int] | ndarray[Any, dtype[Any]]:
        """
        Makes predictions using the loaded model.

        Args:
            X (np.ndarray): Input features for prediction.

        Returns:
            np.ndarray: Predicted labels.
        """
        if self.estimator is not None:
            probabilities: ndarray[Any, Any] = self.estimator.predict_proba(X)
            if self.show_calibration:
                self.evaluate_calibration(
                    true_labels=self.targets, predicted_probabilities=probabilities
                )
            return self.fit(probabilities=probabilities)

        else:
            LOGGER.error(msg="Model not loaded. Unable to make predictions.")
            return np.array(object=[])


# Usage
if __name__ == "__main__":
    predictor = BasePredictor(model_cfg=MODEL_CFG)
    score: list[int] | ndarray[Any, dtype[Any]] = predictor.predict_proba(
        X=predictor.samples
    )

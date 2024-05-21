"""The module defines the BaseTrainer class."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import joblib

from uim.dataset import DerivedFeaturesTransformer
from uim.models import EstimatorWithCalibration
from uim.modules import split_train_calib_test
from uim.utils import DATASET_CFG, LOGGER, MODEL_CFG, set_global_seed

if TYPE_CHECKING:
    from types import SimpleNamespace

    from numpy import ndarray
    from pandas import DataFrame, Series


set_global_seed(seed=42)


class BaseTrainer:
    """A class to encapsulate the training process of a model with configuration loaded from a YAML file.

    This class handles loading data, training a model with specified parameters, and saving the model
    to a directory based on its type.
    """

    def __init__(self: BaseTrainer, model_cfg: SimpleNamespace, data_cfg: SimpleNamespace) -> None:
        """Model trainer with a configuration path.

        Args:
        ----
            model_cfg (SimpleNamespace): Configuration namespace for the model.
            data_cfg (SimpleNamespace): Configuration namespace for the data.

        """
        self._data_cfg: SimpleNamespace = data_cfg
        self._model_cfg: SimpleNamespace = model_cfg

        self.model_path: str = os.path.join(self._model_cfg.model["path"], self._model_cfg.model["type"])  # noqa: PTH118
        os.makedirs(name=self.model_path, exist_ok=True)  # Create the directory if it doesn't exist  # noqa: PTH103

    def _fit(self: BaseTrainer) -> None:
        """Configure the estimator for training."""
        self.estimator = EstimatorWithCalibration()

    def _fit_transform(
        self: BaseTrainer,
    ) -> tuple[DataFrame, DataFrame, DataFrame, Series[Any], Series[Any], Series[Any]]:
        """Fit and transform the data.

        Returns
        -------
            tuple[DataFrame, DataFrame, DataFrame, Series[Any], Series[Any], Series[Any]]: Split data into training, calibration, and testing sets.

        """  # noqa: E501
        data: DataFrame = DerivedFeaturesTransformer(cfg=self._data_cfg).fit_transform()

        traget_name: str = self._data_cfg.features["target"]["name"]

        targets: Series[Any] = data[traget_name]
        samples: DataFrame = data.drop(labels=[traget_name], axis=1)

        (
            samples_train,
            samples_calib,
            samples_test,
            targets_train,
            targets_calib,
            targets_test,
        ) = split_train_calib_test(targets=targets, samples=samples)
        return (
            samples_train,
            samples_calib,
            samples_test,
            targets_train,
            targets_calib,
            targets_test,
        )

    def fit(self: BaseTrainer) -> None:
        """Train the model using parameters specified in the YAML configuration file and save the trained model."""
        self._fit()

        (
            samples_train,
            samples_calib,
            samples_test,
            targets_train,
            targets_calib,
            targets_test,
        ) = self._fit_transform()

        self.estimator.fit(
            samples_train=samples_train,
            samples_calib=samples_calib,
            targets_train=targets_train,
            targets_calib=targets_calib,
        )
        LOGGER.info(msg="Training complete. Model is trained and ready to be saved.")
        self.save_model(samples=samples_test, targets=targets_test)

    def fit_predict(self: BaseTrainer) -> tuple[ndarray[Any, Any], Series[Any]]:
        """Train the model and return predictions.

        Returns
        -------
            tuple[ndarray[Any, Any], Series[Any]]: Predicted probabilities and target labels for the test set.

        """
        self._fit()

        (
            samples_train,
            samples_calib,
            samples_test,
            targets_train,
            targets_calib,
            targets_test,
        ) = self._fit_transform()

        self.estimator.fit(
            samples_train=samples_train,
            samples_calib=samples_calib,
            targets_train=targets_train,
            targets_calib=targets_calib,
        )

        LOGGER.info(msg="Training complete. Model is trained and ready to be saved.")

        probabilities: ndarray[Any, Any] = self.estimator.predict_proba(samples=samples_test)
        return probabilities, targets_test

    def save_model(self: BaseTrainer, samples: ndarray | DataFrame, targets: ndarray | Series[Any]) -> None:
        """Save the trained model to the specified path in the configuration file."""
        if self.estimator is not None:
            model_filename: str = os.path.join(self.model_path, "trained_model.pkl")  # noqa: PTH118
            joblib.dump(value=(self.estimator, samples, targets), filename=model_filename)
            LOGGER.info(msg=f"Model saved to {model_filename}")
        else:
            LOGGER.info(msg="No model to save. Please train the model first.")


# Usage
if __name__ == "__main__":
    BaseTrainer(model_cfg=MODEL_CFG, data_cfg=DATASET_CFG).fit()

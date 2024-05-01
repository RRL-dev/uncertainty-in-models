from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import joblib
from numpy import ndarray
from pandas.core.series import Series

from uim.dataset import DerivedFeaturesTransformer
from uim.models import EstimatorWithCalibration
from uim.modules import split_train_calib_test
from uim.utils import DATASET_CFG, LOGGER, MODEL_CFG, set_global_seed

if TYPE_CHECKING:
    from pandas import Series
    from pandas.core.frame import DataFrame


set_global_seed(seed=42)


class BaseTrainer:
    """
    A class to encapsulate the training process of a model with configuration loaded from a YAML file.

    This class handles loading data, training a model with specified parameters, and saving the model
    to a directory based on its type.
    """

    def __init__(self, model_cfg: SimpleNamespace, data_cfg: SimpleNamespace):
        """
        Initializes the ModelTrainer with a configuration path.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self._data_cfg: SimpleNamespace = data_cfg
        self._model_cfg: SimpleNamespace = model_cfg

        self.model_path: str = os.path.join(
            self._model_cfg.model["path"], self._model_cfg.model["type"]
        )
        os.makedirs(
            name=self.model_path, exist_ok=True
        )  # Create the directory if it doesn't exist

    def _fit(self) -> None:
        # Configure the RandomForestClassifier with parameters from the YAML file
        self.estimator = EstimatorWithCalibration()

    def _fit_transform(
        self,
    ) -> tuple[DataFrame, DataFrame, DataFrame, Series[Any], Series[Any], Series[Any]]:
        data: DataFrame = DerivedFeaturesTransformer(cfg=self._data_cfg).fit_transform()

        traget_name: str = self._data_cfg.features["target"]["name"]

        targets: Series[Any] = data[traget_name]
        samples: DataFrame = data.drop(labels=[traget_name], axis=1)

        samples_train: DataFrame
        samples_calib: DataFrame
        samples_test: DataFrame
        targets_train: Series[Any]
        targets_calib: Series[Any]
        targets_test: Series[Any]

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

    def fit(self) -> None:
        """
        Trains the model using parameters specified in the YAML configuration file and saves the trained model.
        """
        self._fit()

        samples_train: DataFrame
        samples_calib: DataFrame
        samples_test: DataFrame
        targets_train: Series[Any]
        targets_calib: Series[Any]
        targets_test: Series[Any]

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

    def fit_predict(self) -> tuple[ndarray[Any, Any], Series[Any]]:
        self._fit()

        samples_train: DataFrame
        samples_calib: DataFrame
        samples_test: DataFrame
        targets_train: Series[Any]
        targets_calib: Series[Any]
        targets_test: Series[Any]

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

        probabilities: ndarray[Any, Any] = self.estimator.predict_proba(
            samples=samples_test
        )
        return probabilities, targets_test

    def save_model(
        self, samples: ndarray | DataFrame, targets: ndarray | Series[Any]
    ) -> None:
        """
        Saves the trained model to the specified path in the configuration file.
        """
        if self.estimator is not None:
            model_filename: str = os.path.join(self.model_path, "trained_model.pkl")
            joblib.dump(
                value=(self.estimator, samples, targets), filename=model_filename
            )
            LOGGER.info(msg=f"Model saved to {model_filename}")
        else:
            LOGGER.info(msg="No model to save. Please train the model first.")


# Usage
if __name__ == "__main__":
    BaseTrainer(model_cfg=MODEL_CFG, data_cfg=DATASET_CFG).fit()

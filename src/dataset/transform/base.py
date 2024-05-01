from types import SimpleNamespace
from typing import Any

import pandas as pd
from numpy import ndarray
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import DATASET_CFG, LOGGER, read_file


class BaseTransformer:
    """
    Base class for transforming base features. This class handles the initialization,
    reading of data, and transformation (including encoding and normalization) of
    base features defined in the configuration.

    Attributes:
        _cfg (SimpleNamespace): Configuration for the transformer.
        encoders (dict[str, OneHotEncoder]): Dictionary to store OneHotEncoder objects for categorical features.
        scalers (dict[str, StandardScaler]): Dictionary to store StandardScaler objects for numerical features.
        data (pd.DataFrame): DataFrame holding the data to be transformed.
    """

    def __init__(self, cfg: SimpleNamespace) -> None:
        """
        Initialize the BaseTransformer with a given configuration.

        Args:
            cfg (SimpleNamespace): Configuration for the transformer, containing data paths and feature specifications.
        """
        self._cfg: SimpleNamespace = cfg
        self.encoders: dict[str, OneHotEncoder] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.data: pd.DataFrame = pd.DataFrame()

        self._read_file()

    def _read_file(self) -> None:
        """
        Reads the file from the path specified in the configuration.
        """
        LOGGER.info(msg="Start reading file")
        file_path: str = self._cfg.data["file_path"]
        LOGGER.info(msg=f"File path: {file_path}")
        self.data = read_file(
            file_path=file_path, file_format=self._cfg.data["file_format"]
        )
        LOGGER.info(msg="Successfully read file")

    def fit_transform(self) -> pd.DataFrame:
        """
        Fits and transforms the data based on the feature configurations.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        LOGGER.info(msg="Start transforming the data")
        for feature, settings in self._cfg.features.items():
            try:
                if settings["type"] == "categorical":
                    self._encode_feature(feature=feature, settings=settings)
                elif settings["type"] == "numerical":
                    self._normalize_feature(feature=feature, settings=settings)
            except Exception as e:
                LOGGER.error(msg=f"Error processing feature {feature}: {e}")
                raise
        return self.data

    def _encode_feature(self, feature: str, settings: dict) -> None:
        """
        Encodes a categorical feature using OneHotEncoding.

        Args:
            feature (str): The name of the feature to encode.
            settings (dict): Settings that specify encoding details.
        """
        if "encoding" in settings and settings["encoding"] == "onehot":
            if feature not in self.encoders:
                self.encoders[feature] = OneHotEncoder(sparse_output=False)
            transformed: ndarray[Any, Any] = self.encoders[feature].fit_transform(
                X=self.data[[feature]]
            )
            features: ndarray[Any, Any] = self.encoders[feature].get_feature_names_out(
                input_features=[feature]
            )
            encoded_df = pd.DataFrame(
                data=transformed, columns=features, index=self.data.index
            )
            self.data = pd.concat([self.data, encoded_df], axis=1)
            self.data.drop(columns=feature, inplace=True)

    def _normalize_feature(self, feature: str, settings: dict) -> None:
        """
        Normalizes a numerical feature using StandardScaler.

        Args:
            feature (str): The name of the feature to normalize.
            settings (dict): Settings that specify normalization details.
        """
        if "normalization" in settings and settings["normalization"] == "standard":
            try:
                if feature not in self.scalers:
                    self.scalers[feature] = StandardScaler()
                self.data[feature] = (
                    self.scalers[feature].fit_transform(self.data[[feature]]).flatten()
                )
            except KeyError:
                LOGGER.error(
                    msg=f"Feature '{feature}' not found in the DataFrame columns."
                )
                raise


if __name__ == "__main__":
    data: pd.DataFrame = BaseTransformer(cfg=DATASET_CFG).fit_transform()

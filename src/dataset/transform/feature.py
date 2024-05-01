from types import SimpleNamespace

import pandas as pd

from src.utils import DATASET_CFG, LOGGER

from .base import BaseTransformer


class DerivedFeaturesTransformer(BaseTransformer):
    """
    Class for transforming derived features. Extends the BaseTransformer by adding functionality
    to compute features derived from the base features using custom calculations.

    Attributes:
        derived_features (set[str]): Set to track the names of derived features.
    """

    def __init__(self, cfg: SimpleNamespace) -> None:
        """
        Initialize the DerivedFeaturesTransformer with a given configuration.

        Args:
            cfg (SimpleNamespace): Configuration for the transformer, including settings for derived features.
        """
        super().__init__(cfg=cfg)
        self.derived_features: set[str] = set()

    def fit_transform(self) -> pd.DataFrame:
        """
        Extends fit_transform to include processing of derived features.

        Returns:
            DataFrame: The DataFrame with transformed and derived features.
        """
        super().fit_transform()
        for feature, settings in self._cfg.features.items():
            if "derived_from" in settings:
                self.create_derived_feature(feature=feature, settings=settings)
        self._prepare_weekly_churn_target()
        return self.data

    def _prepare_weekly_churn_target(self) -> None:
        """
        Prepares the target variable for weekly churn prediction.
        """
        if (
            "ChurnIn30Days" not in self.data.columns
            or "MemberID" not in self.data.columns
        ):
            LOGGER.error("ChurnIn30Days or MemberID column not found in the data.")
            return

        # Convert the date column to datetime type
        self.data["EffectiveDate"] = pd.to_datetime(arg=self.data["EffectiveDate"])

        # Group by MemberID and week, and aggregate churn
        weekly_churn: pd.DataFrame = self.data.groupby(
            by=["MemberID", pd.Grouper(key="EffectiveDate", freq="W")]
        ).agg(func={"ChurnIn30Days": "max"})

        # Reset the index to flatten the DataFrame
        weekly_churn.reset_index(inplace=True)

        # Create a new target variable indicating churn in each week
        self.data["ChurnThisWeek"] = 0  # Initialize as 0
        self.data.loc[
            self.data["MemberID"].isin(
                values=weekly_churn.loc[weekly_churn["ChurnIn30Days"] == 1, "MemberID"]
            ),
            "ChurnThisWeek",
        ] = 1

        self.data.drop(labels=["ChurnIn30Days", "EffectiveDate"], axis=1, inplace=True)
        LOGGER.info(msg="Weekly churn target prepared successfully")

    def create_derived_feature(self, feature: str, settings: dict):
        """
        Creates and adds a derived feature to the DataFrame.

        Args:
            feature (str): Name of the feature to derive.
            settings (dict): Configuration including the calculation formula.
        """
        try:
            self.data[feature] = self.data.eval(expr=settings["calculation"])
            LOGGER.info(msg=f"Successfully derived feature: {feature}")
        except Exception as e:
            LOGGER.error(msg=f"Failed to create derived feature {feature}: {e}")
            raise


if __name__ == "__main__":
    data: DataFrame = DerivedFeaturesTransformer(cfg=DATASET_CFG).fit_transform()

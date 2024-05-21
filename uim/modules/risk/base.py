"""The module defines the BaseRiskScore class and its methods."""

from __future__ import annotations

from typing import Any

import numpy as np

from uim.utils import LOGGER


class BaseRiskScore:
    """A class to compute risk scores based on predicted probabilities of customer churn using fuzzy set theory.

    The risk levels are categorized into three groups: low, medium, and high risk, each defined by a specific
    membership function. These functions assign a membership score from 0 to 1, indicating the degree to which
    an observation (churn probability) belongs to a fuzzy risk category.

    - Low Risk: Defined for churn probabilities ≤ 0.3 with a linear membership function decreasing from 1 to 0
      as probability increases from 0 to 0.3.
    - Medium Risk: Defined for churn probabilities between 0.2 and 0.7 with a triangular membership function
      peaking at 0.45.
    - High Risk: Defined for churn probabilities ≥ 0.6 with a linear membership function increasing from 0 to 1
      as probability increases from 0.6 to 0.75.

    """

    def _low_risk(self: BaseRiskScore, score: float) -> float:
        """Compute the membership degree for low risk based on churn probability."""
        return max(0, min(1, (0.3 - score) / 0.3))

    def _medium_risk(self: BaseRiskScore, score: float) -> float:
        """Compute the membership degree for medium risk based on churn probability."""
        if score <= 0.2:  # noqa: PLR2004
            return 0
        elif score <= 0.45:  # noqa: PLR2004, RET505
            return (score - 0.2) / 0.25
        elif score <= 0.7:  # noqa: PLR2004
            return (0.7 - score) / 0.25
        else:
            return 0

    def _high_risk(self: BaseRiskScore, score: float) -> float:
        """Compute the membership degree for high risk based on churn probability."""
        return max(0, min(1, (score - 0.6) / 0.15))

    def fit(
        self: BaseRiskScore,
        probabilities: np.ndarray[Any, Any] | float | int | list[float],  # noqa: PYI041 , PGH003# type: ignore
    ) -> list[int]:
        """Determine the risk level for one or more customers based on their churn probabilities."""
        if isinstance(probabilities, float | int):  # Single probability passed in
            probabilities: list[float] | list[int] = [probabilities]

        elif isinstance(probabilities, np.ndarray):
            if probabilities.ndim > 1:
                probabilities = probabilities[:, 1]  # type: ignore
            probabilities = probabilities.tolist()  # type: ignore  # noqa: PGH003

        risk_levels: list[int] = []
        for score in probabilities:
            low_risk_score: float = self._low_risk(score=score)
            medium_risk_score: float = self._medium_risk(score=score)
            high_risk_score: float = self._high_risk(score=score)

            if high_risk_score >= 0.5:  # noqa: PLR2004
                risk_levels.append(2)  # High Risk
            elif medium_risk_score >= low_risk_score:
                risk_levels.append(1)  # Medium Risk
            else:
                risk_levels.append(0)  # Low Risk

        return risk_levels


if __name__ == "__main__":
    # Example usage
    risk_score_calculator = BaseRiskScore()
    single_risk: list[int] = risk_score_calculator.fit(probabilities=0.55)
    batch_risk: list[int] = risk_score_calculator.fit(
        probabilities=np.array(object=[0.2, 0.4, 0.6, 0.8]),
    )
    LOGGER.info(msg=f"Single Risk Level: {single_risk}")
    LOGGER.info(msg=f"Batch Risk Levels: {batch_risk}")

"""The module defines the RiskCurve class and its methods."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class RiskCurve:
    """A class dedicated to plotting membership functions for different risk levels based on churn probabilities.

    This class visualizes the fuzzy membership functions for low, medium, and high risk, helping in understanding
    the decision boundaries and membership degrees for each risk category.

    """

    LOW_RISK_THRESHOLD = 0.3
    MEDIUM_RISK_START = 0.2
    MEDIUM_RISK_PEAK = 0.45
    MEDIUM_RISK_END = 0.7
    HIGH_RISK_START = 0.6
    HIGH_RISK_END = 0.75

    def __init__(self: RiskCurve) -> None:  # noqa: D107
        self.probabilities: np.ndarray = np.linspace(start=0, stop=1, num=100)
        self.low_risk: np.ndarray = self.calculate_low_risk()
        self.medium_risk: np.ndarray = self.calculate_medium_risk()
        self.high_risk: np.ndarray = self.calculate_high_risk()

    def calculate_low_risk(self: RiskCurve) -> np.ndarray:
        """Calculate the membership values for low risk across a range of probabilities."""
        return np.maximum(0, 1 - self.probabilities / self.LOW_RISK_THRESHOLD)

    def calculate_medium_risk(self: RiskCurve) -> np.ndarray:
        """Calculate the membership values for medium risk across a range of probabilities."""
        return np.where(
            self.probabilities <= self.MEDIUM_RISK_START,
            0,
            np.where(
                self.probabilities <= self.MEDIUM_RISK_PEAK,
                (self.probabilities - self.MEDIUM_RISK_START) / (self.MEDIUM_RISK_PEAK - self.MEDIUM_RISK_START),
                np.where(
                    self.probabilities <= self.MEDIUM_RISK_END,
                    (self.MEDIUM_RISK_END - self.probabilities) / (self.MEDIUM_RISK_END - self.MEDIUM_RISK_PEAK),
                    0,
                ),
            ),
        )

    def calculate_high_risk(self: RiskCurve) -> np.ndarray:
        """Calculate the membership values for high risk across a range of probabilities."""
        return np.maximum(
            0, np.minimum(1, (self.probabilities - self.HIGH_RISK_START) / (self.HIGH_RISK_END - self.HIGH_RISK_START))
        )

    def plot_membership_functions(self: RiskCurve) -> None:
        """Plot the membership functions for each risk level."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.probabilities, self.low_risk, "b", label="Low Risk", linewidth=2)
        plt.plot(self.probabilities, self.medium_risk, "g", label="Medium Risk", linewidth=2)
        plt.plot(self.probabilities, self.high_risk, "r", label="High Risk", linewidth=2)
        plt.title(label="Membership Functions for Risk Levels")
        plt.xlabel(xlabel="Probability")
        plt.ylabel(ylabel="Membership Degree")
        self.add_vertical_lines()
        plt.legend()
        plt.grid(visible=True)
        plt.show()

    def add_vertical_lines(self: RiskCurve) -> None:
        """Add vertical lines to the plot to mark key thresholds."""
        plt.axvline(x=self.LOW_RISK_THRESHOLD, linestyle="--", color="gray")
        plt.axvline(x=self.MEDIUM_RISK_START, linestyle="--", color="gray")
        plt.axvline(x=self.MEDIUM_RISK_PEAK, linestyle="--", color="gray")
        plt.axvline(x=self.MEDIUM_RISK_END, linestyle="--", color="gray")
        plt.axvline(x=self.HIGH_RISK_START, linestyle="--", color="gray")
        plt.axvline(x=self.HIGH_RISK_END, linestyle="--", color="gray")


if __name__ == "__main__":
    # Example of usage:
    RiskCurve().plot_membership_functions()

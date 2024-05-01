import matplotlib.pyplot as plt
import numpy as np


class RiskCurve:
    """
    A class dedicated to plotting membership functions for different risk levels based on churn probabilities.

    This class visualizes the fuzzy membership functions for low, medium, and high risk, helping in understanding
    the decision boundaries and membership degrees for each risk category.
    """

    def __init__(self) -> None:
        self.probabilities: np.ndarray = np.linspace(start=0, stop=1, num=100)
        self.low_risk: np.ndarray = self.calculate_low_risk()
        self.medium_risk: np.ndarray = self.calculate_medium_risk()
        self.high_risk: np.ndarray = self.calculate_high_risk()

    def calculate_low_risk(self) -> np.ndarray:
        """Calculate the membership values for low risk across a range of probabilities."""
        return np.maximum(0, 1 - self.probabilities / 0.3)

    def calculate_medium_risk(self) -> np.ndarray:
        """Calculate the membership values for medium risk across a range of probabilities."""
        return np.where(
            self.probabilities <= 0.2,
            0,
            np.where(
                self.probabilities <= 0.45,
                (self.probabilities - 0.2) / 0.25,
                np.where(
                    self.probabilities <= 0.7, (0.7 - self.probabilities) / 0.25, 0
                ),
            ),
        )

    def calculate_high_risk(self) -> np.ndarray:
        """Calculate the membership values for high risk across a range of probabilities."""
        return np.maximum(0, np.minimum(1, (self.probabilities - 0.6) / 0.15))

    def plot_membership_functions(self) -> None:
        """Plot the membership functions for each risk level."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.probabilities, self.low_risk, "b", label="Low Risk", linewidth=2)
        plt.plot(
            self.probabilities, self.medium_risk, "g", label="Medium Risk", linewidth=2
        )
        plt.plot(
            self.probabilities, self.high_risk, "r", label="High Risk", linewidth=2
        )
        plt.title(label="Membership Functions for Risk Levels")
        plt.xlabel(xlabel="Probability")
        plt.ylabel(ylabel="Membership Degree")
        self.add_vertical_lines()
        plt.legend()
        plt.grid(visible=True)
        plt.show()

    def add_vertical_lines(self) -> None:
        """Add vertical lines to the plot to mark key thresholds."""
        plt.axvline(x=0.3, linestyle="--", color="gray")
        plt.axvline(x=0.2, linestyle="--", color="gray")
        plt.axvline(x=0.45, linestyle="--", color="gray")
        plt.axvline(x=0.7, linestyle="--", color="gray")
        plt.axvline(x=0.6, linestyle="--", color="gray")
        plt.axvline(x=0.75, linestyle="--", color="gray")


if __name__ == "__main__":
    # Example of usage:
    RiskCurve().plot_membership_functions()

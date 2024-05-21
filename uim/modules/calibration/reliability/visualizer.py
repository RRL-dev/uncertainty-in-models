"""The module defines the CalibrationVisualizer class and its methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss

from uim.utils import LOGGER

from .base import CalibrationReliability

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from pandas import Series


class CalibrationVisualizer(CalibrationReliability):
    """A class to visualize calibration of classifiers, using reliability diagrams and Expected Calibration Error (ECE) calculations.

    Attributes
    ----------
        estimator (EstimatorWithCalibration): An instance of a calibrated estimator.

    """  # noqa: E501

    def plot_reliability_diagram(self: CalibrationVisualizer, bin_data: dict[str, Any]) -> None:
        """Plot a reliability diagram based on calibration data.

        Args:
        ----
            bin_data (dict): Data dictionary containing binning results from calibration computations.

        """
        ax: Axes

        _, ax = plt.subplots(figsize=(8, 8))
        self._reliability_diagram_subplot(ax=ax, bin_data=bin_data)
        plt.show()

    def evaluate_calibration(
        self: CalibrationVisualizer,
        true_labels: np.ndarray | Series,
        predicted_probabilities: np.ndarray,
        num_bins: int = 10,
    ) -> None:
        """Evaluate and visualize the calibration of the estimator.

        Args:
        ----
            true_labels (np.ndarray | Series): True labels of the data.
            predicted_probabilities (np.ndarray): Probabilities as predicted by the estimator.
            num_bins (int): The number of bins to use for calibration evaluation.

        """
        bin_data: dict[str, Any] = self.compute_calibration(
            probabilities=predicted_probabilities,
            targets=true_labels,
            num_bins=num_bins,
        )

        LOGGER.info(
            msg=f"Negative Log Likelihood: {log_loss(y_true=true_labels, y_pred=predicted_probabilities[:, 1])}"
        )

        LOGGER.info(msg=f"Expected Calibration Error: {bin_data['expected_calibration_error']}")
        LOGGER.info(msg=f"Maximum Calibration Error: {bin_data['max_calibration_error']}")

        self.plot_reliability_diagram(bin_data=bin_data)

    def _reliability_diagram_subplot(  # noqa: PLR0913
        self: CalibrationVisualizer,
        ax: Axes,
        bin_data: dict[str, Any],
        draw_ece: bool = True,  # noqa: FBT001, FBT002
        draw_bin_importance: str | None = None,
        title: str = "Reliability Diagram",
        xlabel: str = "Confidence",
        ylabel: str = "Expected Accuracy",
    ) -> None:
        """Draw a reliability diagram into a subplot.

        Args:
        ----
            ax (Axes): The axes on which to draw the reliability diagram.
            bin_data (dict): Data dictionary containing the calibration analysis results.
            draw_ece (bool): Whether to display the Expected Calibration Error (ECE) on the diagram.
            draw_bin_importance (str | None): Modifies the display of bins to reflect their 'importance',
                where 'importance' can be visualized by varying 'alpha' levels or 'width' of the bins.
                Options are "alpha", "width", or None.
            title (str): Title of the plot.
            xlabel (str): Label for the X-axis, representing confidence levels.
            ylabel (str): Label for the Y-axis, representing accuracy levels.

        """
        bins: np.ndarray = bin_data["bins"]
        counts: np.ndarray = bin_data["bin_counts"]
        accuracies: np.ndarray = bin_data["bin_accuracies"]
        confidences: np.ndarray = bin_data["bin_confidences"]

        bin_size: float = 1.0 / len(counts)
        positions: np.ndarray = bins + bin_size / 2.0

        widths: float = bin_size
        alphas = 0.3  # Default alpha transparency for bars
        min_count: np.float64 = np.min(a=counts)
        max_count: np.float64 = np.max(a=counts)
        normalized_counts: np.ndarray = (counts - min_count) / (max_count - min_count)

        if draw_bin_importance == "alpha":
            alphas = 0.2 + 0.8 * normalized_counts
        elif draw_bin_importance == "width":
            widths = 0.1 * bin_size + 0.9 * bin_size * normalized_counts  # type: ignore  # noqa: PGH003

        colors: np.ndarray = np.zeros(shape=(len(counts), 4))
        colors[:, 0] = 240 / 255.0  # Red color
        colors[:, 1] = 60 / 255.0  # Green color
        colors[:, 2] = 60 / 255.0  # Blue color
        colors[:, 3] = alphas  # Alpha transparency

        # Draw bars for the gaps
        gap_plt: BarContainer = ax.bar(
            x=positions,
            height=np.abs(accuracies - confidences),
            bottom=np.minimum(accuracies, confidences),
            width=widths,
            edgecolor=colors,
            color=colors,
            linewidth=1,
            label="Gap",
        )

        # Draw bars for accuracies
        acc_plt: BarContainer = ax.bar(
            x=positions,
            height=0,
            bottom=accuracies,
            width=widths,
            edgecolor="black",
            color="black",
            alpha=1.0,
            linewidth=3,
            label="Accuracy",
        )

        ax.set_aspect(aspect="equal")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

        if draw_ece:
            ece: np.float64 = bin_data["expected_calibration_error"] * 100
            ax.text(
                x=0.98,
                y=0.02,
                s=f"ECE={ece:.2f}%",
                color="black",
                ha="right",
                va="bottom",
                transform=ax.transAxes,
            )

        ax.set_xlim(left=0, right=1)
        ax.set_ylim(bottom=0, top=1)
        ax.set_title(label=title)
        ax.set_xlabel(xlabel=xlabel)
        ax.set_ylabel(ylabel=ylabel)
        ax.legend(handles=[gap_plt, acc_plt])
        plt.show()

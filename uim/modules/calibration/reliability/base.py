"""The module defines the CalibrationReliability class and its methods."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .statistic import binary_binned_statistics

if TYPE_CHECKING:
    from pandas import Series


class CalibrationReliability:
    """A class to calculate calibration of classifiers, using reliability and Expected Calibration Error (ECE) calculations."""  # noqa: E501

    def compute_calibration(
        self: CalibrationReliability,
        probabilities: np.ndarray[Any, Any],
        targets: np.ndarray[Any, Any] | Series,
        num_bins: int = 21,
    ) -> dict[str, Any]:
        """Compute and return calibration data.

        Args:
        ----
            probabilities (np.ndarray): Probabilities predicted by the model.
            targets (np.ndarray[Any, Any] | Series): True labels of the data.
            num_bins (int): Number of bins for grouping probability predictions.

        Returns:
        -------
            dict[str, Any]: A dictionary containing calibration data including:
                - 'bins': np.ndarray containing the bin edges.
                - 'bin_targets': np.ndarray containing the mean target values in each bin.
                - 'bin_accuracies': np.ndarray containing the accuracy of the predicted labels in each bin.
                - 'bin_confidences': np.ndarray containing the mean probabilities in each bin.
                - 'avg_accuracy': np.floating representing the average accuracy across all bins.
                - 'avg_confidence': np.floating representing the average confidence across all bins.
                - 'max_calibration_error': np.float64 representing the maximum calibration error.
                - 'expected_calibration_error': np.float64 representing the expected calibration error.

        """
        stat_bins: dict[str, Any] = binary_binned_statistics(
            probabilities=probabilities,
            targets=targets,
            num_bins=num_bins,
        )

        avg_accuracy: np.floating[Any] = np.average(a=stat_bins["bin_accuracies"], weights=stat_bins["bin_counts"])

        avg_confidence: np.floating[Any] = np.average(a=stat_bins["bin_confidences"], weights=stat_bins["bin_counts"])

        ece: np.float64 = np.sum(
            a=stat_bins["bin_counts"] * np.abs(stat_bins["bin_accuracies"] - stat_bins["bin_confidences"])
        ) / np.sum(a=stat_bins["bin_counts"])

        mce: np.float64 = np.max(a=np.abs(stat_bins["bin_accuracies"] - stat_bins["bin_confidences"]))

        calib_result: dict[str, Any] = {}
        calib_result.update(**stat_bins)
        calib_result.update({"avg_accuracy": avg_accuracy})
        calib_result.update({"avg_confidence": avg_confidence})
        calib_result.update({"max_calibration_error": mce})
        calib_result.update({"expected_calibration_error": ece})
        return calib_result

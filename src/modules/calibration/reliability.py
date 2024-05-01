from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pandas import Series

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CalibrationReliability:
    """
    A class to calculate calibration of classifiers, using reliability and
    Expected Calibration Error (ECE) calculations.
    """

    def binary_binned_statistics(
        self,
        probabilities: np.ndarray[Any, Any],
        targets: np.ndarray[Any, Any] | Series,
        num_bins: int = 21,
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.floating[Any]]],
        np.ndarray[Any, np.dtype[Any]],
        np.ndarray[Any, np.dtype[Any]],
        np.ndarray[Any, np.dtype[Any]],
    ]:
        """
        Compute the mean target values, counts, and mean probabilities within binned ranges of probabilities.

        Args:
            probabilities (np.ndarray): The probabilities for which binning and statistics are to be calculated.
            targets (np.ndarray[Any, Any] | Series): The target labels corresponding to the probabilities.
            num_bins (int): The number of bins to use for dividing the range of probabilities.

        Returns:
            bins, bin_counts, bin_accuracies, bin_confidences: Arrays containing bin edges, counts, accuracies, and mean probabilities per bin.
        """
        confidences: np.ndarray[Any, Any] = (
            probabilities[:, 1] if probabilities.ndim > 1 else probabilities
        )
        pred_labels: np.ndarray[Any, Any] = (
            np.argmax(a=probabilities, axis=1)
            if probabilities.ndim > 1
            else np.round(a=probabilities)
        )
        bins: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.linspace(
            start=0, stop=1, num=num_bins + 1
        )
        digitized = np.digitize(x=confidences, bins=bins) - 1
        bin_counts: np.ndarray[Any, np.dtype[Any]] = np.array(
            object=[np.sum(a=digitized == i) for i in range(num_bins)]
        )
        bin_accuracies: np.ndarray[Any, np.dtype[Any]] = np.array(
            object=[
                np.mean(targets[digitized == i] == pred_labels[digitized == i])
                if bin_counts[i] > 0
                else 0
                for i in range(num_bins)
            ]
        )
        bin_confidences: np.ndarray[Any, np.dtype[Any]] = np.array(
            object=[
                np.mean(a=confidences[digitized == i]) if bin_counts[i] > 0 else 0
                for i in range(num_bins)
            ]
        )

        return bins, bin_counts, bin_accuracies, bin_confidences

    def compute_calibration(
        self,
        probabilities: np.ndarray[Any, Any],
        targets: np.ndarray[Any, Any] | Series,
        num_bins: int = 21,
    ) -> dict[str, Any]:
        """
        Computes and returns calibration data.

        Args:
            probabilities (np.ndarray): Probabilities predicted by the model.
            targets (np.ndarray[Any, Any] | Series): True labels of the data.
            num_bins (int): Number of bins for grouping probability predictions.

        Returns:
            A dictionary containing calibration data including bins, counts, and errors.
        """
        bins: NDArray[Any]
        bin_counts: NDArray[Any]
        bin_accuracies: NDArray[Any]
        bin_confidences: NDArray[Any]

        bins, bin_counts, bin_accuracies, bin_confidences = (
            self.binary_binned_statistics(
                probabilities=probabilities, targets=targets, num_bins=num_bins
            )
        )
        avg_accuracy: np.floating[Any] = np.average(
            a=bin_accuracies, weights=bin_counts
        )
        avg_confidence: np.floating[Any] = np.average(
            a=bin_confidences, weights=bin_counts
        )
        ece: np.float64 = np.sum(
            a=bin_counts * np.abs(bin_accuracies - bin_confidences)
        ) / np.sum(a=bin_counts)
        mce: np.float64 = np.max(a=np.abs(bin_accuracies - bin_confidences))

        return {
            "bins": bins,
            "counts": bin_counts,
            "accuracies": bin_accuracies,
            "confidences": bin_confidences,
            "avg_accuracy": avg_accuracy,
            "avg_confidence": avg_confidence,
            "expected_calibration_error": ece,
            "max_calibration_error": mce,
        }

"""The module defines the binary_binned_statistics function."""

from typing import Any

import numpy as np
from pandas.core.series import Series


def binary_binned_statistics(
    probabilities: np.ndarray[Any, Any],
    targets: np.ndarray[Any, Any] | Series,
    num_bins: int = 21,
) -> dict[str, Any]:
    """Compute the mean target values, counts, and mean probabilities within binned ranges of probabilities.

    Args:
    ----
        probabilities (np.ndarray): The probabilities for which binning and statistics are to be calculated.
        targets (np.ndarray[Any, Any] | Series): The target labels corresponding to the probabilities.
        num_bins (int): The number of bins to use for dividing the range of probabilities.

    Returns:
    -------
        dict[str, Any]: A dictionary containing:
            - 'bins': np.ndarray containing the bin edges.
            - 'bin_targets': np.ndarray containing the mean target values in each bin.
            - 'bin_accuracies': np.ndarray containing the accuracy of the predicted labels in each bin.
            - 'bin_confidences': np.ndarray containing the mean probabilities in each bin.

    """
    confidences: np.ndarray[Any, Any] = probabilities[:, 1] if probabilities.ndim > 1 else probabilities
    pred_labels: np.ndarray[Any, Any] = (
        np.argmax(a=probabilities, axis=1) if probabilities.ndim > 1 else np.round(a=probabilities)
    )

    bins: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.linspace(start=0, stop=1, num=num_bins + 1)

    digitized: np.ndarray[Any, np.dtype[np.signedinteger[Any]]] = np.searchsorted(a=bins[1:-1], v=confidences)
    bin_counts: np.ndarray[Any, np.dtype[np.signedinteger[Any]]] = np.bincount(digitized, minlength=len(bins))

    bin_acc: np.ndarray[Any, np.dtype[np.signedinteger[Any]]] = np.bincount(
        digitized,
        weights=(targets == pred_labels),
        minlength=len(bins),
    )

    bin_prob: np.ndarray[Any, np.dtype[np.signedinteger[Any]]] = np.bincount(
        digitized,
        weights=confidences,
        minlength=len(bins),
    )

    bin_true: np.ndarray[Any, np.dtype[np.signedinteger[Any]]] = np.bincount(
        digitized,
        weights=targets,
        minlength=len(bins),
    )

    nonzero: np.ndarray = bin_counts != 0
    bin_targets = bin_true[nonzero] / bin_counts[nonzero]
    bin_accuracies = bin_acc[nonzero] / bin_counts[nonzero]
    bin_confidences = bin_prob[nonzero] / bin_counts[nonzero]

    return {
        "bins": bins[nonzero],
        "bin_counts": bin_counts[nonzero],
        "bin_targets": bin_targets,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
    }

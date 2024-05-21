"""The module defines the RBF kernel function."""

import numpy as np
from numpy.typing import NDArray


def rbf_kernel(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    variance: float = 1.0,
    length_scale: float = 1.0,
) -> NDArray[np.float64]:
    """Compute the Radial Basis Function (RBF) kernel (also known as the Gaussian kernel) between two sets of input points.

    The RBF kernel is defined as:
    K(x, y) = variance * exp(-0.5 / length_scale^2 * ||x - y||^2)

    Args:
    ----
        x1 (NDArray[np.float64]): First set of input points, shape (n1, d), where n1 is the number of points and d is the dimensionality.
        x2 (NDArray[np.float64]): Second set of input points, shape (n2, d), where n2 is the number of points and d is the dimensionality.
        variance (float): Variance parameter for the RBF kernel, controls the height of the kernel. Default is 1.0.
        length_scale (float): Length scale parameter for the RBF kernel, controls the width of the kernel. Default is 1.0.

    Returns:
    -------
        NDArray[np.float64]: Kernel matrix, shape (n1, n2), where each element (i, j) represents the RBF kernel value between X1[i] and X2[j].

    """  # noqa: E501
    sqdist = np.sum(a=x1**2, axis=1).reshape(-1, 1) + np.sum(a=x2**2, axis=1) - 2 * np.dot(a=x1, b=x2.T)
    return variance * np.exp(-0.5 / length_scale**2 * sqdist)

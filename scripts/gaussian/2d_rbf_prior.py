"""Plot of 2d rbd prior dimension of 2d gaussian."""  # noqa: N999

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from uim.models.gaussian import rbf_kernel


def plot_gp_prior_samples(  # noqa: PLR0913
    start: float = -5.0,
    stop: float = 5.0,
    num_points: int = 50,
    num_samples: int = 3,
    variance: float = 1.0,
    length_scale: float = 1.0,
) -> None:
    """Generate and plot samples from a Gaussian Process prior using the RBF kernel.

    :param start: The starting value of the input range.
    :param stop: The stopping value of the input range.
    :param num_points: The number of points in the input range.
    :param num_samples: The number of samples to draw from the Gaussian Process prior.
    :param variance: Variance parameter for the RBF kernel.
    :param length_scale: Length scale parameter for the RBF kernel.
    """
    np.random.seed(seed=42)  # noqa: NPY002

    # Generate 1D input points
    x_points: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.linspace(
        start=start,
        stop=stop,
        num=num_points,
    ).reshape(-1, 1)

    # Define the mean of the Gaussian process (zero mean)
    mean: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.zeros(shape=(num_points,))

    # Compute the covariance matrix using the RBF kernel
    covariance: np.ndarray[Any, np.dtype[np.floating[Any]]] = rbf_kernel(
        x1=x_points,
        x2=x_points,
        variance=variance,
        length_scale=length_scale,
    )

    # Sample from the multivariate normal distribution
    prior_samples: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.random.multivariate_normal(  # noqa: NPY002
        mean=mean,
        cov=covariance,
        size=num_samples,
    )

    # Plot the samples
    plt.figure(figsize=(10, 6))
    for i in range(num_samples):
        plt.plot(x_points, prior_samples[i, :], label=f"Sample {i+1}")
    plt.xlim(start, stop)
    plt.ylim(-3, 3)
    plt.xlabel(xlabel="x")
    plt.ylabel(ylabel="f(x)")
    plt.title(label="Samples from the GP prior")
    plt.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    plot_gp_prior_samples()

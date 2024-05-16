from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal

if TYPE_CHECKING:
    from matplotlib.contour import QuadContourSet
    from scipy.stats._multivariate import multivariate_normal_frozen


def plot_2d_gaussian_index() -> None:
    """
    Plot a 2D Gaussian distribution with a sampled point and its values across dimensions.
    """
    # Define the covariance matrix Σ
    sigma: NDArray[np.float64] = np.array(
        object=[[1.0, 0.7], [0.7, 1.0]], dtype=np.float64
    )

    # Define the mean vector (center of the Gaussian)
    mean: NDArray[np.float64] = np.zeros(shape=2, dtype=np.float64)

    # Create a grid for plotting the bivariate distribution
    y1: NDArray[np.float64]
    y2: NDArray[np.float64]
    y1, y2 = np.meshgrid(
        np.linspace(start=-3, stop=3, num=100), np.linspace(start=-3, stop=3, num=100)
    )
    position: NDArray[np.float64] = np.dstack(tup=(y1, y2))

    # Compute the Gaussian PDF values for the grid points
    rv: multivariate_normal_frozen = multivariate_normal(mean=mean, cov=sigma)
    pdf_values: NDArray[np.float64] = rv.pdf(x=position)

    # Sample one point from the multivariate Gaussian distribution
    np.random.seed(seed=42)
    sample: NDArray[np.float64] = np.random.multivariate_normal(
        mean=mean, cov=sigma, size=1
    ).ravel()

    # Create the plot
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(14, 7), gridspec_kw={"width_ratios": [2, 1]}
    )
    fig.suptitle("2D Gaussian Distribution and Sampled Point", fontsize=16)

    # Plot the bivariate distribution for y1 and y2
    contour: QuadContourSet = ax[0].contourf(
        y1, y2, pdf_values, cmap="viridis", alpha=0.7
    )
    ax[0].set_xlabel("$y_1$", fontsize=14)
    ax[0].set_ylabel("$y_2$", fontsize=14)
    ax[0].scatter(sample[0], sample[1], s=100, color="red", label="Sampled Point")
    ax[0].set_xlim([-3, 3])
    ax[0].set_ylim([-3, 3])
    ax[0].legend(loc="upper right", fontsize=12)
    fig.colorbar(
        mappable=contour, ax=ax[0], orientation="vertical", label="Probability Density"
    )

    # Plot the values of the sampled point as a function of their index
    ax[1].plot([1, 2], sample, "o-", markersize=8, color="dodgerblue")
    ax[1].set_xlabel("Variable Index", fontsize=14)
    ax[1].set_ylabel("Value", fontsize=14)
    ax[1].set_ylim([-3, 3])
    ax[1].set_xticks([1, 2])
    ax[1].grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.show()


if __name__ == "__main__":
    # Call the function to plot
    plot_2d_gaussian_index()

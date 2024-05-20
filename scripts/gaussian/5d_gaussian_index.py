from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

if TYPE_CHECKING:
    from matplotlib.contour import QuadContourSet
    from numpy.typing import NDArray
    from scipy.stats._multivariate import multivariate_normal_frozen


def plot_5d_gaussian_index() -> None:
    """Plot a 5D Gaussian distribution with a sampled point and its values across dimensions."""
    # Define the covariance matrix Σ
    sigma: NDArray[np.float64] = np.array(
        object=[
            [1.0, 0.9, 0.8, 0.6, 0.4],
            [0.9, 1.0, 0.9, 0.8, 0.6],
            [0.8, 0.9, 1.0, 0.9, 0.8],
            [0.6, 0.8, 0.9, 1.0, 0.9],
            [0.4, 0.6, 0.8, 0.9, 1.0],
        ],
        dtype=np.float64,
    )

    # Define the mean vector (center of the Gaussian)
    mean: NDArray[np.float64] = np.zeros(shape=5, dtype=np.float64)

    # Sample one point from the multivariate Gaussian distribution
    np.random.seed(seed=42)  # noqa: NPY002
    sample: NDArray[np.float64] = np.random.multivariate_normal(mean=mean, cov=sigma, size=1).ravel()  # noqa: NPY002

    # Create a grid for plotting the bivariate distribution of y1 and y5
    y1: NDArray[np.float64]
    y5: NDArray[np.float64]
    y1, y5 = np.meshgrid(np.linspace(start=-3, stop=3, num=100), np.linspace(start=-3, stop=3, num=100))
    position: NDArray[np.float64] = np.dstack(tup=(y1, y5))

    # Extract the submatrix for y1 and y5
    sigma_sub: NDArray[np.float64] = sigma[[0, 4], :][:, [0, 4]]
    mean_sub: NDArray[np.float64] = mean[[0, 4]]

    # Compute the Gaussian PDF values for the grid points
    rv: multivariate_normal_frozen = multivariate_normal(mean=mean_sub, cov=sigma_sub)  # type: ignore  # noqa: PGH003
    pdf_values: NDArray[np.float64] = rv.pdf(x=position)

    # Create the plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), gridspec_kw={"width_ratios": [2, 1]})

    fig.suptitle(t="5D Gaussian Distribution and Sampled Point", fontsize=16)

    # Plot the bivariate distribution for y1 and y5
    contour: QuadContourSet = ax[0].contourf(y1, y5, pdf_values, cmap="viridis", alpha=0.7)

    ax[0].set_xlabel("$y_1$", fontsize=14)
    ax[0].set_ylabel("$y_5$", fontsize=14)
    ax[0].scatter(sample[0], sample[4], s=100, color="red", label="Sampled Point")
    ax[0].set_xlim([-3, 3])
    ax[0].set_ylim([-3, 3])
    ax[0].legend(loc="upper right", fontsize=12)
    fig.colorbar(mappable=contour, ax=ax[0], orientation="vertical", label="Probability Density")

    # Plot the values of the sampled point as a function of their index
    ax[1].plot(range(1, 6), sample, "o-", markersize=8, color="dodgerblue")
    ax[1].set_xlabel("Variable Index", fontsize=14)
    ax[1].set_ylabel("Value", fontsize=14)
    ax[1].set_ylim([-3, 3])
    ax[1].set_xticks(range(1, 6))
    ax[1].grid(True, which="both", linestyle="--", linewidth=0.5)  # noqa: FBT003

    plt.show()


if __name__ == "__main__":
    # Call the function to plot
    plot_5d_gaussian_index()

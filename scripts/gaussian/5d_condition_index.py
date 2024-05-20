from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

if TYPE_CHECKING:
    from matplotlib.contour import QuadContourSet
    from numpy.typing import NDArray
    from scipy.stats._multivariate import multivariate_normal_frozen


def plot_conditional_index() -> None:
    """Plot the conditional distribution of x_2, x_3, x_4, and x_5 given x_1 from a 5D Gaussian distribution.

    This function:
    1. Defines a 5x5 covariance matrix and a mean vector.
    2. Samples a point from the 5D Gaussian distribution.
    3. Computes the conditional mean and covariance of x_2, x_3, x_4, and x_5 given x_1.
    4. Samples from the conditional distribution.
    5. Plots the bivariate distribution of x_1 and x_5, highlighting the sampled point.
    6. Plots the values of the conditional sample as a function of their index, highlighting the given value of x_1.

    Returns
    -------
        None

    """
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
    sample: NDArray[np.float64] = np.random.multivariate_normal(  # noqa: NPY002
        mean=mean,
        cov=sigma,
        size=1,
    ).ravel()

    # Given value for x_1
    x1: float = sample[0]
    given_value: NDArray[np.float64] = np.array(object=[x1])

    # Extract submatrices for conditional distribution
    sigma_aa: NDArray[np.float64] = sigma[1:, 1:]
    sigma_ab: NDArray[np.float64] = sigma[1:, 0]
    sigma_bb: float = sigma[0, 0]

    mean_a: NDArray[np.float64] = mean[1:]
    mean_b: float = mean[0]

    # Compute the conditional mean and covariance
    sigma_bb_inv: float = 1 / sigma_bb  # Inverse of a scalar
    mean_a_given_b: NDArray[np.float64] = mean_a + sigma_ab * sigma_bb_inv * (given_value - mean_b)
    sigma_a_given_b: NDArray[np.float64] = sigma_aa - np.outer(a=sigma_ab, b=sigma_ab) * sigma_bb_inv

    # Sample from the conditional distribution
    conditional_sample: NDArray[np.float64] = np.random.multivariate_normal(  # noqa: NPY002
        mean=mean_a_given_b,
        cov=sigma_a_given_b,
        size=1,
    ).ravel()

    # Create the plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(t="Conditional Distribution $p(x_2, x_3, x_4, x_5 | x_1)$", fontsize=16)

    # Plot the bivariate distribution for y1 and y5
    y1: NDArray[np.float64]
    y5: NDArray[np.float64]
    y1, y5 = np.meshgrid(np.linspace(start=-3, stop=3, num=100), np.linspace(start=-3, stop=3, num=100))

    position: NDArray[np.float64] = np.dstack(tup=(y1, y5))

    sigma_sub: NDArray[np.float64] = sigma[[0, 4], :][:, [0, 4]]
    mean_sub: NDArray[np.float64] = mean[[0, 4]]

    rv: multivariate_normal_frozen = multivariate_normal(mean=mean_sub, cov=sigma_sub) # type: ignore  # noqa: PGH003

    pdf_values: NDArray[np.float64] = rv.pdf(x=position)
    contour: QuadContourSet = ax[0].contourf(y1, y5, pdf_values, cmap="viridis", alpha=0.7)

    ax[0].set_xlabel("$x_1$", fontsize=14)
    ax[0].set_ylabel("$x_5$", fontsize=14)
    ax[0].scatter(sample[0], sample[4], s=100, color="red", label="Sampled Point")
    ax[0].set_xlim([-3, 3])
    ax[0].set_ylim([-3, 3])
    ax[0].legend(loc="upper right", fontsize=12)
    fig.colorbar(contour, ax=ax[0], orientation="vertical", label="Probability Density")

    # Plot the values of the conditional sample as a function of their index
    indices: list[int] = [1, 2, 3, 4, 5]
    conditional_values: NDArray[np.float64] = np.insert(arr=conditional_sample, obj=0, values=x1)

    ax[1].plot(indices, conditional_values, "o-", markersize=8, color="dodgerblue")
    ax[1].scatter(1, x1, s=100, color="red", label="$x_1$ (Given)")
    ax[1].set_xlabel("Variable Index", fontsize=14)
    ax[1].set_ylabel("Value", fontsize=14)
    ax[1].set_ylim([-3, 3])
    ax[1].set_xticks(indices)
    ax[1].grid(True, which="both", linestyle="--", linewidth=0.5)  # noqa: FBT003
    ax[1].legend(loc="upper right", fontsize=12)

    plt.show()


if __name__ == "__main__":
    # Call the function to plot
    plot_conditional_index()

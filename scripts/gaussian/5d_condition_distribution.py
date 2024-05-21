"""Plot of condition gaussian distribution."""  # noqa: N999

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

if TYPE_CHECKING:
    from matplotlib.contour import QuadContourSet
    from numpy.typing import NDArray
    from scipy.stats._multivariate import multivariate_normal_frozen


def plot_conditional_distribution() -> None:
    """Plot the conditional distribution of x_1 given x_2, x_3, x_4, and x_5."""
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
    mean: NDArray[np.float64] = np.zeros(5, dtype=np.float64)

    # Sample one point from the multivariate Gaussian distribution
    np.random.seed(seed=42)  # noqa: NPY002
    sample: NDArray[np.float64] = np.random.multivariate_normal(  # noqa: NPY002
        mean=mean, cov=sigma, size=1
    ).ravel()

    # Given values for x_2, x_3, x_4, and x_5
    x2: float = sample[1]
    x3: float = sample[2]
    x4: float = sample[3]
    x5: float = sample[4]
    given_values: NDArray[np.float64] = np.array(object=[x2, x3, x4, x5])

    # Extract submatrices for conditional distribution
    sigma_aa: float = sigma[0, 0]
    sigma_ab: NDArray[np.float64] = sigma[0, 1:]
    sigma_bb: NDArray[np.float64] = sigma[1:, 1:]
    sigma_ba: NDArray[np.float64] = sigma[1:, 0]

    mean_a: float = mean[0]
    mean_b: NDArray[np.float64] = mean[1:]

    # Compute the conditional mean and covariance
    sigma_bb_inv: NDArray[np.float64] = np.linalg.inv(a=sigma_bb)
    mean_a_given_b: float = float(
        mean_a + sigma_ab @ sigma_bb_inv @ (given_values - mean_b),
    )

    sigma_a_given_b: float = float(sigma_aa - sigma_ab @ sigma_bb_inv @ sigma_ba)

    # Create a range of x_1 values
    x1_range: NDArray[np.float64] = np.linspace(start=-3, stop=3, num=100)
    conditional_pdf: NDArray[np.float64] = multivariate_normal.pdf(
        x=x1_range,
        mean=mean_a_given_b,
        cov=sigma_a_given_b,  # type: ignore  # noqa: PGH003
    )

    # Create the plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7), gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(t="Conditional Distribution $p(x_1 | x_2, x_3, x_4, x_5)$", fontsize=16)

    # Plot the bivariate distribution for y1 and y5
    y1: NDArray[np.float64]
    y5: NDArray[np.float64]
    y1, y5 = np.meshgrid(np.linspace(start=-3, stop=3, num=100), np.linspace(start=-3, stop=3, num=100))

    position: NDArray[np.float64] = np.dstack((y1, y5))
    sigma_sub: NDArray[np.float64] = sigma[[0, 4], :][:, [0, 4]]
    mean_sub: NDArray[np.float64] = mean[[0, 4]]

    rv: multivariate_normal_frozen = multivariate_normal(mean=mean_sub, cov=sigma_sub)  # type: ignore  # noqa: PGH003
    pdf_values: NDArray[np.float64] = rv.pdf(position)

    contour: QuadContourSet = ax[0].contourf(
        y1,
        y5,
        pdf_values,
        cmap="viridis",
        alpha=0.7,
    )

    ax[0].set_xlabel("$x_1$", fontsize=14)
    ax[0].set_ylabel("$x_5$", fontsize=14)
    ax[0].scatter(sample[0], sample[4], s=100, color="red", label="Sampled Point")
    ax[0].set_xlim([-3, 3])
    ax[0].set_ylim([-3, 3])
    ax[0].legend(loc="upper right", fontsize=12)
    fig.colorbar(mappable=contour, ax=ax[0], orientation="vertical", label="Probability Density")

    # Plot the conditional distribution
    ax[1].plot(x1_range, conditional_pdf, "b-", lw=2, label="$p(x_1 | x_2, x_3, x_4, x_5)$")

    ax[1].set_xlabel("$x_1$", fontsize=14)
    ax[1].set_ylabel("Density", fontsize=14)
    ax[1].legend(loc="upper right", fontsize=12)
    ax[1].grid(True, which="both", linestyle="--", linewidth=0.5)  # noqa: FBT003

    plt.show()


if __name__ == "__main__":
    # Call the function to plot
    plot_conditional_distribution()

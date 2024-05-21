"""The module defines the GaussianProcess class and its methods."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .kernel import rbf_kernel


class GaussianProcess:
    """A class to implement Gaussian Process regression."""

    def __init__(self: GaussianProcess, noise: float = 1e-10, variance: float = 1.0, length_scale: float = 1.0) -> None:
        """Initialize the Gaussian Process.

        Args:
        ----
            noise (float): Variance of the observation noise. Default is 1e-10.
            variance (float): Variance parameter for the RBF kernel. Default is 1.0.
            length_scale (float): Length scale parameter for the RBF kernel. Default is 1.0.

        """
        self.noise: float = noise
        self.variance: float = variance
        self.length_scale: float = length_scale

    def fit(self: GaussianProcess, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the Gaussian Process to the training data.

        Args:
        ----
            x_train (np.ndarray): Training input data, shape (n_train, d).
            y_train (np.ndarray): Training target values, shape (n_train,).

        """
        self.X_train: np.ndarray[Any, Any] = x_train
        self.y_train: np.ndarray[Any, Any] = y_train
        self.covariance: np.ndarray[Any, np.dtype[np.floating[Any]]] = rbf_kernel(
            x1=x_train,
            x2=x_train,
            variance=self.variance,
            length_scale=self.length_scale,
        ) + self.noise * np.eye(N=len(x_train))

        self.decomposition: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.linalg.cholesky(a=self.covariance)

        self.alpha: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.linalg.solve(
            a=self.decomposition.T,
            b=np.linalg.solve(a=self.decomposition, b=y_train),
        )

    def predict(
        self: GaussianProcess,
        x_test: np.ndarray,
    ) -> tuple[np.ndarray[Any, np.dtype[np.floating[Any]]], np.ndarray[Any, np.dtype[Any]]]:
        """Predict using the Gaussian Process.

        Args:
        ----
            x_test (np.ndarray): Test input data, shape (n_test, d).

        Returns:
        -------
            Tuple containing the mean and covariance of the predictions.
            - mean: Predicted mean, shape (n_test,).
            - variance: Predicted variance vector, shape (n_test, n_test).

        """
        covariance_star: np.ndarray[Any, np.dtype[np.floating[Any]]] = rbf_kernel(
            x1=self.X_train,
            x2=x_test,
            variance=self.variance,
            length_scale=self.length_scale,
        )

        covariance_star_star: np.ndarray[Any, np.dtype[np.floating[Any]]] = rbf_kernel(
            x1=x_test,
            x2=x_test,
            variance=self.variance,
            length_scale=self.length_scale,
        )

        mean: np.ndarray[Any, np.dtype[np.floating[Any]]] = covariance_star.T.dot(b=self.alpha)
        outer_covariance: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.linalg.solve(
            a=self.decomposition,
            b=covariance_star,
        )

        sample_covariance: np.ndarray[Any, np.dtype[np.floating[Any]]] = covariance_star_star - outer_covariance.T.dot(
            b=outer_covariance,
        )

        variance: np.ndarray[Any, np.dtype[Any]] = np.sqrt(np.diag(v=sample_covariance))
        return mean, variance


# Example usage
if __name__ == "__main__":
    # Generate sample data
    X_train: np.ndarray[Any, np.dtype[Any]] = np.array(
        object=[[-4.0], [-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]]
    )
    y_train: np.ndarray[Any, np.dtype[Any]] = np.sin(X_train).ravel()

    # Fit the Gaussian Process
    gp = GaussianProcess(length_scale=1.0, variance=1.0, noise=1e-10)
    gp.fit(x_train=X_train, y_train=y_train)

    # Predict over a range of values
    X_test: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.linspace(start=-5, stop=5, num=100).reshape(-1, 1)

    cov: np.ndarray[Any, np.dtype[Any]]
    mean: np.ndarray[Any, np.dtype[np.floating[Any]]]
    mean, cov = gp.predict(x_test=X_test)
    std_dev: np.ndarray[Any, np.dtype[Any]] = np.sqrt(cov)

    # Create the plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

    # Plot the covariance matrix
    K: np.ndarray[Any, np.dtype[np.floating[np._64Bit]]] = rbf_kernel(x1=X_test, x2=X_test)  # type: ignore  # noqa: PGH003
    im = ax[0].imshow(K, interpolation="none", cmap="viridis")
    ax[0].set_title(r"$\Sigma$")
    fig.colorbar(mappable=im, ax=ax[0])

    # Plot the Gaussian Process fit
    ax[1].plot(X_train, y_train, "ro", label="Training data")
    ax[1].plot(X_test, mean, "b-", label="Mean prediction")
    ax[1].fill_between(
        X_test.ravel(),
        mean - 1.96 * std_dev,
        mean + 1.96 * std_dev,
        alpha=0.2,
        label="95% confidence interval",
    )
    ax[1].set_title("Gaussian Process Regression")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("y")
    ax[1].legend()

    # Add horizontal scale annotation
    ax[1].annotate(
        "1",
        xy=(0.5, 1.5),
        xycoords="data",
        xytext=(1, 2),
        textcoords="data",
        arrowprops={"arrowstyle": "->", "connectionstyle": "arc3"},
    )
    plt.show()

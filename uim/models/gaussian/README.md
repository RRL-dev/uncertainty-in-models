# Uncertainty in Models: Gaussian Distribution and Conditional Distribution Visualization

This repository focuses on visualizing uncertainty in models through multivariate Gaussian distributions, conditional distributions, and their sampled points. These visualizations help in understanding how uncertainties propagate and are conditioned within Gaussian processes.


## Overview

The repository includes:

- Visualization of 2D and 5D Gaussian distributions.

- Visualization of conditional distributions $(p(x_1 \mid x_2, x_3, x_4, x_5))$ and $(p(x_2, x_3, x_4, x_5 \mid x_1))$.

- Sampled points from these distributions and their representation as functions of variable indices.


## Mathematical Background

### Gaussian Distribution

A multivariate Gaussian distribution is defined by its mean vector $\( \boldsymbol{\mu} \)$ and covariance matrix $\( \boldsymbol{\Sigma} \)$. The probability density function (PDF) is given by:


$$

f(\mathbf{x}) = \frac{1}{(2\pi)^{k/2} |\boldsymbol{\Sigma}|^{1/2}} \exp \left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)

$$


### Conditional Distribution


Given a multivariate normal distribution $\( \mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \)$, we can partition $\( \mathbf{x} \), \( \boldsymbol{\mu} \)$, and $\( \boldsymbol{\Sigma} \)$ as follows:


$$
\mathbf{x} = \begin{bmatrix} 

\mathbf{x}_a \\ 

\mathbf{x}_b 

\end{bmatrix}, \quad 

\boldsymbol{\mu} = \begin{bmatrix} 

\boldsymbol{\mu}_a \\ 

\boldsymbol{\mu}_b 

\end{bmatrix}, \quad 

\boldsymbol{\Sigma} = \begin{bmatrix} 

\boldsymbol{\Sigma}_{aa} & \boldsymbol{\Sigma}_{ab} \\ 

\boldsymbol{\Sigma}_{ba} & \boldsymbol{\Sigma}_{bb} 

\end{bmatrix}
$$


The conditional distribution of $(\mathbf{x}_a)$ given $(\mathbf{x}_b = \mathbf{b})$ is:

$$
\mathbf{x}_a \mid \mathbf{x}_b = \mathbf{b} \sim \mathcal{N}(\boldsymbol{\mu}_{a \mid b}, \boldsymbol{\Sigma}_{a \mid b})
$$


where


$$
\boldsymbol{\mu}_{a \mid b} = \boldsymbol{\mu}_a + \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1} (\mathbf{b} - \boldsymbol{\mu}_b)
$$



$$
\boldsymbol{\Sigma}_{a \mid b} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1} \boldsymbol{\Sigma}_{ba}
$$



## Visualizations


### 2D Gaussian Distribution


This plot visualizes a 2D Gaussian distribution along with a sampled point. The sampled point is highlighted in red, and the values are shown as functions of their variable indices.

### 5D Gaussian Distribution

This plot visualizes a 5D Gaussian distribution along with a sampled point. The sampled point is highlighted in red, and the values are shown as functions of their variable indices.

### Conditional Distribution $(p(x_1 \mid x_2, x_3, x_4, x_5))$

This plot visualizes the conditional distribution $(p(x_1 \mid x_2, x_3, x_4, x_5))$. The left plot shows the bivariate distribution of $(x_1)$ and $(x_5)$ with the sampled point highlighted in red. The right plot shows the conditional distribution as a function of $(x_1)$.

### Conditional Distribution $(p(x_2, x_3, x_4, x_5 \mid x_1))$

This plot visualizes the conditional distribution $(p(x_2, x_3, x_4, x_5 \mid x_1))$ given a sampled value for $( x_1)$. The left plot shows the bivariate distribution of $(x_1)$ and $(x_5)$ with the sampled point highlighted in red. The right plot shows the values of the sampled point as a function of their variable indices, with $(x_1)$ highlighted as a red point.

## Usage

To run the script and visualize the distributions, simply execute the Python scripts located in the `scripts` folder:


```bash

python scripts/2d_gaussian_index.py

python scripts/5d_condition_distribution.py
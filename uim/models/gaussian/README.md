<h1 align="center">Uncertainty in Models: Gaussian Distribution and Conditional Distribution Visualization</h1>

<p align="center">This repository focuses on visualizing uncertainty in models through multivariate Gaussian distributions, conditional distributions, and their sampled points. These visualizations help in understanding how uncertainties propagate and are conditioned within Gaussian processes.</p>

<h2 align="center">Overview</h2>

<p align="center">The repository includes:</p>

<ul align="center">
  <li>Visualization of 2D and 5D Gaussian distributions.</li>
  <li>Visualization of conditional distributions \( p(x_1 \mid x_2, x_3, x_4, x_5) \) and \( p(x_2, x_3, x_4, x_5 \mid x_1) \).</li>
  <li>Sampled points from these distributions and their representation as functions of variable indices.</li>
</ul>

<h2 align="center">Mathematical Background</h2>

<h3 align="center">Gaussian Distribution</h3>

<p align="center">A multivariate Gaussian distribution is defined by its mean vector \( \boldsymbol{\mu} \) and covariance matrix \( \boldsymbol{\Sigma} \). The probability density function (PDF) is given by:</p>

<p align="center">
$$
f(\mathbf{x}) = \frac{1}{(2\pi)^{k/2} |\boldsymbol{\Sigma}|^{1/2}} \exp \left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right)
$$
</p>

<h3 align="center">Conditional Distribution</h3>

<p align="center">Given a multivariate normal distribution $( \mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) )$, we can partition $( \mathbf{x} ), ( \boldsymbol{\mu} )$, and $( \boldsymbol{\Sigma} )$ as follows:</p>


```math
\mathbf{x} = \begin{pmatrix} 
\mathbf{x}_a \\ 
\mathbf{x}_b 
\end{pmatrix}, \quad 
\boldsymbol{\mu} = \begin{pmatrix} 
\boldsymbol{\mu}_a \\ 
\boldsymbol{\mu}_b 
\end{pmatrix}, \quad 
\boldsymbol{\Sigma} = \begin{pmatrix} 
\boldsymbol{\Sigma}_{aa} & \boldsymbol{\Sigma}_{ab} \\ 
\boldsymbol{\Sigma}_{ba} & \boldsymbol{\Sigma}_{bb} 
\end{pmatrix}
```

<p align="center">The conditional distribution of $( \mathbf{x}_a ) given $( \mathbf{x}_b = \mathbf{b} )$ is:</p>
<p align="center">

```math
\mathbf{x}_a \mid \mathbf{x}_b = \mathbf{b} \sim \mathcal{N}(\boldsymbol{\mu}_{a \mid b}, \boldsymbol{\Sigma}_{a \mid b})
```

<p align="center">where</p>

```math
\boldsymbol{\Sigma}_{a \mid b} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1} \boldsymbol{\Sigma}_{ba}
```

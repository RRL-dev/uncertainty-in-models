<h1 align="center">Uncertainty in Models: Gaussian Distribution and Conditional Distribution Visualization</h1>

<p align="center">This repository focuses on visualizing uncertainty in models through multivariate Gaussian distributions, conditional distributions, and their sampled points. These visualizations help in understanding how uncertainties propagate and are conditioned within Gaussian processes.</p>

<h2 align="center">Overview</h2>

<p align="center">The repository includes:</p>

<ul align="center">
  <li>Visualization of 2D and 5D Gaussian distributions.</li>
  <li>Visualization of conditional distributions $( p(x_1 \mid x_2, x_3, x_4, x_5) )$ and $( p(x_2, x_3, x_4, x_5 \mid x_1) )$.</li>
  <li>Sampled points from these distributions and their representation as functions of variable indices.</li>
</ul>

<h2 align="center">Mathematical Background</h2>

<h3 align="center">Gaussian Distribution</h3>

<p align="center">A multivariate Gaussian distribution is defined by its mean vector $( \boldsymbol{\mu} )$ and covariance matrix $( \boldsymbol{\Sigma} )$. The probability density function (PDF) is given by:</p>

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

<p align="center">The conditional distribution of $( \mathbf{x}_a )$ given $( \mathbf{x}_b = \mathbf{b} )$ is:</p>
<p align="center">

```math
\mathbf{x}_a \mid \mathbf{x}_b = \mathbf{b} \sim \mathcal{N}(\boldsymbol{\mu}_{a \mid b}, \boldsymbol{\Sigma}_{a \mid b})
```

<p align="center">where</p>

```math
\boldsymbol{\mu}_{a \mid b} = \boldsymbol{\mu}_a + \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1} (\mathbf{b} - \boldsymbol{\mu}_b)
```

```math
\boldsymbol{\Sigma}_{a \mid b} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab} \boldsymbol{\Sigma}_{bb}^{-1} \boldsymbol{\Sigma}_{ba}
```

</p>
<h2 align="center">Visualizations</h2>
</p>

<h3 align="center">2D Gaussian Distribution</h3>
<p align="center"><img width="100%" src="https://github.com/RRL-dev/uncertainty-in-models/blob/main/uim/assets/2d_gaussian_index.png?raw=true" alt="2D Gaussian Distribution"></p>
<p align="center">This plot visualizes a 2D Gaussian distribution along with a sampled point. The sampled point is highlighted in red, and the values are shown as functions of their variable indices.
</p>


<h3 align="center">5D Gaussian Distribution</h3>
<p align="center"><img width="100%" src="https://github.com/RRL-dev/uncertainty-in-models/blob/main/uim/assets/5d_gaussian_index.png?raw=true" alt="5D Gaussian Distribution"></p>
<p align="center">This plot visualizes a 5D Gaussian distribution along with a sampled point. The sampled point is highlighted in red, and the values are shown as functions of their variable indices.</p>


<h3 align="center">Conditional Distribution $( p(x_1 \mid x_2, x_3, x_4, x_5) )$</h3>
<p align="center"><img width="100%" src="https://github.com/RRL-dev/uncertainty-in-models/blob/main/uim/assets/conditional_distribution.png?raw=true" alt="Conditional Distribution"></p>
<p align="center">This plot visualizes the conditional distribution $( p(x_1 \mid x_2, x_3, x_4, x_5) )$. The left plot shows the bivariate distribution of $( x_1 )$ and $( x_5 )$ with the sampled point highlighted in red. The right plot shows the conditional distribution as a function of $( x_1 )$.</p>


<h3 align="center">Conditional Distribution $( p(x_2, x_3, x_4, x_5 \mid x_1) )$</h3>
<p align="center"><img width="100%" src="https://github.com/RRL-dev/uncertainty-in-models/blob/main/uim/assets/conditional_distribution_index.png?raw=true" alt="Conditional Distribution Index"></p>


<p align="center">This plot visualizes the conditional distribution $( p(x_2, x_3, x_4, x_5 \mid x_1) )$ given a sampled value for $( x_1 )$. The left plot shows the bivariate distribution of $( x_1 )$ and $( x_5 )$ with the sampled point highlighted in red. The right plot shows the values of the sampled point as a function of their variable indices, with $( x_1 )$ highlighted as a red point.</p>


<h2 align="center">Usage</h2>
<p align="center">To run the script and visualize the distributions, simply execute the Python scripts located in the `scripts` folder:</p>

```bash
python scripts/2d_gaussian_index.py
python scripts/5d_gaussian_index.py
python scripts/5d_condition_index.py
python scripts/5d_condition_distribution.py
```
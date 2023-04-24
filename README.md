# Physics-informed neural networks (PINNs) for dynamic systems

This repo includes a series of physics-informed neural networks for various dynamic systems. The models and their development stage are shown in the below table. Please note, though some are noted as available and ready for use, they are in no way perfect. Please feel free to flag any bugs, or any possible improvements. The letters which define the development stage are:
 - A: Available
 - D: In development
 - P: Proposed


| Model             | Free vibration    | Forced Vibration  | One-step Ahead |
| :----             | :------------:    | :--------------:  | :-----------:  |
| Linear SDOF       | A                 | A                 | D              |
| Duffing SDOF      | A                 | A                 | D              |
| Linear MDOF       | P                 | P                 | P              |
| Duffing MDOF      | P                 | P                 | P              |
| Continuous Beam   | D                 | P                 | P              |
<!-- | Elastic waves     | N/A               | D                 | N/A            | -->

# General PINN Definition

## Artificial Neural Networks
For regression problems, the aim of an ANN is to map from an $n$-dimensional input, $\mathbf{x}$, to  $k$-dimensional output, $\mathbf{y}$. 
For $N$ hidden layers of a neural network, each of the $m$ nodes is passed through an activation function $\sigma$. 
```math
\mathcal{N}_{\mathbf{y}}(\mathbf{x};\mathbf{W}, \mathbf{B}) := \sigma(\mathbf{w}^l x^{l-1} + \mathbf{b}^l), \quad \mathrm{for}\; l = 2,...,N
```
where $\mathbf{W}=\{\mathbf{w}^1,...,\mathbf{w}^N\}$ and $\mathbf{B}=\{\mathbf{b}^1,...,\mathbf{b}^N\}$ are the weights and biases of the network, respectively.
These then form the hyperparameters of the networks $\mathbf{\Theta} = \{\mathbf{W},\mathbf{B}\}$. 

With target output data $\mathbf{y}^*$ from the domain of observations $\mathbf{x}^*\in\Omega_0$, the "optimal" parameters are commonly determined using a simple mean-squared-error objective function,
```math
L_{obs}(\mathbf{x}^*;\mathbf{\Theta}) = \langle \mathbf{y}^* - \mathcal{N}_{\mathbf{y}}(\mathbf{x}^*;\mathbf{\Theta}) \rangle _{\Omega_{0}}, \qquad
    \langle \bullet \rangle _{\Omega_{\kappa}} = \frac{1}{N_{\kappa}}\sum_{x\in\Omega_{\kappa}}\left|\left|\bullet\right|\right|^2
```

## Physics-Informed Neural Network

If the physics of the system is known (or estimated) in the form of ordinary or partial differential equations, then this can be embedded into the objective function over which the NN parameters are optimised. 
Given a general form of the PDE,
```math
\mathcal{F}(\mathbf{y},\mathbf{x};\theta) = 0
```
for some nonlinear operator $\mathcal{F}$ acting on $\mathbf{y}(\mathbf{x})$, where $\theta$ are parameters of the equation. For example, the wave equation (here in it's general form),
```math
\frac{\partial^2 u}{\partial t^2} = c_1^2 \frac{\partial^2}{\partial x_1^2} + c_2^2 \frac{\partial^2}{\partial x_2^2} + ... + c_n^2 \frac{\partial^2}{\partial x_n^2}
```
where $\mathbf{y}=\{u_1,u_2,...,u_n\}$, $\mathbf{x}=\{x_1,x_2,...,x_n\}$, and $\theta = \{c_1,c_2,...,c_n\}$.

When predicting the output from a neural network, we can also create an estimate of the nonlinear operator, $\mathcal{F}(\mathcal{N}_{\mathbf{y}},\mathbf{x};\theta)$. 
This can then be directly used as an additional objective function to be minimised, as when this value equals zero, the PDE is satisfied. 
Given the domain of collocation points, $\mathbf{x}_p \in \Omega_p$, this term is defined as,
```math
L_{pde}(\mathbf{x}_c;\mathbf{\Theta},\theta) = \langle \mathcal{F}(\mathcal{N}_\mathbf{y_p},\mathbf{x}_p;\theta) \rangle _{\Omega_p}, \qquad \mathcal{N}_\mathbf{y_p} = \mathcal{N}_\mathbf{y}(\mathbf{x}_p;\mathbf{\Theta})
```
Then, we can combine the the observation objective function with the pde objective function, and minimise this,

```math
L = L_{obs} + \Lambda L_{pde}
```
where $\Lambda$ is a normalisation parameter required to posit the objective function values in the same magnitude to aid optimisation. In this work, often a combination of the input normalisation parameters are used to set the value of $\Lambda$.


# Physics-informed neural networks for SDOF oscillators

Selection of physics-informed neural networks examples for dynamic systems.

This code is developed in python and detailed examples with theory are given using Jupyter notebooks. 
Below are the packages required, along with the version in which they have been tested. All packages can be installed via pip:
```
python -- version = 3.10.8
pytorch -- version = 2.0
matplotlib -- version = 3.6.2
ipython -- version = 8.7.0
mdof-toybox -- version = 0.2.2
tqdm -- version = 4.64.1
```

This example code set currently has the following notebooks:

## SDOF linear oscillator
`sdof_pinn_linear.ipynb`

This notebook aims to model a single degree of freedom oscillator, with linear stiffness and damping, under free vibration. This system is defined by the ordinary differential equation,
$$
m\ddot{x} + c\dot{x} + kx = 0
$$
More theory on the process is given in the notebooks. A simple guide on generating the ground truth model, and building and training the PINN is given here. For brevity, code snippets which define variables are not shown here but are clearly necessary.
> Preamble
```python
import numpy as np
import torch
from sdof_oscillators import sdof_linear
from sdof_pinn import bbnn, sdof_pinn, add_noise
```
> Generate ground truth and training data
```python
x_gt = sdof_linear(time, x0, v0, k_tild, c_tild)
```
> Normalise data for NN and store normalisation parameters
```python
t_hat, alpha_t = normalise(time, "range")
x_hat, alpha_x = normalise(x, "range")
sub_ind = ... # generate a vector of indices selecting data used for training
t_data, x_data = t[sub_ind], x[sub_ind]
```
> Build a 'black-box' neural network, as an example here, an ANN with 3 layers, each with 32 nodes is generated. Then, train using torch built-in optimiser, here an Adam optimiser with a learning rate of 1e-3, and betas of (0.99, 0.999) is used.
```python
bb_model = bbnn(
    N_INPUT = 1,
    N_OUTPUT = 1,
    N_HIDDEN = 32,
    N_LAYERS = 3
)
optimizer = torch.optim.Adam(bb_model.parameters(), lr=1e-3)

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = bb_model.loss_func(t_data, x_data)
    loss.backward()
    optimizer.step()
```
> Build a PINN, here the same architecture is used, then the system is designated as a linear oscillator.
```python
pi_model = sdof_pinn(
    N_INPUT = 1,
    N_OUTPUT = 1,
    N_HIDDEN = 32,
    N_LAYERS = 3
)

pi_model.nonlinearity("lin_osc")
```
> To set the physical parameters to be optimised, and embed the normalisation parameters
```python
pi_model.set_phys_params(params={"k":1.0, "c":1.0}, par_type="variable")
alphas = {
    "c" : alpha_c,
    "k" : alpha_k,
    "t" : alpha_t,
    "x" : alpha_x,
}
pi_model.set_norm_params(alphas=alphas, "up_time")
```
> Learning stage, `t-physics` is a vector of time points to calculate the ode residual over
```python
optimizer = torch.optim.Adam(pi_model.parameters(), lr=1e-3)

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = pi_model.loss_func(t_physics, t_data, x_data, lambds)
    loss.backward()
    optimizer.step()
```

## SDOF nonlinear oscillator
`sdof_pinn_nonlinear.ipynb`

This notebook aims to model a single degree of freedom oscillator, with nonlinear stiffness, under free vibration. This system is defined by the ordinary differential equation,
$$
m\ddot{x} + c\dot{x} + kx + k_3x^3 = 0
$$
The process is much the same as that for the linear oscillator, the main differences are in generating the ground truth, and in certain lines defining the PINN. The main differences are,
```python
from sdof_oscillators import cubic_duffing_sdof

x_gt = cubic_duffing_sdof(time, x0, v0, k_tild, c_tild, k3_tild)

pi_model.nonlinearity("cubic_stiffness")

pi_model.set_phys_params(params={"k":1.0, "c":1.0, "k3":1.0}, par_type = "variable")
alphas = {
    "c" : alpha_c,
    "k" : alpha_k,
    "k3" : alpha_k3,
    "t" : alpha_t,
    "x" : alpha_x,
}
```
# Physics-informed neural networks for SDOF oscillators

Selection of physics-informed neural networks examples for dynamic systems.

This code is developed in python and detailed examples with theory are given using Jupyter notebooks. 
Below are the packages required, along with the version in which they have been tested. All packages can be installed via pip:
```
python -- version = 3.10.8
numpy -- version = 1.23.5
scipy -- version = 1.10.1
pytorch -- version = 2.0
matplotlib -- version = 3.6.2
ipython -- version = 8.7.0
mdof-toybox -- version = 0.2.2
tqdm -- version = 4.64.1
```
These notebooks aim to model single degree of freedom oscillators, all with linear stiffness and damping, and nonlinearity modelled as a duffing oscillator with cubic stiffness. %
This example code set has the following notebooks, defined by the ordinary differential equations:

## SDOF linear oscillator - free vibration
`sdof_free_linear.ipynb`
```math
m\ddot{x} + c\dot{x} + kx = 0
```
## SDOF Duffing oscillator - free vibration
`sdof_free_cubic.ipynb`
```math
m\ddot{x} + c\dot{x} + kx + k_3x^3 = 0
```
## SDOF linear oscillator - forced vibration
`sdof_forced_linear.ipynb`
```math
m\ddot{x} + c\dot{x} + kx = F_x
```
## SDOF Duffing oscillator - forced vibration
`sdof_forced_cubic.ipynb`
```math
m\ddot{x} + c\dot{x} + kx + k_3x^3 = F_x
```

More theory on the process is given in the notebooks. A simple guide on generating the ground truth model, and building and training the PINN is given here. For brevity, code snippets which define variables are not shown here but are clearly necessary.
> Preamble
```python
import numpy as np
import torch
from sdof_oscillators import sdof_solution, add_noise, generate_excitation
from sdof_pinn import bbnn, sdof_pinn, normalise
```
> The first step is to generate the excitation signal, which is done using a configuration dictionary. The types of excitations available, and the required configuration parameters are given in the table below. A * designates an optional parameter. If using free vibration, set the configuration file to be `None` type.

|Type                 |Req. Parameters                        |Key                              |  Type | Description
|:----                |:------------:                         |:--------------:                 | :--------------:  | :--:
|Sinusoid <br> `sinusoid` |$\omega$ <br> $F_0$                    |`w` <br> `F0`                    |`float` <br> `float` | central frequency <br> amplitude
|White Gaussian<br>`white_gaussian`|$\bar{F}$<br>$F_0$<br>seed*            |`offset`<br>`F0`<br>`seed`       | `float`<br>`float`<br>`int` | offset (mean) <br> amplitude <br> RNG seed
|Sine Sweep<br>`sine_sweep`|$\{\omega_0,\omega_1\}$<br>$F_0$       |`w`<br>`F0`                      |`vector[float]`<br> `float` | start and end freq. <br> amplitude
|Rand. Phase Multisine<br>`rand_phase_ms`|$\mathbf{f}$<br>$S_x$<br>seed*|`freqs`<br>`Sx`<br>`seed`| `vector[float]` <br> `vector[float]` <br> `int` | freq. bins <br> PSD amplitudes <br> RNG seed
> Then the forcing signal is generated, given a time vector and the configuration dictionary, and ensure to put the created signal into the excitation config
```python
force_signal = generate_excitation(t, **excitation)
excitation['F'] = force_signal
```
> Next, we generate ground truth and training data. The ground truth is determined using M. Champneys' `mdof-toybox` package, but is wrapped in a helper function here. The type and parameters of the SDOF system are set using a configuration dictionary. The `sdof_solution` function returns both the displacement and forcing function. This is important as the forcing is normalised by the mass if necessary.

| Key | Type | Options | Description |
|:---:|:----:|:-------:|:-----------:|
| `nonlinearity` | `str` | `"linear"`<br>`"cubic"` | Linear SDOF <br> Duffing SDOF |
| `m_norm` | `bool` |  | Set whether physical parameters are normalised by the mass (i.e. $k$ or $\tilde{k}$) |
| `params` | `dict` | `{k:float,c:float,...}` | Physical parameters defining the gt model |
| `init_conds` | `dict` | `{x0:float,v0:float}` | Initial conditions in terms of displacement and velocity |
| `forcing` | `dict` | `excitation_config`<br>`None` | Definition of excitation signal (use above configuration dictionary)

```python
x, F_tild = sdof_solution(t, **gt_config)
excitation["F_tild"] = F_tild
```
> Normalise data for NN and store normalisation parameters
```python
t_hat, alpha_t = normalise(time, "range")
x_hat, alpha_x = normalise(x, "range")
F_hat, alpha_F = normalise(F_tild, "range")
sub_ind = ... # generate a vector of indices selecting data used for training
t_data, x_data = t[sub_ind], x[sub_ind]
```
> First, we build a 'black-box' neural network, as an example here, an ANN with 3 layers, each with 32 nodes is generated. Then, train using torch built-in optimiser, here an Adam optimiser with a learning rate of 1e-3, and betas of (0.99, 0.999) is used.
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
> Next, we build a PINN, here the same architecture is used, then the system is configured using another configuration dictionary.

| Key | Type | Options | Description |
|:---:|:----:|:-------:|:-----------:|
| `nonlinearity` | `str` | `"linear"`<br>`"cubic"` | Linear SDOF <br> Duffing SDOF |
| `phys_params['par_type']` | `str` | `"constant"`<br>`"variable"` | Sets whether to estimate parameters as well as solution
| `phys_params` | `dict` | `{par_type:str,`<br>`k:float,`<br>`c:float,...}` | Physical parameter values, if variable; an initial estimate |
| `alphas` | `dict` | `{alpha_x:float,`<br>`alpha_t:float,...}` | Normalisation parameters |
| `ode_norm_Lambda` | `float` |  | Scaling of ODE loss function
| `forcing` | `dict` | `excitation_config`<br>`None` | Definition of excitation signal (use above configuration dictionary)

```python
pi_model = sdof_pinn(
    N_INPUT = 1,
    N_OUTPUT = 1,
    N_HIDDEN = 32,
    N_LAYERS = 3
)

pi_model.configure(pinn_config)
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

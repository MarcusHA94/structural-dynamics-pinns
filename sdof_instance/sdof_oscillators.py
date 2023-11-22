import torch
import toybox as tb
import numpy as np
from math import pi
import scipy.signal

# Function to add noise to a tensor
def add_noise(x: torch.Tensor, noise_std: float, seed: int = 160151156156) -> torch.Tensor:
    torch.manual_seed(seed)
    std = noise_std * torch.amax(x)
    x = x + torch.normal(0.0, std, (x.shape[0], 1))
    return x

# Single Degree of Freedom (SDOF) equation solver with optional noise
def sdof_free_linear(
    t: torch.Tensor, x0: float, v0: float, k_tild: float, c_tild: float, noise_std: float = 0.0
) -> torch.Tensor:
    wn = torch.sqrt(k_tild)
    zeta = c_tild / (2 * np.sqrt(k_tild))
    wd = wn * torch.sqrt(1 - zeta ** 2)
    A = x0
    B = (v0 + zeta * wn * x0) / (wd)
    C = zeta * wn
    x = torch.exp(-C * t) * (A * torch.cos(wd * t) + B * torch.sin(wd * t))
    return add_noise(x, noise_std)

# Solve SDOF equation for displacement and velocity given parameters and initial conditions
def sdof_solution(time: torch.Tensor, config: dict) -> (torch.Tensor, torch.Tensor):
    nt = time.shape[0]

    params = config["params"]
    init_conds = config["init_conds"]
    forcing = config["forcing"]
    nonlinearity = config["nonlinearity"]
    m_norm = config["m_norm"]

    x0 = init_conds["x0"]
    v0 = init_conds["v0"]

    # Set up SDOF system based on conditions
    if m_norm:
        sdof_system = tb.symetric(dofs=1, k=params["k_tild"], c=params["c_tild"], m=1.0)
    else:
        sdof_system = tb.symetric(dofs=1, k=params["k"], c=params["c"], m=params["m"])

    # Handle different nonlinearities
    if nonlinearity == "linear":
        sdof_system.N = None
    elif nonlinearity == "cubic":
        def cub_stiff(_, t, y, ydot):
            if m_norm:
                return np.dot(y ** 3, params["k3_tild"])
            else:
                return np.dot(y ** 3, params["k3"])

        sdof_system.N = cub_stiff

    # Handle forcing conditions
    if forcing is None:
        sdof_system.excitation = [None]
    else:
        if m_norm:
            sdof_system.excitation = [forcing["F_tild"]]
        else:
            sdof_system.excitation = [forcing["F"]]

    # Simulate the system and extract results
    data = sdof_system.simulate((nt, time[1]), w0=np.array([x0, v0]), normalise=False)
    x = torch.tensor(data["y1"]).view(-1, 1).to(torch.float32)
    v = torch.tensor(data["ydot1"]).view(-1, 1).to(torch.float32)
    return x, v

# Generate excitation signal based on configuration
def generate_excitation(time: np.ndarray, exc_config: dict) -> torch.Tensor:
    F0 = exc_config["F0"]
    nt = time.shape[0]
    time = time.reshape(-1, 1)

    exc_type = exc_config["type"]
    if exc_type == "sinusoid":
        w = exc_config["w"]
        F = F0 * np.sin(w * time)
    elif exc_type == "white_gaussian":
        u = exc_config["offset"]
        sig = F0
        seed = exc_config.get("seed", 43810)
        np.random.seed(seed)
        F = np.random.normal(u, sig, size=(time.shape[0], 1))
    elif exc_type == "sine_sweep":
        f0 = exc_config["w"][0] / (2 * pi)
        f1 = exc_config["w"][1] / (2 * pi)
        F = F0 * scipy.signal.chirp(time, f0, time[-1], f1, method=exc_config["scale"])
    elif exc_type == "rand_phase_ms":
        freqs = exc_config["freqs"].reshape(-1, 1)
        Sx = exc_config["Sx"].reshape(-1, 1)
        seed = exc_config.get("seed", 43810)
        np.random.seed(seed)
        phases = np.random.rand(freqs.shape[0])
        F_mat = np.sin(time @ freqs.T + phases.T)
        F = F_mat @ Sx

    return F

# Compute the forcing function for a given frequency and time
def forcing_func(F0: float, freq: float, t: torch.Tensor, freq_unit: str = 'rads') -> torch.Tensor:
    if freq_unit == 'rads':
        w = freq
    elif freq_unit == 'hz':
        w = freq * 2 * pi
    F = F0 * torch.sin(w * t)
    return F

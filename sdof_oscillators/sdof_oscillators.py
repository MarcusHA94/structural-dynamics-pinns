import torch
import toybox as tb
import numpy as np
from math import pi
import scipy

def add_noise(x, noise_std, seed=160151156156):
    torch.manual_seed(seed)
    std = noise_std * torch.amax(x)
    x = x + torch.normal(0.0, std, (x.shape[0],1))
    return x

def sdof_free_linear(t, x0, v0, k_tild, c_tild, noise_std=0.0):
    wn = torch.sqrt(k_tild)
    zeta = c_tild/(2*np.sqrt(k_tild))
    wd = wn*torch.sqrt(1-zeta**2)
    A = x0
    B = (v0 + zeta*wn*x0)/(wd)
    C = zeta * wn
    x = torch.exp(-C*t) * (A * torch.cos(wd*t) + B * torch.sin(wd*t))
    return add_noise(x, noise_std)

def sdof_solution(time, **config): #params, init_conds, forcing=None, nonlinearity="linear"):
    nt = time.shape[0]

    params = config["params"]
    init_conds = config["init_conds"]
    forcing = config["forcing"]
    nonlinearity = config["nonlinearity"]
    m_norm = config["m_norm"]

    x0 = init_conds["x0"]
    v0 = init_conds["v0"]

    match m_norm:
        case False:
            sdof_system = tb.symetric(dofs=1, k=params["k"], c=params["c"], m=params["m"])
        case True:
            sdof_system = tb.symetric(dofs=1, k=params["k_tild"], c=params["c_tild"], m=1.0)

    match config:
        case {"nonlinearity":"linear"}:
            sdof_system.N = None
        case {"nonlinearity":"cubic","m_norm":True}:
            def cub_stiff(_, t, y, ydot):
                return np.dot(y**3, params["k3_tild"])
            sdof_system.N = cub_stiff
        case {"nonlinearity":"cubic","m_norm":False}:
            def cub_stiff(_, t, y, ydot):
                return np.dot(y**3, params["k3"])
            sdof_system.N = cub_stiff

    match config:
        case {"forcing":None}:
            sdof_system.excitation = [None]
        case {"m_norm":True,"forcing":dict()}:
            sdof_system.excitation = [forcing["F_tild"]]
        case {"m_norm":False,"forcing":dict()}:
            sdof_system.excitation = [forcing["F"]]
    
    data = sdof_system.simulate((nt, time[1]), w0=np.array([x0, v0]), normalise=False)
    x = torch.tensor(data["y1"]).view(-1,1).to(torch.float32)
    F = torch.tensor(data["x1"]).view(-1,1).to(torch.float32)
    return x

def generate_excitation(time, **exc_config):

    F0 = exc_config["F0"]
    nt = time.shape[0]
    time = time.reshape(-1,1)

    match exc_config["type"]:
        case "sinusoid":
            w = exc_config["w"]
            F = F0 * np.sin(w * time)
        case "white_gaussian":
            u = exc_config["offset"]
            sig = F0
            match exc_config:
                case {"seed" : int() as seed}:
                    np.random.seed(seed)
                case _:
                    np.random.seed(43810)
            F = np.random.normal(u, sig, size=(time.shape[0],1))
        case "sine_sweep":
            f0 = exc_config["w"][0] / (2*pi)
            f1 = exc_config["w"][1] / (2*pi)
            F = F0*scipy.signal.chirp(time, f0, time[-1], f1, method=exc_config["scale"])
        case "rand_phase_ms":
            freqs = exc_config["freqs"].reshape(-1,1)
            Sx = exc_config["Sx"].reshape(-1,1)
            match exc_config:
                case {"seed" : int() as seed}:
                    np.random.seed(seed)
                case _:
                    np.random.seed(43810)
            phases = np.random.rand(freqs.shape[0], 1)
            F_mat = np.sin(time @ freqs.T + phases.T)
            F = F_mat @ Sx
    
    return F
            
def forcing_func(F0, freq, t, freq_unit='rads'):
    match freq_unit:
        case 'rads':
            w = freq
        case 'hz':
            w = freq*2*pi
    F = F0 * torch.sin(w * t)
    return F
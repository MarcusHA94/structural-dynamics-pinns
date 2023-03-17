import torch
import toybox as tb
import numpy as np

def add_noise(x, noise_std, seed=160151156156):
    torch.manual_seed(seed)
    std = noise_std * torch.amax(x)
    x = x + torch.normal(0.0, std, (x.shape[0],1))
    return x

def sdof_linear(t, x0, v0, k_tild, c_tild, noise_std=0.0):
    wn = torch.sqrt(k_tild)
    zeta = c_tild/(2*np.sqrt(k_tild))
    wd = wn*torch.sqrt(1-zeta**2)
    A = x0
    B = (v0 + zeta*wn*x0)/(wd)
    C = zeta * wn
    x = torch.exp(-C*t) * (A * torch.cos(wd*t) + B * torch.sin(wd*t))
    return add_noise(x, noise_std)

def duffing_system(k_tild, c_tild, k3_tild):
    system = tb.symetric(dofs=1, k=k_tild, c=c_tild, m=1.0)
    def cub_stiff(_, t, y, ydot):
        np.dot(y**3, k3_tild)
        system.N = cub_stiff
        system.excitation = [None]
    return system

def cubic_duffing_sdof(t, x0, v0, k_tild, c_tild, k3_tild, noise_std=0.0):
    nt = t.shape[0]
    sdof_system = duffing_system(k_tild, c_tild, k3_tild)
    duff_data = sdof_system.simulate((nt, t[1]), w0=np.array([x0, v0]), normalise=False)
    x = torch.tensor(duff_data["y1"]).view(-1,1).to(torch.float32)
    return add_noise(x, noise_std)
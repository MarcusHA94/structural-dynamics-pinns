import numpy as np
import torch
from math import pi
import scipy

def gen_ndof_cantilever(m_,c_,k_,ndof=None,return_numpy=False,connected_damping=True):
    if torch.is_tensor(m_):
        ndof = m_.shape[0]
    else:
        m_ = m_ * torch.ones((ndof))
        c_ = c_ * torch.ones((ndof))
        k_ = k_ * torch.ones((ndof))
    M = torch.zeros((ndof,ndof), dtype=torch.float32)
    C = torch.zeros((ndof,ndof), dtype=torch.float32)
    K = torch.zeros((ndof,ndof), dtype=torch.float32)
    for i in range(ndof):
        M[i,i] = m_[i]
    for i in range(ndof-1):
        if connected_damping:
            C[i,i] = c_[i] + c_[i+1]
            C[i,i+1] = -c_[i+1]
        else:
            C[i,i] = c_[i]
        K[i,i] = k_[i] + k_[i+1]
        K[i,i+1] = -k_[i+1]
    C[-1,-1] = c_[-1]
    K[-1,-1] = k_[-1]
    C = torch.triu(C) + torch.triu(C, 1).T
    K = torch.triu(K) + torch.triu(K, 1).T
    if return_numpy:
        return M.numpy(), C.numpy(), K.numpy()
    else:
        return M, C, K

def mdof_simulate(time, **config):

    ndof = config['n_dofs']
    dt = time[1]-time[0]
    num_samps = time.shape[0]

    if config['init_state'] is None:
        z0 = np.zeros((2*ndof))
    else:
        z0 = np.concatenate((np.array(config['init_state']['x0']),np.array(config['init_state']['v0'])),axis=0)
    M = config['params']['M']
    C = config['params']['C']
    K = config['params']['K']
    if 'Kn' in config['params']:
        Kn = config['params']['Kn']
    else:
        Kn = None
    
    if config['forcing'] is not None:
        f = config['forcing']['F'].T
    else:
        f = None

    A = np.concatenate((
        np.concatenate((np.zeros((ndof,ndof)),np.eye(ndof)),axis=1),
        np.concatenate((-np.linalg.inv(M)@K,-np.linalg.inv(M)@C),axis=1)
    ),axis=0)

    if f is not None:
        H = np.concatenate((np.zeros((ndof,ndof)),np.linalg.inv(M)),axis=0)
    
    if Kn is not None:
        An = np.concatenate((
                np.zeros((ndof,ndof)),
                -np.linalg.inv(M)@Kn
                ), axis=0)
        
    def rung_f(z):
        match [Kn, f]:
            case [None,None]:
                return A@z
            case [_,None]:
                zn = (z[:ndof] - np.concatenate((np.zeros((1)),z[:ndof-1])))**3
                return A@z + An@zn
            case [None,_]:
                return A@z + H@f
            case [_,_]:
                zn = (z[:ndof] - np.concatenate((np.zeros((1)),z[:ndof-1])))**3
                return A@z + An@zn + H@f[:,t]
    
    z = np.zeros((2*ndof,num_samps))
    z[:,0] = z0
    for t in range(num_samps-1):
        k1 = rung_f(z[:,t])
        k2 = rung_f(z[:,t] + k1*dt/2)
        k3 = rung_f(z[:,t] + k2*dt/2)
        k4 = rung_f(z[:,t] + k3*dt)
        z[:,t+1] = z[:,t] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return torch.tensor(z[:ndof,:],dtype=torch.float32).T, torch.tensor(z[ndof:,:],dtype=torch.float32).T

def add_noise(x, db=-10.0, seed=43810):

    ns = x.shape[0]
    nd = x.shape[1]
    noise_amp = 10.0 ** (db / 10.0)
    x_noisy = torch.zeros_like(x)
    for i in range(nd):
        np.random.seed(seed+i)
        noise_x = noise_amp * np.random.normal(loc=0.0, scale=noise_amp,size=ns)
        x_noisy[:,i] = x[:,i] + noise_x
    return x_noisy


def generate_excitation(time, **exc_config):

    F0 = exc_config["F0"]
    n_modes = exc_config["n_dofs"]
    time = time.reshape(-1,1)
    F = np.zeros((time.shape[0], n_modes))

    match exc_config["type"]:
        case "sinusoid":
            for n in range(n_modes):
                w = exc_config["w"][n]
                F[:,n] = (F0 * np.sin(w * time)).reshape(-1)
        case "white_gaussian":
            u = exc_config["offset"]
            sig = F0
            for n in range(n_modes):
                match exc_config:
                    case {"seed" : int() as seed}:
                        np.random.seed(seed+n)
                    case _:
                        np.random.seed(43810+n)
                F[:,n] = np.random.normal(u[n], sig[n], size=(time.shape[0]))
        case "sine_sweep":
            for n in range(n_modes):
                f0 = exc_config["w"][n][0] / (2*pi)
                f1 = exc_config["w"][n][1] / (2*pi)
                F[:,n] = F0*scipy.signal.chirp(time, f0, time[-1], f1, method=exc_config["scale"])
        case "rand_phase_ms":
            for n in range(n_modes):
                freqs = exc_config["freqs"][n].reshape(-1,1)
                Sx = exc_config["Sx"][n].reshape(-1,1)
                match exc_config:
                    case {"seed" : int() as seed}:
                        np.random.seed(seed+n)
                    case _:
                        np.random.seed(43810+n)
                phases = np.random.rand(freqs.shape[0], 1)
                F_mat = np.sin(time @ freqs.T + phases.T)
                F[:,n] = (F_mat @ Sx).reshape(-1)
    return F


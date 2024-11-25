import numpy as np
import torch
from math import pi
import scipy
from typing import Union, Tuple

Tensor = Union[torch.Tensor, np.ndarray]
TensorFloat = Union[torch.Tensor, float]

class nonlinearity:

    def __init__(self, dofs, gk_exp = None, gc_exp = None):

        self.dofs = dofs
        self.gk_exp = gk_exp
        self.gc_exp = gc_exp
    
    def Kn_func(self, kn_):

        Kn = torch.diag(kn_) - torch.diag(kn_[1:], 1)
        return Kn
    
    def gk_func(self, x, xdot):
        if self.gk_exp is not None:
            return torch.sign(x) * torch.abs(x) ** self.gk_exp
        else:
            return torch.zeros_like(x)
    
    def Cn_func(self, cn_):

        Cn = torch.diag(cn_) - torch.diag(cn_[1:], 1)
        return Cn
    
    def gc_func(self, x, xdot):
        if type(self.gc_exp) == float:
            return torch.sign(xdot) * torch.abs(xdot) ** self.gc_exp
        elif self.gc_exp == 'vdp':
            return (x**2 - 1) * xdot
        else:
            return torch.zeros_like(xdot)
        
    def mat_func(self, kn_, cn_, invM):

        Kn = self.Kn_func(kn_)
        Cn = self.Cn_func(cn_)

        return torch.cat((
            torch.zeros((self.dofs, 2*self.dofs)),
            torch.cat((-invM @ Kn, -invM @ Cn), dim=1)
            ), dim=0)
    
    def gz_func(self, z):
        x_ = z[:self.dofs, :] - torch.cat((torch.zeros((1, z.shape[1])), z[:self.dofs-1, :]), dim=0)
        xdot_ = z[self.dofs:, :] - torch.cat((torch.zeros((1, z.shape[1])), z[self.dofs:-1, :]), dim=0)
        return torch.cat((self.gk_func(x_, xdot_), self.gc_func(x_, xdot_)), dim=0)

def gen_ndof_cantilever(m_: TensorFloat, c_: TensorFloat, k_: TensorFloat, ndof: int = None, return_numpy: bool = False, connected_damping: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
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

def add_noise(x: np.ndarray, db: Tuple[float, None] = None, SNR: Tuple[float, None] = None, seed: int = 43810) -> np.ndarray:

    ns = x.shape[0]
    nd = x.shape[1]
    x_noisy = np.zeros_like(x)

    match [db, SNR]:
        case [float(), None]:
            noise_amp = 10.0 ** (db / 10.0)
            for i in range(nd):
                np.random.seed(seed + i)
                noise_x = np.random.normal(loc=0.0, scale=np.sqrt(noise_amp), size=ns)
                x_noisy[:,i] = x[:,i] + noise_x
        case [None, float()]:
            for i in range(nd):
                np.random.seed(seed + i)
                P_sig = 10 * np.log10(np.mean(x[:, i]**2))
                P_noise = P_sig - SNR
                noise_amp = 10 ** (P_noise / 10.0)
                noise_x = np.random.normal(loc=0.0, scale=np.sqrt(noise_amp), size=ns)
                x_noisy[:,i] = x[:,i] + noise_x
        case [float(), float()]:
            raise Exception("Over specified, please select either db or SNR")
        case [None, None]:
            raise Exception("No noise level specified")
    return x_noisy



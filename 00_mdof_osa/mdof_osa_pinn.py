import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt

from scipy.stats import qmc

from tqdm import tqdm
from tqdm.auto import tqdm as tqdma
from IPython import display

from typing import Tuple, Union
Tensor = Union[torch.Tensor, np.ndarray]

def max_mag_data(data: Tensor, axis: int = None) -> Tensor:
    """
    Compute the maximum magnitude of data along the specified axis.
    """
    if torch.is_tensor(data):
        if axis is None:
            data_max = torch.max(torch.max(torch.abs(data)))
        else:
            data_max = torch.max(torch.abs(data),dim=axis)[0]
    else:
        data_max = np.max(np.abs(data),axis=axis)
    return data_max

def range_data(data: Tensor, axis: int = None) -> Tensor:
    """
    Compute the range of data along the specified axis.
    """
    if torch.is_tensor(data):
        if axis is None:
            data_range = torch.max(torch.max(data)) - torch.min(torch.min(data))
        else:
            data_range = torch.max(data,dim=axis)[0] - torch.min(data,dim=axis)[0]
    else:
        data_range = np.max(data, axis=axis) - np.min(data, axis=axis)
    return data_range

def normalise(data: Tensor, norm_type: str = "var", norm_dir: str = "all") -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
    """
    Normalize data based on the specified normalization type and direction.
    """
    if norm_type=="var":
        if len(data.shape)>1 and norm_dir=="axis":
            mean = data.mean(axis=0)
            std = data.std(axis=0)
        else:
            mean = data.mean()
            std = data.std()
        data_norm = (data-mean)/std
        return data_norm, (mean, std)
    elif norm_type=="range":
        if len(data.shape)>1 and norm_dir=="axis":
            dmax = range_data(data,axis=0)
        else:
            dmax = range_data(data)
        data_norm = data/dmax
        return data_norm, dmax
    elif norm_type=="max":
        if len(data.shape)>1 and norm_dir=="axis":
            dmax = max_mag_data(data,axis=0)
        else:
            dmax = max_mag_data(data)
        data_norm = data/dmax
        return data_norm, dmax

def nonlin_state_transform(z: torch.Tensor) -> torch.Tensor:
    n_dof = int(z.shape[0]/2)
    return (z[:n_dof,:] - torch.cat((torch.zeros(1, z.shape[1]), z[:n_dof-1, :]), dim=0))**3
    

class osa_pinn_mdof(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]
        self.n_dof = config["n_dof"]
        self.activation = nn.Tanh
        self.device = config["device"]

        self.build_net()

        self.configure(config)

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return 0
    
    def build_ed_net(self):
        self.ed_net = nn.Sequential(
            nn.Sequential(*[nn.Linear(1, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.ed_net
    
    def forward(self, x0, v0, t, f0=None):
        if f0 is None:
            x = torch.cat((x0, v0, t.view(-1,1)), dim=1)
        else:
            x = torch.cat((x0, v0, f0, t.view(-1,1)), dim=1)
        y = self.net(x)
        return y
    
    def configure(self, config):

        self.config = config

        self.nct = config["nct"]  # number of time collocation points
        self.col_dt = config["col_dt"]  # delta-t for time collocation points

        self.set_phys_params()
        self.set_norm_params()

    def set_phys_params(self):
        config = self.config
        self.param_attrs = {}
        for param_name, param_dict in config["phys_params"].items():
            self.param_attrs[param_name] = param_dict["type"]
            if param_dict["type"] == "constant":
                setattr(self,param_name,param_dict["value"])
            elif param_dict["type"] == "variable":
                self.register_parameter(param_name, nn.Parameter(torch.ones(self.n_dof)))
        if hasattr(self,"M") and hasattr(self,"C") and hasattr(self,"K"):
            self.A = torch.cat((
                    torch.cat((torch.zeros((self.n_dof,self.n_dof)),torch.eye(self.n_dof)), dim=1),
                    torch.cat((-torch.linalg.inv(self.M)@self.K, -torch.linalg.inv(self.M)@self.C), dim=1)
                    ), dim=0)
        elif hasattr(self,"M"):
            self.m_ = torch.diag(self.M)  # takes diagonal from mass matrix if set as constant
        
        if hasattr(self,"M"):
            self.H = torch.cat((torch.zeros((self.n_dof,self.n_dof)),torch.linalg.inv(self.M)), dim=0)
        if hasattr(self,"Kn") and config["nonlinearity"]=="cubic":
            self.An = torch.cat((
                    torch.zeros((self.n_dof,self.n_dof)),
                    -torch.linalg.inv(self.M)@self.Kn
                ), dim=0)

    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]["t"]
        self.alpha_x = config["alphas"]["x"]
        self.alpha_v = config["alphas"]["v"]
        self.alpha_z = torch.cat((self.alpha_x*torch.ones(self.n_dof,1), self.alpha_v*torch.ones(self.n_dof,1)), dim=0).float().to(self.device)
        self.alpha_f = config["alphas"]["f"]
        for param_name, param_dict in config["phys_params"].items():
            if param_dict["type"] == "variable":
                setattr(self,"alpha_"+param_name[:-1], config["alphas"][param_name[:-1]])

    def set_aux_funcs(self, nonlin_func):
        self.nonlin_func = nonlin_func

    def set_switches(self, lambdas: dict) -> None:
        switches = {}
        for key, value in lambdas:
            switches[key] = value>0.0
        self.switches = switches

    def set_colls_and_obs(self, t_data, x_data, v_data, f_data=None):

        # _data -> [samples, dof]
        n_obs = x_data.shape[0]-1

        # Observation set (uses displacement one data point ahead)
        self.x_obs = x_data[:-1,:]  # initial displacement input
        self.v_obs = v_data[:-1,:]  # initial velocity input
        self.t_obs = torch.zeros((n_obs,1))
        for i in range(n_obs):
            self.t_obs[i] = t_data[i+1] - t_data[i]  # time at end of horizon (window)
        if f_data is not None:
            self.f_obs = f_data[:-1,:]  # force input
        self.z_obs = torch.cat((x_data[1:,:], v_data[1:,:]), dim=1).requires_grad_()  # displacement at end of window (output)

        # Collocation set (sets a copy of the x0, v0 for a vector of time over the time horizon)
        x_col = torch.zeros((n_obs*self.nct,self.n_dof))
        v_col = torch.zeros((n_obs*self.nct,self.n_dof))
        t_col = torch.zeros((n_obs*self.nct,1))
        f_col = torch.zeros((n_obs*self.nct,self.n_dof))
        t_pred = torch.zeros((n_obs*self.nct,1))

        for i in range(n_obs):
            for j in range(self.n_dof):
                x_col[self.nct*i:self.nct*(i+1),j] = x_data[i,j].item()*torch.ones(self.nct)
                v_col[self.nct*i:self.nct*(i+1),j] = v_data[i,j].item()*torch.ones(self.nct)
                if f_data is not None:
                    f_col[self.nct*i:self.nct*(i+1),j] = f_data[i,j].item()*torch.ones(self.nct)
            t_col[self.nct*i:self.nct*(i+1),0] = torch.linspace(0, t_data[i+1].item()-t_data[i].item(), self.nct)

            # generates a vector of the time for the predicted output, by simply adding the total window onto the current time in the data
            t_pred[self.nct*i:self.nct*(i+1),0] = t_data[i] + torch.linspace(0, t_data[i+1].item()-t_data[i].item(), self.nct)

        self.x_col = x_col.requires_grad_()
        self.v_col = v_col.requires_grad_()
        self.t_col = t_col.requires_grad_()
        if f_data is not None:
            self.f_col = f_col.requires_grad_()
        
        self.ic_ids = torch.argwhere(t_col[:,0]==torch.tensor(0.0)).view(-1)

        return t_pred

    def loss_func(self, obs_data: torch.Tensor, col_data: torch.Tensor, lambdas: dict, ic_ids: Tuple[np.ndarray, None] = None) -> Tuple[torch.Tensor, list, dict]:

        z_obs = obs_data[:, :2*self.n_dof]
        x0_obs = obs_data[:, 2*self.n_dof : 3*self.n_dof]
        v0_obs = obs_data[:, 3*self.n_dof : 4*self.n_dof]
        f0_obs = obs_data[:, 4*self.n_dof : 5*self.n_dof]
        t_obs = obs_data[:, -1]

        if self.switches['obs']:
            # generate prediction at observation points
            if f_obs is None:
                zh_obs_hat = self.forward(x0_obs, v0_obs, t_obs)
            else:
                zh_obs_hat = self.forward(x0_obs, v0_obs, t_obs, f0_obs)
            R_obs = torch.sqrt(torch.sum((zh_obs_hat - z_obs)**2, dim=1))

        x0_col = col_data[..., : self.n_dof].reshape(-1, self.n_dof)
        v0_col = col_data[..., self.n_dof : 2*self.n_dof].reshape(-1, self.n_dof)
        f0_col = col_data[..., 2*self.n_dof : 3*self.n_dof].reshape(-1, self.n_dof)
        t_col = col_data[..., -1].reshape(-1, 1)

        if self.switches['ode']:
            # generate prediction over prediction horizon (collocation domain)
            if f_col is None:
                zp_col_hat = self.forward(x0_col, v0_col, t_col)
            else:
                zp_col_hat = self.forward(x0_col, v0_col, t_col, f0_col)
            
            # retrieve derivatives
            dzdt = torch.zeros_like(zp_col_hat)
            for i in range(zp_col_hat.shape[1]):
                dzdt[:, i] = torch.autograd.grad(zp_col_hat[:, i], t_col, torch.ones_like(zp_col_hat[:, i]), create_graph=True)[0][:,0]  # ∂_t-hat N_z-hat

            # retrieve physical parameters
            if hasattr(self,"A"):
                M, C, K = self.M, self.C, self.K
                A = self.A
            else:
                params = {}
                for param_name, param_dict in self.config["phys_params"].items():
                    if param_dict["type"] == "constant":
                        params[param_name] = param_dict["value"]
                    else:
                        params[param_name] = getattr(self,param_name)*getattr(self,"alpha_"+param_name[:-1])
                M, C, K = self.param_func(params["m_"],params["c_"],params["k_"])
                invM = torch.diag(1/torch.diag(M))
                A = torch.cat((
                    torch.cat((torch.zeros((self.n_dof, self.n_dof)), torch.eye(self.n_dof)), dim=1),
                    torch.cat((-invM @ K, -invM @ C), dim=1)
                    ), dim=0).requires_grad_()
            if f_col is not None:
                if hasattr(self,"H"):
                    H = self.H
                else:
                    H = torch.cat((torch.zeros((self.n_dof, self.n_dof)), invM), dim=0)

            if self.nonlinearity is not None:
                An = self.nonlinearity.mat_func(params['kn_'], params['cn_'], invM)

            # calculate ode residual
            match [self.nonlinearity, f_col]:
                case None, None:
                    R_ = (self.alpha_z / self.alpha_t) * dzdt.T - A @ (self.alpha_z * zp_col_hat.T)
                    R_ode = R_[self.n_dof:, :].T
                case [_, None]:
                    gz = self.nonlinearity.gz_func(self.alpha_z*zp_col_hat.T)
                    R_ = (self.alpha_z / self.alpha_t) * dzdt.T - A @ (self.alpha_z * zp_col_hat.T) - An @ gz
                    R_ode = R_[self.n_dof:, :].T
                case [None, torch.Tensor()]:
                    R_ = (self.alpha_z / self.alpha_t) * dzdt.T - A @ (self.alpha_z * zp_col_hat.T) - H @ (self.alpha_f * f_col.T)
                    R_ode = R_[self.n_dof:, :].T
                case [_, torch.Tensor()]:
                    gz = self.nonlinearity.gz_func(self.alpha_z * zp_col_hat.T)
                    R_ = (self.alpha_z / self.alpha_t) * dzdt.T - A @ (self.alpha_z * zp_col_hat.T) - An @ gz - H @ (self.alpha_f * f_col.T)
                    R_ode = R_[self.n_dof:, :].T

        if self.switches['cc']:
            # continuity condition residual
            R_cc = R_[:self.n_dof,:].T
        else:
            R_cc = torch.zeros((2, 2))

        if self.switches['ic']:
            if ic_ids is None:
                raise Exception("Initial condition switch is on but no indexes were given")
            else:
                # initial condition residual
                R_ic = self.alpha_z * z0_col[ic_ids, :] - self.alpha_z * zp_col_hat[ic_ids, :]
        else:
            R_ic = torch.zeros((2, 2))

        L_obs = lambdas['obs'] * torch.mean(R_obs**2)
        L_ic = lambdas['ic'] * torch.sum(torch.mean(R_ic**2, dim=0), dim=0)
        L_cc = lambdas['cc'] * torch.sum(torch.mean(R_cc**2, dim=0), dim=0)
        L_ode = lambdas['ode'] * torch.sum(torch.mean(R_ode**2, dim=0), dim=0)

        loss = L_obs + L_ic + L_cc + L_ode

        if math.isnan(loss):
            raise Exception("Loss is NaN, upsi")

        return loss, [L_obs, L_ic, L_cc, L_ode]
    
    def predict(self, pred_data):
        z0_pred = pred_data['z0_hat']
        t_pred = pred_data["t_hat"]
        f_pred = pred_data["f_hat"]

        if f_pred is None:
            xp = self.forward(z0_pred, t_pred)
        else:
            xp = self.forward(z0_pred, t_pred, f_pred)
        return xp

        # if self.param_discovery:
        #     xp_ed = self.ed_net(self.t_ed_col)
        #     return xp, xp_ed, self.t_ed_col
        # else:
        #     return xp


class osa_mdof_dataset(torch.utils.data.Dataset):

    def __init__(self, t_data, x_data, v_data, f_data = None, data_config = None, device = torch.device("cpu")):

        n_dof = x_data.shape[1]
        if data_config is dict:
            self.subsample = data_config['nct']  # number to subsample
            self.nct = data_config['nct']  # number of collocation points
        else:
            self.subsample = 8
            self.nct = 4
        nct = self.nct
        if x_data.shape[1] != v_data.shape[1]:
            raise Exception("Dimension mismatch for data, please check DOFs dimension of data")

        # normalise data based on range
        t_data, alpha_t = normalise(t_data, "range")
        x_data, alpha_x = normalise(x_data, "range", "all")
        v_data, alpha_v = normalise(v_data, "range", "all")
        if f_data is not None:
            f_data, alpha_f = normalise(f_data, "range", "all")

        # create dataset
        nt = t_data.shape[0]
        sobol_sampler = qmc.Sobol(d=1, seed=43810)
        samples = sobol_sampler.random_base2(m=int(np.log2(nt/32)))
        # Scale samples to the desired range. This example assumes you want integers from 0 to nt-1.
        sub_ind = np.sort((samples * nt).astype(int), axis=0).squeeze()
        # sub_ind = np.sort(qmc.Sobol(d=1, seed=43810).integers(l_bounds=nt, n=int(nt/self.subsample)), axis=0).squeeze()
        t_data_sub = t_data[sub_ind]
        x_data_sub = x_data[sub_ind, :]
        v_data_sub = v_data[sub_ind, :]
        f_data_sub = f_data[sub_ind, :]
        n_obs = x_data_sub.shape[0] - 1

        # observation set (uses state one data point ahead)
        x0_obs = x_data_sub[:-1, :]  # initial displacement input
        v0_obs = v_data_sub[:-1, :]  # initial velocity input
        t_obs = t_data_sub[:-1, :]  # time location in signal for observation (for plotting)
        dt_obs = torch.zeros((n_obs, 1))  # delta t for input to network
        for i in range(n_obs):
            dt_obs[i] = t_data_sub[i+1] - t_data_sub[i]
        if f_data is not None:
            f0_obs = f_data_sub[:-1, :]  # force input
        z_obs = torch.cat((x_data_sub[1:,:], v_data_sub[1:,:]), dim=1).requires_grad_()  # displacement at end of window (output)
        
        # collocation set (sets a copy of x0, v0 for a vector of time over the time horizon)
        x0_col = torch.zeros((n_obs, nct, n_dof))
        v0_col = torch.zeros((n_obs, nct, n_dof))
        dt_col = torch.zeros((n_obs, nct, 1))
        t_col = torch.zeros((n_obs, nct, 1))
        if f_data is not None:
            f0_col = torch.zeros((n_obs, nct, n_dof))

        for i in range(n_obs):
            x0_col[i, :, :] = x_data_sub[i, :]*torch.ones((nct, n_dof))
            v0_col[i, :, :] = v_data_sub[i, :]*torch.ones((nct, n_dof))
            dt_col[i, :, 0] = torch.linspace(0, t_data_sub[i+1].item() - t_data_sub[i].item(), nct)
            if f_data is not None:
                f0_col[i, :, :] = f_data_sub[i, :]*torch.ones((nct, n_dof))

            # generates a vector of the time for the predicted output, by simply adding the total window onto the current time in the data
            t_col[i, :, 0] = t_data_sub[i] + torch.linspace(0, t_data_sub[i+1].item() - t_data_sub[i].item(), nct)

        if f_data is not None:
            # concatenate into one large dataset
            data = torch.cat((x_data, v_data, f_data, t_data), dim=1)
            obs_data = torch.cat((z_obs, x0_obs, v0_obs, f0_obs, dt_obs, t_obs), dim=1)
            col_data = torch.cat((x0_col, v0_col, f0_col, dt_col, t_col), dim=2)
            self.alphas = {
                "x" : alpha_x, 
                "v" : alpha_v,
                "f" : alpha_f, 
                "t" : alpha_t
            }
        else:
            # concatenate into one large dataset
            data = torch.cat((x_data, v_data, t_data), dim=1)
            obs_data = torch.cat((z_obs, t_obs, x0_obs, v0_obs), dim=1)
            col_data = torch.cat((t_col, x0_col, v0_col), dim=2)
            self.alphas = {
                "x" : alpha_x, 
                "v" : alpha_v,
                "t" : alpha_t
            }

        self.ground_truth = data.to(device)
        self.obs_data = obs_data.to(device)
        self.col_data = col_data.to(device)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.obs_data[index, ...], self.col_data[index, ...]

    def get_original(self, index: int) -> np.ndarray:
        return self.ground_truth[index]

    def __len__(self) -> int:
        return self.obs_data.shape[0]

    def __repr__(self) -> str:
        return self.__class__.__name__



class bbnn(nn.Module):
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        self.n_input = N_INPUT
        self.n_output = N_OUTPUT
        self.n_hidden = N_HIDDEN
        self.n_layers = N_LAYERS
        self.activation = nn.Tanh

        self.build_net()
    
    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
        
    def forward(self, x):
        x = self.net(x)
        return x

    def predict(self, tp):
        yp = self.forward(tp)
        return yp

    def loss_func(self, x_obs, y_obs):
        yp_obs = self.forward(x_obs)
        loss = torch.mean((yp_obs - y_obs)**2)
        return loss

class ParamClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, 'phys_params'):
            params = module.phys_params.data
            params = params.clamp(0,1)
            module.phys_params.data = params
            
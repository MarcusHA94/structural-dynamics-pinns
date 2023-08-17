import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple

def max_mag_data(data: np.ndarray, axis: int=None) -> np.ndarray:
    if torch.is_tensor(data):
        if axis==None:
            data_max = torch.max(torch.max(torch.abs(data)))
        else:
            data_max = torch.max(torch.abs(data),dim=axis)[0]
    else:
        data_max = np.max(np.abs(data),axis=axis)
    return data_max

def range_data(data: np.ndarray, axis: int=None) -> np.ndarray:
    if torch.is_tensor(data):
        if axis==None:
            data_range = torch.max(torch.max(data)) - torch.min(torch.min(data))
        else:
            data_range = torch.max(data,dim=axis)[0] - torch.min(data,dim=axis)[0]
    else:
        data_range = np.max(data, axis=axis) - np.min(data, axis=axis)
    return data_range

def normalise(data: np.ndarray, norm_type: str="var",norm_dir: str="all") -> Union[Tuple[np.ndarray, float], Tuple[np.ndarray, np.ndarray]]:
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
    
def gen_state_matrices(M: torch.tensor, C: torch.tensor, K: torch.tensor, Kn: Union[torch.tensor, bool]=False) -> Union[Tuple[torch.tensor, torch.tensor], Tuple[torch.tensor, torch.tensor, torch.tensor]]:
    n_dof = M.shape[0]
    A = torch.cat((
        torch.cat((torch.zeros((n_dof, n_dof)), torch.eye(n_dof)), dim=1),
        torch.cat((-torch.linalg.inv(M)@K, -torch.linalg.inv(M)@C), dim=1)
        ), dim=0).requires_grad_()
    H = torch.cat((torch.zeros((n_dof, n_dof)),torch.linalg.inv(M)), dim=0)
    if Kn:
        An = torch.cat((
            torch.zeros((n_dof, n_dof)),
            -torch.linalg.inv(M)@Kn
            ), dim=0).requires_grad_()
        return A, H, An
    else:
        return A, H

class beam_pinn(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_modes = config['n_modes']

        self.n_input = 1
        self.n_output = self.n_modes * 2
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]
        self.n_modes = config["n_modes"]
        self.activation = nn.Tanh

        self.build_net()

        self.configure(**config)

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def forward(self, t, G=0.0, D=1.0):
        y_pass = self.net(t)
        y_masked = G + D * y_pass
        return y_masked
    
    def configure(self, **config):

        self.config = config

        # set mode shapes
        self.phi_col = config["modes"]["phi"] # [x, mode]
        self.phi_obs = self.phi_col[config["modes"]["sens_ids"], :]

        # set physical parameters
        self.param_attrs = {}
        for param_name, param_dict in config["phys_params"].items():
            self.param_attrs[param_name] = param_dict["type"]
            if param_dict["type"] == "constant":
                setattr(self, param_name, param_dict["value"])
            elif param_dict["type"] == "variable":
                self.register_parameter(param_name, nn.Parameter(param_dict["value"]))
                setattr(self,"alpha_"+param_name[:-1],config["alphas"][param_name[:-1]])
        if hasattr(self, "M") and hasattr(self,"C") and hasattr(self,"K"):
            self.A = torch.cat((
                    torch.cat((torch.zeros((self.n_modes,self.n_modes)),torch.eye(self.n_modes)), dim=1),
                    torch.cat((-torch.linalg.inv(self.M)@self.K, -torch.linalg.inv(self.M)@self.C), dim=1)
                    ), dim=0)
        if hasattr(self,"M"):
            self.H = torch.cat((torch.zeros((self.n_modes,self.n_modes)),torch.linalg.inv(self.M)), dim=0)

        # set normalisation parameters
        self.alpha_t = config["alphas"]["t"]
        self.alpha_x = config["alphas"]["x"]
        self.alpha_w = config["alphas"]["w"]
        self.alpha_wd = config["alphas"]["wd"]
        self.alpha_tau = torch.cat((self.alpha_w * torch.ones((self.n_modes, 1)), self.alpha_wd * torch.ones((self.n_modes, 1))), dim=0)

    def set_colls_and_obs(self, obs_data, coll_data):

        # observation data
        self.t_obs = obs_data['t_hat']
        self.x_obs = obs_data['x_hat']
        self.w_obs = obs_data['w_hat']
        self.wd_obs = obs_data['wd_hat']
        self.zz_obs = torch.cat((obs_data['w_hat'].unsqueeze(0), obs_data['wd_hat'].unsqueeze(0)), dim=0).requires_grad_()

        # collocation set
        self.t_col = coll_data['t_hat'].requires_grad_()
        self.x_col = coll_data['x_hat'].requires_grad_()

    def set_switches(self, lambdas):
        switches = {}
        for key, value in lambdas.items():
            switches[key] = value>0.0
        self.switches = switches
        return 0

    def loss_func(self, lambdas):

        if self.switches['obs']:
            # generate prediction at observation points
            tau_obs_pred = self.forward(self.t_obs)
            q_obs_pred = tau_obs_pred[:,:self.n_modes].T
            qd_obs_pred = tau_obs_pred[:,self.n_modes:].T
            w_obs_pred = torch.sum(torch.matmul(q_obs_pred.unsqueeze(2), self.phi_obs.T.unsqueeze(1)), dim=0).squeeze()
            wd_obs_pred = torch.sum(torch.matmul(qd_obs_pred.unsqueeze(2), self.phi_obs.T.unsqueeze(1)), dim=0).squeeze()
            z_obs_pred = torch.cat((w_obs_pred.unsqueeze(0), wd_obs_pred.unsqueeze(0)), dim=0)
            R_obs = torch.sum(torch.sqrt(torch.sum((z_obs_pred - self.zz_obs)**2, dim=2)), dim=0)
        
        if self.switches['ode'] or self.switches['cc']:
            # generate prediction over collocation domain
            tau_col_pred = self.forward(self.t_col)

            # retrieve derivatives
            dtau_dt = torch.zeros_like(tau_col_pred)
            for i in range(tau_col_pred.shape[1]):
                dtau_dt[:,i] = torch.autograd.grad(tau_col_pred[:,i], self.t_col, torch.ones_like(tau_col_pred[:,i]), create_graph=True)[0][:,0]  # ∂_t-hat N_tau-hat
            
            # retrieve physical parameters
            if hasattr(self, "A"):
                M, C, K = self.M, self.C, self.K
                A = self.A
            else:
                params = {}
                for param_name, param_dict in self.config["phys_params"].items():
                    if param_dict["type"] == "constant":
                        params[param_name] = param_dict["value"]
                    else:
                        params[param_name] = getattr(self,param_name)*getattr(self,"alpha_"+param_name[:-1])
                M, C, K = self.param_func(**params)
                A, _ = gen_state_matrices(M, C, K)
        
        if self.switches['ode']:
            R_ = (self.alpha_tau/self.alpha_t) * dtau_dt.T - A @ (self.alpha_tau * tau_col_pred.T)
            R_ode = R_[self.n_modes:, :].T
        else:
            R_ode = torch.zeros((2, 2))
        
        if self.switches['cc']:
            # continuity condition
            R_cc = R_[:self.n_modes, :].T
        else:
            R_cc = torch.zeros((2, 2))

        residuals = {
            "R_obs" : R_obs,
            "R_cc" : R_cc,
            "R_ode" : R_ode
        }

        L_obs = lambdas['obs'] * torch.mean(R_obs**2)
        L_cc = lambdas['cc'] * torch.sum(torch.mean(R_cc**2, dim=0), dim=0)
        L_ode = lambdas['ode'] * torch.sum(torch.mean(R_ode**2, dim=0), dim=0)

        loss = L_obs + L_cc + L_ode
        return loss, [L_obs, L_cc, L_ode], residuals
    
    def predict_w(self, t_pred):

        tau_pred = self.forward(t_pred)
        q_pred = tau_pred[:,:self.n_modes].T
        qd_pred = tau_pred[:,self.n_modes:].T
        w_pred = torch.sum(torch.matmul(q_pred.unsqueeze(2), self.phi_col.T.unsqueeze(1)), dim=0).squeeze()
        wd_pred = torch.sum(torch.matmul(qd_pred.unsqueeze(2), self.phi_col.T.unsqueeze(1)), dim=0).squeeze()
        z_pred = torch.cat((w_pred.unsqueeze(0), wd_pred.unsqueeze(0)), dim=0)

        return z_pred

        
                

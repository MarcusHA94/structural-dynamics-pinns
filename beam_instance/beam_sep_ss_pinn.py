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

class beam_sep_ss_pinn(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_modes = config['n_modes']

        self.n_input = 1
        self.n_output_state = self.n_modes * 2
        self.n_hidden_state = config["n_hidden"]
        self.n_layers_state = config["n_layers"]

        self.n_hidden_mode = 2 * self.n_modes
        self.n_layers_mode = 2
        self.n_output_mode = self.n_modes
        
        self.activation = nn.Tanh

        self.build_state_net()
        self.build_mode_net()

        self.configure(**config)

    def build_state_net(self):
        self.net_state = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden_state), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden_state, self.n_hidden_state), self.activation()]) for _ in range(self.n_layers_state - 1)]),
            nn.Linear(self.n_hidden_state, self.n_output_state)
            )
    
    def build_mode_net(self):
        self.net_mode = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden_mode), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden_mode, self.n_hidden_mode), self.activation()]) for _ in range(self.n_layers_mode - 1)]),
            nn.Linear(self.n_hidden_mode, self.n_output_mode)
            )
    
    def forward(self, t_span, x_span):

        tau_pred = self.net_state(t_span)
        phi_pred = self.net_mode(x_span)

        return tau_pred, phi_pred
    
    def configure(self, **config):

        self.config = config

        # set physical parameters
        self.param_attrs = {}
        for param_name, param_dict in config["phys_params"].items():
            self.param_attrs[param_name] = param_dict["type"]
            if param_dict["type"] == "constant":
                setattr(self, param_name, param_dict["value"])
            elif param_dict["type"] == "variable":
                self.register_parameter(param_name, nn.Parameter(param_dict["value"]))
                setattr(self,"alpha_"+param_name[:-1],config["alphas"][param_name[:-1]])

        # set normalisation parameters
        self.alpha_t = config["alphas"]["t"]
        self.alpha_x = config["alphas"]["x"]
        self.alpha_w = config["alphas"]["w"]
        self.alpha_wd = config["alphas"]["wd"]
        self.alpha_tau = torch.cat((self.alpha_w * torch.ones((self.n_modes, 1)), self.alpha_wd * torch.ones((self.n_modes, 1))), dim=0)

    def set_colls_and_obs(self, obs_data, coll_data):

        # observation data
        self.t_obs = obs_data['t_hat'].reshape(-1,1)
        self.x_obs = obs_data['x_hat'].reshape(-1,1)
        self.w_obs = obs_data['w_hat']
        self.wd_obs = obs_data['wd_hat']
        self.zz_obs = torch.cat((obs_data['w_hat'].unsqueeze(0), obs_data['wd_hat'].unsqueeze(0)), dim=0).requires_grad_()

        # collocation set
        self.t_col = coll_data['t_hat'].reshape(-1,1).requires_grad_()
        self.x_col = coll_data['x_hat'].reshape(-1,1).requires_grad_()

    def set_switches(self, lambdas):
        switches = {}
        for key, value in lambdas.items():
            switches[key] = value>0.0
        self.switches = switches

        if self.switches['bc']:
            self.bc_ids = torch.argwhere(torch.remainder(self.x_col.reshape(-1), 1.0) == 0.0).squeeze()

        return 0

    def loss_func(self, lambdas):

        if self.switches['obs']:
            # generate prediction at observation points
            tau_obs_pred, phi_obs_pred = self.forward(self.t_obs, self.x_obs)
            q_obs_pred = tau_obs_pred[:,:self.n_modes].T
            qd_obs_pred = tau_obs_pred[:,self.n_modes:].T
            w_obs_pred = torch.sum(torch.matmul(q_obs_pred.unsqueeze(2), phi_obs_pred.T.unsqueeze(1)), dim=0).squeeze()
            wd_obs_pred = torch.sum(torch.matmul(qd_obs_pred.unsqueeze(2), phi_obs_pred.T.unsqueeze(1)), dim=0).squeeze()
            z_obs_pred = torch.cat((w_obs_pred.unsqueeze(0), wd_obs_pred.unsqueeze(0)), dim=0)
            R_obs = torch.sum(torch.sqrt(torch.sum((z_obs_pred - self.zz_obs)**2, dim=2)), dim=0)
        
        if self.switches['ode'] or self.switches['cc'] or self.switches['bc']:
            # generate prediction over collocation domain
            tau_col_pred, phi_col_pred = self.forward(self.t_col, self.x_col)
            tau_col_pred = tau_col_pred.squeeze()

            # retrieve derivatives
            dtau_dt = torch.zeros_like(tau_col_pred)
            for i in range(tau_col_pred.shape[1]):
                dtau_dt[:,i] = torch.autograd.grad(tau_col_pred[:,i], self.t_col, torch.ones_like(tau_col_pred[:,i]), create_graph=True)[0][:,0]  # ∂_t-hat N_τ-hat

            dphi_dx = torch.zeros_like(phi_col_pred)
            d2phi_dx2 = torch.zeros_like(phi_col_pred)
            for i in range(phi_col_pred.shape[1]):
                dphi_dx[:,i] = torch.autograd.grad(phi_col_pred[:,i], self.x_col, torch.ones_like(phi_col_pred[:,i]), create_graph=True)[0][:,0]  # ∂_x-hat N_φ-hat
                d2phi_dx2[:,i] = torch.autograd.grad(dphi_dx[:,i], self.x_col, torch.ones_like(dphi_dx[:,i]), create_graph=True)[0][:,0]  # ∂2_x-hat N_φ-hat^2
            
            # retrieve physical parameters
            M_integ = torch.zeros((self.n_modes, self.n_modes))
            K_integ = torch.zeros((self.n_modes, self.n_modes))
            for i in range(self.n_modes):
                for j in range(self.n_modes):
                    if i < j: continue
                    m_integrand = phi_col_pred[:, i].reshape(-1, 1) * phi_col_pred[:, j].reshape(-1, 1)
                    M_integ[i, j] = torch.trapezoid(m_integrand.reshape(-1), self.x_col.reshape(-1))
                    k_integrand = d2phi_dx2[:, i].reshape(-1, 1) * d2phi_dx2[:, j].reshape(-1, 1)
                    K_integ[i, j] = torch.trapezoid(k_integrand.reshape(-1), self.x_col.reshape(-1))
                    if i != j:
                        M_integ[j, i] = M_integ[i, j]
                        K_integ[j, i] = K_integ[i, j]
            M = self.pA * M_integ
            C = self.pA * self.c * M_integ
            K = self.EI * K_integ
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

        if self.switches['bc']:
            R_bc = phi_col_pred[self.bc_ids, :]
        else:
            R_bc = torch.zeros((2, 2))

        residuals = {
            "R_obs" : R_obs,
            "R_cc" : R_cc,
            "R_ode" : R_ode,
            "R_bc" : R_bc
        }

        L_obs = lambdas['obs'] * torch.mean(R_obs**2)
        L_cc = lambdas['cc'] * torch.sum(torch.mean(R_cc**2, dim=0), dim=0)
        L_ode = lambdas['ode'] * torch.sum(torch.mean(R_ode**2, dim=0), dim=0)
        L_bc = lambdas['bc'] * torch.sum(torch.mean(R_bc**2, dim=0), dim=0)

        loss = L_obs + L_cc + L_ode + L_bc
        return loss, [L_obs, L_cc, L_ode, L_bc], residuals
    
    def predict_w(self, t_pred, x_pred):

        tau_pred, phi_pred = self.forward(t_pred, x_pred)
        q_pred = tau_pred[:,:self.n_modes].T
        qd_pred = tau_pred[:,self.n_modes:].T
        w_pred = torch.sum(torch.matmul(q_pred.unsqueeze(2), phi_pred.T.unsqueeze(1)), dim=0).squeeze()
        wd_pred = torch.sum(torch.matmul(qd_pred.unsqueeze(2), phi_pred.T.unsqueeze(1)), dim=0).squeeze()
        z_pred = torch.cat((w_pred.unsqueeze(0), wd_pred.unsqueeze(0)), dim=0)

        return z_pred, phi_pred

        
                

import torch
import torch.nn as nn
import numpy as np
from math import pi

from typing import Tuple, Union
Tensor = Union[torch.Tensor, np.ndarray]

def max_mag_data(data: Tensor, axis: int = None) -> Tensor:
    """
    Compute the maximum magnitude of data along the specified axis.
    """
    if torch.is_tensor(data):
        if axis==None:
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
        if axis==None:
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

class mdof_dataset(torch.utils.data.Dataset):

    def __init__(self, x_data, v_data, f_data, t_data, subsample):

        self.subsample = subsample  # subsample simulates lower sample rate
        n_dof = x_data.shape[1]
        if x_data.shape[1] != v_data.shape[1]:
            raise Exception("Dimension mismatch for data, please check DOFs dimension of data")

        # normalise data based on range
        t_data, alpha_t = normalise(t_data, "range")
        x_data, alpha_x = normalise(x_data, "range", "all")
        v_data, alpha_v = normalise(v_data, "range", "all")
        f_data, alpha_f = normalise(f_data, "range", "all")
        
        # concatenate into one large dataset
        data = np.concatenate((x_data, v_data, f_data, t_data), axis=1)
        self.alphas = {
            "x" : alpha_x, 
            "v" : alpha_v,
            "f" : alpha_f, 
            "t" : alpha_t
        }

        # reshape to number of batches
        # 2 ndof for state, 1 ndof for force, 1 for time
        self.ground_truth = data
        col_data = data[:(data.shape[0] // (subsample)) * (subsample)]  # cut off excess data
        data = col_data[::self.subsample]
        self.col_data = col_data.reshape((-1, self.subsample, 3 * n_dof + 1))
        self.data = data.reshape((-1, 3 * n_dof + 1))

    def __getitem__(self, index: int) -> np.ndarray:
        return self.data[index, :], self.col_data[index, :]
    
    def get_original(self, index: int) -> np.ndarray:
        return self.ground_truth[index]

    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __repr__(self) -> str:
        return self.__class__.__name__


class mdof_pinn_stoch(nn.Module):

    def __init__(self, config: dict, device=None):
        super().__init__()
        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]
        self.n_dof = config["n_dof"]
        self.activation = nn.Tanh
        self.t_pi = torch.tensor(pi)

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.build_net()

        self.configure(config)

    def build_net(self) -> None:
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            ).to(self.device)
        return self.net
    
    def forward(self, x: torch.Tensor, G: torch.Tensor = torch.tensor(0.0), D: torch.Tensor = torch.tensor(1.0)) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input to network
            G (torch.Tensor): Tensor of BC values
            D (torch.Tensor): Tensor of BC extension mask

        Returns:
            torch.Tensor: Output tensor.
        """
        y = G + D * self.net(x)
        return y
    
    def configure(self, config: dict) -> None:
        """
        Configures neural network

        Args:
            config (dict): Configuration parameters
        """

        self.config = config

        self.nonlinearity = config["nonlinearity"]
        self.param_func = config["param_func"]

        self.nct = config["nct"]  # number of time collocation points

        self.set_phys_params()
        self.set_norm_params()

    def set_phys_params(self) -> None:
        """
        Set physical parameters of model, and adds them as either constants or parameters for optimisation
        """

        config = self.config
        self.param_attrs = {}
        for param_name, param_dict in config["phys_params"].items():
            self.param_attrs[param_name] = param_dict["type"]
            if param_dict["type"] == "constant":
                setattr(self,param_name, param_dict["value"])
            elif param_dict["type"] == "variable":
                self.register_parameter(param_name, nn.Parameter(torch.ones_like(param_dict["value"])))
        if hasattr(self,"M") and hasattr(self,"C") and hasattr(self,"K"):
            self.A = torch.cat((
                    torch.cat((torch.zeros((self.n_dof,self.n_dof)),torch.eye(self.n_dof)), dim=1),
                    torch.cat((-torch.linalg.inv(self.M)@self.K, -torch.linalg.inv(self.M)@self.C), dim=1)
                    ), dim=0).to(self.device)
        elif hasattr(self,"M"):
            self.m_ = torch.diag(self.M).to(self.device)  # takes diagonal from mass matrix if set as constant
        
        if hasattr(self,"M"):
            self.H = torch.cat((torch.zeros((self.n_dof,self.n_dof)),torch.linalg.inv(self.M)), dim=0).to(self.device)
        if hasattr(self,"Kn") and config["nonlinearity"]=="cubic":
            self.An = torch.cat((
                    torch.zeros((self.n_dof,self.n_dof)),
                    -torch.linalg.inv(self.M)@self.Kn
                ), dim=0).to(self.device)

    def set_norm_params(self) -> None:
        """
        Set normalisation parameters of the model
        """
        config = self.config
        self.alpha_t = config["alphas"]["t"]
        self.alpha_x = config["alphas"]["x"]
        self.alpha_v = config["alphas"]["v"]
        self.alpha_z = torch.cat((self.alpha_x*torch.ones(self.n_dof,1), self.alpha_v*torch.ones(self.n_dof,1)), dim=0)
        self.alpha_f = config["alphas"]["f"]
        for param_name, param_dict in config["phys_params"].items():
            if param_dict["type"] == "variable":
                setattr(self,"alpha_"+param_name[:-1],config["alphas"][param_name[:-1]])

    def Kn_func(self, kn_: torch.Tensor) -> torch.Tensor:
        """
        Generates Kn matrix
        
        Args:
            kn_ (torch.Tensor): vector of kn values
    
        """
        Kn = torch.zeros((self.n_dof,self.n_dof), dtype=torch.float32)
        for n in range(self.n_dof):
            Kn[n,n] = kn_[n]
        for n in range(self.n_dof-1):
            Kn[n,n+1] = -kn_[n+1]
        return Kn.requires_grad_()
    
    def set_switches(self, lambdas: dict) -> None:
        """
        Sets switches for residual/loss calculation to improve performance of unecessary calculation
        Args:
            lambdas (dict): dictionary of lambda weighting parameters
        """
        switches = {}
        for key, value in lambdas.items():
            switches[key] = value>0.0
        self.switches = switches


    def loss_func(self, lambdas: dict, t_obs: torch.Tensor, z_obs: torch.Tensor, t_col: torch.Tensor, f_col: torch.Tensor) -> Tuple[torch.Tensor, list, dict]:
        """
        Calculate the loss values.

        Args:
            lambdas (dict): Dictionary of lambda weighting parameters

        Returns:
            torch.Tensor: Total loss
            list: list containing individual losses
            dict: dictionary of residuals
        """

        if self.switches['obs']:
            # generate prediction at observation points
            zh_obs_hat = self.forward(t_obs)
            R_obs = torch.sqrt(torch.sum((zh_obs_hat - z_obs)**2, dim=1))

        if self.switches['ode'] or self.switches['cc']:
            # generate prediction over collocation domain
            zh_col_hat = self.forward(t_col)

            # retrieve derivatives
            dxdt = torch.zeros_like(zh_col_hat)
            for i in range(zh_col_hat.shape[1]):
                dxdt[:,i] = torch.autograd.grad(zh_col_hat[:,i], t_col, torch.ones_like(zh_col_hat[:,i]), create_graph=True)[0][:,0]  # ∂_t-hat N_x-hat

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
                A = torch.cat((
                    torch.cat((torch.zeros((self.n_dof,self.n_dof)),torch.eye(self.n_dof)), dim=1),
                    torch.cat((-torch.linalg.inv(M)@K, -torch.linalg.inv(M)@C), dim=1)
                    ), dim=0).requires_grad_()
            if hasattr(self,"H"):
                H = self.H
            else:
                H = torch.cat((torch.zeros((self.n_dof,self.n_dof)),torch.linalg.inv(M)), dim=0)
            if hasattr(self,"kn_"):
                if self.config["phys_params"]["kn_"]["type"]=="constant":
                    Kn = self.Kn_func(self.kn_)
                else:
                    Kn = self.Kn_func(self.kn_*self.alpha_kn)
                An = torch.cat((
                    torch.zeros((self.n_dof,self.n_dof)),
                    -torch.linalg.inv(M)@Kn
                    ), dim=0).requires_grad_()
                    
        if self.switches['ode']:
            match self.config:
                case {"nonlinearity":"linear","forcing":None}:
                    R_ = (self.alpha_z/self.alpha_t)*dxdt.T - A@(self.alpha_z*zh_col_hat.T)
                    R_ode = R_[self.n_dof:,:].T
                case {"nonlinearity":"cubic","forcing":None}:
                    q = (self.alpha_x*zh_col_hat[:,:self.n_dof] - torch.cat((torch.zeros(zh_col_hat.shape[0],1),self.alpha_x*zh_col_hat[:,:self.n_dof-1]),dim=1))**3
                    R_ = (self.alpha_z/self.alpha_t)*dxdt.T - A@(self.alpha_z*zh_col_hat.T) - An@(q.T)
                    R_ode = R_[self.n_dof:,:].T
                case {"nonlinearity":"linear","forcing":torch.Tensor()}:
                    R_ = (self.alpha_z/self.alpha_t)*dxdt.T - A@(self.alpha_z*zh_col_hat.T) - H@(self.alpha_f*f_col.T)
                    R_ode = R_[self.n_dof:,:].T
                case {"nonlinearity":"cubic","forcing":torch.Tensor()}:
                    q = (self.alpha_x*zh_col_hat[:,:self.n_dof] - torch.cat((torch.zeros(zh_col_hat.shape[0],1),self.alpha_x*zh_col_hat[:,:self.n_dof-1]),dim=1))**3
                    R_ = (self.alpha_z/self.alpha_t)*dxdt.T - A@(self.alpha_z*zh_col_hat.T) - An@(q.T) - H@(self.alpha_f*f_col.T)
                    R_ode = R_[self.n_dof:,:].T
            R_ode = torch.sqrt(torch.sum((R_ode)**2, dim=1))
        else:
            R_ode = torch.ones((2,1))

        if self.switches['cc']:
            # continuity condition residual
            R_cc = R_[:self.n_dof,:].T
        else:
            R_cc = torch.zeros((2,1))

        residuals = {
            "R_obs" : R_obs,
            "R_cc" : R_cc,
            "R_ode" : R_ode
        }

        # L_obs = lambdas['obs'] * torch.mean(R_obs**2)
        N_o = R_obs.shape[0]
        if self.switches['obs']:
            log_likeli_obs = -N_o * torch.log(self.sigma_) - (N_o/2) * torch.log(2*self.t_pi) - 0.5 * torch.sum((R_obs**2/self.sigma_**2))
        else:
            log_likeli_obs = torch.tensor(0.0)
        L_obs = lambdas['obs'] * -log_likeli_obs

        N_c = R_ode.shape[0]
        if self.switches['ode']:
            log_likeli_ode = -N_c * torch.log(self.sigma_s_) - (N_c/2) * torch.log(2*self.t_pi) - 0.5 * torch.sum((R_ode**2/(self.sigma_s_**2)))
        else:
            log_likeli_ode = torch.tensor(0.0)
        # L_ode = lambdas['ode'] * torch.sum(torch.mean(R_ode**2, dim=0), dim=0)
        L_ode = lambdas['ode'] * -log_likeli_ode

        L_cc = lambdas['cc'] * torch.sum(torch.mean(R_cc**2, dim=0), dim=0)

        loss = L_obs + L_ode
        return loss, [L_obs, L_cc, L_ode], residuals
    
    def predict(self, t_pred: torch.Tensor, theta_s = None) -> torch.Tensor:
        """
        Predict state values
        """
        zp = self.forward(t_pred)
        # retrieve derivatives
        dxdt = torch.zeros_like((zp))
        for i in range(zp.shape[1]):
            dxdt[:,i] = torch.autograd.grad(zp[:,i], t_pred, torch.ones_like(zp[:,i]), create_graph=True)[0][:,0]  # ∂_t-hat N_x-hat

        # retrieve physical parameters
        if hasattr(self,"A") and (theta_s is None):
            M, C, K = self.M, self.C, self.K
            A = self.A
        else:
            if theta_s is None:
                params = {}
                for param_name, param_dict in self.config["phys_params"].items():
                    if param_dict["type"] == "constant":
                        params[param_name] = param_dict["value"]
                    else:
                        params[param_name] = getattr(self,param_name)*getattr(self,"alpha_"+param_name[:-1])
                M, C, K = self.param_func(params["m_"],params["c_"],params["k_"])
            else:
                M, C, K = self.param_func(self.m_, theta_s[:self.n_dof]*self.alpha_c, theta_s[self.n_dof:2*self.n_dof]*self.alpha_k)
            A = torch.cat((
                torch.cat((torch.zeros((self.n_dof,self.n_dof)),torch.eye(self.n_dof)), dim=1),
                torch.cat((-torch.linalg.inv(M)@K, -torch.linalg.inv(M)@C), dim=1)
                ), dim=0).requires_grad_()
                
        if hasattr(self,"H"):
            H = self.H
        else:
            H = torch.cat((torch.zeros((self.n_dof,self.n_dof)),torch.linalg.inv(M)), dim=0)
        if hasattr(self,"kn_"):
            if self.config["phys_params"]["kn_"]["type"]=="constant":
                Kn = self.Kn_func(self.kn_)
            elif theta_s is None:
                Kn = self.Kn_func(self.kn_*self.alpha_kn)
            else:
                Kn = self.Kn_func(theta_s[2*self.n_dof:]*self.alpha_kn)
            An = torch.cat((
                torch.zeros((self.n_dof,self.n_dof)),
                -torch.linalg.inv(M)@Kn
                ), dim=0).requires_grad_()
        
        match self.config:
            case {"nonlinearity":"linear"}:
                Hf_pred = (self.alpha_z/self.alpha_t)*dxdt.T - A@(self.alpha_z*zp.T)
            case {"nonlinearity":"cubic"}:
                q = (self.alpha_x*zp[:,:self.n_dof] - torch.cat((torch.zeros(zp.shape[0],1),self.alpha_x*zp[:,:self.n_dof-1]),dim=1))**3
                Hf_pred = (self.alpha_z/self.alpha_t)*dxdt.T - A@(self.alpha_z*zp.T) - An@(q.T)
        f_pred = torch.linalg.inv(M) @ Hf_pred[self.n_dof:,:] / self.alpha_f
        
        return zp, f_pred.T

    def physics_likelihood(self, theta_s=None):

        z_pred, f_pred = self.predict(theta_s)

        N_c = f_pred.shape[0]
        f_res = torch.sqrt(torch.sum((self.f_col - self.f_pred)**2, dim=1))
        log_likeli_ode = - N_c * torch.log(self.sigma_s_) - (N_c/2) * torch.log(2*self.t_pi) - 0.5 * torch.sum((f_res**2/(self.sigma_s_**2)))
        return log_likeli_ode


class ParamClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, 'c_'):
            params_c = module.c_.data
            params_c = params_c.clamp(0,None)
            module.c_.data = params_c
        if hasattr(module, 'k_'):
            params_k = module.k_.data
            params_k = params_k.clamp(0,None)
            module.k_.data = params_k
        if hasattr(module, 'kn_'):
            params_kn = module.kn_.data
            params_kn = params_kn.clamp(0,None)
            module.kn_.data = params_kn
            
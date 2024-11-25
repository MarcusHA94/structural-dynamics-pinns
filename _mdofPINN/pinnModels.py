import torch
import torch.nn as nn
import numpy as np
import math
from pinnUtils import normalise

from typing import Tuple, Union, Optional
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
        if isinstance(self.gc_exp, float):
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
        if len(z.shape) == 3:
            dofs = int(z.shape[1]/2)
            x_    = z[:, :dofs, :] - torch.cat((torch.zeros((z.shape[0], 1, z.shape[2])), z[:, :dofs-1, :]), dim=1)
            xdot_ = z[:, dofs:, :] - torch.cat((torch.zeros((z.shape[0], 1, z.shape[2])), z[:, dofs:-1, :]), dim=1)
            return torch.cat((self.gk_func(x_, xdot_), self.gc_func(x_, xdot_)), dim=1)
        else:
            dofs = int(z.shape[0]/2)
            x_    = z[:dofs, :] - torch.cat((torch.zeros((1, z.shape[1])), z[:dofs-1, :]), dim=0)
            xdot_ = z[dofs:, :] - torch.cat((torch.zeros((1, z.shape[1])), z[dofs:-1, :]), dim=0)
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

class mdof_pinn_model(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]
        self.seq_len = config["seq_len"]
        self.n_dof = config["n_dof"]
        if 'activation' in config.keys():
            self.activation = getattr(nn, config["activation"])
        else:
            # self.activation = nn.Tanh
            self.activation = nn.SiLU
        if 'net_split' in config.keys():
            self.net_split = config["net_split"]
        else:
            self.net_split = False
        self.device = config["device"]

        self.build_nets()
        self.config = config
        
        self.pinn_type = 'normal'

    def gather_params(self):
        self.net_params_list = []
        if self.net_split:
            for net in self.nets:
                for net_ in net: self.net_params_list.append(net_.parameters())
        else:
            for net in self.nets:
                self.net_params_list.append(net.parameters())

    def build_nets(self):
        if self.net_split:
            nets = [[None] * 2] * self.seq_len
            for net_n in range(self.seq_len):
                nets[net_n][0] = nn.Sequential(
                    nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
                    nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
                    nn.Linear(self.n_hidden, self.n_dof)
                    ).to(self.device)
                nets[net_n][1] = nn.Sequential(
                    nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
                    nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
                    nn.Linear(self.n_hidden, self.n_dof)
                    ).to(self.device)
            self.nets = tuple(nets)
            self.network_parameters = []
            for net in self.nets:
                for net_ in net: self.network_parameters += list(net_.parameters())
            pass
        else:
            nets = [None] * self.seq_len
            for net_n in range(self.seq_len):
                nets[net_n] = nn.Sequential(
                    nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
                    nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
                    nn.Linear(self.n_hidden, self.n_output)
                    ).to(self.device)
            self.nets = tuple(nets)
            # self.nets = nn.ModuleList(self.nets)
            self.network_parameters = []
            for net in self.nets: self.network_parameters += list(net.parameters())

    def build_net(self) -> int:
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input * self.seq_len, self.n_hidden * self.seq_len), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden * self.seq_len, self.n_hidden * self.seq_len), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden * self.seq_len, self.n_output * self.seq_len),
            nn.Unflatten(dim=1, unflattened_size = (self.seq_len, self.n_output))
            )
        return 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Args:
            x [(torch.Tensor)]: List of inputs to networks [samples x n_input] * seq_len

        Returns:
            torch.Tensor: Output tensor.
        """
        if self.net_split:
            y = []
            for nq in range(self.seq_len):
                z1 = self.nets[nq][0](x[nq])
                z2 = self.nets[nq][1](x[nq])
                y.append(torch.cat((z1, z2), dim=1))
        else:
            y = [torch.zeros((x[0].shape[0], self.n_output)).to(self.device)] * self.seq_len
            for nq in range(self.seq_len):
                y[nq] = self.nets[nq](x[nq])
        return tuple(y)  # [samples x n_output] * seq_len
    
    def forward_new(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input to network  [seq_len x samples x n_input]

        Returns:
            torch.Tensor: Output tensor.
        """
        y = torch.zeros((self.seq_len, x.shape[1], self.n_output)).to(self.device)
        for nq in range(self.seq_len):
            y[nq] = self.nets[nq](x[nq].unsqueeze(1))
        return y  # [seq_len x samples x n_output]
    
    def configure(self, param_func, nonlin_func) -> None:
        """
        Configures neural network

        Args:
            config (dict): Configuration parameters
        """
        
        self.param_func = param_func
        self.nonlinearity = nonlin_func

        self.set_phys_params()
        self.set_norm_params()
        
        self.gather_params()
        self.set_switches(self.config['lambds'])

    def set_phys_params(self) -> None:
        """
        Set physical parameters of model, and adds them as either constants or parameters for optimisation
        """
        config = self.config
        self.param_attrs = {}
        self.system_parameters = []
        
        #TODO: implement sparse matrices to see if it speeds up computation
        
        if all (param["type"]  == 'constant' for param in config["phys_params"].values()):
            params = {}
            for param_name, param_dict in self.config["phys_params"].items():
                params[param_name] = param_dict["value"]
            m_vec, c_vec, k_vec, kn_vec, cn_vec = self.param_parser(params)
            M, C, K = self.param_func(m_vec, c_vec, k_vec)
            self.M = M
            self.C = C
            self.K = K
            invM = torch.diag(1/torch.diag(M))
            # state matrices
            self.A = torch.cat((
                torch.cat((torch.zeros((self.n_dof,self.n_dof)),torch.eye(self.n_dof)), dim=1),
                torch.cat((-invM @ K, -invM @ C), dim=1)
                ), dim=0)
            self.H = torch.cat((torch.zeros((self.n_dof, self.n_dof)), invM), dim=0)
            if self.nonlinearity is not None:
                self.An = self.nonlinearity.mat_func(kn_vec, cn_vec, invM)
            # observation matrices
            self.B = torch.cat((-invM @ K, -invM @ C), dim=1)
            self.D = invM
            if self.nonlinearity is not None:
                self.Bn = self.nonlinearity.mat_func(kn_vec, cn_vec, invM)[self.n_dof:, :]
        
        for param_name, param_dict in config["phys_params"].items():
            self.param_attrs[param_name] = param_dict["type"]
            if param_dict["type"] == "constant":
                setattr(self,param_name,param_dict["value"])
            elif param_dict["type"] == "variable":
                self.register_parameter(param_name, nn.Parameter(param_dict["value"]))
                self.system_parameters.append(getattr(self, param_name))
        if hasattr(self, "M") and hasattr(self, "C") and hasattr(self, "K"):
            self.A = torch.cat((
                    torch.cat((torch.zeros((self.n_dof,self.n_dof)),torch.eye(self.n_dof)), dim=1),
                    torch.cat((-torch.linalg.inv(self.M)@self.K, -torch.linalg.inv(self.M)@self.C), dim=1)
                    ), dim=0)
        elif hasattr(self,"M"):
            self.m_ = torch.diag(self.M)  # takes diagonal from mass matrix if set as constant
        
        obs_dropouts = config["dropouts"]
        obs_keep = [j for j in range(self.n_dof) if j not in obs_dropouts]
        self.Sa = torch.diag(torch.tensor(obs_keep))

    def set_norm_params(self) -> None:
        """
        Set normalisation parameters of the model
        """
        config = self.config
        
        # signal value norms
        self.alpha_t = config["alphas"]["t"].clone().detach().to(self.device)
        self.alpha_x = config["alphas"]["x"].clone().detach().to(self.device)
        self.alpha_v = config["alphas"]["v"].clone().detach().to(self.device)
        self.alpha_z = torch.cat((self.alpha_x.item()*torch.ones(self.n_dof,1), self.alpha_v.item()*torch.ones(self.n_dof,1)), dim=0).float().to(self.device)
        self.alpha_a = config["alphas"]["a"].clone().detach().to(self.device)
        self.alpha_f = config["alphas"]["f"].clone().detach().to(self.device)
        
        # system parameter norms
        self.alpha_c = config["alphas"]["c"].clone().detach().to(self.device)
        self.alpha_k = config["alphas"]["k"].clone().detach().to(self.device)
        self.alpha_kn = config["alphas"]["kn"].clone().detach().to(self.device)
        self.alpha_cn = config["alphas"]["cn"].clone().detach().to(self.device)
        # for param_name, param_dict in config["phys_params"].items():
        #     if param_dict["type"] == "variable":
        #         setattr(self,"alpha_"+param_name[:-2], config["alphas"][param_name[:-2]])
        #     else:
        #         setattr(self, "alpha_"+param_name[:-2], 1.0)

    def set_aux_funcs(self, param_func, nonlin_func):
        self.param_func = param_func
        self.nonlinearity = nonlin_func
    
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

    def param_parser(self, params) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parses physical parameters into matrices
        
        Args:
            params (dict): dictionary of physical parameters

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: mass, damping, stiffness, nonlinear stiffnes and nonlinear damping vectors
        """

        m_vec = torch.zeros(self.n_dof)
        c_vec = torch.zeros(self.n_dof)
        k_vec = torch.zeros(self.n_dof)
        kn_vec = torch.zeros(self.n_dof)
        cn_vec = torch.zeros(self.n_dof)

        for i in range(self.n_dof):
            m_vec[i] = params[f'm_{i}']
            c_vec[i] = params[f'c_{i}']
            k_vec[i] = params[f'k_{i}']
            kn_vec[i] = params[f'kn_{i}']
            cn_vec[i] = params[f'cn_{i}']

        return m_vec, c_vec, k_vec, kn_vec, cn_vec

    def retrieve_state_matrices(self, theta_s: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if theta_s is None:
            if hasattr(self,"A"):
                M, C, K = self.M, self.C, self.K
                A = self.A
            else:
                params = {}
                for param_name, param_dict in self.config["phys_params"].items():
                    if param_dict["type"] == "constant":
                        params[param_name] = param_dict["value"]
                    else:
                        if param_name[-2] == "_":
                            params[param_name] = getattr(self,param_name)*getattr(self,"alpha_"+param_name[:-2])
                        else:
                            params[param_name] = getattr(self,param_name)*getattr(self,"alpha_"+param_name[:-3])
                m_vec, c_vec, k_vec, kn_vec, cn_vec = self.param_parser(params)
                M, C, K = self.param_func(m_vec, c_vec, k_vec)
                invM = torch.diag(1/torch.diag(M))
                A = torch.cat((
                    torch.cat((torch.zeros((self.n_dof, self.n_dof)), torch.eye(self.n_dof)), dim=1),
                    torch.cat((-invM @ K, -invM @ C), dim=1)
                    ), dim=0).requires_grad_()
        else:
            if hasattr(self,"A"):
                M, C, K = self.M, self.C, self.K
                A = self.A
            else:
                M, C, K = self.param_func(self.m_, theta_s[:self.n_dof]*self.alpha_c, theta_s[self.n_dof:2*self.n_dof]*self.alpha_k)
                invM = torch.diag(1/torch.diag(M))
                A = torch.cat((
                    torch.cat((torch.zeros((self.n_dof, self.n_dof)), torch.eye(self.n_dof)), dim=1),
                    torch.cat((-invM @ K, -invM @ C), dim=1)
                    ), dim=0).requires_grad_()
        if hasattr(self,"H"):
            H = self.H
        else:
            H = torch.cat((torch.zeros((self.n_dof, self.n_dof)), invM), dim=0)

        # nonlinear parameters
        if self.nonlinearity is not None:
            if hasattr(self,"An"):
                An = self.An
            else:
                An = self.nonlinearity.mat_func(kn_vec, cn_vec, invM)
        else:
            An = torch.zeros((self.n_dof, self.n_dof))
                
        return A, H, An
    
    def retrieve_obs_matrices(self, theta_s: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if theta_s is None:
            if hasattr(self,"B"):
                B = self.B
            else:
                params = {}
                for param_name, param_dict in self.config["phys_params"].items():
                    if param_dict["type"] == "constant":
                        params[param_name] = param_dict["value"]
                    else:
                        if param_name[-2] == "_":
                            params[param_name] = getattr(self,param_name)*getattr(self,"alpha_"+param_name[:-2])
                        else:
                            params[param_name] = getattr(self,param_name)*getattr(self,"alpha_"+param_name[:-3])
                m_vec, c_vec, k_vec, kn_vec, cn_vec = self.param_parser(params)
                M, C, K = self.param_func(m_vec, c_vec, k_vec)
                invM = torch.diag(1/torch.diag(M))
                B = torch.cat((-invM @ K, -invM @ C), dim=1).requires_grad_()
        else:
            if hasattr(self,"B"):
                B = self.B
            else:
                M, C, K = self.param_func(self.m_, theta_s[:self.n_dof]*self.alpha_c, theta_s[self.n_dof:2*self.n_dof]*self.alpha_k)
                invM = torch.diag(1/torch.diag(M))
                B = torch.cat((-invM @ K, -invM @ C), dim=1).requires_grad_()
        if hasattr(self,"D"):
            D = self.D
        else:
            D = invM
        if self.nonlinearity is not None:
            if hasattr(self,"Bn"):
                Bn = self.Bn
            else:
                Bn = self.nonlinearity.mat_func(kn_vec, cn_vec, invM)[self.n_dof:, :]
        else:
            Bn = torch.zeros((self.n_dof, 2 * self.n_dof))
                
        return B, D, Bn
    
    def loss_func(
        self,
        t_obs_list: list[torch.Tensor],
        acc_obs: torch.Tensor,
        f_obs: Tuple[torch.Tensor, None],
        t_col_list: Tuple[list[torch.Tensor], None],  # when None use same domain as observations
        f_col: Tuple[torch.Tensor, None],  # when None use same domain as observations
        lambdas: dict,
        obs_dropouts: Optional[list] = False,
        acc_obs_method: str = 'obs_model'
        ) -> Tuple[torch.Tensor, list, dict]:
        """
        Calculates residuals for loss functions

        Args:
            t_obs_list [(torch.Tensor)]: time values for observation domain [[sample x 1]] * sequence
            acc_obs (torch.Tensor): observations of state [sample x sequence x dof]
            f_obs (torch.Tensor): observations of force [sample x sequence x dof]
            t_col [(torch.Tensor)]: time values for collocation domain [[sample x 1]] * sequence
            f_col (torch.Tensor): measurements of force in collocation domain [sample x sequence x dof]
            lambdas (dict): dictionary of loss weighting parameters

        Returns:
            loss: total loss
            losses: list of individual losses
            residuals: dictionary of residuals

        """
        
        if self.switches['obs'] or self.switches['occ']:
            # generate prediction at observation points
            zp_obs_hat = self.forward(t_obs_list)  # list of predicted states [samples x state_dim] * seq_len
                
            # retrieve system matrices
            B, D, Bn = self.retrieve_obs_matrices()
            
        if self.switches['obs']:
            # calculate residuals
            #TODO: use Sa matrix for dropouts
            if obs_dropouts:
                idx_obs = [j for j in range(self.n_dof) if j not in obs_dropouts]  # indices of observed DOFs
                R_obs = torch.zeros((t_obs_list[0].shape[0])).to(self.device)  # initialise observation residuals [samples]
                for nq in range(self.seq_len):
                    B_ = B[idx_obs, :]  # reduced observation matrix
                    Bn_ = Bn[idx_obs, :]  # reduced nonlinear state matrix
                    D_ = D[idx_obs, :][:, idx_obs]  # reduced force matrix
                    acc_obs_ = self.alpha_a * acc_obs[:, nq, idx_obs].T  # acceleration observations [n_dof_obs x samples]
                    f_obs_ = self.alpha_f * f_obs[:, nq, idx_obs].T if f_obs is not None else None  # force observations [n_dof_obs x samples]
                    alpha_z_ = self.alpha_z
                    if self.nonlinearity is None:
                        if f_obs is None:
                            R_obs_seq = (B_ @ (alpha_z_ * zp_obs_hat[nq].T) - acc_obs_).T
                        else:
                            R_obs_seq = (B_ @ (alpha_z_ * zp_obs_hat[nq].T) + D_ @ f_obs_ - acc_obs_).T
                    else:
                        gz = self.nonlinearity.gz_func(alpha_z_ * zp_obs_hat[nq].T)
                        if f_obs is None:
                            R_obs_seq = (B_ @ (alpha_z_ * zp_obs_hat[nq].T) + Bn_ @ gz - acc_obs_).T
                        else:
                            R_obs_seq = (B_ @ (alpha_z_ * zp_obs_hat[nq].T) + Bn_ @ gz + D_ @ f_obs_ - acc_obs_).T
                    R_obs += torch.sqrt(torch.sum(R_obs_seq**2, dim=1))
            else:
                R_obs = torch.zeros((t_obs_list[0].shape[0])).to(self.device)
                for nq in range(self.seq_len):
                    acc_obs_ = self.alpha_a * acc_obs[:, nq, :].T
                    f_obs_ = self.alpha_f * f_obs[:, nq, :].T
                    alpha_z_ = self.alpha_z
                    match [self.nonlinearity, f_obs]:
                        case None, None:
                            R_obs_seq = (B @ (alpha_z_ * zp_obs_hat[nq].T) - acc_obs_).T
                            R_obs += torch.sqrt(torch.sum(R_obs_seq**2, dim=1))
                        case [_, None]:
                            R_obs_seq = (B @ (alpha_z_ * zp_obs_hat[nq].T) + Bn @ self.nonlinearity.gz_func(alpha_z_ * zp_obs_hat[nq].T) - acc_obs_).T
                            R_obs += torch.sqrt(torch.sum(R_obs_seq**2, dim=1))
                        case [None, torch.Tensor()]:
                            R_obs_seq = (B @ (alpha_z_ * zp_obs_hat[nq].T) + D @ f_obs_ - acc_obs_).T
                            R_obs += torch.sqrt(torch.sum(R_obs_seq**2, dim=1))
                        case [_, torch.Tensor()]:
                            R_obs_seq = (B @ (alpha_z_ * zp_obs_hat[nq].T) + Bn @ self.nonlinearity.gz_func(alpha_z_ * zp_obs_hat[nq].T) + D @ f_obs_ - acc_obs_).T
                            R_obs += torch.sqrt(torch.sum(R_obs_seq**2, dim=1))
        else:
            R_obs = torch.zeros((2))
            
            # if acc_obs_method in ['deriv_continuity', 'both']:
        if self.switches['occ']:
            if obs_dropouts:
                idx_keep = [j for j in range(self.n_dof) if j not in obs_dropouts]  # indices to keep in terms of DOFs
                idx_keep_2 = idx_keep + [j+self.n_dof for j in idx_keep]  # indices to keep in terms of state vector
                R_occ = torch.zeros((t_obs_list[0].shape[0])).to(self.device)
                dzdt_list_obs = [torch.zeros((t_obs_list[0].shape[0], len(idx_keep))).to(self.device) for _ in range(self.seq_len)]
                for nq in range(self.seq_len):
                    acc_obs_ = self.alpha_a * acc_obs[:, nq, idx_keep]  # acceleration observations [n_dof x No]
                    for i, idx in enumerate(idx_keep):
                        dzdt_list_obs[nq][:, i] = torch.autograd.grad(zp_obs_hat[nq][:, idx+self.n_dof], t_obs_list[nq], torch.ones_like(zp_obs_hat[nq][:, i]), create_graph=True)[0][:, 0]  # first derivative of velocity
                    R_occ += torch.sqrt(torch.sum(((self.alpha_z[-1]/self.alpha_t) * dzdt_list_obs[nq] - acc_obs_)**2, dim=1))
            else:
                R_occ = torch.zeros((t_obs_list[0].shape[0])).to(self.device)
                dzdt_list_obs = [torch.zeros((t_obs_list[0].shape[0], 2*self.n_dof)).to(self.device) for _ in range(self.seq_len)]
                for nq in range(self.seq_len):
                    acc_obs_ = self.alpha_a * acc_obs[:, nq, :].T
                    for i in range(self.n_dof):
                        dzdt_list_obs[nq][:, i+self.n_dof] = torch.autograd.grad(zp_obs_hat[nq][:, i], t_obs_list[nq], torch.ones_like(zp_obs_hat[nq][:, i]), create_graph=True)[0][:, 0]
                    R_occ += torch.sqrt(torch.sum(((self.alpha_z[-1]/self.alpha_t) * dzdt_list_obs[nq] - acc_obs_)**2, dim=1))
        else:
            R_occ = torch.zeros((2))
            
            # force contribution checker
            # t_pred_stack = torch.cat(t_obs_list, dim=0).detach()
            # zp_pred_stack = self.alpha_z * torch.cat(zp_obs_hat, dim=0).detach().T
            # f_obs_stack = self.alpha_f * torch.cat([f_obs[:, nq, :] for nq in range(self.seq_len)], dim=0).detach().T
            # xx = zp_pred_stack[:self.n_dof, :]
            # vv = zp_pred_stack[self.n_dof:, :]
            # zn = nonlin_state_transform(zp_pred_stack)
            # C, K, Kn, = self.C, self.K, -self.Bn[:, :self.n_dof] * 10
            # lin_spring_contr = K @ xx
            # lin_damp_contr = C @ vv
            # nonlin_spring_contr = Kn @ zn[:self.n_dof, :]
            # acc_check = D @ (-lin_spring_contr - lin_damp_contr - nonlin_spring_contr + f_obs_stack)
            
            # match acc_obs_method:
            #     case 'obs_model':
            #         R_obs = R_obs_obs
            #     case 'deriv_continuity':
            #         R_obs = R_obs_dcc
            #     case 'both':
            #         R_obs = R_obs_obs + R_obs_dcc
                            
        if self.switches['ode'] or self.switches['cc']:
            
            # generate or retrieve prediction over collocation domain
            dzdt_list_col = [torch.zeros((t_obs_list[0].shape[0], 2*self.n_dof)).to(self.device) for _ in range(self.seq_len)]

            if t_col_list is None:
                zp_col_hat = zp_obs_hat
                t_col_list = t_obs_list
                # if acc_obs_method in ['deriv_continuity', 'both']: 
                if self.switches['occ']:# have already calculated velocity derivatives in observation loss
                    if obs_dropouts:  # there are some dropouts so recover only the derivatives that were not dropped out
                        for nq in range(self.seq_len):
                            dzdt_list_col[nq][:, idx_keep] = dzdt_list_obs[nq]
                            for i in obs_dropouts:
                                dzdt_list_col[nq][:, i + self.n_dof] = torch.autograd.grad(zp_obs_hat[nq][:, i+self.n_dof], t_obs_list[nq], torch.ones_like(zp_obs_hat[nq][:, i]), create_graph=True)[0][:, 0]
                    else:  # no dropouts so retrieve all derivatives of velocity
                        for nq in range(self.seq_len):
                            dzdt_list_col[nq][self.n_dof:] = dzdt_list_obs[nq]
                    for nq in range(self.seq_len): # generate derivatives of displacement
                        for i in range(self.n_dof):
                            dzdt_list_col[nq][:, i] = torch.autograd.grad(zp_obs_hat[nq][:, i], t_obs_list[nq], torch.ones_like(zp_obs_hat[nq][:, i]), create_graph=True)[0][:, 0]
                else:
                    # generate all derivatives
                    dzdt_list_col = [torch.zeros((t_obs_list[0].shape[0], 2*self.n_dof)) for _ in range(self.seq_len)]
                    for nq in range(self.seq_len):
                        for i in range(2*self.n_dof):
                            dzdt_list_col[nq][:, i] = torch.autograd.grad(zp_obs_hat[nq][:, i], t_obs_list[nq], torch.ones_like(zp_obs_hat[nq][:, i]), create_graph=True)[0][:, 0]
                if f_obs is not None:
                    f_col = f_obs
            else:  # collocation domain is different to observation domain, so must generate all derivatives
                zp_col_hat = self.forward(t_col_list)
                # generate derivatives
                dzdt_list_col = [torch.zeros((t_col_list[0].shape[0], 2*self.n_dof)) for _ in range(self.seq_len)]
                for nq in range(self.seq_len):
                    for i in range(2*self.n_dof):
                        dzdt_list_col[nq][:, i] = torch.autograd.grad(zp_col_hat[nq][:, i], t_col_list[nq], torch.ones_like(zp_col_hat[nq][:, i]), create_graph=True)[0][:, 0]

            # retrieve physical parameters
            A, H, An = self.retrieve_state_matrices()
    
            R_ode = torch.zeros((t_col_list[0].shape[0])).to(self.device)
            R_cc = torch.zeros((t_col_list[0].shape[0])).to(self.device)
            for nq in range(self.seq_len):
                f_col_ = self.alpha_f * f_col[:, nq, :].T
                dzdt_ = dzdt_list_col[nq]
                match [self.nonlinearity, f_col]:
                    case None, None:
                        R_ = (self.alpha_z / self.alpha_t) * dzdt_.T - A @ (self.alpha_z * zp_col_hat[nq].T)
                    case [_, None]:
                        gz = self.nonlinearity.gz_func(self.alpha_z*zp_col_hat[nq].T)
                        R_ = (self.alpha_z / self.alpha_t)*dzdt_.T - A @ (self.alpha_z * zp_col_hat[nq].T) - An @ gz
                    case [None, torch.Tensor()]:
                        R_ = (self.alpha_z / self.alpha_t) * dzdt_.T - A @ (self.alpha_z * zp_col_hat[nq].T) - H @ (f_col_)
                    case [_, torch.Tensor()]:
                        gz = self.nonlinearity.gz_func(self.alpha_z * zp_col_hat[nq].T)
                        R_ = (self.alpha_z / self.alpha_t) * dzdt_.T - A @ (self.alpha_z * zp_col_hat[nq].T) - An @ gz - H @ (f_col_)
                R_cc += torch.sqrt(torch.sum((R_[:self.n_dof, :])**2, dim=0)) / self.seq_len
                R_ode += torch.sqrt(torch.sum((R_[self.n_dof:, :])**2, dim=0)) / self.seq_len

            # continuity condition residual
            # R_cc = R_[:self.n_dof, :].T
        else:
            R_ode = torch.zeros((2))
            R_cc = torch.zeros((2))

        residuals = {
            "R_obs" : R_obs,
            "R_occ" : R_occ,
            "R_cc" : R_cc,
            "R_ode" : R_ode
        }

        L_obs = lambdas['obs'] * torch.mean(R_obs**2)
        L_occ = lambdas['occ'] * torch.mean(R_occ**2)
        L_cc = lambdas['cc'] * torch.mean(R_cc**2)
        L_ode = lambdas['ode'] * torch.mean(R_ode**2)
 
        loss = L_obs + L_occ + L_cc + L_ode

        if math.isnan(loss):
            raise Exception("Loss is NaN, upsi")

        return loss, [L_obs, L_occ, L_cc, L_ode], residuals
    
    def predict(self, t_pred_list, theta_s=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict state values
        
        Arguments:
        t_pred: [samples] * seq_len
        theta_s: [n_dof] * 3
        """
        zp_list = self.forward(t_pred_list)
        # retrieve derivatives
        dzdt_list = [torch.zeros((t_pred_list[0].shape[0], 2*self.n_dof)) for _ in range(self.seq_len)]
        for nq in range(self.seq_len):
            for i in range(2*self.n_dof):
                dzdt_list[nq][:, i] = torch.autograd.grad(zp_list[nq][:, i], t_pred_list[nq], torch.ones_like(zp_list[nq][:, i]), create_graph=True)[0][:, 0]

        # # reshape
        t_pred_flat, sort_ids = torch.sort(torch.cat(t_pred_list, dim=0).reshape(-1))
        zp_flat = torch.cat(zp_list, dim=0)[sort_ids, :]
        dzdt_ = torch.cat(dzdt_list, dim=0)
        dzdt = dzdt_[sort_ids, :]

        # retrieve physical parameters
        A, H, An = self.retrieve_state_matrices(theta_s)
        
        match self.nonlinearity:
            case None:
                Hf_pred = (self.alpha_z / self.alpha_t) * dzdt.T - A @ (self.alpha_z * zp_flat.T)
            case _:
                gz = self.nonlinearity.gz_func(self.alpha_z * zp_flat.T)
                Hf_pred = (self.alpha_z / self.alpha_t) * dzdt.T - A @ (self.alpha_z * zp_flat.T) - An @ gz
        M = torch.diag(1/torch.diag(H[self.n_dof:, :]))
        f_pred = M @ Hf_pred[self.n_dof:, :] / self.alpha_f
        
        #TODO: check acceleration prediction with derivative of state instead using observer model
        B, D, Bn = self.retrieve_obs_matrices(theta_s)
        match self.nonlinearity:
            case None:
                a_pred = (B @ (self.alpha_z * zp_flat.T) + D @ (self.alpha_f * f_pred)) / self.alpha_a
            case _:
                a_pred = (B @ (self.alpha_z * zp_flat.T) + Bn @ gz + D @ (self.alpha_f * f_pred)) / self.alpha_a
        
        return zp_flat, f_pred.T, a_pred.T, t_pred_flat
    

class mdof_stoch_pinn(nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]
        self.seq_len = config["seq_len"]
        self.n_dof = config["n_dof"]
        if 'activation' in config.keys():
            self.activation = getattr(nn, config["activation"])
        else:
            # self.activation = nn.Tanh
            self.activation = nn.SiLU
        self.activation = nn.ELU
        self.device = config["device"]
        self.t_pi = torch.tensor(math.pi)

        self.build_nets()

        self.configure(config)
        self.gather_params()
        self.set_switches(config['lambds'])

    def gather_params(self):
        self.net_params_list = []
        for net in self.nets:
            self.net_params_list.append(net.parameters())

    def build_nets(self) -> int:
        nets = [None] * self.seq_len
        for net_n in range(self.seq_len):
            nets[net_n] = nn.Sequential(
                nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
                nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
                nn.Linear(self.n_hidden, self.n_output)
                )
            nets[net_n].to(self.device)
        self.nets = tuple(nets)
        self.network_parameters = []
        for net in self.nets:
            self.network_parameters += list(net.parameters())
        return 0

    def build_net(self) -> int:
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input * self.seq_len, self.n_hidden * self.seq_len), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden * self.seq_len, self.n_hidden * self.seq_len), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden * self.seq_len, self.n_output * self.seq_len),
            nn.Unflatten(dim=1, unflattened_size = (self.seq_len, self.n_output))
            )
        return 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input to network

        Returns:
            torch.Tensor: Output tensor.
        """
        # y = self.net(x)
        y = [torch.zeros((x[0].shape[0], self.n_output), device=self.device)] * self.seq_len
        # y = torch.zeros((self.seq_len, x[0].shape[0], self.n_output), device=self.device)
        for nq in range(self.seq_len):
            y[nq] = self.nets[nq](x[nq])
        return tuple(y)
    
    def configure(self, config: dict) -> None:
        """
        Configures neural network

        Args:
            config (dict): Configuration parameters
        """

        self.config = config

        self.nonlinearity = False
        # self.param_func = config["param_func"]

        self.set_phys_params()
        self.set_norm_params()
        self.set_noise_params()

    def set_phys_params(self) -> None:
        """
        Set physical parameters of model, and adds them as either constants or parameters for optimisation
        """
        config = self.config
        self.param_attrs = {}
        self.system_parameters = []
        for param_name, param_dict in config["phys_params"].items():
            self.param_attrs[param_name] = param_dict["type"]
            if param_dict["type"] == "constant":
                setattr(self,param_name,param_dict["value"])
            elif param_dict["type"] == "variable":
                self.register_parameter(param_name, nn.Parameter(param_dict["value"]))
                self.system_parameters.append(getattr(self, param_name))
        if hasattr(self, "M") and hasattr(self, "C") and hasattr(self, "K"):
            self.A = torch.cat((
                    torch.cat((torch.zeros((self.n_dof,self.n_dof)), torch.eye(self.n_dof)), dim=1),
                    torch.cat((-torch.linalg.inv(self.M)@self.K, -torch.linalg.inv(self.M)@self.C), dim=1)
                    ), dim=0)
        elif hasattr(self,"M"):
            self.m_ = torch.diag(self.M)  # takes diagonal from mass matrix if set as constant
    
    def set_noise_params(self) -> None:
        """
        Set noise parameters in likelihood equations
        """
        config = self.config
        self.noise_parameters = []
        for param_name, param_val in config["noise_params"].items():
            self.register_parameter(param_name, nn.Parameter(param_val))
            self.noise_parameters.append(getattr(self, param_name))

    def set_norm_params(self) -> None:
        """
        Set normalisation parameters of the model
        """
        config = self.config
        self.alpha_t = config["alphas"]["t"].clone().detach().to(self.device)
        self.alpha_x = config["alphas"]["x"].clone().detach().to(self.device)
        self.alpha_v = config["alphas"]["v"].clone().detach().to(self.device)
        self.alpha_z = torch.cat((config["alphas"]["x"]*torch.ones(self.n_dof,1), config["alphas"]["v"]*torch.ones(self.n_dof,1)), dim=0).float().to(self.device)
        self.alpha_f = config["alphas"]["f"].clone().detach().to(self.device)
        for param_name, param_dict in config["phys_params"].items():
            if param_dict["type"] == "variable":
                setattr(self,"alpha_"+param_name[:-1],config["alphas"][param_name[:-1]])
            else:
                setattr(self, "alpha_"+param_name[:-1], 1.0)

    def set_aux_funcs(self, param_func, nonlin_func):
        self.param_func = param_func
        self.nonlinearity = nonlin_func
    
    def set_switches(self, lambdas: dict) -> None:
        """
        Sets switches for residual/loss calculation to improve performance of unecessary calculation
        Args:
            lambdas (dict): dictionary of lambda weighting parameters
        """
        switches = {}
        for key, value in lambdas.items():
            switches[key] = value>0.0
        if self.seq_len == 1: 
            switches['ncc'] = 0.0
        self.switches = switches
    
    def loss_func(self, t_obs: torch.Tensor, z_obs: torch.Tensor, t_col: torch.Tensor, f_col: torch.Tensor, lambdas: dict, obs_dropouts: Tuple[list, bool]=False) -> Tuple[torch.Tensor, list, dict]:
        """
        Calculates residuals for loss functions

        Args:
            t_obs list(torch.Tensor): time values for observation domain [No x 1] x seq_len
            z_obs (torch.Tensor): observations of state [No x seq_len x 2*ndof]
            t_col list(torch.Tensor): time values for collocation domain [Nc x 1] x seq_len
            f_col list(torch.Tensor): measurements of force in collocation domain [Nc x ndof] x seq_len
            lambdas (dict): dictionary of loss weighting parameters

        Returns:
            loss: total loss
            losses: list of individual losses
            residuals: dictionary of residuals

        """

        if self.switches['obs']:
            # generate prediction at observation points
            zp_obs_hat = self.forward(t_obs)
            if obs_dropouts:
                idx = []  # ids for displacements that should be included
                idx2 = []  # ids for velocities that should be included
                for j in range(self.n_dof):
                    if j not in obs_dropouts:
                        idx.append(j)
                        idx2.append(j+self.n_dof)
                idx.extend(idx2)  # all ids in states that should be included
                n_obs_sq = t_obs[0].shape[0]  # number of observations per sequence
                n_obs_state = 2*(self.n_dof-len(obs_dropouts))  # number of observations in the states
                R_obs = torch.zeros((n_obs_sq*self.seq_len, n_obs_state))  # empty matrix for residuals [n_samps, 2n_dof]
                for nq in range(self.seq_len):
                    # R_obs += torch.sqrt(torch.sum((zp_obs_hat[nq][:, idx].reshape(-1, 2*(self.n_dof-len(obs_dropouts))) - z_obs[:, nq, idx].reshape(-1, 2*(self.n_dof-len(obs_dropouts))))**2, dim=1))
                    R_obs[nq*n_obs_sq:(nq+1)*n_obs_sq, :] = zp_obs_hat[nq][:, idx].reshape(-1, n_obs_state) - z_obs[:, nq, idx].reshape(-1, n_obs_state)
            else:
                # R_obs = torch.zeros((t_obs.shape[0]))
                # for nq in range(self.seq_len):
                #     R_obs += torch.sqrt(torch.sum((zp_obs_hat[nq] - z_obs[:, nq, :])**2, dim=1))
                # R_obs = torch.zeros((n_obs_sq*self.seq_len))
                n_obs_sq = t_obs[0].shape[0]  # number of observations per sequence
                R_obs = torch.zeros((n_obs_sq*self.seq_len, 2*self.n_dof))
                for nq in range(self.seq_len):
                    # R_obs[nq*n_obs_sq:(nq+1)*n_obs_sq] = torch.sqrt(torch.sum((zp_obs_hat[nq] - z_obs[:, nq, :])**2, dim=1))
                    R_obs[nq*n_obs_sq:(nq+1)*n_obs_sq, :] = zp_obs_hat[nq] - z_obs[:, nq, :]
                    
        if self.switches['ncc']:
            net_in_ids = [torch.argmin(t_obs[n]) for n in range(self.seq_len)]
            net_out_ids = [torch.argmax(t_obs[n]) for n in range(self.seq_len)]
            R_ncc = torch.zeros((self.seq_len - 1, 1))

            for nq in range(self.seq_len-1):
                R_ncc[nq] = torch.sqrt(torch.sum((zp_obs_hat[nq][net_out_ids[nq], :] - zp_obs_hat[nq + 1][net_in_ids[nq + 1], :])**2, dim=0))
        else:
            R_ncc = torch.ones((self.n_dof, 1))

        if self.switches['ode'] or self.switches['cc']:
            # generate prediction over collocation domain
            zp_col_hat_ = self.forward(t_col)

            # retrieve derivatives
            dxdt_list = [torch.zeros((t_col[0].shape[0], 2*self.n_dof)) for _ in range(self.seq_len)]
            for nq in range(self.seq_len):
                for i in range(2*self.n_dof):
                    dxdt_list[nq][:, i] = torch.autograd.grad(zp_col_hat_[nq][:, i], t_col[nq], torch.ones_like(zp_col_hat_[nq][:, i]), create_graph=True)[0][:, 0]

            # reshape
            t_pred_flat, sort_ids = torch.sort(torch.cat(t_col, dim=0).reshape(-1))
            zp_col_hat = torch.cat(zp_col_hat_, dim=0)[sort_ids, :]
            f_col = torch.cat(f_col, dim=0)[sort_ids, :]
            dxdt_ = torch.cat(dxdt_list, dim=0)
            dxdt = dxdt_[sort_ids, :]

            # retrieve physical parameters
            # linear parameters
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
                M, C, K = self.param_func(params["m_"], params["c_"], params["k_"])
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

            # nonlinear parameters
            if self.nonlinearity is not None:
                An = self.nonlinearity.mat_func(params['kn_'], params['cn_'], invM)
                    
        if self.switches['ode'] or self.switches['cc']:
            match [self.nonlinearity, f_col]:
                case None, None:
                    R_ = (self.alpha_z / self.alpha_t) * dxdt.T - A @ (self.alpha_z * zp_col_hat.T)
                case [_, None]:
                    gz = self.nonlinearity.gz_func(self.alpha_z*zp_col_hat.T)
                    R_ = (self.alpha_z / self.alpha_t)*dxdt.T - A @ (self.alpha_z * zp_col_hat.T) - An @ gz
                case [None, torch.Tensor()]:
                    R_ = (self.alpha_z / self.alpha_t) * dxdt.T - A @ (self.alpha_z * zp_col_hat.T) - H @ (self.alpha_f * f_col.T)
                case [_, torch.Tensor()]:
                    gz = self.nonlinearity.gz_func(self.alpha_z * zp_col_hat.T)
                    R_ = (self.alpha_z / self.alpha_t) * dxdt.T - A @ (self.alpha_z * zp_col_hat.T) - An @ gz - H @ (self.alpha_f * f_col.T)
            R_ode = R_[self.n_dof:, :].T / self.alpha_f
        else:
            R_ode = torch.zeros((self.n_dof, 1))

        if self.switches['cc']:
            # continuity condition residual
            R_cc = R_[:self.n_dof, :].T / self.alpha_v
        else:
            R_cc = torch.zeros((self.n_dof, 1))

        residuals = {
            "R_obs" : R_obs,
            "R_ncc" : R_ncc,
            "R_cc" : R_cc,
            "R_ode" : R_ode
        }

        # likelihoods
        N_o = R_obs.shape[0]
        # Sigma = torch.diag(torch.cat((self.sigma_x * torch.ones(self.n_dof), self.sigma_v * torch.ones(self.n_dof))))
        if self.switches['obs']:
            ## same sigma for all states
            # log_likeli_obs = -N_o * torch.log(self.sigma_z) - (N_o/2) * torch.log(2*self.t_pi) - 0.5 * torch.sum((R_obs**2/self.sigma_z**2))

            ## looping with separate sigma_x and sigma_v
            sigmas = [self.sigma_x, self.sigma_v]
            log_likeli_obs = torch.tensor(0.0)
            for d in range(2):
                if obs_dropouts:
                    log_likeli_obs += (- 0.5 * N_o * torch.log(2*self.t_pi) - N_o * torch.log(sigmas[d]) - 0.5 * torch.sum(torch.sum(R_obs[:, int(d*n_obs_state/2):int((d+1)*n_obs_state/2)]**2, dim=1)/sigmas[d]**2, dim=0))
                else:
                    log_likeli_obs += (- 0.5 * N_o * torch.log(2*self.t_pi) - N_o * torch.log(sigmas[d]) - 0.5 * torch.sum(torch.sum(R_obs[:, d*self.n_dof:(d+1)*self.n_dof]**2, dim=1)/sigmas[d]**2, dim=0))
            L_obs = lambdas['obs'] * - log_likeli_obs
            
            ## full multivariate calc
            # dist_log_term = torch.tensor(0., dtype=torch.float32)
            # for i in range(R_obs.shape[0]):
            #     dist_log_term += R_obs[i,:].T @ torch.linalg.inv(Sigma) @ R_obs[i,:]
            # log_likeli_obs = -(N_o/2) * (2 * self.n_dof * torch.log(2*self.t_pi) + torch.log(torch.linalg.det(Sigma))) - 0.5 * dist_log_term
        else:
            log_likeli_obs = torch.tensor(0.0)
            L_obs = torch.tensor(0.0)

        N_c = R_ode.shape[0]
        log_likeli_ode = torch.tensor(0.0)
        if self.switches['ode']:
            log_likeli_ode = -0.5 * N_c * torch.log(2*self.t_pi) - N_c * torch.log(self.sigma_f) - 0.5 * torch.sum(torch.sum(R_ode**2, dim=1)/self.sigma_f**2, dim=0)
            L_ode = lambdas['ode'] * - log_likeli_ode
        else:
            log_likeli_ode = torch.tensor(0.0)
            L_ode = torch.tensor(0.0)
        
        if self.switches['cc']:
            log_likeli_cc = - 0.5 * N_c * torch.log(2*self.t_pi) - N_c * torch.log(self.sigma_v) - 0.5 * torch.sum(torch.sum(R_cc**2, dim=1)/self.sigma_v**2, dim=0)
            L_cc = lambdas['cc'] * - log_likeli_cc
        else:
            L_cc = torch.tensor(0.0)

        L_ncc = lambdas['ncc'] * torch.mean(R_ncc*2, dim=0)
 
        loss = L_obs + L_ode + L_cc + L_ncc

        if math.isnan(loss):
            raise Exception("Nan again for some bloody reason")

        return loss, [L_obs, L_cc, L_ode, L_ncc], residuals
    
    def predict(self, t_pred, theta_s=None, f_col=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict state values
        """
        zp_ = self.forward(t_pred)
        # tp_sort = [None] * self.seq_len
        # zp_sort = [None] * self.seq_len
        # fc_sort = [None] * self.seq_len
        # retrieve derivatives
        dxdt_list = [torch.zeros((t_pred[0].shape[0], 2*self.n_dof)) for _ in range(self.seq_len)]
        for nq in range(self.seq_len):
            for i in range(2*self.n_dof):
                dxdt_list[nq][:, i] = torch.autograd.grad(zp_[nq][:, i], t_pred[nq], torch.ones_like(zp_[nq][:, i]), create_graph=True)[0][:, 0]
            # tp_sort[nq], sort_ids = torch.sort(t_pred[nq].reshape(-1))
            # zp_sort[nq] = zp_[nq][sort_ids, :]
            # if f_col is not None:
            #     fc_sort[nq] = f_col[nq][sort_ids, :]
            # dxdt_list[nq] = dxdt_list[nq][sort_ids, :]

        # reshape
        # t_pred_flat = torch.cat(tp_sort, dim=0)
        t_pred_flat, sort_ids = torch.sort(torch.cat(t_pred, dim=0).reshape(-1))
        zp_flat = torch.cat(zp_, dim=0)[sort_ids, :]
        if f_col is not None:
            f_col_ = torch.cat(f_col, dim=0)[sort_ids, :]
        dxdt = torch.cat(dxdt_list, dim=0)[sort_ids, :]

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
                        # params[param_name] = self.param_transforms[param_name](getattr(self,param_name))
                        params[param_name] = getattr(self,param_name)*getattr(self,"alpha_"+param_name[:-1])
                M, C, K = self.param_func(params["m_"], params["c_"], params["k_"])
            else:
                M, C, K = self.param_func(self.m_, theta_s[:self.n_dof]*self.alpha_c, theta_s[self.n_dof:2*self.n_dof]*self.alpha_k)
            invM = torch.diag(1/torch.diag(M))
            A = torch.cat((
                torch.cat((torch.zeros((self.n_dof, self.n_dof), device=self.device), torch.eye(self.n_dof, device=self.device)), dim=1),
                torch.cat((-invM @ K, -invM @ C), dim=1)
                ), dim=0).requires_grad_()
                
        # nonlinear parameters
        if self.nonlinearity is not None:
            if theta_s is None:
                An = self.nonlinearity.mat_func(params['kn_'], params['cn_'], invM)
            else:
                kn_ = torch.tensor(theta_s[2*self.n_dof:3*self.n_dof], dtype=torch.float32)
                cn_ = torch.zeros_like(kn_)
                An = self.nonlinearity.mat_func(kn_, cn_, invM)
        
        match self.nonlinearity:
            case None:
                Hf_pred = (self.alpha_z / self.alpha_t) * dxdt.T - A @ (self.alpha_z * zp_flat.T)
            case _:
                gz = self.nonlinearity.gz_func(self.alpha_z * zp_flat.T)
                Hf_pred = (self.alpha_z / self.alpha_t) * dxdt.T - A @ (self.alpha_z * zp_flat.T) - An @ gz
        f_pred = M @ Hf_pred[self.n_dof:, :]
        
        if f_col is None:
            return zp_flat, f_pred.T, t_pred_flat, dxdt
        else:
            return zp_flat, f_pred.T, t_pred_flat, dxdt, f_col_
        
    def locked_force_pred(self, theta_s, z_pred, dzdt_pred):

        # retrieve physical parameters
        M, C, K = self.param_func(self.m_, theta_s[:self.n_dof]*self.alpha_c, theta_s[self.n_dof:2*self.n_dof]*self.alpha_k)
        invM = torch.diag(1/torch.diag(M))
        # M1, C1, K1 = self.param_func(self.m_, torch.ones(self.n_dof), 15.0*torch.ones(self.n_dof))
        A = torch.cat((
            torch.cat((torch.zeros((self.n_dof, self.n_dof), device=self.device), torch.eye(self.n_dof, device=self.device)), dim=1),
            torch.cat((-invM @ K, -invM@C), dim=1)
            ), dim=0).requires_grad_()
        # A1 = torch.cat((
        #     torch.cat((torch.zeros((self.n_dof, self.n_dof), device=self.device), torch.eye(self.n_dof, device=self.device)), dim=1),
        #     torch.cat((-invM @ K1, -invM@C1), dim=1)
        #     ), dim=0).requires_grad_()
                
        # nonlinear parameters
        if self.nonlinearity is not None:
            if self.nonlinearity.gk_exp is not None:
                kn__ = theta_s[2*self.n_dof:3*self.n_dof]
                cn__ = torch.zeros(self.n_dof)
            elif self.nonlinearity.gc_exp is not None:
                kn__ = torch.zeros(self.n_dof)
                cn__ = theta_s[2*self.n_dof:3*self.n_dof]
                # cn__1 = 0.75 * torch.ones(self.n_dof) / self.alpha_cn
            #cn__[0] = theta_s[-2]

            An = self.nonlinearity.mat_func(self.alpha_kn * kn__, self.alpha_cn * cn__, invM)
            # An1 = self.nonlinearity.mat_func(self.alpha_kn * kn__, self.alpha_cn * cn__1, invM)

        match self.nonlinearity:
            case None:
                Hf_pred = (self.alpha_z / self.alpha_t) * dzdt_pred.T - A @ (self.alpha_z * z_pred.T)
            case _:
                gz = self.nonlinearity.gz_func(self.alpha_z * z_pred.T)
                Hf_pred = (self.alpha_z / self.alpha_t) * dzdt_pred.T - A @ (self.alpha_z * z_pred.T) - An @ gz
                # Hf_pred1 = (self.alpha_z / self.alpha_t) * dzdt_pred.T - A1 @ (self.alpha_z * z_pred.T) - An1 @ gz
        f_pred = M @ Hf_pred[self.n_dof:, :]
        # f_pred1 = M @ Hf_pred1[self.n_dof:, :]
        self.f_pred = f_pred.T

        return f_pred.T / self.alpha_f
    
    def phys_log_likeli(self, theta_s):

        z_pred = self.z_pred
        dzdt_pred = self.dzdt_pred
        f_col = self.f_col

        # retrieve physical parameters
        M, C, K = self.param_func(self.m_, theta_s[:self.n_dof]*self.alpha_c, theta_s[self.n_dof:2*self.n_dof]*self.alpha_k)
        invM = torch.diag(1/torch.diag(M))
        A = torch.cat((
            torch.cat((torch.zeros((self.n_dof, self.n_dof), device=self.device), torch.eye(self.n_dof, device=self.device)), dim=1),
            torch.cat((-invM @ K, -invM@C), dim=1)
            ), dim=0).requires_grad_()
                
        # nonlinear parameters
        if self.nonlinearity is not None:
            # An = self.nonlinearity.mat_func(theta_s[2*self.n_dof:3*self.n_dof], theta_s[3*self.n_dof:4*self.n_dof], invM)
            kn__ = torch.tensor(theta_s[2*self.n_dof:3*self.n_dof], dtype=torch.float32)
            cn__ = torch.zeros(self.n_dof)
            An = self.nonlinearity.mat_func(self.alpha_kn * kn__, self.alpha_cn * cn__, invM)

        match self.nonlinearity:
            case None:
                Hf_pred = (self.alpha_z / self.alpha_t) * dzdt_pred.T - A @ (self.alpha_z * z_pred.T)
            case _:
                gz = self.nonlinearity.gz_func(self.alpha_z * z_pred.T)
                Hf_pred = (self.alpha_z / self.alpha_t) * dzdt_pred.T - A @ (self.alpha_z * z_pred.T) - An @ gz
        f_pred = M @ Hf_pred[self.n_dof:, :]

        sigma_f = theta_s[-1]

        N_c = f_pred.shape[0]
        f_res = torch.sqrt(torch.sum((f_col * self.alpha_f - f_pred.T)**2, dim=1))
        log_likeli_ode = - N_c * torch.log(sigma_f) - (N_c/2) * torch.log(2*self.t_pi) - 0.5 * torch.sum((f_res**2/(sigma_f**2)))
        return log_likeli_ode


class mdof_dataset(torch.utils.data.Dataset):

    def __init__(self, t_data, acc_data, f_data = None, data_config = None, device = torch.device("cpu"), force_drop = False):

        if data_config is not None:
            self.subsample = data_config['subsample']  # subsample simulates lower sample rate
            self.seq_len = data_config['seq_len']  # subsample simulates lower sample rate
        else:
            self.subsample = 1
            self.seq_len = 1
        n_dof = acc_data.shape[1]
        if acc_data.shape[1] != f_data.shape[1]:
            raise Exception("Dimension mismatch for data, please check DOFs dimension of data")

        # normalise data based on range
        t_data, alpha_t = normalise(t_data, "range")
        acc_data, alpha_a = normalise(acc_data, "range", "all")
        if f_data is not None:
            if not force_drop:
                f_data, alpha_f = normalise(f_data, "range", "all")
            else:
                f_data, alpha_f = torch.zeros_like(f_data), torch.tensor(0.0)
        
            # concatenate into one large dataset
            data = torch.cat((acc_data, f_data, t_data), dim=1)
            self.alphas = {
                "a" : alpha_a, 
                "f" : alpha_f, 
                "t" : alpha_t
            }

            # reshape to batches and sequences
            # ndof for acc, 1 ndof for force, 1 for time
            self.ground_truth = data.to(device)
            col_data = data[:(data.shape[0] // (self.seq_len * self.subsample)) * (self.subsample * self.seq_len)]  # cut off excess data

            # create observation data from subsample
            obs_data = col_data[::self.subsample]
            # self.obs_data = obs_data.reshape((-1, self.seq_len, 3 * n_dof + 1)).to(device)
            self.obs_data = obs_data.T.reshape((2*n_dof+1, self.seq_len, -1)).permute(2, 1, 0).to(device)

            # create collocation data
            # self.col_data = col_data.reshape((-1, self.seq_len, self.subsample, 3 * n_dof + 1)).to(device)
            self.col_data = col_data.T.reshape((2*n_dof+1, self.seq_len, self.subsample, -1)).permute(3, 2, 1, 0).to(device)
        else:
            # concatenate into one large dataset
            data = torch.cat((acc_data, t_data), dim=1)
            self.alphas = {
                "a" : alpha_a, 
                "t" : alpha_t
            }

            # reshape to number of batches
            # 2 ndof for state, 1 for time
            self.ground_truth = data.to(device)
            col_data = data[:(data.shape[0] // (self.seq_len * self.subsample)) * (self.seq_len * self.subsample)]  # cut off excess data

            # create obervation data from subsample
            obs_data = col_data[::self.subsample]
            # self.data = data.reshape((-1, self.seq_len, 2 * n_dof + 1)).to(device)
            self.obs_data = obs_data.T.reshape((n_dof+1, self.seq_len, -1)).T.to(device)

            # self.col_data = col_data.reshape((-1, self.subsample, self.seq_len, 2 * n_dof + 1)).to(device)
            self.col_data = col_data.T.reshape((n_dof+1, self.seq_len, self.subsample, -1)).T.to(device)

    def __getitem__(self, index: int) -> np.ndarray:
        return self.obs_data[index, ...], self.col_data[index, ...]
    
    def get_original(self, index: int) -> np.ndarray:
        return self.ground_truth[index]

    def __len__(self) -> int:
        return self.obs_data.shape[0]
    
    def __repr__(self) -> str:
        return self.__class__.__name__


class mdof_stoch_dataset(torch.utils.data.Dataset):

    def __init__(self, t_data, x_data, v_data, f_data = None, snr = 50.0, num_repeats = 1, data_config = None, device = torch.device("cpu")):

        self.num_repeats = num_repeats

        if data_config is not None:
            self.subsample = data_config['subsample']  # subsample simulates lower sample rate
            self.seq_len = data_config['seq_len']  # subsample simulates lower sample rate
        else:
            self.subsample = 1
            self.seq_len = 1
        n_dof = x_data.shape[1]
        self.n_dof = n_dof
        if x_data.shape[1] != v_data.shape[1]:
            raise Exception("Dimension mismatch for data, please check DOFs dimension of data")

        # add noise and collate to shapes [num_samps, dof, num_repeats]
        t_noisy = torch.cat([t_data.unsqueeze(2) for _ in range(num_repeats)], dim=2)
        xx_noisy_list = [torch.tensor(add_noise(x_data, SNR=snr, seed = 42 + j)).unsqueeze(2) for j in range(num_repeats)]
        vv_noisy_list = [torch.tensor(add_noise(v_data, SNR=snr, seed = 8 + j)).unsqueeze(2) for j in range(num_repeats)]
        xx_noisy = torch.cat(xx_noisy_list, dim=2)
        vv_noisy = torch.cat(vv_noisy_list, dim=2)
        if f_data is not None: 
            ff_noisy_list = [torch.tensor(add_noise(f_data, SNR=snr, seed = 16 + j)).unsqueeze(2) for j in range(num_repeats)]
            ff_noisy = torch.cat(ff_noisy_list, dim=2)

        # normalise data based on range
        t_obs, alpha_t = normalise(t_noisy, "range", "all")
        x_obs, alpha_x = normalise(xx_noisy, "range", "all")
        v_obs, alpha_v = normalise(vv_noisy, "range", "all")
        if f_data is not None:
            f_obs, alpha_f = normalise(ff_noisy, "range", "all")
            self.alphas = {
                "x" : alpha_x, 
                "v" : alpha_v,
                "f" : alpha_f, 
                "t" : alpha_t
            }

            # dimension - 2 ndof for state, 1 ndof for force, 1 for time

            # create observation set from noisy data
            # concatenate and subsample into observation dataset [num_samps, dimension, num_repeats]
            obs_data_ = torch.cat((x_obs, v_obs, f_obs, t_obs), dim=1)[::self.subsample]
            # cutoff excess data before reshape
            obs_data_ = obs_data_[:(obs_data_.shape[0] // self.seq_len) * self.seq_len]
            # reshape to [num_samps_per_seq, num_repeats, seq_len, dimension]
            obs_data = torch.zeros((obs_data_.shape[0] // self.seq_len, num_repeats, self.seq_len, 3*n_dof+1))
            for i in range(self.num_repeats):
                obs_data[:, i, :, :] = obs_data_[..., i].T.reshape((3*n_dof+1, self.seq_len, -1)).T
            self.obs_data = obs_data.to(device)

            # create collocation data
            # concatenate into [num_samps, dimension]
            col_data_ = torch.tensor(np.concatenate((x_data/alpha_x, v_data/alpha_v, f_data/alpha_f, t_data/alpha_t), axis=1), dtype=torch.float32)
            # set ground truth
            self.ground_truth = col_data_.to(device)
            # cutoff excess data before reshape
            col_data = col_data_[:(col_data_.shape[0] // (self.seq_len)) * (self.seq_len)]  # cut off excess data
            # reshape to [num_samps_per_seq, subsample, seq_len, dimension]
            self.col_data = col_data.T.reshape((3*n_dof+1, self.seq_len, self.subsample, -1)).T.to(device)

        else:
            pass  # sort this out once you have figured what you want to do
            
            raise NotImplementedError("Lack of force data not implemented yet")

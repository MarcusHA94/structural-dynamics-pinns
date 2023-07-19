import torch
import torch.nn as nn
import numpy as np
from mdof_oscillators import gen_ndof_cantilever

def max_mag_data(data,axis=None):
    if torch.is_tensor(data):
        if axis==None:
            data_max = torch.max(torch.max(torch.abs(data)))
        else:
            data_max = torch.max(torch.abs(data),dim=axis)[0]
    else:
        data_max = np.max(np.abs(data),axis=axis)
    return data_max

def normalise(data,norm_type="var",norm_dir="all"):
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
            dmax = max_mag_data(data,axis=0)
        else:
            dmax = max_mag_data(data)
        data_norm = data/dmax
        return data_norm, dmax
    

class mdof_pinn(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]
        self.n_dof = config["n_dof"]
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
        x = self.net(t)
        y = G + D * x
        return y
    
    def configure(self, **config):

        self.config = config

        self.nonlinearity = config["nonlinearity"]
        self.param_func = config["param_func"]

        self.nct = config["nct"]  # number of time collocation points

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

        match config["forcing"]:
            case dict():
                self.force = torch.tensor(config["forcing"])

    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]["t"]
        self.alpha_x = config["alphas"]["x"]
        self.alpha_v = config["alphas"]["v"]
        self.alpha_z = torch.cat((self.alpha_x*torch.ones(self.n_dof,1), self.alpha_v*torch.ones(self.n_dof,1)), dim=0)
        if config["forcing"] != None:
            self.alpha_F = config["alphas"]["F"]
        for param_name, param_dict in config["phys_params"].items():
            if param_dict["type"] == "variable":
                setattr(self,"alpha_"+param_name[:-1],config["alphas"][param_name[:-1]])

    def set_colls_and_obs(self, data, prediction):

        # Observation set 
        self.x_obs = data['x_hat']  # displacement input
        self.v_obs = data['v_hat']  # velocity input
        self.t_obs = data['t_hat']  # time input
        if data['F_hat'] is not None:
            self.f_obs = data['F_hat']  # force input
        self.zz_obs = torch.cat((data['x_hat'], data['v_hat']), dim=1).requires_grad_()

        # Collocation set
        t_col = prediction['t_hat']
        self.t_col = t_col.requires_grad_()
        self.f_col = prediction['F_hat']
        if prediction['F_hat'] is not None:
            self.f_col = prediction['F_hat'].requires_grad_()
        
        self.ic_id = torch.argwhere(t_col[:,0]==torch.tensor(0.0)).view(-1)

    def Kn_func(self, kn_):
        Kn = torch.zeros((self.n_dof,self.n_dof), dtype=torch.float32)
        for n in range(self.n_dof):
            Kn[n,n] = kn_[n]
        for n in range(self.n_dof-1):
            Kn[n,n+1] = -kn_[n+1]
        return Kn.requires_grad_()
    
    def calc_residuals(self):

        if self.switches['obs']:
            # generate prediction at observation points
            zh_obs_hat = self.forward(self.t_obs)
            R_obs = torch.sqrt(torch.sum((zh_obs_hat - self.zz_obs)**2,dim=1))

        if self.switches['ode'] or self.switches['cc'] or self.switches['ic']:
            # generate prediction over collocation domain
            zh_col_hat = self.forward(self.t_col)

        if self.switches['ic']:
            # calculate ic residual
            R_ic1 = zh_col_hat[self.ic_id,:].T
            R_ic2 = dxdt[self.ic_id,:].T
            R_ic = torch.cat((R_ic1,R_ic2),dim=1)
        else:
            R_ic = torch.zeros((2,2))
        
        if self.switches['ode'] or self.switches['cc']:
            # retrieve derivatives
            dxdt = torch.zeros((zh_col_hat.shape[0],zh_col_hat.shape[1]))
            for i in range(zh_col_hat.shape[1]):
                dxdt[:,i] = torch.autograd.grad(zh_col_hat[:,i], self.t_col, torch.ones_like(zh_col_hat[:,i]), create_graph=True)[0][:,0]  # ∂_t-hat N_x-hat

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
            if self.config["forcing"] is not None:
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
                    R_ = (self.alpha_z/self.alpha_t)*dxdt.T - A@(self.alpha_z*zh_col_hat.T) - H@(self.alpha_F*self.f_col.T)
                    R_ode = R_[self.n_dof:,:].T
                case {"nonlinearity":"cubic","forcing":torch.Tensor()}:
                    q = (self.alpha_x*zh_col_hat[:,:self.n_dof] - torch.cat((torch.zeros(zh_col_hat.shape[0],1),self.alpha_x*zh_col_hat[:,:self.n_dof-1]),dim=1))**3
                    R_ = (self.alpha_z/self.alpha_t)*dxdt.T - A@(self.alpha_z*zh_col_hat.T) - An@(q.T) - H@(self.alpha_F*self.f_col.T)
                    R_ode = R_[self.n_dof:,:].T
        else:
            R_ode = torch.zeros((2,2))

        if self.switches['cc']:
            # continuity condition residual
            R_cc = R_[:self.n_dof,:].T
        else:
            R_cc = torch.zeros((2,2))

        return {
            "R_obs" : R_obs,
            "R_ic" : R_ic,
            "R_cc" : R_cc,
            "R_ode" : R_ode
        }
    
    def set_switches(self, lambdas):
        switches = {}
        for key, value in lambdas.items():
            switches[key] = value>0.0
        self.switches = switches

    def loss_func(self, lambdas):
        residuals = self.calc_residuals()
        R_obs = residuals["R_obs"]
        R_ic = residuals["R_ic"]
        R_cc = residuals["R_cc"]
        R_ode = residuals["R_ode"]

        L_obs = lambdas['obs'] * torch.mean(R_obs**2)
        L_ic = lambdas['ic'] * torch.sum(torch.mean(R_ic**2, dim=0), dim=0)
        L_cc = lambdas['cc'] * torch.sum(torch.mean(R_cc**2, dim=0), dim=0)
        L_ode = lambdas['ode'] * torch.sum(torch.mean(R_ode**2, dim=0), dim=0)

        loss = L_obs + L_ic + L_cc + L_ode
        return loss, [L_obs, L_ic, L_cc, L_ode], residuals
    
    def predict(self):
        xp = self.forward(self.t_col)
        return xp

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
            
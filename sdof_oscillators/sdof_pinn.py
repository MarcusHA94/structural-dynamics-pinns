import torch
import torch.nn as nn
import numpy as np

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

class bbnn(nn.Module):
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        self.n_input = N_INPUT
        self.n_output = N_OUTPUT
        self.n_hidden = N_HIDDEN
        self.n_layers = N_LAYERS
        self.activation = nn.Tanh

        self.build()
    
    def build(self):
        self.fcs = nn.Sequential(*[
                        nn.Linear(self.n_input, self.n_hidden),
                        self.activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(self.n_hidden, self.n_hidden),
                            self.activation()]) for _ in range(self.n_layers-1)])
        self.fce = nn.Linear(self.n_hidden, self.n_output)
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

    def loss_func(self, x_obs, y_obs):
        yp_obs = self.forward(x_obs)
        loss = torch.mean((yp_obs - y_obs)**2)
        return loss


class sdof_pinn(nn.Module):
    
    def __init__(self, N_INPUT, N_OUTPUT, N_HIDDEN, N_LAYERS):
        super().__init__()
        self.n_input = N_INPUT
        self.n_output = N_OUTPUT
        self.n_hidden = N_HIDDEN
        self.n_layers = N_LAYERS
        self.activation = nn.Tanh

        self.build()
    
    def build(self):
        self.fcs = nn.Sequential(*[
                        nn.Linear(self.n_input, self.n_hidden),
                        self.activation()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(self.n_hidden, self.n_hidden),
                            self.activation()]) for _ in range(self.n_layers-1)])
        self.fce = nn.Linear(self.n_hidden, self.n_output)
    
    def nonlinearity(self, nonlin_style="lin_osc"):
        self.nonlin_style = nonlin_style

    def set_phys_params(self, params, par_type):
        self.param_type = par_type
        match par_type:
            case "constant":
                match self.nonlin_style:
                    case "lin_osc":
                        self.k = params['k']
                        self.c = params['c']
                        self.phys_params = torch.tensor([self.c, self.k])
                    case "cubic_stiffness":
                        self.k = params['k']
                        self.k3 = params['k3']
                        self.c = params['c']
                        self.phys_params = torch.tensor([self.c, self.k, self.k3])
            case "variable":
                match self.nonlin_style:
                    case "lin_osc":
                        self.register_parameter("phys_params", nn.Parameter(torch.tensor([params["c"], params["k"]])))
                    case "cubic_stiffness":
                        self.register_parameter("phys_params", nn.Parameter(torch.tensor([params["c"], params["k"], params["k3"]])))

    def set_norm_params(self, alphas, ode_norm_type):
        self.alpha_t = alphas["t"]
        self.alpha_x = alphas["x"]
        match self.nonlin_style:
            case "lin_osc":
                self.alpha_k = alphas["k"]
                self.alpha_c = alphas["c"]
                
                alpha_d0 = 1.0
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                self.ode_alphas = {
                    "d0" : alpha_d0,
                    "d1" : alpha_d1,
                    "d2" : alpha_d2
                }
                
            case "cubic_stiffness":
                self.alpha_k = alphas["k"]
                self.alpha_k3 = alphas["k3"]
                self.alpha_c = alphas["c"]

                alpha_d0 = 1.0
                alpha_d0_3 = self.alpha_x**2
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                self.ode_alphas = {
                    "d0" : alpha_d0,
                    "d0_3" : alpha_d0_3,
                    "d1" : alpha_d1,
                    "d2" : alpha_d2
                }

        match ode_norm_type:
            case "up_time":
                self.ode_alphas.update((k, v*self.alpha_t) for k,v in self.ode_alphas.items())
            case "up_time2":
                self.ode_alphas.update((k, v*self.alpha_t**2) for k,v in self.ode_alphas.items())
        
    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x

    def ode_residual(self, t_pde_hat, t_obs, x_obs):

        match self.param_type:
            case "constant":
                self.m_hat = self.ode_alphas["d2"]
                self.c_hat = self.ode_alphas["d1"] * self.c
                self.k_hat = self.ode_alphas["d0"] * self.k
                if self.nonlin_style == "cubic_stiffness":
                    self.k3_hat = self.ode_alphas["d0_3"] * self.k3

            case "variable":
                self.m_hat = self.ode_alphas["d2"]
                self.c_hat = self.ode_alphas["d1"] * self.phys_params[0] * self.alpha_c
                self.k_hat = self.ode_alphas["d0"] * self.phys_params[1] * self.alpha_k
                if self.nonlin_style == "cubic_stiffness":
                    self.k3_hat = self.ode_alphas["d0_3"] * self.phys_params[2] * self.alpha_k3

        # observation loss
        xh_obs = self.forward(t_obs)  # N_y-hat or N_y (in Ω_a)
        R_obs = xh_obs - x_obs

        # pde loss
        xh_pde_hat = self.forward(t_pde_hat)   # N_y-hat (in Ω_ode)
        dx = torch.autograd.grad(xh_pde_hat, t_pde_hat, torch.ones_like(xh_pde_hat), create_graph=True)[0]  # ∂_t-hat N_y-hat
        dx2 = torch.autograd.grad(dx, t_pde_hat, torch.ones_like(dx), create_graph=True)[0]  # ∂^2_t-hat N_y-hat
        match self.nonlin_style:
            case "lin_osc":
                R_pde = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_pde_hat
            case "cubic_stiffness":
                R_pde = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_pde_hat + self.k3_hat * xh_pde_hat**3

        return {
            "R_obs" : R_obs,
            "R_pde" : R_pde
        }

    def loss_func(self, t_pde, t_obs, x_obs, lambds):
        residuals = self.ode_residual(t_pde, t_obs, x_obs)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]

        L_obs = lambds[0] * torch.mean(R_obs**2)
        L_pde = lambds[1] * torch.mean(R_pde**2)
        loss = L_obs + L_pde
        return loss, [L_obs, L_pde]


class ParamClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, 'phys_params'):
            params = module.phys_params.data
            params = params.clamp(0,1)
            module.phys_params.data = params

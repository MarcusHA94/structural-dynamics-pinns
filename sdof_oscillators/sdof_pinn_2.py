import torch
import torch.nn as nn
import numpy as np


class sdof_pinn(nn.Module):

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

    def configure(self, config):

        self.config = config

        self.nonlinearity = config["nonlinearity"]
        self.forcing = config["forcing"]
        self.param_type = config["phys_params"]["par_type"]

        self.set_phys_params()
        self.set_norm_params()

    def set_phys_params(self):
        config = self.config
        match config:
            case {"phys_params":{"par_type":"constant"},"nonlinearity":"linear"}:
                self.k = config["phys_params"]['k']
                self.c = config["phys_params"]['c']
                self.phys_params = torch.tensor([self.c, self.k])
            case {"phys_params":{"par_type":"constant"},"nonlinearity":"cubic"}:
                self.k = config["phys_params"]['k']
                self.c = config["phys_params"]['c']
                self.k3 = config["phys_params"]['k3']
                self.phys_params = torch.tensor([self.c, self.k])
                self.phys_params = torch.tensor([self.c, self.k, self.k3])
            case {"phys_params":{"par_type":"variable"},"nonlinearity":"linear"}:
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]["c"], config["phys_params"]["k"]])))
            case {"phys_params":{"par_type":"variable"},"nonlinearity":"cubic"}:
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]["c"], config["phys_params"]["k"], config["phys_params"]["k3"]])))
        match config["forcing"]:
            case {}:
                self.force = config["forcing"]["values"]

    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]["t"]
        self.alpha_x = config["alphas"]["x"]
        if config["forcing"] != None:
            self.alpha_F = config["alphas"]["F"]
        
        match config:
            case {"nonlinearity":"linear","forcing":None}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                
                alpha_d0 = 1.0
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"]
                }
            case {"nonlinearity":"linear","forcing":{}}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                
                alpha_d0 = 1.0
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                alpha_ff = self.alpha_F/self.alpha_x
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"],
                    "ff" : alpha_ff * config["ode_norm_Lambda"]
                }
            case {"nonlinearity":"cubic","forcing":None}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                
                alpha_d0 = 1.0
                alpha_d0_3 = self.alpha_x**2
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d0_3" : alpha_d0_3 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"]
                }
            case {"nonlinearity":"cubic","forcing":{}}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                
                alpha_d0 = 1.0
                alpha_d0_3 = self.alpha_x**2
                alpha_d1 = 1.0 / self.alpha_t
                alpha_d2 = 1.0 / (self.alpha_t**2)
                alpha_ff = self.alpha_F/self.alpha_x
                self.ode_alphas = {
                    "d0" : alpha_d0 * config["ode_norm_Lambda"],
                    "d0_3" : alpha_d0_3 * config["ode_norm_Lambda"],
                    "d1" : alpha_d1 * config["ode_norm_Lambda"],
                    "d2" : alpha_d2 * config["ode_norm_Lambda"],
                    "ff" : alpha_ff * config["ode_norm_Lambda"]
                }

    def calc_residuals(self, t_pde_hat, t_obs, x_obs):

        # observation loss
        xh_obs = self.forward(t_obs)  # N_y-hat or N_y (in Ω_a)
        R_obs = xh_obs - x_obs

        # ode values
        xh_pde_hat = self.forward(t_pde_hat)   # N_y-hat (in Ω_ode)
        dx = torch.autograd.grad(xh_pde_hat, t_pde_hat, torch.ones_like(xh_pde_hat), create_graph=True)[0]  # ∂_t-hat N_y-hat
        dx2 = torch.autograd.grad(dx, t_pde_hat, torch.ones_like(dx), create_graph=True)[0]  # ∂^2_t-hat N_y-hat

        # retrieve ode loss parameters
        self.m_hat = self.ode_alphas["d2"]
        match self.config["phys_params"]["par_type"]:
            case "constant":
                self.c_hat = self.ode_alphas["d1"] * self.c
                self.k_hat = self.ode_alphas["d0"] * self.k
                match self.config["nonlinearity"]:
                    case "cubic":
                        self.k3_hat = self.ode_alphas["d0_3"] * self.k3
            case "variable":
                self.c_hat = self.ode_alphas["d1"] * self.phys_params[0] * self.alpha_c
                self.k_hat = self.ode_alphas["d0"] * self.phys_params[1] * self.alpha_k
                match self.config["nonlinearity"]:
                    case "cubic":
                        self.k3_hat = self.ode_alphas["d0_3"] * self.phys_params[2] * self.alpha_k3
        match self.config["forcing"]:
            case {}:
                self.eta = self.ode_alphas["ff"]

        # calculate ode residual
        match self.config:
            case {"nonlinearity":"linear","forcing":None}:
                R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_pde_hat
            case {"nonlinearity":"cubic","forcing":None}:
                R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_pde_hat + self.k3_hat * xh_pde_hat**3
            case {"nonlinearity":"linear","forcing":{}}:
                R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_pde_hat - self.eta * self.force
            case {"nonlinearity":"cubic","forcing":{}}:
                R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_pde_hat + self.k3_hat * xh_pde_hat**3 - self.eta * self.force

        return {
            "R_obs" : R_obs,
            "R_ode" : R_ode
        }

    def loss_func(self, t_pde, t_obs, x_obs, lambdas):
        residuals = self.calc_residuals(t_pde, t_obs, x_obs)
        R_obs = residuals["R_obs"]
        R_pde = residuals["R_pde"]

        L_obs = lambdas['obs'] * torch.mean(R_obs**2)
        L_ode = lambdas['ode'] * torch.mean(R_pde**2)
        loss = L_obs + L_ode

        return loss, [L_obs, L_ode]

    def predict(self, tp):
        yp = self.forward(tp)
        return yp


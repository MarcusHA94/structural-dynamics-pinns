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
    

class osa_pinn_sdof(nn.Module):

    def __init__(self, n_input, n_output, n_hidden, n_layers):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = nn.Tanh

        self.build_net()

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def forward(self, x0, v0, t, G=0.0, D=1.0):
        # y_ = self.net(x)
        # y = self.G + self.D * y_
        x = torch.cat((x0.view(-1,1), v0.view(-1,1), t.view(-1,1)), dim=1)
        y = G + D * self.net(x)
        return y
    
    def configure(self, **config):

        self.config = config

        self.nonlinearity = config["nonlinearity"]
        self.forcing = config["forcing"]
        self.param_type = config["phys_params"]["par_type"]

        self.T = config["T"]  # time interval (not normalised)
        self.T_hat = config["T_hat"]  # time interval (normalised)
        self.nct = config["nct"]  # number of time collocation points
        self.t_wind = torch.linspace(0, self.T_hat, self.nct)  # time collocation points in each window

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
                self.phys_params = torch.tensor([self.c, self.k, self.k3])
            case {"phys_params":{"par_type":"variable"},"nonlinearity":"linear"}:
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]["c"], config["phys_params"]["k"]])))
            case {"phys_params":{"par_type":"variable"},"nonlinearity":"cubic"}:
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]["c"], config["phys_params"]["k"], config["phys_params"]["k3"]])))
        match config["forcing"]:
            case dict():
                self.force = torch.tensor(config["forcing"]["F_tild"])

    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]["t"]
        self.alpha_x = config["alphas"]["x"]
        self.alpha_v = config["alphas"]["v"]
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
            case {"nonlinearity":"linear","forcing":dict()}:
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
                self.alpha_k3 = config["alphas"]["k3"]
                
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
            case {"nonlinearity":"cubic","forcing":dict()}:
                self.alpha_k = config["alphas"]["k"]
                self.alpha_c = config["alphas"]["c"]
                self.alpha_k3 = config["alphas"]["k3"]
                
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

    def set_colls_and_obs(self, t_data, x_data, v_data):

        n_obs = x_data.shape[0]-1

        # Observation set (uses displacement one data point ahead)
        self.x_obs = x_data[:-1].view(-1,1)  # initial displacement input
        self.v_obs = v_data[:-1].view(-1,1)  # initial velocity input
        self.t_obs = self.T_hat*torch.ones((n_obs,1))  # time at end of horizon (window)
        self.yy_obs = x_data[1:].view(-1,1).requires_grad_()  # displacement at end of window (output)

        # Collocation set (sets a copy of the x0, v0 for a vector of time over the time horizon)
        x_col = torch.zeros((n_obs*self.nct,1))
        v_col = torch.zeros((n_obs*self.nct,1))
        t_col = torch.zeros((n_obs*self.nct,1))
        D_col = torch.ones((n_obs*self.nct,1))
        G_col = torch.zeros((n_obs*self.nct,1))
        t_pred = torch.zeros((n_obs*self.nct,1))

        for i in range(n_obs):
            x_col[self.nct*i:self.nct*(i+1),0] = x_data[i].item()*torch.ones(self.nct)
            v_col[self.nct*i:self.nct*(i+1),0] = v_data[i].item()*torch.ones(self.nct)
            t_col[self.nct*i:self.nct*(i+1),0] = self.t_wind.clone()

            D_col[self.nct*i,0] = 0.0
            G_col[self.nct*i,0] = x_data[i]

            # generates a vector of the time for the predicted output, by simply adding the total window onto the current time in the data
            t_pred[self.nct*i:self.nct*(i+1),0] = t_data[i] + self.t_wind

        self.x_col = x_col.requires_grad_()
        self.v_col = v_col.requires_grad_()
        self.t_col = t_col.requires_grad_()
        self.G_col = G_col
        self.D_col = D_col
        
        self.ic_ids = torch.argwhere(t_col[:,0]==torch.tensor(0.0))

        return t_pred
    
    def calc_residuals(self):

        # generate prediction at observation points
        xh_obs_hat = self.forward(self.x_obs, self.v_obs, self.t_obs)
        R_obs = xh_obs_hat - self.yy_obs

        # generate prediction over prediction horizon
        xh_coll_hat = self.forward(self.x_col, self.v_col, self.t_col)#, self.G_col, self.D_col)

        # retrieve derivatives
        dx = torch.autograd.grad(xh_coll_hat, self.t_col, torch.ones_like(xh_coll_hat), create_graph=True)[0]  # ∂_t-hat N_y-hat
        dx2 = torch.autograd.grad(dx, self.t_col, torch.ones_like(dx), create_graph=True)[0]  # ∂^2_t-hat N_y-hat

        # initial condition residual
        R_ic1 = self.alpha_v * self.v_col[self.ic_ids] - (self.alpha_x/self.alpha_t) * dx[self.ic_ids]
        R_ic2 = self.x_col[self.ic_ids] - xh_coll_hat[self.ic_ids]
        R_ic = torch.cat((R_ic1,R_ic2), dim=0).view(-1)

        # retrieve ode loss parameters
        self.m_hat = self.ode_alphas["d2"]
        match self.param_type:
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
            case dict():
                self.eta = self.ode_alphas["ff"]

        # calculate ode residual
        match self.config:
            case {"nonlinearity":"linear","forcing":None}:
                R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat
            case {"nonlinearity":"cubic","forcing":None}:
                R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat + self.k3_hat * xh_coll_hat**3
            case {"nonlinearity":"linear","forcing":{}}:
                R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat - self.eta * self.force
            case {"nonlinearity":"cubic","forcing":{}}:
                R_ode = self.m_hat * dx2 + self.c_hat * dx + self.k_hat * xh_coll_hat + self.k3_hat * xh_coll_hat**3 - self.eta * self.force

        return {
            "R_obs" : R_obs,
            "R_ic" : R_ic,
            "R_ode" : R_ode
        }

    def loss_func(self, lambdas):
        residuals = self.calc_residuals()
        R_obs = residuals["R_obs"]
        R_ic = residuals["R_ic"]
        R_ode = residuals["R_ode"]

        L_obs = lambdas['obs'] * torch.mean(R_obs**2)
        L_ic = lambdas['ic'] * torch.mean(R_ic**2)
        L_ode = lambdas['ode'] * torch.mean(R_ode**2)
        loss = L_obs + L_ic + L_ode

        return loss, [L_obs, L_ic, L_ode]
    
    def predict(self):
        xp = self.forward(self.x_col, self.v_col, self.t_col)#, self.G_col, self.D_col)
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


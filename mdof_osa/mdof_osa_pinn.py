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
    

class osa_pinn_mdof(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_input = config["n_input"]
        self.n_output = config["n_output"]
        self.n_hidden = config["n_hidden"]
        self.n_layers = config["n_layers"]
        self.n_dof = config["n_dof"]
        self.activation = nn.Tanh

        self.build_net()
        if config["phys_params"]["par_type"] == "variable":
            self.build_ed_net()

        self.configure(**config)

    def build_net(self):
        self.net = nn.Sequential(
            nn.Sequential(*[nn.Linear(self.n_input, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.net
    
    def build_ed_net(self):
        self.ed_net = nn.Sequential(
            nn.Sequential(*[nn.Linear(1, self.n_hidden), self.activation()]),
            nn.Sequential(*[nn.Sequential(*[nn.Linear(self.n_hidden, self.n_hidden), self.activation()]) for _ in range(self.n_layers-1)]),
            nn.Linear(self.n_hidden, self.n_output)
            )
        return self.ed_net
    
    def forward(self, x0, v0, t, f0=None, G=0.0, D=1.0):
        if f0 is None:
            x = torch.cat((x0, v0, t.view(-1,1)), dim=1)
        else:
            x = torch.cat((x0, v0, t.view(-1,1), f0), dim=1)
        y = G + D * self.net(x)
        return y
    
    def configure(self, **config):

        self.config = config

        self.nonlinearity = config["nonlinearity"]
        self.param_type = config["phys_params"]["par_type"]

        self.nct = config["nct"]  # number of time collocation points

        self.set_phys_params()
        self.set_norm_params()

    def set_phys_params(self):
        config = self.config
        match config:
            case {"phys_params":{"par_type":"constant"},"nonlinearity":"linear"}:
                self.K = config["phys_params"]['K']
                self.C = config["phys_params"]['C']
                self.M = config["phys_params"]['M']
                self.A = torch.cat((
                    torch.cat((torch.zeros((self.n_dof,self.n_dof)),torch.eye(self.n_dof)), dim=1),
                    torch.cat((-torch.linalg.inv(self.M)@self.K, -torch.linalg.inv(self.M)@self.C), dim=1)
                    ), dim=0)
                self.H = torch.cat((torch.zeros((self.n_dof,self.n_dof)),torch.linalg.inv(self.M)), dim=0)
            case {"phys_params":{"par_type":"constant"},"nonlinearity":"cubic"}:
                self.K = config["phys_params"]['K']
                self.C = config["phys_params"]['C']
                self.M = config["phys_params"]['M']
                self.Kn = config["phys_params"]['Kn']
                self.A = torch.cat((
                    torch.cat((torch.zeros((self.n_dof,self.n_dof)),torch.eye(self.n_dof)), dim=1),
                    torch.cat((-torch.linalg.inv(self.M)@self.K, -torch.linalg.inv(self.M)@self.C), dim=1)
                    ), dim=0)
                self.H = torch.cat((torch.zeros((self.n_dof,self.n_dof)),torch.linalg.inv(self.M)), dim=0)
                self.An = torch.cat((torch.zeros((self.n_dof,self.n_dof)),-torch.linalg.inv(self.M)@self.Kn), dim=0)
            #TODO: Add variable physical parameters
            case {"phys_params":{"par_type":"variable"},"nonlinearity":"linear"}:
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]["c"], config["phys_params"]["k"]])))
            case {"phys_params":{"par_type":"variable"},"nonlinearity":"cubic"}:
                self.register_parameter("phys_params", nn.Parameter(torch.tensor([config["phys_params"]["c"], config["phys_params"]["k"], config["phys_params"]["k3"]])))
        match config["forcing"]:
            case torch.Tensor():
                self.force = config["forcing"]

    def set_norm_params(self):
        config = self.config
        self.alpha_t = config["alphas"]["t"]
        self.alpha_x = config["alphas"]["x"]
        self.alpha_v = config["alphas"]["v"]
        self.alpha_z = torch.cat((self.alpha_x*torch.ones(self.n_dof,1), self.alpha_v*torch.ones(self.n_dof,1)), dim=0)
        if config["forcing"] != None:
            self.alpha_F = config["alphas"]["F"]
        # for param_name, param_dict in config["phys_params"].items():
        #     if param_dict["type"] == "variable":
        #         setattr(self,"alpha_"+param_name[:-1],config["alphas"][param_name[:-1]])

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

        #TODO: Add variable phys params and equation discovery domain data
        # if self.config["phys_params"]["par_type"] == "variable":
        #     self.t_ed_col = torch.linspace(torch.min(t_data),torch.max(t_data),self.config["phys_params"]["ed_nct"]).reshape(-1,1).requires_grad_()
        #     self.t_ed_obs = t_data.clone().requires_grad_()
        #     self.y_ed_obs = x_data.clone().requires_grad_() 

        return t_pred
    
    def calc_residuals(self):

        # generate prediction at observation points
        if self.config["forcing"] is None:
            zh_obs_hat = self.forward(self.x_obs, self.v_obs, self.t_obs)
        else:
            zh_obs_hat = self.forward(self.x_obs, self.v_obs, self.t_obs, self.f_obs)
        R_obs = torch.sqrt(torch.sum((zh_obs_hat - self.z_obs)**2,dim=1))

        # generate prediction over prediction horizon (collocation domain)
        if self.config["forcing"] is None:
            zh_col_hat = self.forward(self.x_col, self.v_col, self.t_col)
        else:
            zh_col_hat = self.forward(self.x_col, self.v_col, self.t_col, self.f_col)
        
        # retrieve derivatives
        dzdt = torch.zeros((zh_col_hat.shape[0],zh_col_hat.shape[1]))
        for i in range(zh_col_hat.shape[1]):
            dzdt[:,i] = torch.autograd.grad(zh_col_hat[:,i], self.t_col, torch.ones_like(zh_col_hat[:,i]), create_graph=True)[0][:,0]  # ∂_t-hat N_x-hat

        # generate predictions for equation discovery domain
        #TODO: include equation discovery domain for state space
        if self.param_type=="variable":
            zh_ed_obs = self.ed_net(self.t_ed_obs)
            pass

        # initial condition residual
        # R_ic1 = self.alpha_v * self.v_col[self.ic_ids,:] - (self.alpha_x/self.alpha_t) * dxdt[self.ic_ids,:self.n_dof]
        R_ic2 = self.alpha_x * self.x_col[self.ic_ids,:] - self.alpha_x * zh_col_hat[self.ic_ids,:self.n_dof]
        R_ic3 = self.alpha_v * self.v_col[self.ic_ids,:] - self.alpha_v * zh_col_hat[self.ic_ids,self.n_dof:]
        R_ic = torch.cat((R_ic2.squeeze(),R_ic3.squeeze()), dim=1)

        # calculate ode residual
        match self.config:
            case {"nonlinearity":"linear","forcing":None}:
                R_ = (self.alpha_z/self.alpha_t)*dzdt.T - self.A@(self.alpha_z*zh_col_hat.T)
                R_ode = R_[self.n_dof:,:].T
            case {"nonlinearity":"cubic","forcing":None}:
                zn = (self.alpha_x*zh_col_hat[:,:self.n_dof] - torch.cat((torch.zeros(zh_col_hat.shape[0],1),self.alpha_x*zh_col_hat[:,:self.n_dof-1]),dim=1))**3
                R_ = (self.alpha_z/self.alpha_t)*dzdt.T - self.A@(self.alpha_z*zh_col_hat.T) - self.An@(zn.T)
                R_ode = R_[self.n_dof:,:].T
            case {"nonlinearity":"linear","forcing":torch.Tensor()}:
                R_ = (self.alpha_z/self.alpha_t)*dzdt.T - self.A@(self.alpha_z*zh_col_hat.T) - self.H@(self.alpha_F*self.f_col.T)
                R_ode = R_[self.n_dof:,:].T
            case {"nonlinearity":"cubic","forcing":torch.Tensor()}:
                zn = (self.alpha_x*zh_col_hat[:,:self.n_dof] - torch.cat((torch.zeros(zh_col_hat.shape[0],1),self.alpha_x*zh_col_hat[:,:self.n_dof-1]),dim=1))**3
                R_ = (self.alpha_z/self.alpha_t)*dzdt.T - self.A@(self.alpha_z*zh_col_hat.T) - self.An@(zn.T) - self.H@(self.alpha_F*self.f_col.T)
                R_ode = R_[self.n_dof:,:].T

        # continuity condition residual
        R_cc = R_[:self.n_dof,:].T

        if self.param_type == "variable":
            #TODO: Add equation discovery domain loss
            pass

            return {
                "R_obs" : R_obs,
                "R_ic" : R_ic,
                "R_cc" : R_cc,
                "R_ode" : R_ode,
                "R_ed_b" : None,
                "R_ed_p" : None
            }
        else:
            return {
            "R_obs" : R_obs,
            "R_ic" : R_ic,
            "R_cc" : R_cc,
            "R_ode" : R_ode
            }

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

        match self.param_type:
            case "variable":
                R_ed_b = residuals["R_ed_b"]
                L_ed_b = lambdas['ed_b'] * torch.mean(R_ed_b**2)
                R_ed_p = residuals["R_ed_p"]
                L_ed_p = lambdas['ed_p'] * torch.mean(R_ed_p**2)
                loss = L_obs + L_ic + L_ode + L_ed_b + L_ed_p

                return loss, [L_obs, L_ic, L_ode, L_ed_b, L_ed_p]
            case "constant":
                loss = L_obs + L_ic + L_cc + L_ode
                return loss, [L_obs, L_ic, L_cc, L_ode]
    
    def predict(self):
        if self.config['forcing'] is None:
            xp = self.forward(self.x_col, self.v_col, self.t_col)
        else:
            xp = self.forward(self.x_col, self.v_col, self.t_col, self.f_col)
        match self.param_type:
            case "variable":
                xp_ed = self.ed_net(self.t_ed_col)
                return xp, xp_ed, self.t_ed_col
            case "constant":
                return xp
    
    def predict_new(self, x0, v0, t, f0=None):
        if not torch.is_tensor(t):
            t = torch.tensor([[t]])
        if len(x0.shape) < 2:
            x0 = x0.reshape(1,-1)
        if len(v0.shape) < 2:
            v0 = v0.reshape(1,-1)
        if len(f0.shape) < 2:
            f0 = f0.reshape(1,-1)
        xp = self.forward(x0, v0, t, f0)
        # if f0 is None:
        #     X = torch.tensor([x0, v0, t])
        #     xp = self.forward(x0, v0, t, f0)
        # else:
        #     X = torch.tensor([x0, v0, t, f0])
        #     xp = self.forward(X)
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
        if hasattr(module, 'phys_params'):
            params = module.phys_params.data
            params = params.clamp(0,1)
            module.phys_params.data = params
            
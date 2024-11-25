import numpy as np
import torch
from typing import Union, Tuple

import matplotlib.pyplot as plt
from IPython import display

from tqdm import tqdm
from tqdm.auto import tqdm as tqdma

Tensor = Union[torch.Tensor, np.ndarray]
TensorFloat = Union[torch.Tensor, float]

class ParamClipper(object):

    def __init__(self, param_lims: dict=None):
        self.param_lims = param_lims

    def __call__(self, module):

        for i in range(module.n_dof):
            if hasattr(module, f'c_{i}'):
                params_c = getattr(module, f'c_{i}').data
                params_c = params_c.clamp(0, None)
                getattr(module, f'c_{i}').data = params_c
            if hasattr(module, f'k_{i}'):
                params_k = getattr(module, f'k_{i}').data
                params_k = params_k.clamp(0, None)
                getattr(module, f'k_{i}').data = params_k
            if hasattr(module, f'kn_{i}'):
                params_kn = getattr(module, f'kn_{i}').data
                params_kn = params_kn.clamp(0, None)
                getattr(module, f'kn_{i}').data = params_kn
            if hasattr(module, f'cn_{i}'):
                params_cn = getattr(module, f'cn_{i}').data
                params_cn = params_cn.clamp(0, None)
                getattr(module, f'cn_{i}').data = params_cn

def dropout(dropouts, *data_):
    data_dropped = [None] * len(data_)
    for i, data in enumerate(data_):
        data_dropped[i] = data.clone()
        for j in dropouts:
            data_dropped[i][:,j] = torch.zeros_like(data[:,j])
    
    return data_dropped

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

def range_data(data: Tensor, axis: Tuple[int, None] = None) -> Tensor:
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


class mdof_pinn_plotter:

    def __init__(self, n_dof, n_cols, figsize=(18,16)):

        # if n_dof > n_cols:
        #     sub_rows = n_dof // 3 + int((n_dof%3)!=0)
        #     sub_cols = n_cols
        # else:
        #     sub_rows = 1
        #     sub_cols = n_dof

        # mosaic_key = [[None]] * (sub_rows * 2)
        # for j in range(sub_rows):
        #     mosaic_key[2 * j] = [f'dsp_dof_{3*j+d:d}' for d in range(3)]
        #     mosaic_key[2 * j + 1] = [f'vel_dof_{3*j+d:d}' for d in range(3)]
        # mosaic_key.extend([['loss_plot'] * 3] * 5)
        
        mosaic_key = [[f'dsp_dof_{d:d}', f'vel_dof_{d:d}', f'acc_dof_{d:d}', f'frc_dof_{d:d}'] for d in range(n_dof)]
        mosaic_key.extend([['loss_plot'] * 4] * 5)

        self.fig, self.axs = plt.subplot_mosaic(
            mosaic_key,
            figsize=figsize,
            facecolor='w'
        )

    def plot_joint_loss_hist(self, ax, loss_hist, pinn_type='normal'):
        n_epoch = len(loss_hist)
        indices = np.arange(1,n_epoch+1)
        if n_epoch > 20000:
            step = int(np.floor(n_epoch/10000))
            loss_hist = loss_hist[::step,:]
            indices = indices[::step]
        if pinn_type == 'normal':
            labels = ["L_obs", "L_occ", "L_cc", "L_ode", "L"]
        else:
            labels = ["L_obs", "L_f", "L_cc", "L_ode", "L"]
        colors = ["tab:blue", "tab:purple", "tab:red", "tab:green", "black"]
        ax.cla()
        for i in range(len(labels)):
            ax.plot(indices, loss_hist[:,i], color=colors[i], label=labels[i])
        ax.set_yscale('log')
        ax.legend()

    def sort_data(self, vec2sort: np.ndarray, *data_: tuple[np.ndarray,...]):
        sort_ids = np.argsort(vec2sort)
        sorted_data_ = [None] * len(data_)
        for i, data in enumerate(data_):
            sorted_data_[i] = np.zeros_like(data)
            if len(data.shape) > 1:
                for j in range(data.shape[1]):
                    sorted_data_[i][:,j] = data[sort_ids,j].squeeze()
            else:
                sorted_data_[i] = data[sort_ids]
        if len(data_) > 1:
            return tuple(sorted_data_), sort_ids
        else:
            return sorted_data_[0], sort_ids
        
    def plot_result(self, axs_m, ground_truth, obs_data, prediction, alphas, n_dof, eq_pred = False):
        for ax in axs_m:
            axs_m[ax].cla()
        xL = np.amax(ground_truth["t"])
        for dof in range(n_dof):
            axs_m[f'dsp_dof_{dof:d}'].plot(ground_truth["t"], ground_truth["x_hat"][:,dof], color="grey", linewidth=0.5, alpha=0.5, label="Exact solution")
            axs_m[f'dsp_dof_{dof:d}'].plot(prediction["t_hat"] * alphas["t"].item(), prediction["x_hat"][:,dof]*alphas["x"].item(), color="tab:blue", linewidth=0.5, alpha=0.8, linestyle='--', label="Neural network prediction")
            yLx = np.amax(np.abs(ground_truth["x_hat"][:,dof]))
            axs_m[f'dsp_dof_{dof:d}'].set_xlim(-0.05*xL, 1.05*xL)
            axs_m[f'dsp_dof_{dof:d}'].set_ylim(-1.1*yLx, 1.1*yLx)

            axs_m[f'vel_dof_{dof:d}'].plot(ground_truth["t"], ground_truth["v_hat"][:,dof], color="grey", linewidth=0.5, alpha=0.5, label="Exact solution")
            axs_m[f'vel_dof_{dof:d}'].plot(prediction["t_hat"] * alphas["t"].item(), prediction["v_hat"][:,dof]*alphas["v"].item(), color="tab:red", linewidth=0.5, alpha=0.8, linestyle='--', label="Neural network prediction")
            yLv = np.amax(np.abs(ground_truth["v_hat"][:,dof]))
            axs_m[f'vel_dof_{dof:d}'].set_xlim(-0.05*xL, 1.05*xL)
            axs_m[f'vel_dof_{dof:d}'].set_ylim(-1.1*yLv, 1.1*yLv)
        
            axs_m[f'acc_dof_{dof:d}'].plot(obs_data["t_hat"] * alphas["t"].item(), obs_data["a_hat"][:, dof] * alphas["a"].item(), color="tab:olive", linewidth=0.5, alpha=0.8, label='Training Data')
            axs_m[f'acc_dof_{dof:d}'].plot(ground_truth["t"], ground_truth["a_hat"][:,dof], color="grey", linewidth=0.5, alpha=0.5, label="Exact solution")
            axs_m[f'acc_dof_{dof:d}'].plot(prediction["t_hat"] * alphas["t"].item(), prediction["a_hat"][:,dof]*alphas["a"].item(), color="tab:orange", linewidth=0.5, alpha=0.8, linestyle='--', label="Neural network prediction")
            yla = np.amax(np.abs(ground_truth["a_hat"][:,dof]))
            axs_m[f'acc_dof_{dof:d}'].set_xlim(-0.05*xL, 1.05*xL)
            axs_m[f'acc_dof_{dof:d}'].set_ylim(-1.1*yla, 1.1*yla)
        
            axs_m[f'frc_dof_{dof:d}'].plot(obs_data["t_hat"] * alphas["t"].item(), obs_data["f_hat"][:, dof] * alphas["f"].item(), color="tab:olive", linewidth=0.5, alpha=0.8, label='Training Data')
            axs_m[f'frc_dof_{dof:d}'].plot(ground_truth["t"], ground_truth["f_hat"][:,dof], color="grey", linewidth=0.5, alpha=0.5, label="Exact solution")
            axs_m[f'frc_dof_{dof:d}'].plot(prediction["t_hat"] * alphas["t"].item(), prediction["f_hat"][:,dof]*alphas["f"].item(), color="tab:purple", linewidth=0.5, alpha=0.8, linestyle='--', label="Neural network prediction")
            if eq_pred:
                axs_m[f'frc_dof_{dof:d}'].plot(prediction["t_hat"] * alphas["t"].item(), prediction["f_hat_eq"][:,dof]*alphas["f"].item(), color="tab:cyan", linewidth=0.5, alpha=0.8, linestyle='--', label="Equation prediction")
            ylf = np.amax(np.abs(ground_truth["f_hat"][:,dof]))
            axs_m[f'frc_dof_{dof:d}'].set_xlim(-0.05*xL, 1.05*xL)
            axs_m[f'frc_dof_{dof:d}'].set_ylim(-1.1*ylf, 1.1*ylf)
            
    def plot_train_update(self, ground_truth, obs_data, prediction, alphas, n_dof, loss_hist, eq_pred=False):

        self.plot_result(self.axs, ground_truth, obs_data, prediction, alphas, n_dof, eq_pred)
        self.plot_joint_loss_hist(self.axs['loss_plot'], np.array(loss_hist), pinn_type='spi' if eq_pred else 'normal')

class mdof_pinn_trainer:

    def __init__(self, train_dataset, data_config, n_dof, device, train_loader, col_domain=True):

        self.train_dataset = train_dataset
        self.device = device
        self.data_config = data_config
        self.train_loader = train_loader
        self.n_dof = n_dof
        self.col_domain = col_domain

        self.num_obs_samps = len(train_dataset) * data_config['seq_len']
        self.num_col_samps = len(train_dataset) * data_config['subsample'] * data_config['seq_len']

    def print_params(self, mdof_model, gt_params) -> str:
        write_str = 'c :                       k :                     cn :                     kn : \n'
        for i in range(mdof_model.n_dof):
            for param in ['m_', 'c_', 'k_', 'kn_', 'cn_']:
                if mdof_model.config["phys_params"][f'{param}{i}']["type"] == 'variable':
                    write_str += f'{param}{i}: {(getattr(mdof_model, f"{param}{i}").item())*(getattr(mdof_model, f"alpha_{param[:-1]}")):.3f} '
                else:
                    write_str += f'{param}{i}: {getattr(mdof_model, f"{param}{i}"):.3f} '
                write_str += f'[{gt_params[param][i]:.2f}]       '
            write_str += '\n'
        return write_str
    
    def train(self,
              num_epochs,
              mdof_model,
              print_step,
              net_optimisers,
              plotter,
              ground_truth,
              pinn_config,
              param_optimiser = None,
              loss_hist = None,
              print_params = False,
              param_clipper = None,
              schedulers = None,
              profile = False,
              ):

        self.prediction = {
            "t_hat" : None,
            "x_hat" : None,
            "v_hat" : None,
            "a_hat" : None,
            "f_hat" : None
        }

        self.obs_data_dict = {
            "t_hat" : None,
            "a_hat" : None,
            "f_hat" : None
        }
        
        if 'acc_obs_method' in pinn_config.keys():
            acc_obs_method = pinn_config['acc_obs_method']
        else:
            acc_obs_method = 'obs_model'

        epoch = 0
        if loss_hist is None:
            self.loss_hist = []
        else:
            self.loss_hist = loss_hist
        progress_bar = tqdm(total=num_epochs)

        if profile:
            profile.start()
        try:
            while epoch < num_epochs:

                write_string = ''
                write_string += f'Epoch {epoch:d}\n'
                phase_loss = 0.
                losses = [0.0] * 4
                mdof_model.train()
                for i, (obs_data, col_data) in enumerate(self.train_loader):
                    if profile:
                        profile.step()
                    
                    ### parse data
                    acc_obs = obs_data[..., :self.n_dof].float().to(self.device).requires_grad_()  # [sample, sequence, dof]
                    f_obs = obs_data[..., self.n_dof:2*self.n_dof].float().to(self.device).requires_grad_()  # [sample, sequence, dof]
                    time_obs_ = [obs_data[:, nq, -1].reshape(-1, 1).to(self.device).requires_grad_() for nq in range(self.data_config['seq_len'])]
                    
                    ### parse data into lists
                    # acc_obs = [obs_data[:, nq, :self.n_dof].float().to(self.device).requires_grad_() for nq in range(self.data_config['seq_len'])]  # [samples, n_dof] * seq_len
                    # f_obs = [obs_data[:, nq, self.n_dof:2*self.n_dof].float().to(self.device).requires_grad_() for nq in range(self.data_config['seq_len'])]
                    # time_obs_ = [obs_data[:, nq, -1].reshape(-1, 1).to(self.device).requires_grad_() for nq in range(self.data_config['seq_len'])]

                    # unroll collocation data
                    if self.col_domain:
                        time_col_ = [col_data[:, :, nq, -1].reshape(-1, 1).to(self.device).requires_grad_() for nq in range(self.data_config['seq_len'])]
                        force_col_ = [col_data[:, :, nq, self.n_dof:-1].reshape(-1, self.n_dof).float().to(self.device) for nq in range(self.data_config['seq_len'])]
                    else:
                        time_col_ = None
                        force_col_ = None

                    ### Calculate loss and backpropagate
                    for optim in net_optimisers:
                        optim.zero_grad()
                    if param_optimiser is not None:
                        param_optimiser.zero_grad()
                    loss, losses_i, _ = mdof_model.loss_func(time_obs_, acc_obs, f_obs, time_col_, force_col_, pinn_config['lambds'], pinn_config['dropouts'], acc_obs_method)
                    phase_loss += loss.item()
                    losses = [losses[j] + loss_i for j, loss_i in enumerate(losses_i)]
                    loss.backward()
                    for optim in net_optimisers: 
                        optim.step()
                    if param_optimiser is not None:
                        param_optimiser.step()
                        if param_clipper is not None:
                            mdof_model.apply(param_clipper)
                        
                self.loss_hist.append([loss_it.item() for loss_it in losses] + [phase_loss])
                write_string += f'\tLoss {loss:.4e}\n'
                if schedulers is not None:
                    for scheduler in schedulers:
                        scheduler.step()
                    write_string += f'\tLearning rate: {schedulers[0].get_last_lr()}\n'

                if (epoch + 1) % print_step == 0:
                    
                    mdof_model.eval()
                        
                    t_obs = np.zeros((self.num_obs_samps, 1))
                    a_obs = np.zeros((self.num_obs_samps, self.n_dof))
                    f_obs = np.zeros((self.num_obs_samps, self.n_dof))

                    t_pred = np.zeros((self.num_col_samps, 1))
                    z_pred = np.zeros((self.num_col_samps, 2*self.n_dof))
                    a_pred = np.zeros((self.num_col_samps, self.n_dof))
                    f_pred = np.zeros((self.num_col_samps, self.n_dof))
                    if mdof_model.pinn_type == 'spi':
                        f_pred_eq = np.zeros((self.num_col_samps, self.n_dof))

                    for i, (obs_data, col_data) in enumerate(self.train_loader):

                        inpoint_o = i * self.data_config['batch_size'] * self.data_config['seq_len']
                        outpoint_o = (i+1) * self.data_config['batch_size'] * self.data_config['seq_len']
                        t_obs[inpoint_o:outpoint_o] = obs_data[..., -1].cpu().reshape(-1,1)
                        a_obs[inpoint_o:outpoint_o] = obs_data[..., :self.n_dof].cpu().reshape(-1, self.n_dof)
                        f_obs[inpoint_o:outpoint_o] = obs_data[..., self.n_dof:2*self.n_dof].cpu().reshape(-1, self.n_dof)

                        inpoint_ = i * self.data_config['batch_size'] * self.data_config['subsample'] * self.data_config['seq_len']
                        outpoint_ = (i + 1) * self.data_config['batch_size'] * self.data_config['subsample'] * self.data_config['seq_len']
                        if self.col_domain:
                            t_pred_list = [obs_data[:, nq, -1].reshape(-1, 1).to(self.device).requires_grad_() for nq in range(self.data_config['seq_len'])]
                        else:
                            t_pred_list = [col_data[:, :, nq, -1].reshape(-1, 1).to(self.device).requires_grad_() for nq in range(self.data_config['seq_len'])]
                        if mdof_model.pinn_type == 'spi':
                            z_pred_, f_pred_, a_pred_, t_pred_, f_pred_eq_ = mdof_model.predict(t_pred_list)
                            f_pred_eq[inpoint_:outpoint_, :] = f_pred_eq_.detach().cpu().reshape(-1, self.n_dof).numpy()
                        else:
                            z_pred_, f_pred_, a_pred_, t_pred_ = mdof_model.predict(t_pred_list)

                        t_pred[inpoint_:outpoint_] = t_pred_.detach().cpu().numpy().reshape(-1,1)
                        z_pred[inpoint_:outpoint_, :] = z_pred_.detach().cpu().reshape(-1, 2*self.n_dof).numpy()
                        a_pred[inpoint_:outpoint_, :] = a_pred_.detach().cpu().reshape(-1, self.n_dof).numpy()
                        f_pred[inpoint_:outpoint_, :] = f_pred_.detach().cpu().reshape(-1, self.n_dof).numpy()
                    
                    (a_obs, f_obs, t_obs), _ = plotter.sort_data(t_obs[:,0], a_obs, f_obs, t_obs)
                    if mdof_model.pinn_type == 'spi':
                        (z_pred, f_pred, a_pred, t_pred, f_pred_eq), _ = plotter.sort_data(t_pred[:,0], z_pred, f_pred, a_pred, t_pred, f_pred_eq)
                        self.prediction["f_hat_eq"] = f_pred_eq
                    else:
                        (z_pred, f_pred, a_pred, t_pred), _ = plotter.sort_data(t_pred[:,0], z_pred, f_pred, a_pred, t_pred)

                    self.prediction['t_hat'] = t_pred
                    self.prediction["x_hat"] = z_pred[:, :self.n_dof]
                    self.prediction["v_hat"] = z_pred[:, self.n_dof:]
                    self.prediction["a_hat"] = a_pred
                    self.prediction["f_hat"] = f_pred

                    self.obs_data_dict['t_hat'] = t_obs
                    self.obs_data_dict['a_hat'] = a_obs
                    self.obs_data_dict['f_hat'] = f_obs

                    if mdof_model.pinn_type == 'spi':
                        plotter.plot_train_update(ground_truth, self.obs_data_dict, self.prediction, pinn_config['alphas'], self.n_dof, self.loss_hist, eq_pred=True)
                    else:
                        plotter.plot_train_update(ground_truth, self.obs_data_dict, self.prediction, pinn_config['alphas'], self.n_dof, self.loss_hist)
                    
                    display.clear_output(wait=True)
                    display.display(plt.gcf())
                    if print_params:
                        write_string += self.print_params(mdof_model, ground_truth['params'])
                    tqdma.write(write_string)
                epoch += 1
                progress_bar.update(1)
        except KeyboardInterrupt:
            progress_bar.close()
        
        if profile:
            profile.stop()
            
        display.clear_output()

        print(write_string)

class mdof_stoch_pinn_plotter:

    def __init__(self, n_dof, n_cols, figsize=(18,16), plot_force=False):

        self.plot_force = plot_force

        if n_dof > n_cols:
            sub_rows = n_dof // n_cols + int((n_dof%n_cols)!=0)
        else:
            sub_rows = 1

        if plot_force:
            mosaic_key = [[None]] * (sub_rows * 3)
            for j in range(sub_rows):
                mosaic_key[3 * j] = [f'dsp_dof_{n_cols*j+d:d}' for d in range(n_cols)]
                mosaic_key[3 * j + 1] = [f'vel_dof_{n_cols*j+d:d}' for d in range(n_cols)]
                mosaic_key[3 * j + 2] = [f'frc_dof_{n_cols*j+d:d}' for d in range(n_cols)]
        else:
            mosaic_key = [[None]] * (sub_rows * 2)
            for j in range(sub_rows):
                mosaic_key[2 * j] = [f'dsp_dof_{n_cols*j+d:d}' for d in range(n_cols)]
                mosaic_key[2 * j + 1] = [f'vel_dof_{n_cols*j+d:d}' for d in range(n_cols)]
        mosaic_key.extend([['loss_plot'] * n_cols] * sub_rows)

        self.fig, self.axs = plt.subplot_mosaic(
            mosaic_key,
            figsize=figsize,
            facecolor='w'
        )

    def plot_joint_loss_hist(self, ax, loss_hist):
        n_epoch = len(loss_hist)
        indices = np.arange(1,n_epoch+1)
        if n_epoch > 20000:
            step = int(np.floor(n_epoch/10000))
            loss_hist = loss_hist[::step,:]
            indices = indices[::step]
        labels = ["L_obs", "L_cc", "L_ode", "L_nc", "L"]
        colors = ["tab:blue", "tab:red", "tab:green", "tab:purple", "black"]
        ax.cla()
        for i in range(len(labels)):
            ax.plot(indices, loss_hist[:,i], color=colors[i], label=labels[i])
        ax.set_yscale('symlog')
        # if np.amin(loss_hist) < 1e-3:
        ax.set_ylim(-1e5, -1e3)
        ax.legend()

    def sort_data(self, vec2sort: np.ndarray, *data_: tuple[np.ndarray,...]):
        sort_ids = np.argsort(vec2sort)
        sorted_data_ = [None] * len(data_)
        for i, data in enumerate(data_):
            sorted_data_[i] = np.zeros_like(data)
            if len(data.shape) > 1:
                for j in range(data.shape[1]):
                    sorted_data_[i][:,j] = data[sort_ids,j].squeeze()
            else:
                sorted_data_[i] = data[sort_ids]
        if len(data_) > 1:
            return tuple(sorted_data_), sort_ids
        else:
            return sorted_data_[0], sort_ids
        
    def plot_result(self, axs_m, ground_truth, obs_data, prediction, alphas, n_dof):
        for ax in axs_m:
            axs_m[ax].cla()
        xL = np.amax(ground_truth["t"])
        for dof in range(n_dof):
            # displacement
            axs_m[f'dsp_dof_{dof:d}'].plot(obs_data["t_hat"] * alphas["t"].item(), obs_data["x_hat"][:, dof] * alphas["x"].item(), color="tab:olive", linewidth=0.5, alpha=0.8, label='Training Data')
            axs_m[f'dsp_dof_{dof:d}'].plot(ground_truth["t"], ground_truth["x_hat"][:,dof], color="grey", linewidth=0.5, alpha=0.5, label="Exact solution")
            axs_m[f'dsp_dof_{dof:d}'].plot(prediction["t_hat"] * alphas["t"].item(), prediction["x_hat"][:,dof]*alphas["x"].item(), color="tab:blue", linewidth=0.5, alpha=0.8, linestyle='--', label="Neural network prediction")
            axs_m[f'dsp_dof_{dof:d}'].fill_between((prediction["t_hat"]*alphas["t"].item()).squeeze(), (prediction["x_hat"][:,dof]-2*prediction['sigma_x'])*alphas["x"].item(), (prediction["x_hat"][:,dof]+2*prediction['sigma_x'])*alphas["x"].item(), alpha=0.25, color="tab:blue", label=r"$2\sigma$ Range")
            yLx = np.amax(np.abs(ground_truth["x_hat"][:,dof]))
            axs_m[f'dsp_dof_{dof:d}'].set_xlim(-0.05*xL, 1.05*xL)
            axs_m[f'dsp_dof_{dof:d}'].set_ylim(-1.1*yLx, 1.1*yLx)

            # velocity
            axs_m[f'vel_dof_{dof:d}'].plot(obs_data["t_hat"] * alphas["t"].item(), obs_data["v_hat"][:, dof] * alphas["v"].item(), color="tab:olive", linewidth=0.5, alpha=0.8, label='Training Data')
            axs_m[f'vel_dof_{dof:d}'].plot(ground_truth["t"], ground_truth["v_hat"][:,dof], color="grey", linewidth=0.5, alpha=0.5, label="Exact solution")
            axs_m[f'vel_dof_{dof:d}'].plot(prediction["t_hat"] * alphas["t"].item(), prediction["v_hat"][:,dof]*alphas["v"].item(), color="tab:red", linewidth=0.5, alpha=0.8, linestyle='--', label="Neural network prediction")
            axs_m[f'vel_dof_{dof:d}'].fill_between((prediction["t_hat"]*alphas["t"].item()).squeeze(), (prediction["v_hat"][:,dof]-2*prediction['sigma_v'])*alphas["v"].item(), (prediction["v_hat"][:,dof]+2*prediction['sigma_v'])*alphas["v"].item(), alpha=0.25, color="tab:blue", label=r"$2\sigma$ Range")
            yLv = np.amax(np.abs(ground_truth["v_hat"][:,dof]))
            axs_m[f'vel_dof_{dof:d}'].set_xlim(-0.05*xL, 1.05*xL)
            axs_m[f'vel_dof_{dof:d}'].set_ylim(-1.1*yLv, 1.1*yLv)

            # force
            if self.plot_force:
                axs_m[f'frc_dof_{dof:d}'].plot(
                    prediction["t_hat"] * alphas["t"].item(), 
                    obs_data["f_hat"][:, dof] * alphas["f"].item(), 
                    color="tab:olive", linewidth=0.5, alpha=0.8, label='Observation Data'
                    )
                axs_m[f'frc_dof_{dof:d}'].plot(
                    ground_truth["t"], 
                    ground_truth["f_hat"][:,dof], 
                    color="grey", linewidth=0.5, alpha=0.5, label="Exact solution"
                    )
                axs_m[f'frc_dof_{dof:d}'].plot(
                    prediction["t_hat"] * alphas["t"].item(), 
                    prediction["f_hat"][:,dof], 
                    color="tab:green", linewidth=0.5, alpha=0.8, linestyle='--', label="Prediction"
                    )
                # axs_m[f'frc_dof_{dof:d}'].fill_between(
                #     (prediction["t_hat"] * alphas["t"].item()).squeeze(), 
                #     (prediction["f_hat"][:,dof]) - (2*prediction['sigma_f'])*alphas["f"].item(), 
                #     (prediction["f_hat"][:,dof]) + (2*prediction['sigma_f'])*alphas["f"].item(), 
                #     alpha=0.25, color="tab:blue", label=r"$2\sigma$ Range"
                #     )
                # yLf = np.amax(np.abs(ground_truth["f_hat"][:,dof]))
                yLf = np.amax(np.abs(prediction["f_hat"][:,dof]))
                axs_m[f'frc_dof_{dof:d}'].set_xlim(-0.05*xL, 1.05*xL)
                axs_m[f'frc_dof_{dof:d}'].set_ylim(-1.1*yLf, 1.1*yLf)

        
    def plot_train_update(self, ground_truth, obs_data, prediction, alphas, n_dof, loss_hist):

        self.plot_result(self.axs, ground_truth, obs_data, prediction, alphas, n_dof)
        self.plot_joint_loss_hist(self.axs['loss_plot'], np.array(loss_hist))


class mdof_stoch_pinn_trainer:

    def __init__(self, train_dataset, data_config, n_dof, device, train_loader):

        self.train_dataset = train_dataset
        self.device = device
        self.data_config = data_config
        self.train_loader = train_loader
        self.n_dof = n_dof

        self.num_obs_samps = len(train_dataset) * data_config['seq_len'] * data_config['num_repeats']
        self.num_col_samps = len(train_dataset) * data_config['subsample'] * data_config['seq_len']

    def train(self, num_epochs, mdof_model, print_step, optimisers, plotter, ground_truth, pinn_config, loss_hist=None, print_params=False):

        net_optimisers = optimisers['nets']
        noise_optimiser = optimisers['noise']

        t_obs = np.zeros((self.num_obs_samps, 1))
        z_obs = np.zeros((self.num_obs_samps, 2*self.n_dof))

        t_pred = np.zeros((self.num_col_samps, 1))
        z_pred = np.zeros((self.num_col_samps, 2*self.n_dof))
        f_pred = np.zeros((self.num_col_samps, self.n_dof))

        self.prediction = {
            "t_hat" : None,
            "x_hat" : None,
            "v_hat" : None,
            "f_hat" : None
        }

        self.obs_data_dict = {
            "t_hat" : None,
            "x_hat" : None,
            "v_hat" : None
        }

        epoch = 0
        if loss_hist is None:
            self.loss_hist = []
        else:
            self.loss_hist = loss_hist
        if 'progress_bar' in globals():
            del progress_bar  # noqa: F821
        display.clear_output()
        progress_bar = tqdm(total=num_epochs)

        try:
            while epoch < num_epochs:

                write_string = ''
                write_string += f'Epoch {epoch:d}\n'
                phase_loss = 0.
                losses = [0.0] * 4
                for i, (obs_data, col_data) in enumerate(self.train_loader):

                    # parse observation domain data
                    time_obs_ = [obs_data[:, :, nq, -1].reshape(-1, 1).requires_grad_() for nq in range(self.data_config['seq_len'])]
                    state_obs = [obs_data[:, :, nq, :2*self.n_dof].reshape(-1, 2*self.n_dof).float().to(self.device) for nq in range(self.data_config['seq_len'])]
                    state_obs = torch.cat([state_obs[nq].unsqueeze(1) for nq in range(self.data_config['seq_len'])], dim=1)

                    # parse collocation domain data
                    time_col_ = [col_data[:, :, nq, -1].reshape(-1, 1).requires_grad_() for nq in range(self.data_config['seq_len'])]
                    force_col_ = [col_data[:, :, nq, 2*self.n_dof:-1].reshape(-1, self.n_dof).float().to(self.device) for nq in range(self.data_config['seq_len'])]

                    # network_optimizer.zero_grad()
                    for optim in net_optimisers:
                        optim.zero_grad()
                    noise_optimiser.zero_grad()
                    loss, losses_i, _ = mdof_model.loss_func(time_obs_, state_obs, time_col_, force_col_, pinn_config['lambds'], pinn_config['dropouts'])
                    phase_loss += loss.item()
                    losses = [losses[j] + loss_i for j, loss_i in enumerate(losses_i)]
                    loss.backward()
                    # network_optimizer.step()
                    for optim in net_optimisers:
                        optim.step()
                    noise_optimiser.step()

                self.loss_hist.append([loss_it.item() for loss_it in losses] + [phase_loss])
                write_string += f'\tLoss {loss.item():.4e}\n'
                write_string += f'Obs loss: {losses[0].item():.4e}, CC loss: {losses[1].item():.4e}, Ode loss: {losses[2].item():.4e}, Nc loss: {losses[3].item():.4e}\n'

                if (epoch + 1) % print_step == 0:

                    obs_step = self.data_config['batch_size'] * self.data_config['seq_len'] * self.data_config['num_repeats']
                    col_step = self.data_config['batch_size'] * self.data_config['subsample'] * self.data_config['seq_len']

                    for i, (obs_data, col_data) in enumerate(self.train_loader):

                        inpoint_o = i * obs_step
                        outpoint_o = (i+1) * obs_step
                        t_obs[inpoint_o:outpoint_o] = obs_data[..., -1].cpu().reshape(-1,1)
                        z_obs[inpoint_o:outpoint_o] = obs_data[..., :2*self.n_dof].cpu().reshape(-1, 2*self.n_dof)

                        inpoint_ = i * col_step
                        outpoint_ = (i + 1) * col_step
                        t_col_ = [col_data[:, :, nq, -1].reshape(-1, 1).requires_grad_() for nq in range(self.data_config['seq_len'])]
                        force_col_ = [col_data[:, :, nq, 2*self.n_dof:-1].reshape(-1, self.n_dof).float().to(self.device) for nq in range(self.data_config['seq_len'])]
                        # z_pred_, f_pred_, t_pred_, _ = mdof_model.predict(t_col_)
                        z_pred_, f_pred_, t_pred_, _, force_col = mdof_model.predict(t_col_, f_col=force_col_)

                        t_pred[inpoint_:outpoint_] = t_pred_.detach().cpu().numpy().reshape(-1,1)
                        z_pred[inpoint_:outpoint_, :] = z_pred_.detach().cpu().reshape(-1, 2*self.n_dof).numpy()
                        f_pred[inpoint_:outpoint_, :] = f_pred_.detach().cpu().reshape(-1, self.n_dof).numpy()
                    
                    (z_obs, t_obs), _ = plotter.sort_data(t_obs[:,0], z_obs, t_obs)
                    (z_pred, f_pred, t_pred), _ = plotter.sort_data(t_pred[:,0], z_pred, f_pred, t_pred)

                    self.prediction['t_hat'] = t_pred
                    self.prediction["x_hat"] = z_pred[:, :self.n_dof]
                    self.prediction["v_hat"] = z_pred[:, self.n_dof:]
                    self.prediction["f_hat"] = f_pred
                    # self.prediction["sigma_z"] = mdof_model.sigma_z.detach().item()
                    self.prediction["sigma_x"] = mdof_model.sigma_x.detach().item()
                    self.prediction["sigma_v"] = mdof_model.sigma_v.detach().item()
                    self.prediction["sigma_f"] = mdof_model.sigma_f.detach().item()

                    self.obs_data_dict['t_hat'] = t_obs
                    self.obs_data_dict['x_hat'] = z_obs[:, :self.n_dof]
                    self.obs_data_dict['v_hat'] = z_obs[:, self.n_dof:]
                    self.obs_data_dict['f_hat'] = force_col.detach()

                    plotter.plot_train_update(ground_truth, self.obs_data_dict, self.prediction, pinn_config['alphas'], self.n_dof, self.loss_hist)
                    
                    display.clear_output(wait=True)
                    display.display(plt.gcf())

                    # write_string += f'State noise: {self.prediction["sigma_z"]:.4e}\n'
                    write_string += f'Displ noise: {self.prediction["sigma_x"]:.4e}\n'
                    write_string += f'Veloc noise: {self.prediction["sigma_v"]:.4e}\n'
                    write_string += f'Force noise: {self.prediction["sigma_f"]:.4e}\n'

                    if print_params:
                        write_string += 'c :                       k :                     cn :                     kn : \n'
                        for j in range(self.n_dof):
                            write_string = '%d : ' % (j+1)
                            for param in ['c_','k_','cn_','kn_']:
                                if pinn_config['phys_params'][param]['type']=='constant':
                                    if len(getattr(mdof_model, param).shape) == 0:
                                        write_string += '%.4f ' % getattr(mdof_model, param)
                                    else:
                                        write_string += '%.4f ' % getattr(mdof_model, param)[j]
                                elif pinn_config['phys_params'][param]['type']=='variable':
                                    if len(getattr(mdof_model, param).shape) == 0:
                                        write_string += '%.4f ' % (getattr(mdof_model, param)*pinn_config['alphas'][param[:-1]])
                                    else:
                                        write_string += '%.4f ' % (getattr(mdof_model, param)[j]*pinn_config['alphas'][param[:-1]])
                                write_string += '[%.4f]       ' % ground_truth['params'][param][j]
                            write_string += (write_string + '\n')
                    tqdma.write(write_string)
                epoch += 1
                progress_bar.update(1)
        except KeyboardInterrupt:
            progress_bar.close()
        
        display.clear_output()

        print(write_string)

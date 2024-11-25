import torch
import numpy as np
import copy
from datetime import date
from typing import Union, Tuple

class test_parser:

    def update_date_id(self, date_id):
        self.date_id = date_id
        self.file_id = f'{self.date_id}__{self.test_id}'

class sparse_recov_parser(test_parser):

    def __init__(self, sparsity_type, nonlin_type, error_type, snr, dofs, p_obs_drop):

        self.nonlin_type = nonlin_type

        self.sparsity_type = sparsity_type
        a = np.arange(dofs)
        match sparsity_type:
            case 'domain_interpolation':
                if p_obs_drop == 40.0:
                    self.dropouts = [1, 3, 6, 8, 11, 13, 16, 18]
                    self.dropouts = np.delete(self.dropouts, np.argwhere(self.dropouts<dofs)).tolist()
                else:
                    # self.dropouts = [1, 3]
                    step = round(100/(100-p_obs_drop))
                    # self.dropouts = a[a%step==1].tolist()
                    dropout_dropouts = a[::step].tolist()
                    self.dropouts = np.delete(a, dropout_dropouts).tolist()
                testid1 = 'inter'
            case 'domain_extension':
                # self.dropouts = [0, 1]
                self.dropouts = a[a>int(dofs * (100 - p_obs_drop)/100)].tolist()
                testid1 = 'exten'
            case list():
                self.dropouts = a[sparsity_type].tolist()
                p_obs_drop = 100 * len(self.dropouts) / dofs
                testid1 = 'custom'

        self.error_type = error_type
        match error_type:
            case 'No error':
                testid4 = 'nonerr'
            case 'Value error':
                testid4 = 'valerr'
            case 'Linear model':
                testid4 = 'linmod'
            case 'Force missing':
                testid4 = 'frcmia'

        testid2 = f'{int(p_obs_drop):d}dr'
        testid3 = f'{dofs:d}dof'
        
        # self.nonlin_type = nonlin_type
        # match self.nonlin_type:
        #     case 'exponent_damping':
        #         testid2 = 'exd'
        #     case 'vanDerPol_damping':
        #         testid2 = 'vdp'
        #     case 'duffing_stiffness':
        #         testid2 = 'duf'
        self.snr = snr
        testid5 = f'{int(snr):d}snr'

        self.test_id = f'sr_{testid1}_{testid2}_{testid3}_{testid4}_{testid5}'
        
        self.date_id = date.today().strftime("%Y%m%d")
        
        self.file_id = f'{self.date_id}__{self.test_id}'
        

    def prescribe_params(self, true_params, min_error = 0.1, max_error = 0.25):

        prescr_params = copy.deepcopy(true_params)
        gt_params = copy.deepcopy(true_params)

        if self.error_type == 'Value error':
            params_error = ['c_', 'cn_', 'k_', 'kn_']
            for param_key in params_error:
                # randomly add or a subtract an error on the value between the min and max error
                for i in range(prescr_params[param_key].shape[0]):
                    np.random.seed(43810+i)
                    gt_params[param_key][i] += gt_params[param_key][i] * np.random.uniform(low = min_error, high = max_error) * [-1,1][np.random.randint(0, 2)]
        elif self.error_type == 'Linear model':
            prescr_params['kn_'] = np.zeros_like(true_params['kn_'])
            prescr_params['cn_'] = np.zeros_like(true_params['cn_'])
        
        return prescr_params, gt_params

    def pinn_param_dict(self, prescr_params):
        
        param_dict = {}
        n_dof = prescr_params['m_'].shape[0]
        for i in range(n_dof):
            for param in ['m_', 'c_', 'k_']:
                param_dict.update({f"{param}{i}" : {"type" : "constant", "value" : torch.tensor(prescr_params[param][i], dtype=torch.float32)}})
        if self.error_type != 'Linear model':
            match self.nonlin_type:
                case 'vanDerPol_damping' | 'exponent_damping':
                    for i in range(n_dof):
                        param_dict.update({f"cn_{i}" : {"type" : "constant", "value" : torch.tensor(prescr_params['cn_'][i], dtype=torch.float32)}})
                        param_dict.update({f"kn_{i}" : {"type" : "constant", "value" : torch.tensor(prescr_params['kn_'][i], dtype=torch.float32)}})
                case 'duffing_stiffness':
                    for i in range(n_dof):
                        param_dict.update({f"kn_{i}" : {"type" : "constant", "value" : torch.tensor(prescr_params['kn_'][i], dtype=torch.float32)}})
                        param_dict.update({f"cn_{i}" : {"type" : "constant", "value" : torch.tensor(prescr_params['cn_'][i], dtype=torch.float32)}})
        else:
            param_dict.update({f"cn_{i}" : {"type" : "constant", "value" : torch.tensor(0.0, dtype=torch.float32)} for i in range(n_dof)})
            param_dict.update({f"kn_{i}" : {"type" : "constant", "value" : torch.tensor(0.0, dtype=torch.float32)} for i in range(n_dof)})
        
        return param_dict
            

class param_est_parser(test_parser):

    def __init__(self, system_type, nonlin_type, n_dof, force_loc, snr, num_time_samps, num_repeats):

        self.system_type = system_type
        self.nonlin_type = nonlin_type
        self.n_dof = n_dof
        self.force_loc = force_loc
        self.snr = snr

        match system_type:
            case 'first_nonlin':
                testid1 = 'firstnln'
            case 'inter_nonlin':
                testid1 = 'internln'
            case 'fully_nonlin':
                if nonlin_type == 'vanDerPol_damping':
                    testid1 = 'vandpd'
                else:
                    testid1 = 'fullnln'

        testid2 = f'{n_dof:d}dof'

        if force_loc == -1:
            testid3 = 'fn'
        elif force_loc == 0:
            testid3 = 'f1'
        
        self.test_id = f'sr_{testid1}_{testid2}_{testid3}_snr{int(snr):d}'
        
        self.date_id = date.today().strftime("%Y%m%d")
        
        self.file_id = f'{self.date_id}__{self.test_id}'

    def prescribe_params(self, cn, kn, dofs):

        match self.system_type:
            case 'first_nonlin':
                cn_ = np.zeros((dofs))
                cn_[0] = cn
                kn_ = np.zeros((dofs))
                kn_[0] = kn
            case 'inter_nonlin':
                cn_ = np.zeros((dofs))
                for i in range(0, cn_.shape[0], 2): 
                    cn_[i] = cn
                kn_ = np.zeros((dofs))
                for i in range(0, kn_.shape[0], 2): 
                    kn_[i] = kn
            case 'fully_nonlin':
                cn_ = cn * np.ones((dofs))
                kn_ = kn * np.ones((dofs))
        return cn_, kn_
    
    def pinn_param_dict(self, m_, c_, k_, cn_, kn_):

        param_dict = {
            "m_" : {
                "type" : "constant",
                "value" : torch.tensor(m_, dtype=torch.float32)
            },
            "c_" : {
                "type" : "variable",
                "value" : torch.ones(c_.shape[0], dtype=torch.float32)
            },
            "k_" : {
                "type" : "variable",
                "value" : torch.ones(k_.shape[0], dtype=torch.float32)
            },
        }
        match self.system_type:
            case 'first_nonlin':
                param_dict['cn_'] = {
                    'type' : 'variable',
                    'value' : torch.tensor(0.0, dtype=torch.float32)
                }
                param_dict['kn_'] = {
                    'type' : 'variable',
                    'value' : torch.tensor(1.0, dtype=torch.float32)
                }
            case 'inter_nonlin' | 'fully_nonlin':
                if self.nonlin_type == 'duffing_stiffness':
                    param_dict['cn_'] = {
                        'type' : 'variable',
                        'value' : torch.zeros(self.n_dof, dtype=torch.float32)
                    }
                    param_dict['kn_'] = {
                        'type' : 'variable',
                        'value' : torch.ones(self.n_dof, dtype=torch.float32)
                    }
                elif self.nonlin_type == 'vanDerPol_damping' | 'exponent_damping':
                    param_dict['cn_'] = {
                        'type' : 'variable',
                        'value' : torch.ones(self.n_dof, dtype=torch.float32)
                    }
                    param_dict['kn_'] = {
                        'type' : 'variable',
                        'value' : torch.zeros(self.n_dof, dtype=torch.float32)
                    }
        
        return param_dict
    
    def pinn_explc_dict(self, m_, c_, k_, cn_, kn_):

        param_dict = {
            "m_" : {
                "type" : "constant",
                "value" : torch.tensor(m_, dtype=torch.float32)
            },
            "c_" : {
                "type" : "constant",
                "value" : torch.tensor(c_, dtype=torch.float32)
            },
            "k_" : {
                "type" : "constant",
                "value" : torch.tensor(k_, dtype=torch.float32)
            },
        }
        match self.system_type:
            case 'first_nonlin':
                param_dict['cn_'] = {
                    'type' : 'variable',
                    'value' : torch.tensor(0.0, dtype=torch.float32)
                }
                param_dict['kn_'] = {
                    'type' : 'variable',
                    'value' : torch.tensor(1.0, dtype=torch.float32)
                }
            case 'inter_nonlin' | 'fully_nonlin':
                param_dict['cn_'] = {
                    'type' : 'constant',
                    'value' : torch.tensor(cn_, dtype=torch.float32)
                }
                param_dict['kn_'] = {
                    'type' : 'constant',
                    'value' : torch.tensor(kn_, dtype=torch.float32)
                }
        
        return param_dict


class state_param_parser(test_parser):

    def __init__(self, sparsity_type, nonlin_type, error_type, snr, dofs, p_obs_drop):

        self.nonlin_type = nonlin_type

        self.sparsity_type = sparsity_type
        a = np.arange(dofs)
        match sparsity_type:
            case 'domain_interpolation':
                if p_obs_drop == 40.0:
                    self.dropouts = np.array([1, 3, 6, 8, 11, 13, 16, 18])
                    self.dropouts = np.delete(self.dropouts, np.argwhere(self.dropouts>dofs)).tolist()
                else:
                    # self.dropouts = [1, 3]
                    step = round(100/(100-p_obs_drop))
                    # self.dropouts = a[a%step==1].tolist()
                    dropout_dropouts = a[1::step].tolist()
                    self.dropouts = np.delete(a, dropout_dropouts).tolist()
                testid1 = 'inter'
            case 'domain_extension':
                # self.dropouts = [0, 1]
                self.dropouts = a[a<round(dofs * p_obs_drop/100)].tolist()
                testid1 = 'exten'
            case list():
                self.dropouts = a[sparsity_type].tolist()
                p_obs_drop = 100 * len(self.dropouts) / dofs
                testid1 = 'custom'

        self.error_type = error_type
        match error_type:
            case 'No error':
                testid3 = 'nonerr'
            case 'Value error':
                testid3 = 'valerr'
            case 'Linear model':
                testid3 = 'linmod'
            case 'Force missing':
                testid3 = 'frcmia'

        match nonlin_type:
            case 'exponent_damping':
                testid4 = 'exd'
            case 'vanDerPol_damping':
                testid4 = 'vdp'
            case 'duffing_stiffness':
                testid4 = 'duf'

        testid2 = f'{int(p_obs_drop):d}dr'
        testid5 = f'{dofs:d}dof'

        self.snr = snr
        testid6 = f'{int(snr):d}snr'

        self.test_id = f'spe_{testid1}_{testid2}_{testid3}_{testid4}_{testid5}_{testid6}'
        
        self.date_id = date.today().strftime("%Y%m%d")
        
        self.file_id = f'{self.date_id}__{self.test_id}'

    def prescribe_params(self, true_params, min_error = 0.1, max_error = 0.25):

        prescr_params = copy.deepcopy(true_params)
        gt_params = copy.deepcopy(true_params)

        if self.error_type == 'Value error':
            params_error = ['c_', 'cn_', 'k_', 'kn_']
            for param_key in params_error:
                # randomly add or a subtract an error on the value between the min and max error
                for i in range(gt_params[param_key].shape[0]):
                    np.random.seed(43810+i*100)
                    gt_params[param_key][i] += gt_params[param_key][i] * np.random.uniform(low = min_error, high = max_error) * [-1,1][np.random.randint(0, 2)]
        elif self.error_type == 'Linear model':
            prescr_params['kn_'] = np.zeros_like(true_params['kn_'])
            prescr_params['cn_'] = np.zeros_like(true_params['cn_'])
        
        return prescr_params, gt_params

    def pinn_param_dict(self, prescr_params, param_dropouts):

        param_dict = {}
        n_dof = prescr_params['m_'].shape[0]
        for i in range(n_dof):
            for param in ['m_', 'c_', 'k_']:
                if i in param_dropouts[param]:
                    param_dict.update({f"{param}{i}" : {"type" : "variable", "value" : torch.tensor(1.0, dtype=torch.float32)}})
                else:
                    param_dict.update({f"{param}{i}" : {"type" : "constant", "value" : torch.tensor(prescr_params[param][i], dtype=torch.float32)}})
        if self.error_type != 'Linear model':
            match self.nonlin_type:
                case 'vanDerPol_damping' | 'exponent_damping':
                    for i in range(n_dof):
                        if i in param_dropouts['cn_']:
                            param_dict.update({f"cn_{i}" : {"type" : "variable", "value" : torch.tensor(1.0, dtype=torch.float32)}})
                        else:
                            param_dict.update({f"cn_{i}" : {"type" : "constant", "value" : torch.tensor(prescr_params['cn_'][i], dtype=torch.float32)}})
                        param_dict.update({f"kn_{i}" : {"type" : "constant", "value" : torch.tensor(prescr_params['kn_'][i], dtype=torch.float32)}})
                case 'duffing_stiffness':
                    for i in range(n_dof):
                        if i in param_dropouts['kn_']:
                            param_dict.update({f"kn_{i}" : {"type" : "variable", "value" : torch.tensor(1.0, dtype=torch.float32)}})
                        else:
                            param_dict.update({f"kn_{i}" : {"type" : "constant", "value" : torch.tensor(prescr_params['kn_'][i], dtype=torch.float32)}})
                        param_dict.update({f"cn_{i}" : {"type" : "constant", "value" : torch.tensor(prescr_params['cn_'][i], dtype=torch.float32)}})
            # param_dict['lin_model'] = False
        else:
            param_dict.update({f"cn_{i}" : {"type" : "constant", "value" : torch.tensor(0.0, dtype=torch.float32)} for i in range(n_dof)})
            param_dict.update({f"kn_{i}" : {"type" : "constant", "value" : torch.tensor(0.0, dtype=torch.float32)} for i in range(n_dof)})
            # param_dict['lin_model'] = True
        
        return param_dict

def add_noise(x: np.ndarray, db: Tuple[float, None] = None, SNR: Tuple[float, None] = None, seed: int = 43810) -> np.ndarray:

    ns = x.shape[0]
    nd = x.shape[1]
    x_noisy = np.zeros_like(x)

    match [db, SNR]:
        case [float(), None]:
            noise_amp = 10.0 ** (db / 10.0)
            for i in range(nd):
                np.random.seed(seed + i)
                noise_x = np.random.normal(loc=0.0, scale=np.sqrt(noise_amp), size=ns)
                x_noisy[:,i] = x[:,i] + noise_x
        case [None, float()]:
            P_sig_ = 10 * np.log10(np.mean(np.mean(x**2, axis=1), axis=0))
            P_noise_ = P_sig_ - SNR
            noise_amp_ = 10 ** (P_noise_ / 10.0)
            for i in range(nd):
                np.random.seed(seed + i)
                if np.mean(x[:, i]**2) == 0:
                    # noise_x = np.random.normal(loc=0.0, scale=np.sqrt(noise_amp_), size=ns)
                    noise_x = np.zeros(ns)
                else:
                    P_sig = 10 * np.log10(np.mean(x[:, i]**2))
                    P_noise = P_sig - SNR
                    noise_amp = 10 ** (P_noise / 10.0)
                    noise_x = np.random.normal(loc=0.0, scale=np.sqrt(noise_amp), size=ns)
                x_noisy[:,i] = x[:,i] + noise_x
        case [float(), float()]:
            raise Exception("Over specified, please select either db or SNR")
        case [None, None]:
            raise Exception("No noise level specified")
    return x_noisy


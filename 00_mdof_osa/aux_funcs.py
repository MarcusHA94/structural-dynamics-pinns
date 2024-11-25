import torch
import numpy as np
import copy

def dropout(dropouts, *data_):
    data_dropped = [None] * len(data_)
    for i, data in enumerate(data_):
        data_dropped[i] = data.clone()
        for j in dropouts:
            data_dropped[i][:,j] = torch.zeros_like(data[:,j])
    
    return data_dropped

class test_parser:

    pass

class sparse_recov_parser(test_parser):

    def __init__(self, sparsity_type, nonlin_type, error_type, snr, dofs, p_obs_drop):

        self.nonlin_type = nonlin_type

        self.sparsity_type = sparsity_type
        a = np.arange(dofs)
        match sparsity_type:
            case 'domain_interpolation':
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

        self.test_id = f'sr_{testid1}_{testid2}_{testid3}_{testid4}'

    def prescribe_params(self, true_params, min_error = 0.1, max_error = 0.25):

        prescr_params = copy.deepcopy(true_params)

        if self.error_type == 'Value error':
            params_error = ['c_', 'cn_', 'k_', 'kn_']
            for param_key in params_error:
                # randomly add or a subtract an error on the value between the min and max error
                for i in range(prescr_params[param_key].shape[0]):
                    np.random.seed(43810+i)
                    prescr_params[param_key][i] += prescr_params[param_key][i] * np.random.uniform(low = min_error, high = max_error) * [-1,1][np.random.randint(0, 2)]
        elif self.error_type == 'Linear model':
            prescr_params['kn_'] = np.zeros_like(true_params['kn_'])
            prescr_params['cn_'] = np.zeros_like(true_params['cn_'])

        # match self.nonlin_type:
        #     case 'duffing_stiffness':
        #         prescr_params['cn_'] = torch.zeros_like(prescr_params['cn_'])
        #     case 'vanDerPol_damping' | 'exponent_damping':
        #         prescr_params['kn_'] = torch.zeros_like(prescr_params['kn_'])
        
        return prescr_params

class param_est_parser(test_parser):

    def __init__(self, system_type, nonlin_type, force_loc, snr):

        self.system_type = system_type
        self.nonlin_type = nonlin_type
        self.force_loc = force_loc
        self.snr = snr

        match system_type:
            case 'linear':
                testid1 = 'lin'
            case 'locally_nonlin':
                testid1 = 'lon'
            case 'fully_nonlin':
                testid1 = 'fun'

        match nonlin_type:
            case 'linear':
                testid2 = 'lin'
            case 'vanDerPol_damping':
                testid2 = 'vdp'
            case 'duffing_stiffness':
                testid2 = 'duf'

        self.test_id = f'sr_{testid1}_{testid2}_f{int(force_loc):d}_snr{int(snr):d}'

    def prescribe_params(self, cn, kn, dofs):

        match self.system_type:
            case 'linear':
                cn_ = np.zeros((dofs))
                kn_ = np.zeros((dofs))
            case 'locally_nonlin':
                cn_ = np.zeros((dofs))
                cn_[0] = cn
                kn_ = np.zeros((dofs))
                kn_[0] = kn
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
        match self.system_type, self.nonlin_type:
            case 'linear', _:
                param_dict['cn_'] = {
                    'type' : 'constant',
                    'value' : torch.tensor(0.0, dtype=torch.float32)
                }
                param_dict['kn_'] = {
                    'type' : 'constant',
                    'value' : torch.tensor(0.0, dtype=torch.float32)
                }
            case _, 'vanDerPol_damping' | 'exponent_damping':
                param_dict['cn_'] = {
                    'type' : 'variable',
                    'value' : torch.tensor(1.0, dtype=torch.float32)
                }
                param_dict['kn_'] = {
                    'type' : 'constant',
                    'value' : torch.tensor(0.0, dtype=torch.float32)
                }
            case _, 'duffing_stiffness':
                param_dict['cn_'] = {
                    'type' : 'constant',
                    'value' : torch.tensor(0.0, dtype=torch.float32)
                }
                param_dict['kn_'] = {
                    'type' : 'variable',
                    'value' : torch.tensor(1.0, dtype=torch.float32)
                }
        # match self.system_type, self.nonlin_type:
        #     case 'linear', _:
        #         param_dict['cn_'] = {
        #             'type' : 'constant',
        #             'value' : torch.zeros(cn_.shape[0], dtype=torch.float32)
        #         }
        #         param_dict['kn_'] = {
        #             'type' : 'constant',
        #             'value' : torch.zeros(kn_.shape[0], dtype=torch.float32)
        #         }
        #     case _, 'vanDerPol_damping' | 'exponent_damping':
        #         param_dict['cn_'] = {
        #             'type' : 'variable',
        #             'value' : torch.tensor(cn_, dtype=torch.float32)
        #         }
        #         param_dict['kn_'] = {
        #             'type' : 'constant',
        #             'value' : torch.zeros(kn_.shape[0], dtype=torch.float32)
        #         }
        #     case _, 'duffing_stiffness':
        #         param_dict['cn_'] = {
        #             'type' : 'constant',
        #             'value' : torch.zeros(cn_.shape[0], dtype=torch.float32)
        #         }
        #         param_dict['kn_'] = {
        #             'type' : 'variable',
        #             'value' : torch.tensor(kn_, dtype=torch.float32)
        #         }
        
        return param_dict




import numpy as np
import warnings

class nonlinearity:

    pass

class exponent_stiffness(nonlinearity):

    def __init__(self, kn_, exponent=3, dofs=None):
        self.exponent = exponent
        match kn_:
            case np.ndarray():
                self.kn_ = kn_
                self.dofs = kn_.shape[0]
            case None:
                warnings.warn('No nonlinear stiffness parameters provided, proceeding with zero', UserWarning)
                self.kn_ = None
            case _:
                if dofs is None:
                    warnings.warn('Under defined nonlinearity, proceeding with zero nonlinearity')
                    self.kn_ = None
                else:
                    self.kn_ = kn_ * np.ones(dofs)
                    self.dofs = dofs
        
        self.Cn = np.zeros((dofs,dofs+1))
        self.Kn = np.concatenate((np.diag(kn_), np.zeros((dofs,1))), axis=1) - np.concatenate((np.zeros((dofs,1)), np.diag(kn_[1:],1)), axis=1)
        
    def z_func(self, x, xdot):
        return np.concatenate((x**self.exponent,np.zeros_like(x[:self.dofs])),axis=0)
    

class exponent_damping(nonlinearity):

    def __init__(self, cn_, exponent=0.5, dofs=None):
        self.exponent = exponent
        match cn_:
            case np.ndarray():
                self.cn_ = cn_
                self.dofs = cn_.shape[0]
            case None:
                warnings.warn('No quadratic damping parameters provided, proceeding with zero', UserWarning)
                self.cn_ = None
            case _:
                if dofs is None:
                    warnings.warn('Under defined nonlinearity, proceeding with zero nonlinearity')
                    self.cn_ = None
                else:
                    self.cn_ = cn_ * np.ones(dofs)
                    self.dofs = dofs
        
        self.Cn = np.concatenate((np.diag(cn_), np.zeros((dofs,1))), axis=1) - np.concatenate((np.zeros((dofs,1)), np.diag(cn_[1:],1)), axis=1)
        self.Kn = np.zeros((dofs,dofs+1))
    
    def z_func(self, x, xdot):
        return np.concatenate((np.zeros_like(x[:self.dofs]),xdot**self.exponent),axis=0)

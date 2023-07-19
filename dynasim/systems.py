import numpy as np
from dynasim.base import mdof_system

class symmetric(mdof_system):
    '''
    Generic symmetric system
    '''

    def __init__(self, m_, c_, k_, dofs=None, nonlinearity=None):

        if type(m_) is np.ndarray:
            dofs = m_.shape[0]
        elif dofs is not None:
            m_ = m_ * np.ones((dofs))
            c_ = c_ * np.ones((dofs))
            k_ = k_ * np.ones((dofs))
        else:
            raise Exception('Under defined system, please provide either parameter vectors or number of degrees of freedom')
        
        self.m_ = m_
        self.c_ = c_
        self.k_ = k_

        M = np.diag(m_) 
        C = np.diag(c_[:-1]+c_[1:]) + np.diag(-c_[1:],k=1) + np.diag(-c_[1:],k=-1)
        K = np.diag(k_[:-1]+k_[1:]) + np.diag(-k_[1:],k=1) + np.diag(-k_[1:],k=-1)

        if nonlinearity is not None:
            self.nonlin_transform = lambda z : nonlinearity.z_func(
                np.concatenate((z[:dofs],2*z[dofs-1])) - np.concatenate((np.zeros_like(z[:1]),z[:dofs])),
                np.concatenate((z[dofs:],2*z[-1])) - np.concatenate((np.zeros_like(z[:1]),z[dofs:]))
                )
            Cn = nonlinearity.Cn
            Kn = nonlinearity.Kn
            super().__init__(M, C, K, Cn, Kn)
        else:
            super().__init__(M, C, K)


class cantilever(mdof_system):
    '''
    Generic "cantilever" system
    '''

    def __init__(self, m_, c_, k_, dofs=None, nonlinearity=None):

        if type(m_) is np.ndarray:
            dofs = m_.shape[0]
        elif dofs is not None:
            m_ = m_ * np.ones((dofs))
            c_ = c_ * np.ones((dofs))
            k_ = k_ * np.ones((dofs))
        else:
            raise Exception('Under defined system, please provide either parameter vectors or number of degrees of freedom')
        
        self.m_ = m_
        self.c_ = c_
        self.k_ = k_

        M = np.diag(m_)
        C = np.diag(np.concatenate((c_[:-1]+c_[1:],np.array([c_[-1]])),axis=0)) + np.diag(-c_[1:],k=1) + np.diag(-c_[1:],k=-1)
        K = np.diag(np.concatenate((k_[:-1]+k_[1:],np.array([k_[-1]])),axis=0)) + np.diag(-k_[1:],k=1) + np.diag(-k_[1:],k=-1)

        if nonlinearity is not None:
            self.nonlin_transform = lambda z : nonlinearity.z_func(
                z[:dofs] - np.concatenate((np.zeros_like(z[:1]),z[:dofs-1])),
                z[dofs:] - np.concatenate((np.zeros_like(z[:1]),z[dofs:-1]))
                )
            
            Cn = nonlinearity.Cn[:,:-1]
            Kn = nonlinearity.Kn[:,:-1]
            super().__init__(M, C, K, Cn, Kn)
        else:
            super().__init__(M, C, K)
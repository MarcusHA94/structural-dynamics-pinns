import numpy as np
import scipy
import scipy.signal
from math import pi

class excitation():

    def generate(self, time):
        return self._generate(time)
    
class sinusoid(excitation):
    '''
    Single sinusoidal signal with central frequency w, amplitude f0, and phase phi
    '''

    def __init__(self, w, f0=1.0, phi=0):
        self.w = w
        self.f0 = f0
        self.phi = phi

    def _generate(self, time, seed=43810):
        return self.f0 * np.sin(self.w*time + self.phi)
    
class white_gaussian(excitation):
    '''
    White Gaussian noise with variance f0, and mean
    '''

    def __init__(self, f0, mean=0.0):
        self.f0 = f0
        self.u = mean
    
    def _generate(self, time, seed=43810):
        ns = time.shape[0]
        np.random.seed(seed)
        return np.random.normal(self.u, self.f0*np.ones((ns)))
    
class sine_sweep(excitation):
    '''
    Sine sweep signal
    '''

    def __init__(self, w_l, w_u, F0=1.0, scale='linear'):
        self.w_l = w_l
        self.w_u = w_u
        self.F0 = F0
        self.scale = scale

    def _generate(self, time, seed=43810):
        f0 = self.w_l / (2*pi)
        f1 = self.w_u / (2*pi)
        F =  self.F0 * scipy.signal.chirp(time, f0, time[-1], f1, method=self.scale)
        return F
    
class rand_phase_ms(excitation):
    '''
    Random-phase multi-sine
    '''

    def __init__(self, freqs, Sx):
        self.freqs = freqs
        self.Sx = Sx

    def _generate(self, time, seed=43810):

        np.random.seed(seed)
        phases = np.random.randn(self.freqs.shape[0]) * pi/2
        F_mat = np.sin(time.reshape(-1,1) @ self.freqs.reshape(1,-1) + phases.T)

        return (F_mat @ self.Sx).reshape(-1)

class shaker():
    '''
    Shaker class generates force signals at each DOF using excitation class
    '''

    def __init__(self, excitations=None, seed=43810):

        self.excitations = excitations
        self.dofs = len(excitations)
        self.seed = seed

    def generate(self, time):
        nt = time.shape[0]
        self.f = np.zeros((self.dofs,nt))
        for n, excite in enumerate(self.excitations):
            match excite:
                case excitation():
                    self.f[n,:] = self.excitations[n]._generate(time, self.seed+n)
                case np.ndarray():
                    self.f[n,:] = self.excitations[n]
                case None:
                    self.f[n,:] = np.zeros(nt)

        return self.f
        



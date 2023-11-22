import numpy as np
import scipy

class simulator():

    def __init__(self, system):
        self.A = system.A
        self.H = system.H
        self.f = system.f
        self.t = system.t
        self.An = system.An
        self.dofs = system.dofs
        self.nonlin_transform = system.nonlin_transform
    
    def sim(self, time, z0):
        raise Exception('No simulator selected')
    
    def ode_f(self, t, z):
        match [self.An, self.f]:
            case [None,None]:
                return self.A@z
            case [_,None]:
                zn = self.nonlin_transform(z)
                return self.A@z + self.An@zn
            case [None,_]:
                t_id = np.argmin(np.abs(self.t-t))
                return self.A@z + self.H@self.f[:,t_id]
            case [_,_]:
                zn = self.nonlin_transform(z)
                t_id = np.argmin(np.abs(self.t-t))
                return self.A@z + self.An@zn + self.H@self.f[:,t_id]
    

class scipy_ivp(simulator):

    def __init__(self, system):
        super().__init__(system)

    def sim(self, time, z0):

        tspan = (time[0], time[-1])
        results = scipy.integrate.solve_ivp(
            fun = self.ode_f,
            t_span = tspan,
            y0 = z0,
            t_eval = time
            )
        z = results.y
        return {
            'x' : np.array(z[:self.dofs,:]),
            'xdot' : np.array(z[self.dofs:,:])
        }


class rk4(simulator):

    def __init__(self, system):
        super().__init__(system)
        self.system = system

    def sim_one(self, dt, z0, t_point):

        k1 = self.ode_f(t_point, z0)
        k2 = self.ode_f(t_point, z0 + k1*dt/2)
        k3 = self.ode_f(t_point, z0 + k2*dt/2)
        k4 = self.ode_f(t_point, z0 + k3*dt)
        zplus1 = z0 + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        return zplus1

    def sim(self, time, z0):

        self.ns = time.shape[0]
        dt = time[1]-time[0]
        z = np.zeros((2*self.system.dofs,self.ns))
        z[:,0] = z0
        for t in range(self.ns-1):
            k1 = self.ode_f(time[t], z[:,t])
            k2 = self.ode_f(time[t], z[:,t] + k1*dt/2)
            k3 = self.ode_f(time[t], z[:,t] + k2*dt/2)
            k4 = self.ode_f(time[t], z[:,t] + k3*dt)
            z[:,t+1] = z[:,t] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        
        return {
            'x' : np.array(z[:self.system.dofs,:]),
            'xdot' : np.array(z[self.system.dofs:,:])
        }
    
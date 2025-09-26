import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from model import Potential

# Clase para manejar la evolución del background
class Background:
    def __init__(self, potential: Potential):
        self.potential = potential
        self.solution = None 

    def H(self, phi, y):
        V = self.potential.evaluate(phi)
        return np.sqrt((0.5 * y**2 + V) / 3)
    
    def dHdt(self, y):
        return - 0.5*y**2
    
    def epsilon(self, phi, y):
        H = self.H(phi, y)
        dHdt = self.dHdt(y)
        return - dHdt/H**2
    
    def eta(self, phi, y):
        H = self.H(phi, y)
        dVdphi = self.potential.first_derivative(phi)
        return -(3*H*y + dVdphi)/(y*H)
    
    def R(self, phi, y):
        H = self.H(phi, y)
        dHdt = self.dHdt(y)
        return 6*(2*H**2 + dHdt)

        
    def EDOs(self, t, Y):
        """
        Ecuaciones diferenciales en función del tiempo t.
        """

        [phi, y, H, a, N] = Y
    
        # Definir la dinámica
        dVdphi = self.potential.first_derivative(phi)
    
        dphidt = y
        dydt = - 3*H*y - dVdphi
        dHdt = - 0.5*y**2
        dadt = a*H
        dNdt = H
    
        return [dphidt, dydt, dHdt, dadt, dNdt]
    
    def Initial_conditions(self, phi_0):
        """
        Calcula las condiciones iniciales para a0, H0 y N0 asumiendo el régimen slow-roll
        """
        y0 = 0
        a0 = 1e-3
        H0 = self.H(phi_0, y0)
        N0 = 0
        return [phi_0, y0, H0, a0, N0]
    

    def solver(self, phi_0):
        """
        Resuelve el sistema con condiciones iniciales adecuadas.
        """
        Y0 = self.Initial_conditions(phi_0)

        t_span = [0, 3e7]
        t_eval = np.linspace(0, 3e7, 10000)

        self.solution = solve_ivp(
            self.EDOs, t_span, Y0, t_eval=t_eval, method="DOP853",
            rtol=1e-6, atol=1e-12, dense_output=True, max_step = 0.1)
        return self.solution
    

    @property
    def data(self):
        
        if self.solution is None:
            raise ValueError('Primero debes resolver el sistema de ecuaciones')

        t = self.solution.t
        phi, dphidt, H, a, N = self.solution.y
        eps_H = -np.gradient(H, t)/H**2
        eta = self.eta(phi, dphidt)
        R = self.R(phi, dphidt)
        aH = a*H

        return {
        't': t, 'N': N, 'phi': phi, 'dphidt': dphidt, 'H': H, 'a': a, 'aH': aH, 'R': R,
            'eps_H': eps_H, 'eta': eta}    

        

    @property
    def N_end(self):

        '''
        Calculate the total number of efolds at the end of inflation, imposing the drive eps_H(t_end) = 1.
        '''
        
        if self.solution is None:
            raise ValueError('First you have to solve the system with the solver method of the Background class.')
        
        N = self.data['N']
        eps_H = self.data['eps_H']    

        # Interpolation to find where epsilon_H = 1
        eps_H_interp = interp1d(N, eps_H - 1, kind='cubic', fill_value='extrapolate', bounds_error=False)
        
        try:
            N_tot = root_scalar(eps_H_interp, bracket=[self.N_in, self.N_fin]).root
            return N_tot
        except:
            if np.max(eps_H) < 1:
                return None 
            else:
                idx = np.argmin(np.abs(eps_H - 1))
                return N[idx]
            
    @property
    def Ne(self):
        
        ''' 
        Calculate the number of efolds before the end of inflation. I.e. 
         Ne = Nend - N.
        '''
    
        Nend = self.N_end
        
        if Nend is None:
            raise ValueError('Inflation does not end in the given range')
        Ne = Nend - self.data['N']
        return Ne   
    

    def interpolation(self, x ='t'):

        coords = {'t': self.data['t'], 'N': self.data['N']}
        if x not in coords:
            raise ValueError("with_respect_to must be 't' o 'N'")

        x_vals = coords[x]
        variables = ['phi', 'dphidt', 'H', 'a', 'aH', 'R', 'eps_H', 'eta_H']
        return {
            var: interp1d(x_vals, self.data[var], kind='cubic', fill_value='extrapolate', bounds_error=False)
        for var in variables
        }


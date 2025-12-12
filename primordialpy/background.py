import numpy as np 
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os
from primordialpy.model import Potential

class Background:
    
    """
    Solves the background dynamics of a single-field inflationary model.

    This class integrates the evolution equations of the inflaton field, its velocity,
    and the Hubble parameter as functions of the number of e-folds N.
    It computes relevant background quantities such as the slow-roll parameters,
    the scale factor, and the comoving Hubble radius.
    """

    def __init__(self,
                  potential: Potential, 
                  phi0, 
                  N_in=0, 
                  N_fin=80, 
                  dphidN_0=None):
        
        self.potential = potential
        self.phi0 = phi0
        self.N_in = N_in
        self.N_fin = N_fin
        
        self.dphidN_0 = dphidN_0         
        self.solution = None
        self._derived_data = None 

         #Folder creation 
        os.makedirs('Data', exist_ok=True)
        os.makedirs('Figures', exist_ok=True)




    def _H(self, phi, dphidN):
        V = self.potential.evaluate(phi)
        kinetic_term = 3 - 0.5 * dphidN**2
        if kinetic_term <= 0:
            raise ValueError("Inflation ended (kinetic dominance reached within solver steps).")
        return np.sqrt(V / kinetic_term)

    def _EDOs(self, N, Y):

        phi, dphidN = Y  
        H = self._H(phi, dphidN)
        
        epsilon = 0.5 * dphidN**2
        dVdphi = self.potential.first_derivative(phi)

        d2phidN2 = -(3 - epsilon)*dphidN - (dVdphi / H**2)
        
        return [dphidN, d2phidN2] 
    


    def solver(self, method='DOP853', rtol=1e-10, atol=1e-12):
        """
        Public method to trigger the solution. 
        By default the methos is DOP853, but the user can try another like RK45, LSODA or Radau. 
        """
    
        if self.dphidN_0 is None:
            # Slow-roll initial condition: 3H*dot_phi approx -V'
            # dphi/dN approx -V'/V
            V0 = self.potential.evaluate(self.phi0)
            dV0 = self.potential.first_derivative(self.phi0)
            y_phi_prime = -dV0 / V0
        else:
            y_phi_prime = self.dphidN_0

        Y0 = [self.phi0, y_phi_prime]
        
        # 2. Solver
        N_eval = np.linspace(self.N_in, self.N_fin, 2000) 
        
        def end_inflation(N, Y):
            return 0.5 * Y[1]**2 - 1.0
        end_inflation.terminal = True
        end_inflation.direction = 1

        self.solution = solve_ivp(
            self._EDOs, 
            [self.N_in, self.N_fin],
            Y0,
            t_eval=N_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            events=end_inflation 
        )        
        self._derived_data = None



    def data(self, save = False, filename= None):
        if self.solution is None:
            raise RuntimeError("Model not solved yet. Call .solve() first.")
        
        if self._derived_data is not None:
            return self._derived_data

        N = self.solution.t
        phi = self.solution.y[0]
        dphidN = self.solution.y[1]
        
        V = self.potential.evaluate(phi)
        H = np.sqrt(V / (3 - 0.5 * dphidN**2))
        
        a = np.exp(N)
        aH = a * H        
        eps_H = 0.5 * dphidN**2
        dVdphi = self.potential.first_derivative(phi)
        d2phidN2 = -(3 - eps_H)*dphidN - (dVdphi / H**2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            eta_H = eps_H - (dphidN * d2phidN2) / (2 * eps_H)
            eta_H = eps_H - (d2phidN2 / dphidN) 

        if save:
            if filename is None:
                filename = 'background_data.txt'
            filepath = os.path.join('Data', filename)
            np.savetxt(filepath, 
                       np.column_stack([N, phi, dphidN, H, a, aH, eps_H, eta_H]),
                       header ='N phi dphidN H a aH eps_H eta_H',
                       fmt = '%.16e')

        self._derived_data = {
            'N': N, 'phi': phi, 'dphidN': dphidN, 'H': H, 
            'a': a, 'aH': aH, 'eps_H': eps_H, 'eta_H': eta_H
        }
        return self._derived_data
    
            
    
    def interpolation(self, x ='N'):

        coords = {'N': self.data()['N']}
        if x not in coords:
            raise ValueError("with_respect_to must be 'N' o 'Ne'")

        x_vals = coords[x]
        variables = ['phi', 'dphidN', 'H', 'a', 'aH', 'eps_H', 'eta_H']
        return {
            var: interp1d(x_vals, self.data()[var], kind='cubic', fill_value='extrapolate', bounds_error=False)
        for var in variables
        }

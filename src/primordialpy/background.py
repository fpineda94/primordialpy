import numpy as np 
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from primordialpy.model import Potential
import os



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
                  N_fin=100, 
                  dphidN_0=None):
        
        """
        Parameters
        ----------
        potential : Potential
            Inflationary potential instance.
        phi0 : float
            Initial value of the inflaton field.
        N_in : float, optional
            Initial number of e-folds. Default is 0.
        N_fin : float, optional
            Maximum number of e-folds. Default is 100.
        dphidN_0 : float, optional
            Initial field velocity. If None, slow-roll initial
            conditions are assumed.
        """
        
        self.potential = potential
        self.phi0 = phi0
        self.N_in = N_in
        self.N_fin = N_fin
        
        self.dphidN_0 = dphidN_0         
        self.solution = None
        self._derived_data = None 


    def _H(self, phi, dphidN):
        V = self.potential.evaluate(phi)
        kinetic_term = 3 - 0.5 * dphidN**2
        return np.sqrt(V / kinetic_term)

    def _ODEs(self, N, Y):

        phi, dphidN = Y  
        H = self._H(phi, dphidN)
        
        epsilon = 0.5 * dphidN**2
        dVdphi = self.potential.first_derivative(phi)

        d2phidN2 = -(3 - epsilon)*dphidN - (dVdphi / H**2)
        
        return [dphidN, d2phidN2] 
    
    def _end_inflation(self, N, Y):
        _, dphidN = Y
        eps_H = 0.5 * dphidN**2
        return eps_H - 1


    def solver(self, method='DOP853', rtol=1e-10, atol=1e-12):

        """
        Public method to trigger the solution. 
        By default the methos is DOP853, but the user can try another like RK45, LSODA or Radau. 
        """
    
        if self.dphidN_0 is None:
            V0 = self.potential.evaluate(self.phi0)
            dV0 = self.potential.first_derivative(self.phi0)
            y_phi_prime = -dV0 / V0
        else:
            y_phi_prime = self.dphidN_0

        Y0 = [self.phi0, y_phi_prime]
        
        # 2. Solver
        N_eval = np.linspace(self.N_in, self.N_fin, 10000) 


        self.solution = solve_ivp(
            self._ODEs, 
            [self.N_in, self.N_fin],
            Y0,
            t_eval=N_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            events=self._end_inflation,
            dense_output=True
        )        
        self._derived_data = None



    def data(self):

        if self.solution is None:
            raise RuntimeError("Model not solved yet. Call .solver() first.")
        
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
        d2phidN2 = np.gradient(dphidN, N)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            eta_H = eps_H - (d2phidN2 / dphidN) 

        self._derived_data = {
            'N': N, 'phi': phi, 'dphidN': dphidN, 'H': H, 
            'a': a, 'aH': aH, 'eps_H': eps_H, 'eta_H': eta_H
        }
        return self._derived_data
    



    def save_data(self, filename: str = "background.dat", path: str = "."):

        """
        Parameters
        ----------
        filename : str, optional
            Output filename. Default is 'background.dat'.
        path : str, optional
            Directory where the file will be saved. Default is current directory.
        """
        
        if self.solution is None:
            raise RuntimeError("Model not solved yet. Call .solver() first.")
        
        full_path = os.path.join(path, filename)
        
        d = self.data()
        header = "N  phi  dphidN  H  a  aH  eps_H  eta_H"
        data_matrix = np.column_stack([
            d['N'], d['phi'], d['dphidN'], d['H'],
            d['a'], d['aH'], d['eps_H'], d['eta_H']
        ])
        
        np.savetxt(full_path, data_matrix, header=header, comments='# ')
        print(f"Saved to {full_path}")
                
        
    # Si por ahora solo va N, simplificalo:
    def interpolation(self):
        x_vals = self.data()['N']
        variables = ['phi', 'dphidN', 'H', 'a', 'aH', 'eps_H', 'eta_H']
        return {
            var: interp1d(x_vals, self.data()[var], kind='cubic', 
                        fill_value='extrapolate', bounds_error=False)
            for var in variables
        }

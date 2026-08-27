import os
import numpy as np 
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from typing import Optional, Dict, List
# Assuming Potential is defined in primordialpy.model
from primordialpy.model import Potential

class Background:
    """
    Solves the background dynamics of a single-field inflationary model.

    This class integrates the evolution equations of the inflaton field, its velocity,
    and the Hubble parameter as functions of the number of e-folds N.
    It computes relevant background quantities such as the slow-roll parameters,
    the scale factor, and the comoving Hubble radius.

    Parameters
    ----------
    potential : Potential
        An instance of the inflationary potential.
    phi0 : float
        Initial value of the inflaton field.
    N_in : float, optional
        Initial number of e-folds. Default is 0.
    N_fin : float, optional
        Maximum number of e-folds to compute. Default is 100.
    dphidN_0 : float, optional
        Initial field velocity (dphi/dN). If None, standard slow-roll 
        initial conditions are assumed. Default is None.
    """

    def __init__(self,
                 potential: Potential, 
                 phi0: float, 
                 N_in: float = 0.0, 
                 N_fin: float = 100.0, 
                 dphidN_0: Optional[float] = None):
        
        self.potential = potential
        self.phi0 = phi0
        self.N_in = N_in
        self.N_fin = N_fin
        self.dphidN_0 = dphidN_0         
        self.solution = None
        self._derived_data: Optional[Dict[str, np.ndarray]] = None 

    def _H(self, phi: float, dphidN: float) -> float:
        """Computes the Hubble parameter in Planck units."""
        V = self.potential.evaluate(phi)
        kinetic_term = 3.0 - 0.5 * dphidN**2
        return np.sqrt(V / kinetic_term)

    def _odes(self, N: float, Y: List[float]) -> List[float]:
        """Defines the system of ODEs for the background dynamics."""
        phi, dphidN = Y  
        H = self._H(phi, dphidN)
        
        epsilon = 0.5 * dphidN**2
        dVdphi = self.potential.first_derivative(phi)

        d2phidN2 = -(3.0 - epsilon) * dphidN - (dVdphi / H**2)
        return [dphidN, d2phidN2] 
    
    def _end_inflation(self, N: float, Y: List[float]) -> float:
        """Event function to terminate integration when inflation ends (eps_H = 1)."""
        _, dphidN = Y
        eps_H = 0.5 * dphidN**2
        return eps_H - 1.0

    def solver(self, method: str = 'DOP853', rtol: float = 1e-10, atol: float = 1e-12) -> None:
        """
        Integrates the background equations of motion.

        By default, the high-order DOP853 method is used, which is excellent for 
        smooth equations, but RK45, LSODA, or Radau can also be specified.

        Parameters
        ----------
        method : str, optional
            Integration method for solve_ivp. Default is 'DOP853'.
        rtol : float, optional
            Relative tolerance for the solver. Default is 1e-10.
        atol : float, optional
            Absolute tolerance for the solver. Default is 1e-12.
        """
        if self.dphidN_0 is None:
            V0 = self.potential.evaluate(self.phi0)
            dV0 = self.potential.first_derivative(self.phi0)
            y_phi_prime = -dV0 / V0
        else:
            y_phi_prime = self.dphidN_0

        Y0 = [self.phi0, y_phi_prime]
        
        N_eval = np.linspace(self.N_in, self.N_fin, 10000) 

        def end_event(N: float, Y: List[float]) -> float:
            return self._end_inflation(N, Y)
        end_event.terminal = True

        self.solution = solve_ivp(
            self._odes, 
            [self.N_in, self.N_fin],
            Y0,
            t_eval=N_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            events=end_event,
            dense_output=True
        )        
        # Reset derived data cache in case solver is called multiple times
        self._derived_data = None

    def data(self) -> Dict[str, np.ndarray]:
        """
        Extracts and computes derived background quantities.

        Returns
        -------
        dict
            A dictionary containing numpy arrays for N, phi, dphidN, H, 
            a, aH, eps_H, and eta_H.

        Raises
        ------
        RuntimeError
            If the solver has not been run yet.
        """
        if self.solution is None:
            raise RuntimeError("Model not solved yet. Call .solver() first.")
        
        if self._derived_data is not None:
            return self._derived_data

        N = self.solution.t
        phi = self.solution.y[0]
        dphidN = self.solution.y[1]
        
        V = self.potential.evaluate(phi)
        H = np.sqrt(V / (3.0 - 0.5 * dphidN**2))
        a = np.exp(N)
        aH = a * H        
        eps_H = 0.5 * dphidN**2
        
        # Exact analytical second derivative avoids np.gradient grid errors
        dVdphi = self.potential.first_derivative(phi)
        d2phidN2 = -(3.0 - eps_H) * dphidN - (dVdphi / H**2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            eta_H = eps_H - (d2phidN2 / dphidN) 

        self._derived_data = {
            'N': N, 'phi': phi, 'dphidN': dphidN, 'H': H, 
            'a': a, 'aH': aH, 'eps_H': eps_H, 'eta_H': eta_H
        }
        return self._derived_data

    def save_background(self, filename: str = "background.dat", path: str = ".") -> None:
        """
        Saves the computed background data to a text file.

        Parameters
        ----------
        filename : str, optional
            Output filename. Default is 'background.dat'.
        path : str, optional
            Directory where the file will be saved. Default is the current directory.
        """
        if self.solution is None:
            raise RuntimeError("Model not solved yet. Call .solver() first.")
        
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        
        d = self.data()
        header = "N  phi  dphidN  H  a  aH  eps_H  eta_H"
        data_matrix = np.column_stack([
            d['N'], d['phi'], d['dphidN'], d['H'],
            d['a'], d['aH'], d['eps_H'], d['eta_H']
        ])
        
        np.savetxt(full_path, data_matrix, header=header, comments='# ')
        print(f"Saved to {full_path}")
                
    def interpolation(self, x: str = 'N') -> Dict[str, CubicSpline]:
        """
        Creates cubic spline interpolators for all background variables.

        Parameters
        ----------
        x : str, optional
            The independent variable for interpolation. Currently, only 'N' 
            is supported. Default is 'N'.

        Returns
        -------
        dict
            A dictionary of CubicSpline objects for each background variable.
        """
        if x != 'N':
            raise ValueError("The independent variable 'x' must be 'N'.")
        
        d = self.data()
        x_vals = d['N']
        variables = ['phi', 'dphidN', 'H', 'a', 'aH', 'eps_H', 'eta_H']
        
        return {
            var: CubicSpline(x_vals, d[var], extrapolate=True)
            for var in variables
        }
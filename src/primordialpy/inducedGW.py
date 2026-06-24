import numpy as np
from scipy.interpolate import interp1d
import os
from typing import Optional
from primordialpy.perturbations import Perturbations

class InducedGW:

    """
    Computes the scalar-induced gravitational wave (SIGW) spectrum.

    Uses the kernel from Kohri & Terada (arXiv:1804.08577) and follows
    the numerical approach of Rezazadeh (arXiv:2110.01482).

    Parameters
    ----------
    perturbations : Perturbations, optional
        Perturbations instance with a computed power spectrum.
    filename : str, optional
        Path to a file with columns [k, Ps].
    k_modes : array-like, optional
        Comoving wavenumber array [Mpc^-1].
    Ps : array-like, optional
        Primordial power spectrum array.
    k_col : int, optional
        Column index for k in the file. Default is 0.
    Ps_col : int, optional
        Column index for Ps in the file. Default is 1.
    delimiter : str, optional
        File delimiter. Default is None (whitespace).
    logk : bool, optional
        If True, k column is in log10 scale. Default is False.
    """

    def __init__(
        self,
        perturbations: Optional[Perturbations] = None,
        filename: Optional[str] = None,
        k_modes=None,
        Ps=None,
        k_col: int = 0,
        Ps_col: int = 1,
        delimiter=None,
        logk: bool = False
    ):
        
        self.f_hz = None
        self.omega_gw = None

        # -------- Case 1: From Perturbations object --------
        if perturbations is not None:
            self.pert = perturbations
            self.k_modes = perturbations.k_modes
            self.Ps = perturbations._P_s_array

        # -------- Case 2: From file --------
        elif filename is not None:
            self.k_modes, self.Ps = self._load_from_file(
                filename,
                k_col,
                Ps_col,
                delimiter,
                logk
            )
            self.pert = None

        # -------- Case 3: From arrays --------
        elif k_modes is not None and Ps is not None:
            self.k_modes = np.asarray(k_modes)
            self.Ps = np.asarray(Ps)
            self.pert = None

        else:
            raise ValueError(
                "You must provide either `perturbations`, `filename`, "
                "or (`k_modes`, `Ps`)."
            )

        self._validate_input()
        self._build_interpolator()

    
    def _load_from_file(
        self,
        filename,
        k_col,
        Ps_col,
        delimiter,
        logk
    ):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        data = np.loadtxt(filename, delimiter=delimiter)

        k = data[:, k_col]
        Ps = data[:, Ps_col]

        if logk:
            k = 10.0 ** k

        return k, Ps
    
    def _validate_input(self):
        if len(self.k_modes) != len(self.Ps):
            raise ValueError("k_modes and Ps must have the same length")

        if np.any(self.k_modes <= 0):
            raise ValueError("k_modes must be strictly positive")

        if not np.all(np.diff(self.k_modes) > 0):
            raise ValueError("k_modes must be strictly increasing")
        
    def _build_interpolator(self):

        mask = self.Ps > 0
        k_safe = self.k_modes[mask]
        Ps_safe = self.Ps[mask]

        if np.sum(mask) < 2:
            raise ValueError("Not enough positive Ps values to build interpolator")

        self.Ps_interp = interp1d(
            np.log(k_safe),
            np.log(Ps_safe),
            kind="linear",
            bounds_error=False,
            fill_value= -np.inf
        )

    def Ps_of_k(self, k):

        k = np.asarray(k, dtype=float)
        Ps = np.zeros_like(k)

        mask = k > 0
        Ps[mask] = np.exp(self.Ps_interp(np.log(k[mask])))

        return Ps
    

    def _kernel_averaged(self, u, v):
          
            x = u**2 + v**2 - 3.0
            
            num = 3.0 - (u + v)**2
            den = 3.0 - (u - v)**2
            
            with np.errstate(divide='ignore', invalid='ignore'):
                log_arg = np.abs(num / den)
                log_term = np.log(log_arg)
            log_term[den == 0] = 0.0 

            term1 = (-4.0 * u * v + x * log_term)**2
            
            theta = np.zeros_like(u)
            theta[(u + v) > np.sqrt(3.0)] = 1.0
            term2 = (np.pi**2) * (x**2) * theta
            
            numerator = 9.0 * (x**2) * (term1 + term2)
            denominator = 32.0 * (u**6) * (v**6)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                kernel = numerator / denominator
            kernel[denominator == 0] = 0.0
            
            return kernel

    def compute(self, k_output=None, n_int=200):
      
        if k_output is None:
            k_output = self.k_modes

        omega_gw = np.zeros_like(k_output)

        Omega_r0_h2 = 4.18e-5 
        prefactor_const = 0.83 * (106.75/10.75)**(-1/3) * (Omega_r0_h2) * (1/6.0)

        u_vals = np.logspace(-2, 2, n_int)
        v_vals = np.logspace(-2, 2, n_int)
        U, V = np.meshgrid(u_vals, v_vals, indexing='ij')
        
        geom_factor = (4.0 * V**2 - (1.0 - U**2 + V**2)**2)**2 / (16.0 * U**2 * V**2)
        
        mask_triangle = (np.abs(U - V) < 1.0) & ((U + V) > 1.0)
        Kernel_RD = self._kernel_averaged(U, V)
        Integrand_Base = geom_factor * Kernel_RD * mask_triangle
        
        dln_u = np.log(u_vals[1]/u_vals[0])
        dln_v = np.log(v_vals[1]/v_vals[0])
        diff_area = (U * V) * dln_u * dln_v

        print(f"Computing SIGW for {len(k_output)} modes...")

        # main Loop 
        for i, k_curr in enumerate(k_output):
            P_u = self.Ps_of_k(k_curr * U)
            P_v = self.Ps_of_k(k_curr * V)
            
            Total_Integrand = Integrand_Base * P_u * P_v
            
            integral_val = np.sum(Total_Integrand * diff_area)
            omega_gw[i] = prefactor_const * integral_val

        # Conversion k [Mpc^-1] -> f [Hz]
        # f = c k / (2 pi a0).  1 Mpc^-1 approx 1.54e-15 Hz
        self.f_hz = k_output * 1.546e-15
        self.omega_gw = omega_gw

        idx_peak = np.argmax(self.omega_gw)
        print(f'Omega_GW_peak = {self.omega_gw[idx_peak]}')
        print(f'f_peak = {self.f_hz[idx_peak]} Hz')

        return self.f_hz, self.omega_gw
    
    def save_gw(self, filename='gw_data.dat', path = "."):

        """
        Parameters
        ----------
        filename : str, optional
            Output filename. Default is 'gw_data.dat'.
        path : str, optional
            Directory where the file will be saved. Default is current directory.
        """

        if self.f_hz is None:
            raise RuntimeError('f_hz not computed yet. Call .compute() first')
        
        if self.omega_gw is None:
            raise RuntimeError('Omega_gw not computed yet, Call .compute() first')
        
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        
        header = 'f_hz Omega_gw'
        data = np.column_stack([self.f_hz, self.omega_gw])
        np.savetxt(full_path, data, header=header, comments='#')
        print(f'Saved to {full_path}')
        
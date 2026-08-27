import os
import numpy as np
from scipy.interpolate import interp1d
from typing import Optional, Tuple, Union
from joblib import Parallel, delayed
from tqdm import tqdm

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
        Path to a file containing [k, Ps] columns.
    k_modes : np.ndarray, optional
        Comoving wavenumber array in Mpc^-1.
    Ps : np.ndarray, optional
        Primordial scalar power spectrum array.
    k_col : int, optional
        Column index for k in the file. Default is 0.
    Ps_col : int, optional
        Column index for Ps in the file. Default is 1.
    delimiter : str, optional
        File delimiter. Default is None (whitespace).
    logk : bool, optional
        If True, the k column in the file is assumed to be in log10 scale. Default is False.
    """

    def __init__(
        self,
        perturbations: Optional[Perturbations] = None,
        filename: Optional[str] = None,
        k_modes: Optional[Union[np.ndarray, list]] = None,
        Ps: Optional[Union[np.ndarray, list]] = None,
        k_col: int = 0,
        Ps_col: int = 1,
        delimiter: Optional[str] = None,
        logk: bool = False
    ):
        
        self.f_hz: Optional[np.ndarray] = None
        self.omega_gw: Optional[np.ndarray] = None

        if perturbations is not None:
            self.pert = perturbations
            self.k_modes = perturbations.k_modes
            self.Ps = perturbations._P_s_array

        elif filename is not None:
            self.k_modes, self.Ps = self._load_from_file(filename, k_col, Ps_col, delimiter, logk)
            self.pert = None

        elif k_modes is not None and Ps is not None:
            self.k_modes = np.asarray(k_modes, dtype=float)
            self.Ps = np.asarray(Ps, dtype=float)
            self.pert = None

        else:
            raise ValueError("You must provide either `perturbations`, `filename`, or (`k_modes`, `Ps`).")

        self._validate_input()
        self._build_interpolator()
    
    def _load_from_file(self, filename: str, k_col: int, Ps_col: int, delimiter: Optional[str], logk: bool) -> Tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        data = np.loadtxt(filename, delimiter=delimiter)
        k = data[:, k_col]
        Ps = data[:, Ps_col]

        if logk:
            k = 10.0 ** k
        return k, Ps
    
    def _validate_input(self) -> None:
        if len(self.k_modes) != len(self.Ps):
            raise ValueError("k_modes and Ps must have the same length.")
        if np.any(self.k_modes <= 0):
            raise ValueError("k_modes must be strictly positive.")
        if not np.all(np.diff(self.k_modes) > 0):
            raise ValueError("k_modes must be strictly increasing.")
        
    def _build_interpolator(self) -> None:
        mask = self.Ps > 0
        k_safe = self.k_modes[mask]
        Ps_safe = self.Ps[mask]

        if np.sum(mask) < 2:
            raise ValueError("Not enough positive Ps values to build interpolator.")

        self.Ps_interp = interp1d(
            np.log(k_safe),
            np.log(Ps_safe),
            kind="linear",
            bounds_error=False,
            fill_value=-np.inf
        )

    def _kernel_averaged(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Computes the radiation domination kernel."""
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

    def compute(self, k_output: Optional[np.ndarray] = None, n_int: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrates the SIGW spectrum.
        """
        if k_output is None:
            k_output = self.k_modes

        Omega_r0_h2 = 4.18e-5 
        prefactor_const = 0.83 * (106.75 / 10.75)**(-1.0/3.0) * Omega_r0_h2 * (1.0 / 6.0)

        u_vals = np.logspace(-2, 2, n_int)
        v_vals = np.logspace(-2, 2, n_int)
        U, V = np.meshgrid(u_vals, v_vals, indexing='ij')
        
        geom_factor = (4.0 * V**2 - (1.0 - U**2 + V**2)**2)**2 / (16.0 * U**2 * V**2)
        mask_triangle = (np.abs(U - V) < 1.0) & ((U + V) > 1.0)
        
        Kernel_RD = self._kernel_averaged(U, V)
        Integrand_Base = geom_factor * Kernel_RD * mask_triangle
        
        dln_u = np.log(u_vals[1] / u_vals[0])
        dln_v = np.log(v_vals[1] / v_vals[0])
        diff_area = (U * V) * dln_u * dln_v

        print(f"Computing SIGW for {len(k_output)} modes...")

        # Parallelized Integration
        integral_vals = Parallel(n_jobs=-1)(
            delayed(_compute_single_k)(
                k_curr, U, V, Integrand_Base, diff_area, self.Ps_interp
            )
            for k_curr in tqdm(k_output, desc="Integrating SIGWs")
        )
        
        omega_gw = prefactor_const * np.array(integral_vals)

        # Conversion to contemporary frequencies
        self.f_hz = k_output * 1.546e-15
        self.omega_gw = omega_gw

        idx_peak = np.argmax(self.omega_gw)
        print(f'Omega_GW_peak = {self.omega_gw[idx_peak]:.4e}')
        print(f'f_peak = {self.f_hz[idx_peak]:.4e} Hz')

        return self.f_hz, self.omega_gw
    
    def save_gw(self, filename: str = 'gw_data.dat', path: str = ".") -> None:
        """Saves the computed SIGW data."""
        if self.f_hz is None or self.omega_gw is None:
            raise RuntimeError('SIGWs not computed yet. Call .compute() first.')
        
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        
        header = 'f_hz  Omega_gw'
        data = np.column_stack([self.f_hz, self.omega_gw])
        np.savetxt(full_path, data, header=header, comments='# ')
        print(f'Saved to {full_path}')

# =====================================================================
# EXTERNAL FUNCTIONS (Optimized for Multiprocessing)
# =====================================================================

def _compute_single_k(k_curr: float, U: np.ndarray, V: np.ndarray, Integrand_Base: np.ndarray, diff_area: np.ndarray, Ps_interp) -> float:
    """Helper function to isolate the integrand evaluation for multiprocessing."""
    def get_ps(q):
        ps = np.zeros_like(q)
        mask = q > 0
        ps[mask] = np.exp(Ps_interp(np.log(q[mask])))
        return ps
        
    P_u = get_ps(k_curr * U)
    P_v = get_ps(k_curr * V)
    
    Total_Integrand = Integrand_Base * P_u * P_v
    return float(np.sum(Total_Integrand * diff_area))
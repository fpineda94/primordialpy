import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from typing import Optional, Tuple, Union, Any

from primordialpy.perturbations import Perturbations


class PBHAbundance:
    """
    Calculates Primordial Black Hole (PBH) abundance from a generic inflationary model.

    This class computes the variance of density contrast, the mass fraction of PBHs 
    at formation (using the Press-Schechter formalism), and the current PBH abundance. 
    It can be initialized either from a Perturbations instance or directly from arrays 
    of k-modes and power spectrum values.

    Parameters
    ----------
    delta_c : float
        Critical density contrast for PBH formation.
    gamma : float
        Efficiency parameter for the PBH mass (fraction of horizon mass).
    gstar : float
        Effective number of relativistic degrees of freedom at formation.
    perturbations : Perturbations, optional
        Instance of the Perturbations class containing the power spectrum.
    k_modes : np.ndarray, optional
        Array of comoving wavenumbers in Mpc^-1. Required if perturbations is None.
    Ps : np.ndarray, optional
        Array of scalar power spectrum values. Required if perturbations is None.
    window : str, optional
        Type of window function to use. Options are 'gaussian' or 'top-hat'. 
        Default is 'gaussian'.

    Attributes
    ----------
    mpbh : np.ndarray or None
        Array of PBH masses in Solar masses.
    fPBH_arr : np.ndarray or None
        Array of PBH fractional abundances f_PBH.
    """

    def __init__(
        self,  
        delta_c: float, 
        gamma: float, 
        gstar: float, 
        perturbations: Optional[Perturbations] = None,
        k_modes: Optional[Union[np.ndarray, list]] = None,
        Ps: Optional[Union[np.ndarray, list]] = None,
        window: str = "gaussian"
    ):
        
        # Validate input configuration
        if perturbations is None and (k_modes is None or Ps is None):
            raise ValueError('Either perturbations or both k_modes and Ps must be provided.')
        
        if perturbations is not None and (k_modes is not None or Ps is not None):
            raise ValueError('Provide either perturbations or k_modes/Ps, not both.')
        
        if perturbations is not None:
            if perturbations._P_s_array is None:
                raise RuntimeError('Power spectrum not computed yet. Call .power_spectrum() first.')
            self.pert = perturbations        
            self.Ps = self.pert._P_s_array
            self.k_modes = self.pert.k_modes  
        else: 
            self.k_modes = np.asarray(k_modes, dtype=float)
            self.Ps = np.asarray(Ps, dtype=float)
            
        self.window = window   
        self.delta_c = delta_c
        self.gamma = gamma
        self.gstar = gstar  
        self.Msun = 1.0  

        # State attributes
        self.mpbh: Optional[np.ndarray] = None
        self.fPBH_arr: Optional[np.ndarray] = None
        
        self.Ps_interp = CubicSpline(self.k_modes, self.Ps, extrapolate=True)

        # Peak values
        idx_peak = np.argmax(self.Ps)
        self.P_peak = float(self.Ps[idx_peak])
        self.k_peak = float(self.k_modes[idx_peak])

    def _window_function(self, x: np.ndarray) -> np.ndarray:
        """Applies the vectorized window function."""
        if self.window == "gaussian":
            return np.exp(-x**2 / 2.0)
        
        elif self.window == "top-hat":
            W = np.ones_like(x)
            mask = x != 0
            xm = x[mask]
            W[mask] = 3.0 * (np.sin(xm) - xm * np.cos(xm)) / xm**3
            return W
        else:
            raise ValueError(f"Window function '{self.window}' not implemented.")

    def sigma_squared(self, k: np.ndarray) -> np.ndarray:
        """
        Computes the variance of the density contrast at different scales.

        Parameters
        ----------
        k : np.ndarray
            Array of comoving wavenumbers.

        Returns
        -------
        np.ndarray
            The variance sigma^2 evaluated at scales R = 1/k.
        """
        r = 1.0 / k 
        Pk = self.Ps_interp(k)  
        ln_k = np.log(k)

        R, K = np.meshgrid(r, k, indexing="ij")
        W = self._window_function(K * R)
        integrand = W**2 * (K * R)**4 * Pk

        integral = np.trapezoid(integrand, ln_k, axis=1)
        return (16.0 / 81.0) * integral

    def beta(self, k: np.ndarray) -> np.ndarray:
        """
        Computes the mass fraction of the universe collapsing into PBHs.
        """
        sigma_sq = self.sigma_squared(k)
        
        beta_val = (
            np.sqrt(2.0 / np.pi) 
            * (np.sqrt(sigma_sq) / self.delta_c)
            * np.exp(-self.delta_c**2 / (2.0 * sigma_sq))
        )
        return beta_val

    def Mpbh(self, k: np.ndarray) -> np.ndarray:
        """
        Computes the relation between PBH mass and horizon mass.

        Returns
        -------
        np.ndarray
            PBH masses in units of Solar masses.
        """
        return 3.68 * (self.gamma / 0.2) * (self.gstar / 10.75)**(-1.0 / 6.0) / (k / 1e6)**2

    def fPBH(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the current PBH abundance as a fraction of dark matter.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the PBH masses and their corresponding abundances.
        """
        k = self.k_modes
        mpbh = self.Mpbh(k)
        beta_val = self.beta(k)

        fPBH_val = (beta_val / 3.94e-9) * np.sqrt(self.gamma / 0.2) * (self.gstar / 10.75)**(-0.25) / np.sqrt(mpbh)

        idx_peak = np.argmax(fPBH_val)
        mpbh_peak = mpbh[idx_peak]

        print(f'fPBH_peak = {fPBH_val[idx_peak]:.4e}')
        print(f'MPBH_peak = {mpbh_peak:.4e} M_sun')

        self.mpbh = mpbh
        self.fPBH_arr = fPBH_val
        return mpbh, fPBH_val
    
    def save_pbh(self, filename: str = 'pbh_abundance.dat', path: str = ".") -> None:
        """
        Saves the computed PBH abundance data to a text file.

        Parameters
        ----------
        filename : str, optional
            Output filename. Default is 'pbh_abundance.dat'.
        path : str, optional
            Directory where the file will be saved. Default is current directory.
        """
        if self.mpbh is None or self.fPBH_arr is None:
            raise RuntimeError('PBH abundance not computed yet. Call .fPBH() first.')
        
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)

        header = "MPBH  fPBH"
        data = np.column_stack([self.mpbh, self.fPBH_arr])
        np.savetxt(full_path, data, header=header, comments='# ')
        print(f'Saved to {full_path}')

    def plot_abundance(self, ax: Any = None, save: bool = False, filename: str = 'fPBH.pdf', ylim_bottom: float = 1e-25, **kwargs: Any) -> Any:
        """
        Plots the PBH abundance f_PBH against mass M_PBH (Solar Masses).
        """
        if self.mpbh is None or self.fPBH_arr is None:
            m_pbh, f_pbh = self.fPBH()
        else:
            m_pbh, f_pbh = self.mpbh, self.fPBH_arr
        
        if ax is None:
            try:
                from .plot_style import style
                style()
            except ImportError:
                pass
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.loglog(m_pbh, f_pbh, **kwargs)

        ax.set_xlabel(r'$M_{\rm PBH} \, [M_\odot]$', fontsize=14)
        ax.set_ylabel(r'$f_{\rm PBH}(M)$', fontsize=14)
        
        ax.set_xlim([np.min(m_pbh), np.max(m_pbh)])
        ax.set_ylim(bottom=ylim_bottom) 

        lines = [line.get_label() for line in ax.get_lines()]
        if r'$f_{\rm PBH} = 1$' not in lines:
            ax.axhline(1, color='gray', linestyle='--', alpha=0.5, linewidth=1, label=r'$f_{\rm PBH} = 1$')

        if save:
            save_dir = 'Figures' 
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Abundance plot saved to {filepath}")

        return ax
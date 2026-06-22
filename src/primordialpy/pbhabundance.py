import numpy as np
from scipy.interpolate import interp1d
import os 
import matplotlib.pyplot as plt
from primordialpy.perturbations import Perturbations


class PBHAbundance:
    
    """
    Calculates PBH abundance from a generic inflationary model.

    Can be initialized either from a Perturbations instance or 
    directly from arrays of k modes and power spectrum values.
    """

    def __init__(
        self,  
        delta_c: float, 
        gamma: float, 
        gstar: float, 
        perturbations: Perturbations = None,
        k_modes: np.ndarray = None,
        Ps: np.ndarray = None,
        window: str = "gaussian"
        ):

        #Validate input
        if perturbations is None and (k_modes is None or Ps is None):
            raise ValueError ('Either perturbations or both k_modes and Ps must be provided')
        
        if perturbations is not None and (k_modes is not None or Ps is not None):
            raise ValueError('Provide either perturbations or k_modes/Ps, not both')
        
        if perturbations is not None:
            if perturbations._P_s_array is None:
                raise RuntimeError('Power spectrum not computed yet. Call .power_spectrum() first.')
            
            self.Ps = self.pert._P_s_array
            self.k_modes = self.pert.k_modes  # in Mpc^-1
        else: 
            self.k_modes = np.asarray(k_modes)
            self.Ps = np.asarray(Ps)
            
        

        self.window = window   
        self.delta_c = delta_c
        self.gamma = gamma
        self.gstar = gstar  
        self.Msun = 1.0  # work in solar mass units

        
        self.Ps_interp = interp1d(
            self.k_modes,
            self.Ps,
            kind="cubic",
            bounds_error=False,
            fill_value= 0.0,
        )

        # Peak values
        self.P_peak = self.Ps[np.argmax(self.Ps)]
        self.k_peak = self.k_modes[np.argmax(self.Ps)]

        os.makedirs("Data", exist_ok=True)

    # ---------------- Window functions ----------------

    def _window_function(self, x):
        """Vectorized window functions"""

        if self.window == "gaussian":
            return np.exp(-x**2 / 2)
        
        elif self.window == "top-hat":
            W = np.ones_like(x)
            mask = x != 0
            xm = x[mask]
            W[mask] = 3 * (np.sin(xm) - xm * np.cos(xm)) / xm**3
            return W
        else:
            raise ValueError(f"Window function {self.window} not implemented")

    # ---------------- Variance ----------------

    def sigma_squared(self, k):

        """
        Compute variance of density contrast at different scales (vectorized).
        """
        r = 1.0 / k  # horizon scale
        Pk = self.Ps_interp(k)  # interpolated spectrum
        ln_k = np.log(k)

        R, K = np.meshgrid(r, k, indexing="ij")

        W = self._window_function(K * R)
        integrand = W**2 * (K * R) ** 4 * Pk

        integral = np.trapz(integrand, ln_k, axis=1)

        return (16.0 / 81.0) * integral

    # ---------------- Beta function ----------------

    def beta(self, k):
        sigma = self.sigma_squared(k)
        beta = (
            np.sqrt(2/np.pi)*(sigma**0.5 /(self.delta_c))
            * np.exp(-self.delta_c**2 / (2 * sigma))
        )
        return beta

    # ---------------- Mass function ----------------

    def Mpbh(self, k):
        """
        Relation between PBH mass and horizon mass in solar masses.
        """
        kCMB = 0.05  # Mpc^-1
        M = 3.68*(self.gamma/0.2)*(self.gstar/10.75)**(-1/6)/(k/1e6)**2
        return M

    # ---------------- PBH abundance ----------------

    def fPBH(self, save = False, filename = None):

        k = self.k_modes
        mpbh = self.Mpbh(k)  # solar masses
        beta = self.beta(k)

        fPBH = (beta/3.94e-9)*np.sqrt(self.gamma/0.2)*(self.gstar/10.75)**(-0.25)/(np.sqrt(mpbh))

        idx_peak = np.argmax(fPBH)
        mpbh_peak = mpbh[idx_peak]


        print(f'fPBH_peak = {fPBH[idx_peak]}')
        print(fr'MPBH_peak = {mpbh_peak} M⊙')

        return mpbh, fPBH

# ---------------- Plotting ----------------

    def plot_abundance(self, ax=None, save=False, filename='fPBH.pdf', ylim_bottom=1e-25, **kwargs):
        """
        Plots the PBH abundance f_PBH against mass M_PBH (Solar Masses).
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on. If None, a new figure is created.
        save : bool
            If True, saves the figure to the 'Figures' folder.
        filename : str
            Name of the output file.
        ylim_bottom : float
            Minimum value for the y-axis to avoid plotting numerical noise.
        **kwargs :
            Arguments passed to ax.loglog (color, linestyle, label, etc.).
        """
        
        m_pbh, f_pbh = self.fPBH(save=False)
        
        if ax is None:
            try:
                from .plot_style import style
                style()
            except ImportError:
                pass
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.loglog(m_pbh, f_pbh, **kwargs)

        ax.set_xlabel(r'$M_{\rm PBH} [M_\odot]$', fontsize=14)
        ax.set_ylabel(r'$f_{\rm PBH} (M)$', fontsize=14)
        

        ax.set_xlim([np.min(m_pbh), np.max(m_pbh)])
        ax.set_ylim(bottom=ylim_bottom) 

        lines = [l.get_label() for l in ax.get_lines()]
        if r'$f_{\rm PBH} = 1$' not in lines:
            ax.axhline(1, color='gray', linestyle='--', alpha=0.5, linewidth=1, label=r'$f_{\rm PBH} = 1$')

        if save:
            import os
            save_dir = 'Figures' 
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', dpi=300)
            print(f"Abundance plot saved to {filepath}")

        return ax
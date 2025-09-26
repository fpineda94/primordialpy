import numpy as np
from scipy.interpolate import interp1d
import os 

from primordialpy.perturbations import Perturbations


class PBHAbundance:
    """
    Calculates PBH abundance from a generic inflationary model.
    """

    def __init__(self, perturbations: Perturbations, delta_c, gamma, gstar, window: str = "gaussian"):
        
        self.window = window
        self.pert = perturbations

        # Power spectrum
        self.Ps = self.pert._P_s_array
        self.k_modes = self.pert.k_modes  # in Mpc^-1
        self.Ps_interp = interp1d(
            self.k_modes,
            self.Ps,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )

        # Peak values
        self.P_peak = self.Ps[np.argmax(self.Ps)]
        self.k_peak = self.k_modes[np.argmax(self.Ps)]

        # PBH parameters
        self.delta_c = delta_c
        self.gamma = gamma
        self.gstar = gstar

        # Units
        self.Msun = 1.0  # work in solar mass units

        # Folder creation
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

        # Crear grillas 2D: r[i], k[j]
        R, K = np.meshgrid(r, k, indexing="ij")

        # Integrando en toda la grilla
        W = self._window_function(K * R)
        integrand = W**2 * (K * R) ** 4 * Pk

        # Integramos sobre k (eje 1)
        integral = np.trapz(integrand, ln_k, axis=1)

        return (16.0 / 81.0) * integral

    # ---------------- Beta function ----------------

    def beta(self, k):
        sigma = self.sigma_squared(k)
        beta = (
            self.gamma
            * (sigma**0.5 /(np.sqrt(2*np.pi)*self.delta_c))
            * np.exp(-self.delta_c**2 / (2 * sigma))
        )
        return beta

    # ---------------- Mass function ----------------

    def Mpbh(self, k):
        """
        Relation between PBH mass and horizon mass in solar masses.
        """
        kCMB = 0.05  # Mpc^-1
        M = (
            1.13e15
            * (self.gamma / 0.2)
            * (self.gstar / 106.75) ** (-1/6)
            * (kCMB / k) ** 2
        )  # solar masses
        return M

    # ---------------- PBH abundance ----------------

    def fPBH(self, save = False, filename = None):

        k = self.k_modes
        mpbh = self.Mpbh(k)  # solar masses
        beta = self.beta(k)

        fPBH = (
            1.68e8*(self.gamma / 0.2) ** (0.5)
            * (self.gstar / 106.75) ** (-1/4)
            * (mpbh / self.Msun) ** (-0.5)
            * beta
        )

        idx_peak = np.argmax(fPBH)
        mpbh_peak = mpbh[idx_peak]

        fPBH_tot = np.trapz(fPBH, np.log(mpbh))

        print(f'fPBH_peak = {fPBH[idx_peak]}')
        print(fr'MPBH_peak = {mpbh_peak} M⊙')
        print(f'Total fPBH abundance: {fPBH_tot}')


        if save:
            if filename is None:
                filename = 'abundance_PBHs_data.txt'
            filepath = os.path.join('Data', filename)
            header = (f'#Parameters: delta_c ={self.delta_c}, gamma = {self.gamma}, gstar = {self.gstar}'
                      f'#Columns: mPBH [Msun, fPBH]')
            np.savetxt(filepath, 
                       np.column_stack([mpbh, fPBH]),
                       header = header,
                       fmt = '%.16e'
                       )
        return mpbh, fPBH, fPBH_tot
    


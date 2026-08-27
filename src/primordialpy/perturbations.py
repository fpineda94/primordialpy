import numpy as np 
from scipy.integrate import solve_ivp
from scipy.optimize import brentq 
from joblib import Parallel, delayed
from tqdm import tqdm
import os

from primordialpy.background import Background
from primordialpy.model import Potential



class Perturbations: 

    """
    Computes the evolution of scalar and tensor primordial perturbations during inflation.

    This class numerically solves the Mukhanov-Sasaki equation for curvature perturbations `R_k`
    and tensor modes `h_k`, using Bunch-Davies initial conditions and a given inflationary background.

    Supports computation of the scalar and tensor power spectra, spectral tilts, and plotting tools
    for visualizing inflationary observables across a range of scales (e.g., CMB, PBHs).

    Parameters
    ----------
    potential : Potential
        Instance of the inflationary potential class.
    background : Background
        Instance of the background class with precomputed inflationary dynamics.
    scale : str
        Type of scale to analyze: 'CMB' or 'PBH'.
    N_CMB : float
        Number of e-folds before the end of inflation at which the pivot scale (k_CMB) exits the horizon.
    k_CMB : float, optional
        Pivot scale value in Mpc⁻¹ (default is 0.05 Mpc⁻¹).
    N_range : float, optional
        Range of e-folds around N_CMB to define the relevant k window (default is 7).
    N_inside : float, optional
        Number of e-folds before horizon exit used to initialize perturbation evolution (default is 5).

    Attributes
    ----------
    solution : OdeResult or ndarray
        Result of the integration for the pivot scale.
    k_modes : ndarray
        Array of comoving modes in Mpc⁻¹ corresponding to the specified scale.
    _P_s_array : ndarray
        Scalar power spectrum computed for k_modes (after calling `Power_spectrum()`).
    _P_t_array : ndarray
        Tensor power spectrum computed for k_modes (after calling `Power_spectrum()`).

    Methods
    -------
    solver()
        Solves the perturbation equations for the pivot scale.
    Power_spectrum()
        Computes scalar and tensor power spectra over `k_modes`.
    _Compute_Power_spectrum(k)
        Computes P_s and P_t for a given wavenumber `k`.
    Spectral_tilts
        Computes spectral indices n_s and n_t at the pivot scale.
    Plot_spectrum(dpi, spectrum, save=False, filename=None)
        Plots scalar or tensor power spectrum.
    Plot_r(dpi, save=False, filename='tensor_to_scalar_ratio.png')
        Plots tensor-to-scalar ratio r(k).
    """

    def __init__(self,
                 potential : Potential, 
                 background: Background, 
                 scale: str, 
                 N_CMB: float,  
                 k_CMB: float = 0.05, 
                N_inside: float = 5):
        
        # Basic configuration
        self.potential = potential     
        self.background = background
        self.scale = scale
        self.solution = None
        self._data_interpolated()

        # Efolds configuration
        self.N_CMB = N_CMB 
        self.N_inside = N_inside 
        self.Nend = self.background.data()['N'][-1]
        self.Nhc = self.Nend - self.N_CMB
     
        # Configuration of k modes
        self.k_CMB = k_CMB 
        self.k_pivot = self.aH(self.Nhc) 
        self.norma = self.k_CMB/self.k_pivot    

        self._ai_cached = self.k_CMB / (np.exp(self.Nhc) * self.H(self.Nhc))

        if hasattr(self, 'scale') and self.scale == 'CMB':
                self.k_min, self.k_max = self.norma*self.aH(self.Nhc - 7), self.norma*self.aH(self.Nhc + 7)
        elif hasattr(self, 'scale') and self.scale == 'PBH':
                self.k_min, self.k_max = self.norma*self.aH(self.Nhc - 7), 1e20
      
        self.k_modes = np.logspace(np.log10(self.k_min), np.log10(self.k_max), num = 1000)

    def _data_interpolated(self, vars = None, x = 'N'):
        if vars is None:
            vars =  ['phi', 'dphidN', 'H', 'a', 'aH', 'eps_H', 'eta_H']
        bg_interp = self.background.interpolation(x)
        for i in vars:
            if i not in bg_interp:
                raise ValueError(f'The variable {i} is not available')
            setattr(self, i, bg_interp[i])
    
    @property
    def _ai(self):
        return self._ai_cached
    
    def _z(self, a, dphidN):
        return a*dphidN
    

    def _odes(self, N, Y, k):

        return _rhs(N, Y, k, self._ai_cached, self.H, self.potential.evaluate,
                    self.potential.first_derivative)


    def N_hc(self, k=None, include_invalid=True):
        def func_to_root(N_val, k_val):
            return k_val - self.norma*self.aH(N_val)

        if k is not None:
            try:
                N_val = brentq(lambda N: func_to_root(N, k), 0, self.Nend)
                return (N_val, k)
            except ValueError as e:
                print(f"Warning: Could not find horizon crossing for k={k}. Error: {e}")
                return (np.nan, k) if include_invalid else None
        else:
            results = []
            for k_val in self.k_modes:
                try:
                    N_val = brentq(lambda N: func_to_root(N, k_val), 0, self.Nend)
                    results.append((N_val, k_val))
                except ValueError as e:
                    print(f"Warning: Could not find horizon crossing for k={k_val}. Error: {e}")
                    if include_invalid:
                        results.append((np.nan, k_val))
            return results
        

            
    def N_ini(self, k=None):
        if k is not None:
            n_hc = self.N_hc(k)[0] 
            return n_hc - self.N_inside if not np.isnan(n_hc) else np.nan
        else:
            return [N_hc - self.N_inside if not np.isnan(N_hc) else np.nan for N_hc, _ in self.N_hc()]
        


    def N_shs(self, k =None):
        if k is not None:
            n_hc = self.N_hc(k)[0]
            return n_hc + 5 if not np.isnan(n_hc) else np.nan
        else:
            return [N_hc + 5 if not np.isnan(N_hc) else np.nan for N_hc, _ in self.N_hc()]
        

    def initial_conditions(self, k, N_hc_val=None):
        if N_hc_val is None:
            N_hc_val = self.N_hc(k)[0]
        
        if np.isnan(N_hc_val):
            return [np.nan] * 10

        N0 = N_hc_val - self.N_inside  
        a0 = self._ai*np.exp(N0)

        phi0 = self.phi(N0)
        dphidN0 = self.dphidN(N0)
        H0 = self.H(N0)
        Y0 = [phi0, dphidN0]
        _, d2phidN20 = self.background._odes(N0, Y0)
        z0 = self._z(a0, dphidN0)

        Rk_re_ic = (1/(np.sqrt(2*k)))/z0
        Rk_im_ic = 0
        Rk_re_N_ic = -Rk_re_ic*((d2phidN20/dphidN0) + 1)
        Rk_im_N_ic = - np.sqrt(k/2)/(a0*H0*z0)

        hk_re_ic = (1/(np.sqrt(2*k)))/a0
        hk_im_ic = 0
        hk_re_N_ic = -hk_re_ic
        hk_im_N_ic = -np.sqrt(k/2)/(a0**2*H0)  
        
        return [phi0, dphidN0, Rk_re_ic, Rk_re_N_ic, Rk_im_ic, Rk_im_N_ic, hk_re_ic, hk_re_N_ic, hk_im_ic, hk_im_N_ic]
    


    def _solver(self, k = None):
        if k is None:
            k = self.k_CMB
    
        Y0 = self.initial_conditions(k)
        N_ini = self.N_ini(k)
        Nshs = self.N_shs(k)
        # Cap consistente con solver_k_mode: nunca integrar más allá de Nend.
        N_end = min(Nshs, self.Nend)
        N_span = [N_ini, N_end]
        N_eval = np.linspace(N_ini, N_end, 10000)

        rtol = _rtol_for_scale(self.scale)
        atol_vec = _tolerances(Y0, rtol=rtol)

        self.solution = solve_ivp(lambda N, Y: self._odes(N, Y, k),  
                        N_span, 
                        Y0, 
                        t_eval= N_eval, 
                        method ='LSODA',
                        rtol = rtol, 
                        atol = atol_vec, 
                        dense_output= True)   
        return self.solution
    

    
    def power_spectra_pivot(self, k= None):
        if k is None:
            k = self.k_CMB
        
        sol = self._solver(k)

        R_re, R_im = sol.y[2], sol.y[4]
        h_re, h_im = sol.y[6], sol.y[8]

        P_s = k**3*(R_re[-1]**2 + R_im[-1]**2)/(2*np.pi**2)
        P_t = 8*k**3*(h_re[-1]**2 + h_im[-1]**2)/(2*np.pi**2)
        r = P_t/P_s

        print(f'Curvature power spectrum at pivot scale is {P_s}')
        print(f'Tensor to scalar ratio at pivot scale is {r}')

        return P_s, P_t, r
    


    def power_spectrum(self):
        """ 
        Calcula el espectro usando multiprocesamiento eficiente aislando la clase principal.
        """
        N_hc_list = self.N_hc()           
        self._N_hc_cache = N_hc_list      
        N_hc_vals = [N for N, _ in N_hc_list]   

       
        Y0_list = [self.initial_conditions(k, N_hc_val=N_val) for k, N_val in zip(self.k_modes, N_hc_vals)]

        H_func = self.H
        V_func = self.potential.evaluate
        dV_func = self.potential.first_derivative
        ai_cached = self._ai_cached
        scale = self.scale
        N_inside = self.N_inside
        Nend = self.Nend

        results = Parallel(n_jobs=-1)(
            delayed(solver_k_mode)(
                k, N_val, N_inside, Nend, Y0, scale, ai_cached, H_func, V_func, dV_func
            )
            for k, N_val, Y0 in tqdm(zip(self.k_modes, N_hc_vals, Y0_list),
                                     total=len(self.k_modes),
                                     desc="Computing P(k)")
        )

        PS = np.array([r[0] for r in results])
        PT = np.array([r[1] for r in results])

        self._P_s_array = PS
        self._P_t_array = PT


        if np.all(np.isnan(PS)):
            self.Ps_peak = np.nan
            self.k_peak = np.nan
            print("Warning: todos los valores de P_s son NaN, no se pudo determinar el pico.")
        else:
                    i_peak = np.nanargmax(PS)
                    self.Ps_peak = PS[i_peak]
                    self.k_peak = self.k_modes[i_peak]

                    ps_base, ps_exp = f"{self.Ps_peak:.4e}".split('e')
                    k_base, k_exp = f"{self.k_peak:.4e}".split('e')

                    print(fr'$Ps_peak = {ps_base}\times 10^{{{ps_exp}}}$')
                    print(fr'$k_peak = {k_base}\times 10^{{{k_exp}}}\,\mathrm{{Mpc}}^{-1}$')
    
        return PS, PT 
    


    @property
    def spectral_tilts(self):
        from scipy.interpolate import interp1d

        if not hasattr(self, '_P_s_array') or not hasattr(self, '_P_t_array'):
            raise RuntimeError("First you must run the Power_spectrum method to calculate the spectra.")

        k = self.k_modes
        P_s = self._P_s_array
        P_t = self._P_t_array
        k_pivot = self.k_CMB

        log_k = np.log(k)
        dlogPs = np.gradient(np.log(P_s), log_k)
        dlogPt = np.gradient(np.log(P_t), log_k)

        n_s_interp = interp1d(k, 1 + dlogPs, kind='cubic', bounds_error = False, fill_value="extrapolate")
        n_t_interp = interp1d(k, dlogPt, kind='cubic', bounds_error = False, fill_value="extrapolate")

        n_s_pivot = float(n_s_interp(k_pivot))
        n_t_pivot = float(n_t_interp(k_pivot))

        return {'n_s': n_s_pivot, 'n_t': n_t_pivot}
    


    def save_power_spectra(self, filename: str = 'power_spectra.dat', path: str = '.'):
        if self._P_s_array is None:
            raise RuntimeError('Power spectra not computed yet. Call .power_spectrum() first.')
        
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        
        header = "k  Ps  Pt"
        data = np.column_stack([self.k_modes, self._P_s_array, self._P_t_array])
        np.savetxt(full_path, data, header=header, comments='# ')
        print(f'Saved to {full_path}')



# =====================================================================
# EXTERNAL FUNCTIONS (Optimized for Multiprocessing)
# =====================================================================


def _rtol_for_scale(scale, rtol_cmb=1e-8, rtol_pbh=1e-10):

    if scale == 'CMB':
        return rtol_cmb
    elif scale == 'PBH':
        return rtol_pbh
    return rtol_cmb


def _tolerances(Y0, rtol=1e-8, floor=1e-300):

    Y0 = np.asarray(Y0, dtype=float)
    scale = np.abs(Y0).copy()

    scale[0] = max(scale[0], 1.0)
    scale[1] = max(scale[1], 1.0)

    r_scale = max(scale[2], scale[4])
    scale[2] = scale[4] = r_scale

    rn_scale = max(scale[3], scale[5])
    scale[3] = scale[5] = rn_scale

    h_scale = max(scale[6], scale[8])
    scale[6] = scale[8] = h_scale

    hn_scale = max(scale[7], scale[9])
    scale[7] = scale[9] = hn_scale

    return np.maximum(rtol * scale, floor)


def _rhs(N, Y, k, ai_cached, H_func, V_func, dV_func):

    [phi, dphidN, Rk_re, Rk_re_N, Rk_im, Rk_im_N, hk_re, hk_re_N, hk_im, hk_im_N] = Y

    V = V_func(phi)
    dVdphi = dV_func(phi)
    d2phidN2 = -(3 - 0.5*(dphidN**2))*dphidN - (6 - (dphidN**2))*dVdphi/(2*V)

    a = ai_cached * np.exp(N)
    H = np.sqrt(V / (3 - 0.5*(dphidN**2)))

    z = a * dphidN
    z_N = a * (dphidN + d2phidN2)

    k_aH_sq = (k / (a * H))**2
    term_s = 1 - 0.5*(dphidN**2) + 2*(z_N/z)
    term_t = 3 - 0.5*(dphidN**2)

    with np.errstate(over='ignore', invalid='ignore'):
        Rk_re_NN = -term_s * Rk_re_N - k_aH_sq * Rk_re
        Rk_im_NN = -term_s * Rk_im_N - k_aH_sq * Rk_im
        hk_re_NN = -term_t * hk_re_N - k_aH_sq * hk_re
        hk_im_NN = -term_t * hk_im_N - k_aH_sq * hk_im

    if not (np.isfinite(Rk_re_NN) and np.isfinite(Rk_im_NN)
            and np.isfinite(hk_re_NN) and np.isfinite(hk_im_NN)):
        Rk_re_NN = Rk_im_NN = 0.0
        hk_re_NN = hk_im_NN = 0.0

    return [dphidN, d2phidN2, Rk_re_N, Rk_re_NN, Rk_im_N, Rk_im_NN, hk_re_N, hk_re_NN, hk_im_N, hk_im_NN]


def solver_k_mode(k, N_hc_val, N_inside, Nend, Y0, scale, ai_cached, H_func, V_func, dV_func):
    
    """
    It solves the perturbation system for a single mode. 
    Being outside the class avoids the serialization overhead (pickling) of joblib.
    """

    if np.isnan(N_hc_val):
        return np.nan, np.nan

    N_ini_val = N_hc_val - N_inside

    if scale == 'PBH':
        N_end_val = Nend
    else:
        N_end_val = min(N_hc_val + 5.0, Nend)

    rtol = _rtol_for_scale(scale)
    atol_vec = _tolerances(Y0, rtol=rtol)

    sol = solve_ivp(
        lambda N, Y: _rhs(N, Y, k, ai_cached, H_func, V_func, dV_func),
        t_span=(N_ini_val, N_end_val),
        y0=Y0,
        method='LSODA',
        rtol=rtol,
        atol=atol_vec,
    )

    if not sol.success:
        return np.nan, np.nan

    Y_hc = sol.y[:, -1]          

    Rk_re, Rk_im = Y_hc[2], Y_hc[4]
    hk_re,  hk_im = Y_hc[6], Y_hc[8]

    P_s = k**3 * (Rk_re**2 + Rk_im**2) / (2 * np.pi**2)
    P_t = 8 * k**3 * (hk_re**2 + hk_im**2) / (2 * np.pi**2)

    return P_s, P_t
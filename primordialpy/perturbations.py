import numpy as np 
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import brentq 
import matplotlib.pyplot as plt
import os

from primordialpy.background import Background
from primordialpy.model import Potential
from primordialpy.plot_style import style   




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

    def __init__(self, potential : Potential, background: Background, scale, N_CMB,  k_CMB = 0.05,  N_inside = 4):

        #Basic configuration
        self.potential = potential     
        self.background = background
        self.scale = scale
        self.solution = None
        self._data_interpolated()

        #Folder creation 
        os.makedirs('Data', exist_ok=True)
        os.makedirs('Figures', exist_ok=True)


        #Efolds configuration
        self.N_CMB = N_CMB 
        self.N_inside = N_inside 
        self.Nend = self.background.N_end 
        self.Nhc = self.Nend - self.N_CMB
  
        #Configuration of k modes
        self.k_CMB = k_CMB #CMB scale
        self.k_pivot = self.aH(self.Nhc) 
        self.norma = self.k_CMB/self.k_pivot    #Normalization factor to convert k modes in Mpc^-1

        if hasattr(self, 'scale') and self.scale == 'CMB':
                self.k_min, self.k_max = self.norma*self.aH(self.Nhc - 7), self.norma*self.aH(self.Nhc + 7)
        elif hasattr(self, 'scale') and self.scale == 'PBH':
                self.k_min, self.k_max = self.norma*self.aH(self.Nhc - 7), self.norma*self.aH(self.Nend - 4)
      
        self.k_modes = np.logspace(np.log10(self.k_min), np.log10(self.k_max), num = 1000)  #List modes in Mpc^-1



        # #Background data
        # bg_data = self.background.data
        # vars = ['N', 'phi', 'dphidN', 'H', 'a', 'aH', 'eps_H', 'eta_H']
        # self.N, self.phi, self.dphidN, self.H, self.a, self.aH, self.eps_H, self.eta_H = (bg_data[i] for i in vars)


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
        '''
        Next we need to fix the initial scale factor. 
        The initial scale factor is fixed by demanding a certain mode (pivot mode) leaves the Hubble scale at a particular time during the evolution.
        We will impose 0.05 $Mpc^{-1}$ mode leaves the Hubble radius  efolds_ before the end of inflation.
        '''
        return self.k_CMB/(np.exp(self.Nhc)*self.H(self.Nhc))
    

    def _z(self, a, dphidN):
        return a*dphidN
    


    def _ODEs(self, N, Y, k):

        '''
        System of equations including the background and perturbation equations
        primordial for $\mathchal{R}_k$ and tensor modes $h_k$. We separate real and imaginary parts for
        numerical stability. 
        '''

        [phi, dphidN, Rk_re, Rk_re_N, Rk_im, Rk_im_N, hk_re, hk_re_N, hk_im, hk_im_N] = Y
        
        #Background
        V = self.potential.evaluate(phi)
        dVdphi = self.potential.first_derivative(phi)
        d2phidN2 = -(3 - 0.5*(dphidN**2))*dphidN - (6 - (dphidN**2))*dVdphi/(2*V)

        #Perturbations
        a = self._ai*np.exp(N)
        H = np.sqrt(V/(3 - 0.5*dphidN))

        z = self._z(a, dphidN)
        z_N = a*(dphidN + d2phidN2)

        #Scalar perturbations
        Rk_re_NN = - (1 - 0.5*(dphidN**2) + 2*(z_N/z))*Rk_re_N - ((k/(a*H))**2)*Rk_re
        Rk_im_NN = - (1 - 0.5*(dphidN**2) + 2*(z_N/z))*Rk_im_N - ((k/(a*H))**2)*Rk_im

        #Tensor perturbations
        hk_re_NN = - (3-(dphidN**2)*0.5)*hk_re_N-((k/(a*H))**2)*hk_re
        hk_im_NN = - (3-(dphidN**2)*0.5)*hk_im_N-((k/(a*H))**2)*hk_im


        return[dphidN, d2phidN2, Rk_re_N, Rk_re_NN, Rk_im_N, Rk_im_NN, hk_re_N, hk_re_NN, hk_im_N, hk_im_NN]

    

    def N_hc(self, k=None, include_invalid=True):
        '''
        Find the efold N at which the k-mode crosses the horizon.
        Returns (N_hc, k) for each mode.
        '''

        def func_to_root(N_val, k_val):
            return k_val - self.norma*self.aH(N_val)

        if k is not None:
            try:
                N_val = brentq(lambda N: func_to_root(N, k), 0, self.Nend)
                return (N_val, k)
            except ValueError as e:
                print(f"Warning: Could not find horizon crossing for k={k} in [0, {self.Nend}]. Error: {e}")
                return (np.nan, k) if include_invalid else None
        else:
            results = []
            for k_val in self.k_modes:
                try:
                    N_val = brentq(lambda N: func_to_root(N, k_val), 0, self.Nend)
                    results.append((N_val, k_val))
                except ValueError as e:
                    print(f"Warning: Could not find horizon crossing for k={k_val} in [0, {self.Nend}]. Error: {e}")
                    if include_invalid:
                        results.append((np.nan, k_val))
                    # Si no se incluyen inválidos, simplemente se omiten
            return results
            


    def N_ini(self, k=None):
        '''
        Find the efold N_ini for a given k mode, 5 efolds before horizon crossing.
        '''
        if k is not None:
            n_hc = self.N_hc(k)[0] 
            return n_hc - self.N_inside if not np.isnan(n_hc) else np.nan
        else:
            return [
                N_hc - self.N_inside if not np.isnan(N_hc) else np.nan
                for N_hc, _ in self.N_hc()
        ]


    def N_shs(self, k =None):

        if k is not None:
            n_hc = self.N_hc(k)[0]
            return n_hc + 5 if not np.isnan(n_hc) else np.nan
        else:
            return [N_hc + 5 if not np.isnan(N_hc) else np.nan
                    for N_hc, _ in self.N_hc()]





    def Initial_conditions(self, k):

        '''
        Suitable initial conditions. We choose Bunch-Davies vacuum for scalar and tensor perturbations
        '''

        N0 = self.N_ini(k)  #E-folds at the beginning 5-folds before the horizon crossing 
        a0 = self._ai*np.exp(N0)

        #Initial condition for the background
        phi0 = self.phi(N0)
        dphidN0 = self.dphidN(N0)
        H0 = self.H(N0)
        Y0 = [phi0, dphidN0, H0]
        _, d2phidN20, _ = self.background._EDOs(N0, Y0)
        z0 = self._z(a0, dphidN0)


        #Bunch-Davies vacuum for R perturbations
        Rk_re_ic = (1/(np.sqrt(2*k)))/z0
        Rk_im_ic = 0
        Rk_re_N_ic = -Rk_re_ic*((d2phidN20/dphidN0) + 1)
        Rk_im_N_ic = - np.sqrt(k/2)/(a0*H0*z0)

        #Initial conditions for tensor perturbations
        hk_re_ic = (1/(np.sqrt(2*k)))/a0
        hk_im_ic = 0
        hk_re_N_ic = -hk_re_ic
        hk_im_N_ic = -np.sqrt(k/2)/(a0**2*H0)  
        
        return [phi0, dphidN0, Rk_re_ic, Rk_re_N_ic, Rk_im_ic, Rk_im_N_ic, hk_re_ic, hk_re_N_ic, hk_im_ic, hk_im_N_ic]

    

    def solver(self):

        '''
        Solves the scalar perturbation equation for the pivot mode k = 0.05 Mpc^-1
        '''

        k = self.k_CMB
        Y0 = self.Initial_conditions(k)
        N_ini = self.N_ini(k)
        Nshs = self.N_shs(k)
        N_span = [N_ini, Nshs]
        N_eval = np.linspace(N_ini, Nshs, 10000)

        self.solution = solve_ivp(lambda N, Y: self._ODEs(N, Y, k),  
                        N_span, 
                        Y0, 
                        t_eval= N_eval, 
                        method ='Radau',
                        rtol = 1e-8, 
                        atol = 1e-12, 
                        dense_output= True)   
        return self.solution
    

    @property
    def data(self):
        
        '''
        Extract the data of the commuting curvature perturbation and its derivative as a function of the number of e-folds N
        and store them in a dictionary.
        '''

        if self.solution is None:
            raise ValueError('First you have to solve the system with solver method')
        k = self.k_CMB
        N = self.solution.t
        R_re = self.solution.y[2]
        dRdN_re = self.solution.y[3] 
        R_im = self.solution.y[4]
        dRdN_im = self.solution.y[5]
        h_re = self.solution.y[6]
        dhdN_re = self.solution.y[7]
        h_im = self.solution.y[8]
        dhdN_im = self.solution.y[9]

        #Power spectrum
        P_s = k**3*(R_re**2 + R_im**2)/(2*np.pi**2)
        P_t = 8*k**3*(h_re**2 + h_im**2)/(2*np.pi**2)
        

        #Primordial power spectrum and tensor to scalar ratio at pivot scale
        Y_hc = self.solution.sol(self.N_shs(k = k))

        Rk_re_hc = Y_hc[2]
        Rk_im_hc = Y_hc[4]
        h_re_hc = Y_hc[6]
        h_im_hc = Y_hc[8]


        P_s_pivot = k**3*(Rk_re_hc**2 + Rk_im_hc**2)/(2*np.pi**2)
        P_t_pivot = 8*k**3*(h_re_hc**2 + h_im_hc**2)/(2*np.pi**2)
        r_pivot = P_t_pivot/P_s_pivot

        

        return {'N': N, 'R_re' : R_re, 'dRdN_re': dRdN_re ,'R_im': R_im, 'dRdN_im': dRdN_im, 
                'h_re' : h_re, 'dhdN_re' : dhdN_re, 'h_im' : h_im, 'dhdN_im' : dhdN_im,'P_s': P_s, 
                    'P_t': P_t,  'P_s_pivot': P_s_pivot, 'P_t_pivot': P_t_pivot, 'r_pivot': r_pivot}
    

    def _Compute_Power_spectrum(self, k):
        Y0 = self.Initial_conditions(k)
        N_ini = self.N_ini(k)

        # For odeint we need the time as the first argument in the ODE        
        def ode_func(Y, N, k):
            return self._ODEs(N, Y, k)
        #We use an adaptative tolerance for the very small modes (k >> aH)
        tol = 1e-16 / k

        # Solve the system with odeint (LSODA optimised in FORTRAN)
        sol = odeint(
            ode_func,
            Y0,
            np.linspace(N_ini, self.Nend, 1000),  
            args=(k,),
            atol=tol,
            mxstep= 10000000
            )   
        
        Y_hc = sol[-1]
        Rk_re, Rk_im, hk_re, hk_im = Y_hc[2], Y_hc[4], Y_hc[6], Y_hc[8]
        
        P_s = k**3 * (Rk_re**2 + Rk_im**2) / (2 * np.pi**2)
        P_t = 8 * k**3 * (hk_re**2 + hk_im**2) / (2 * np.pi**2)
        
        return P_s, P_t


    def Power_spectrum(self, save=False, filename=None):

        PS = np.zeros_like(self.k_modes)
        PT = np.zeros_like(self.k_modes)

        for i, k in enumerate(self.k_modes):
            PS[i], PT[i] = self._Compute_Power_spectrum(k)

        self._P_s_array = PS
        self._P_t_array = PT

        if save:
            if filename is None:
                filename = 'power_spectrum_data.txt'

            filepath = os.path.join('Data', filename)
            np.savetxt(filepath,
                    np.column_stack([self.k_modes, PS, PT]),
                    header='k_modes P_scalar P_tensor',
                    fmt='%.16e')

        return PS, PT    

    @property
    def Spectral_tilts(self):
        
        '''
        Calculates the spectral indices n_s and n_t evaluated on the pivot scale k_pivot,
        using the spectrum already calculated with Power_spectrum().
        '''
        
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

        # Interpolation
        n_s_interp = interp1d(k, 1 + dlogPs, kind='cubic', bounds_error = False, fill_value="extrapolate")
        n_t_interp = interp1d(k, dlogPt, kind='cubic', bounds_error = False, fill_value="extrapolate")

        n_s_pivot = float(n_s_interp(k_pivot))
        n_t_pivot = float(n_t_interp(k_pivot))

        return {'n_s': n_s_pivot, 'n_t': n_t_pivot}

    
    
    def Plot_spectrum(self, dpi, spectrum, save=False, filename=None, show_efolds=True):
        """
        Plots the power spectrum with optional dual axis showing e-folds at horizon crossing.
        
        Parameters:
        -----------
        show_efolds : bool, optional
            If True, adds a secondary x-axis showing the number of e-folds at horizon crossing.
        """
        if not hasattr(self, '_P_s_array') or not hasattr(self, '_P_t_array'):
            raise ValueError('First you must run the Power_spectrum method to calculate the spectra.')
        
        style(dpi=dpi)
        
        # Create figure and primary axis
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        if spectrum == 'scalar':
            idx_peak = np.argmax(self._P_s_array)
            ax1.loglog(self.k_modes, self._P_s_array)

            k_peak = self.k_modes[idx_peak]
            k_peak_str = r"{:.2e}".format(k_peak).replace("e+0", "e").replace("e+","e").replace("e","\\times 10^{") + "}"
            label_kpeak = r"$k_\text{peak} = " + k_peak_str + r"\, \text{Mpc}^{-1}$"
            
            if hasattr(self, 'scale') and self.scale == 'PBH':


                ax1.axvline(k_peak, color='r', linestyle='dashed', 
                        linewidth=1, label=label_kpeak)
                ax1.axvspan(1e-4, 1e0, color='gray', alpha=0.2)
                y_min, y_max = ax1.get_ylim()
                y_text_pos = 10**(0.9 * np.log10(y_min) + 0.1 * np.log10(y_max))
                ax1.text(
                    x=np.sqrt(1e-3 * 1e-1),
                    y=y_text_pos,
                    s='PLANCK',
                    ha='center',
                    va='center',
                    rotation=0,
                    color='black'
                )
                
                k_peak = self.k_modes[idx_peak]
                N_peak, _ = self.N_hc(k=k_peak)
                k_peak_str = r"{:.2e}".format(k_peak).replace("e", r"\times 10^{") + "}"
                print(f'k_peak = {k_peak_str} Mpc^-1')
                print(f'N_peak = {N_peak}')
                print(f'P_s(k_peak) = {self._P_s_array[idx_peak]}')
                
            elif hasattr(self, 'scale') and self.scale == 'CMB':
                ax1.set_xlim(1e-4, 1e0)
                ax1.set_ylim(2e-9, 3e-9)
                
        elif spectrum == 'tensor':
            ax1.loglog(self.k_modes, self._P_t_array)
        else:
            raise ValueError('Spectrum must be scalar or tensor')
        
        # Set labels for primary axis
        ax1.set_xlabel(r'$k$ [Mpc$^{-1}$]')
        if spectrum == 'scalar':
            ax1.set_ylabel(r'$\mathcal{P}_\mathcal{R}(k)$')
        elif spectrum == 'tensor':
            ax1.set_ylabel(r'$\mathcal{P}_\mathcal{T}(k)$')


        # Add secondary axis for e-folds if requested
        if show_efolds:
            ax2 = ax1.twiny()
            
            # Calculate N_hc for all k modes (only valid ones)
            try:
                N_hc_results = self.N_hc()  # Get all horizon crossing e-folds
                
                # Filter out invalid results (NaN values)
                valid_results = [(N, k) for N, k in N_hc_results if not np.isnan(N)]
                
                if valid_results:
                    N_values = np.array([N for N, k in valid_results])
                    k_values = np.array([k for N, k in valid_results])
                    
                    # Create a smooth interpolation for the secondary axis
                    from scipy.interpolate import interp1d
                    
                    # Sort by N values for interpolation
                    sorted_indices = np.argsort(N_values)
                    N_sorted = N_values[sorted_indices]
                    k_sorted = k_values[sorted_indices]
                    
                    # Create interpolation function (N -> k)
                    N_to_k_interp = interp1d(N_sorted, k_sorted, kind='linear', 
                                        bounds_error=False, fill_value='extrapolate')
                    
                    # Define nice e-fold tick positions
                    N_min, N_max = N_sorted.min(), N_sorted.max()
                    N_range = N_max - N_min
                    
                    # Create approximately 5-8 ticks
                    if N_range > 50:
                        N_step = 10
                    elif N_range > 20:
                        N_step = 5
                    else:
                        N_step = max(1, int(N_range / 6))
                    
                    N_ticks = np.arange(int(np.ceil(N_min)), int(np.floor(N_max)) + 1, N_step)
                    
                    # Convert N ticks to k values
                    k_ticks = N_to_k_interp(N_ticks)
                    
                    # Filter ticks that are within the plot range
                    x_min, x_max = ax1.get_xlim()
                    valid_tick_mask = (k_ticks >= x_min) & (k_ticks <= x_max) & (~np.isnan(k_ticks))
                    N_ticks_valid = N_ticks[valid_tick_mask]
                    k_ticks_valid = k_ticks[valid_tick_mask]
                    
                    # Set up the secondary axis
                    ax2.set_xscale('log')
                    ax2.set_xlim(ax1.get_xlim())
                    
                    if len(k_ticks_valid) > 0:
                        ax2.set_xticks(k_ticks_valid)
                        ax2.set_xticklabels([f'{int(N)}' for N in N_ticks_valid])
                        ax2.set_xlabel('e-folds $N$', fontsize=12)
                    else:
                        print("Warning: No valid e-fold ticks found for the current k range")
                else:
                    print("Warning: No valid horizon crossing data found")
                    show_efolds = False
                    
            except Exception as e:
                print(f"Warning: Could not create e-folds axis. Error: {e}")
                show_efolds = False
        
        # # Set title
        # if title is None:
        #     if spectrum == 'scalar':
        #         title = r'Scalar power spectrum $\mathcal{P}_\mathcal{R}(k)$'
        #     else:
        #         title = r'Tensor power spectrum $\mathcal{P}_T(k)$'
        
        # if show_efolds:
        #     # Adjust title position to account for secondary axis
        #     plt.title(title, pad=20)
        # else:
        #     plt.title(title)
        
        # Set filename
        if filename is None:
            filename = f'spectrum_{spectrum}.png'
        elif not filename.endswith('.png'):
            filename += '.png'
        
        ax1.legend(loc='best')
        plt.tight_layout()
        
        if save:
            filepath = os.path.join('Figures', filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Figure saved as: {filepath}")
        
        plt.show()

    
    def Plot_r(self, dpi, save = False, filename = 'tensor_to_scalar_ratio.png', title = None):   

        if not hasattr(self, '_P_s_array') or not hasattr(self, '_P_t_array'):
            raise ValueError('First you must run the Power_spectrum method to calculate the spectra.')
        
        P_S = self._P_s_array
        P_T = self._P_t_array
        r = P_T/P_S

        style(dpi = dpi)
        plt.semilogx(self.k_modes, r)
        plt.axvline(self.k_CMB, color = 'k', linestyle = 'dashed', linewidth= 0.8, label = r'$k_*$')
        plt.xlabel(r'$k$ [Mpc$^{-1}$]')
        plt.ylabel(r'$r(k)$')
        plt.legend()
        
        if title is None:
            plt.title(r'Tensor to scalar ratio $r$')
        
        plt.tight_layout()

        if save:            

            filepath = os.path.join('Figures', filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Figure saved as: {filepath}")
            
        plt.show()


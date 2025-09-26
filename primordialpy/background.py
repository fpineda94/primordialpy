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

    Upon instantiation, the background equations are solved automatically.

    Parameters
    ----------
    potential : Potential
        An instance of a class implementing the inflationary potential interface.
    phi0 : float
        Initial value of the inflaton field.
    N_in : float, optional
        Initial number of e-folds (default is 0).
    N_fin : float, optional
        Final number of e-folds to integrate up to (default is 80).

    Attributes
    ----------
    solution : OdeResult
        Solution of the ODE system from `scipy.integrate.solve_ivp`, includes phi(N), dphi/dN and H(N).
    k_CMB : float
        Pivot scale in Mpc⁻¹ used to identify horizon crossing (default is 0.05 Mpc⁻¹).

    Properties
    ----------
    data : dict
        Dictionary containing background quantities: N, phi, dphidN, H, a, aH, eps_H, eta_H.
    N_end : float
        Number of e-folds at the end of inflation, defined by eps_H = 1.
    Ne : ndarray
        Array of remaining e-folds before the end of inflation: Ne = N_end - N.
    N_CMB : float
        Number of e-folds N when the pivot scale crosses the horizon: k = a(N) * H(N).

    Methods
    -------
    interpolation(x='Ne')
        Returns interpolating functions for background quantities as functions of 'N' or 'Ne'.
    """
    

    def __init__(self, potential:Potential, phi0, N_in = 0, N_fin = 80):

        self.potential = potential
        self.N_in = N_in
        self.N_fin = N_fin
        self.phi0 = phi0
        self.solution = None
        self._solver()

        #Folder creation
        os.makedirs('Data', exist_ok=True)


    #The following functions are defined in terms of the number of e-folds N    

    def _H(self, phi, dphidN):
        V = self.potential.evaluate(phi)
        return np.sqrt(V /(3 - 0.5*dphidN**2))

    def _eps_H(self, dphidN):
        return 0.5*dphidN**2
    
    def _eta_H(self, de_HdN, dphidN):
        e_H = self._eps_H(dphidN)
        return e_H + 0.5*de_HdN/e_H    
    
    # def _eps_V(self, phi):
    #     V = self.potential.evaluate(phi)
    #     dVdphi = self.potential.first_derivative(phi)
    #     return 0.5*(dVdphi/V)**2
    
    # def _eta_V(self, phi):
    #     V = self.potential.evaluate(phi)
    #     d2Vdphi2 = self.potential.second_derivative(phi)
    #     return d2Vdphi2/V
    
    
    def _EDOs(self, N, Y):

        '''
        System of differential equations defining the dynamics of the inflationary model. 
        The result gives phi, dphidN and H as function of N
        '''

        [phi, dphidN, H] = Y

        epsilon = self._eps_H(dphidN)
        dVdphi = self.potential.first_derivative(phi)

        #Background 
        d2phidN2 = -(3 - epsilon)*dphidN - dVdphi/H**2
        dHdN = - 0.5*H*dphidN**2
        
        return[dphidN, d2phidN2, dHdN]
    
    
    
    def _InitialConditions(self, phi0):

        '''
        Initial system conditions. The user must set the initial condition of the inflaton, which depends on the model
        chosen.
        '''
        
        dVdphi = self.potential.first_derivative(phi0)
        V = self.potential.evaluate(phi0)
        dphidN_0 = - dVdphi / V
        H0 = self._H(phi0, dphidN_0)

        return [phi0, dphidN_0, H0]

    


    def _solver(self):

        """
        Solve the background given an interval of efolds and given the 
        initial conditions Y0. When the class is instantiated, the system is automatically solved.
        """

        Y0 = self._InitialConditions(self.phi0)
        N_span = [self.N_in, self.N_fin]
        N_eval = np.linspace(self.N_in, self.N_fin, 1000)

        self.solution = solve_ivp(
            self._EDOs, N_span,
            Y0,
            t_eval = N_eval,
            method = 'DOP853',
            rtol = 1e-6,
            atol = 1e-12,
            max_step = 0.01,
            dense_output = True)
    
    
    def data(self, save = False, filename = None):

        '''
        It extracts the solution data from the ODE system and stores them in a dictionary. 
        Relevant derived quantities are also calculated, such as the Hubble comoving radius aH.
        '''

        if self.solution is None:
            raise ValueError('First youhave to solve the system with _solver method')
        
        N = self.solution.t
        phi, dphidN, H = self.solution.y
        a = np.exp(N)
        eps_H = self._eps_H(dphidN)   #First slow-roll parameter
        deps_HdN = np.gradient(eps_H, N)
        eta_H = self._eta_H(deps_HdN, dphidN) #Second slow-roll parameter
        aH = a*H    #Comoving Hubble radius

        if save:
            if filename is None:
                filename = 'background_data.txt'
            filepath = os.path.join('Data', filename)
            np.savetxt(filepath, 
                       np.column_stack([N, phi, dphidN, H, a, aH, eps_H, eta_H]),
                       header ='N phi dphidN H a aH eps_H eta_H',
                       fmt = '%.16e')

        return {'N': N, 'phi': phi, 'dphidN' : dphidN, 'H': H, 'a': a, 'aH': aH, 'eps_H' : eps_H, 'eta_H': eta_H}
    

    @property
    def N_end(self):

        '''
        Calculate the total number of efolds at the end of inflation, imposing the drive eps_H(t_end) = 1.
        '''
        
        if self.solution is None:
            raise ValueError('First you have to solve the system with the solver method of the Background class.')
        
        N = self.data()['N']
        eps_H = self.data()['eps_H']    
        idx = np.argmax(eps_H >= 1)
        Nend = N[idx]
        return Nend


    @property
    def Ne(self):
        
        ''' 
        Calculate the number of efolds before the end of inflation, i.e. 
         Ne = Nend - N.
        '''
    
        Nend = self.N_end
        
        if Nend is None:
            raise ValueError('Inflation does not end in the given range')
        Ne = Nend - self.data()['N']
        return Ne


    def interpolation(self, x ='Ne'):

        coords = {'N': self.data()['N'], 'Ne': self.Ne}
        if x not in coords:
            raise ValueError("with_respect_to must be 'N' o 'Ne'")

        x_vals = coords[x]
        variables = ['phi', 'dphidN', 'H', 'a', 'aH', 'eps_H', 'eta_H']
        return {
            var: interp1d(x_vals, self.data()[var], kind='cubic', fill_value='extrapolate', bounds_error=False)
        for var in variables
        }

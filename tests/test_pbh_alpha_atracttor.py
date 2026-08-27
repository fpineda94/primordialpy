import pytest
import numpy as np
from primordialpy.model import PotentialFunction
from primordialpy.background import Background
from primordialpy.perturbations import Perturbations
from primordialpy.pbhabundance import PBHAbundance
from primordialpy.inducedGW import InducedGW


def test_pbh_alpha_attractor_inflation():
    """
    Integration test for an Alpha-Attractor potential with a localized bump.
    Validates background solving, power spectrum peak generation, and 
    the correct computation of PBH mass and abundance fractions.
    """
    
    # 1. Setup Potential
    V = 'V0*(tanh(phi/sqrt(6)))**2*(1 + A*(sech((phi - phi0)/sigma))**2)'
    parameters = {'V0': 1.448e-10, 'A': 2.044880e-3, 'phi0': 4.850001, 'sigma': 2.524999e-2}
    potential = PotentialFunction.function(V, param_values=parameters) 

    # 2. Background Dynamics
    bg = Background(potential, phi0=6.3) 
    bg.solver()
    
    data = bg.data()
    assert data['eps_H'][-1] >= 0.97, 'Inflation did not finish correctly'

    # 3. Perturbations & Power Spectrum
    pert = Perturbations(potential, bg, scale='PBH', N_CMB=60)
    
    # Unpack the tuple properly
    Ps, Pt = pert.power_spectrum()
    _, _, r = pert.power_spectra_pivot()
    ns = pert.spectral_tilts['n_s']

    print(f"\nSimulated Perturbations -> n_s: {ns:.4f}, r: {r:.4f}")
    assert ns == pytest.approx(0.96031, rel=0.02)
    assert r == pytest.approx(0.00465, rel=0.2)

    # 4. PBH Abundance
    pbh = PBHAbundance(perturbations=pert, delta_c=0.4, gamma=0.2, gstar=107.5)
    mpbh, fpbh = pbh.fPBH()

    fpbh_peak = np.max(fpbh)
    mpbh_peak = mpbh[np.argmax(fpbh)]

    print(f"\nSimulated PBH -> f_PBH: {fpbh_peak:.4e}, Mass: {mpbh_peak:.4e} M_sun")

    assert fpbh_peak == pytest.approx(0.9550, rel=0.05)
    assert mpbh_peak == pytest.approx(1.66e-13, rel=0.05)

    # 5. SIGWs

    gw = InducedGW(perturbations=pert)  
    f, Omega = gw.compute(n_int= 1000)

    omega_peak = np.max(Omega)
    f_peak = f[np.argmax(Omega)]

    print(f"\nSimulated SIGWs -> Omega_GW: {omega_peak:.4e}, f_peak: {f_peak:.4e} Hz")
    
    assert omega_peak == pytest.approx(2e-8, rel=0.05) 
    assert f_peak == pytest.approx(8e-3, rel=0.05)
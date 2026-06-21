import pytest
import numpy as np
from src.primordialpy.model import PotentialFunction
from src.primordialpy.background import Background
from src.primordialpy.perturbations import Perturbations
from src.primordialpy.pbhabundance import PBHAbundance


def test_pbh_alpha_atracttor_inflation():

    V = 'V0*(tanh(phi/sqrt(6)))**2*(1 + A*(sech((phi - phi0)/sigma))**2)'
    parameters = {'V0' : 1.448e-10, 'A' : 2.044880e-3, 'phi0' : 4.850001, 'sigma': 2.524999e-2}
    potential = PotentialFunction.from_string(V, param_values=parameters) 

    #Creando instancia de la clase Background. 
    background = Background(potential, phi0 = 6.3) 
    background.solver()

    data = background.data()

    assert data['eps_H'][-1] >= 0.97, 'inflation did not finish correctly'

    pert = Perturbations(potential, background, scale= 'PBH', N_CMB = 60)

    pert.Power_spectrum()
    ns = pert.Spectral_tilts['n_s']
    _, _, r = pert.Power_spectra_pivot()

    print(f"\nResultados Simulados -> n_s: {ns:.4f}, r: {r:.4f}")
    assert ns == pytest.approx(0.96031, rel = 0.02)
    assert r == pytest.approx(0.00465, rel = 0.2)

    phb = PBHAbundance(pert, delta_c=0.4, gamma=0.2, gstar=107.5)

    mpbh, fpbh = phb.fPBH()

    fpbh_peak = np.max(fpbh)
    mpbh_peak = mpbh[np.argmax(fpbh)]

    print(f"\nResultados PBH -> f_PBH: {fpbh_peak:.4e}, Masa: {mpbh_peak:.4f} M_sun")

    assert fpbh_peak == pytest.approx(0.9550, rel = 0.1)
    assert mpbh_peak == pytest.approx(1.66e-13, rel = 0.2)
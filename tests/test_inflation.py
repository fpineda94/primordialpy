import pytest
import numpy as np
from primordialpy.model import PotentialFunction
from primordialpy.background import Background
from primordialpy.perturbations import Perturbations

def test_quadratic_inflation():

    pot = PotentialFunction.from_string("0.5 * m**2 * phi**2", {'m': 5.9e-6})

    bg = Background(pot, phi0=17.5, N_in=0, N_fin=80)
    bg.solver()
    
    assert bg.data()['eps_H'][-1] >= 0.9, "La inflación no terminó correctamente"
    
    pert = Perturbations(pot, bg, scale = 'CMB', N_CMB=60.0)
    pert.Power_spectrum()
    
    observables = pert.Spectral_tilts
    ns = observables['n_s']
    r = pert.data['r_pivot'] 
    
    print(f"Calculado: n_s = {ns:.4f}, r = {r:.4f}")

    assert ns == pytest.approx(0.966, rel=1e-2)
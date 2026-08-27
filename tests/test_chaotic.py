import pytest
import numpy as np
from primordialpy.model import PotentialFunction
from primordialpy.background import Background
from primordialpy.perturbations import Perturbations

def test_quadratic_inflation():
  
    pot = PotentialFunction.function("0.5 * m**2 * phi**2", {'m': 6e-6})

    bg = Background(pot, phi0=16.5)
    bg.solver() 


    N, phi, dphidN, H, a, aH, eps_H, eta_H = bg.data().values()
    
    assert eps_H[-1] >= 0.97, "inflation did not finish correctly"
    
    pert = Perturbations(pot, bg, scale='CMB', N_CMB=60.0)
    _, _, r = pert.power_spectra_pivot()
    PS = pert.power_spectrum()
    ns = pert.spectral_tilts['n_s']
    

    print(f"\nSimulated Results -> n_s: {ns:.4f}, r: {r:.4f}")

    assert ns == pytest.approx(0.966, rel=0.02)
    assert r == pytest.approx(0.133, rel=0.15)
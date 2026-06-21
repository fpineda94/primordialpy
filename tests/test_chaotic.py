import pytest
import numpy as np
from src.primordialpy.model import PotentialFunction
from src.primordialpy.background import Background
from src.primordialpy.perturbations import Perturbations

def test_quadratic_inflation():
  
    pot = PotentialFunction.from_string("0.5 * m**2 * phi**2", {'m': 6e-6})

    bg = Background(pot, phi0=16.5, N_in=0, N_fin=75)
    bg.solver() 


    bg_data = bg.data()
    assert bg_data['eps_H'][-1] >= 0.95, "inflation did not finish correctly"
    
    pert = Perturbations(pot, bg, scale='CMB', N_CMB=60.0)
    
    pert.Power_spectrum()
    
    ns = pert.Spectral_tilts['n_s']
    _, _, r = pert.Power_spectra_pivot()

    print(f"\nResultados Simulados -> n_s: {ns:.4f}, r: {r:.4f}")

    assert ns == pytest.approx(0.966, rel=0.02)
    assert r == pytest.approx(0.133, rel=0.15)
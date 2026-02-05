import pytest
import numpy as np
from primordialpy.model import PotentialFunction
from primordialpy.background import Background
from primordialpy.perturbations import Perturbations

def test_starobinsky_inflation():

    pot = PotentialFunction.from_string('3*M**2/4*(1 - exp(-sqrt(2/3)*phi) )**2', {'M':1.3e-5})

    bg = Background(pot, phi0=5.7)
    bg.solver()

    data = bg.data()

    assert data['eps_H'][-1] >= 0.94, 'inflation did not finish correctly'

    pert = Perturbations(pot, bg, scale= 'CMB', N_CMB=54.37)

    pert.Power_spectrum()
    ns = pert.Spectral_tilts['n_s']
    _, _, r = pert.Power_spectra_pivot()

    print(f"\nResultados Simulados -> n_s: {ns:.4f}, r: {r:.4f}")
    assert ns == pytest.approx(0.965, rel = 0.02)
    assert r == pytest.approx(0.0036, rel = 0.02)


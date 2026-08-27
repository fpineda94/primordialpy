import pytest
from primordialpy.model import PotentialFunction
from primordialpy.background import Background
from primordialpy.perturbations import Perturbations

def test_starobinsky_inflation():

    pot = PotentialFunction.function('3*M**2/4*(1 - exp(-sqrt(2/3)*phi) )**2', {'M':1.3e-5})

    bg = Background(pot, phi0=5.7)
    bg.solver()

    N, phi, dphidN, H, a, aH, eps_H, eta_H = bg.data().values()


    assert eps_H[-1] >= 0.97, 'inflation did not finish correctly'

    pert = Perturbations(pot, bg, scale= 'CMB', N_CMB=54.37)
    _, _, r = pert.power_spectra_pivot()
    PS = pert.power_spectrum()
    ns = pert.spectral_tilts['n_s']

    print(f"\nSimulated results -> n_s: {ns:.4f}, r: {r:.4f}")
    assert ns == pytest.approx(0.965, rel = 0.02)
    assert r == pytest.approx(0.0036, rel = 0.02)


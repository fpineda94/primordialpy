import pytest
from primordialpy.model import PotentialFunction
from primordialpy.background import Background
from primordialpy.perturbations import Perturbations

def test_Higgs_inflation():

    pot = PotentialFunction.function('lambda_h*(1 - exp(-sqrt(2/3)*phi) )**2/(4*xi**2)', {'lambda_h':0.13, 'xi' : 17000})

    bg = Background(pot, phi0=5.7)
    bg.solver()

    N, phi, dphidN, H, a, aH, eps_H, eta_H = bg.data().values()

    assert eps_H[-1] >= 0.97, 'inflation did not finish correctly'

    pert = Perturbations(pot, bg, scale= 'CMB', N_CMB=57.6)
    _, _, r = pert.power_spectra_pivot()
    PS = pert.power_spectrum()
    ns = pert.spectral_tilts['n_s']

    print(f"\nStimulated Results -> n_s: {ns:.4f}, r: {r:.4f}")

    assert ns == pytest.approx(0.966, rel = 0.02)
    assert r == pytest.approx(0.0032, rel = 0.02)



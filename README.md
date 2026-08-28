# primordialpy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python library for computing inflationary dynamics, primordial perturbations, PBH abundance, and the signal of scalar-induced gravitational waves (SIGWs) from any single-field inflation model.

## Description

`primordialpy` allows users to analyze single-field inflationary models by writing the inflationary potential as a simple text string expression. The code automatically handles the following:

*   **Solve the background dynamics** using highly accurate numerical integrators.
*   **Compute scalar and tensor primordial perturbations** by solving the equation of comoving curvature perturbation $\mathcal{R}_k$.
*   **Determine and plot the primordial power spectrum**.
*   **Estimate the abundance of Primordial Black Holes (PBHs)** using the Press-Schechter formalism.
*   **Compute the signal of scalar-induced Gravitational Waves (SIGWs)**.

---

## Installation

### Requirements
*   Python >= 3.9
*   numpy >= 1.26.0
*   scipy >= 1.7.0
*   matplotlib >= 3.3.0
*   sympy >= 1.8.0
*   joblib >= 1.0.0
*   tqdm >= 4.60.0

### Installation from GitHub
It is recommended to install `primordialpy` inside a virtual environment to avoid dependency conflicts with other packages:

```bash
python -m venv primordialpy
source primordialpy/bin/activate  # On Windows: venv\Scripts\activate
```

Or if you prefer to use Conda:

```bash
conda create --name primordialpy 
conda activate primordiapy  
```

Then, install the latest development version directly from the repository:

```bash
git clone https://github.com/fpineda94/primordialpy.git
cd primordialpy
pip install -e .
```

---

## Basic Usage

```python
import matplotlib.pyplot as plt
from primordialpy.model import PotentialFunction
from primordialpy.background import Background
from primordialpy.perturbations import Perturbations

# 1. Define your potential as a string
# For example, chaotic inflation: V(phi) = (m^2 / 2) phi^2
V_str = '0.5 * m**2 * phi**2'
params = {'m': 5.9e-6}

# 2. Initialize the model
potential = PotentialFunction.function(V_str, param_values=params)

# 3. Solve Background dynamics
bg = Background(potential, phi0=17.5)
bg.solver()

# 4. Compute perturbations
pert = Perturbations(potential, bg, scale='CMB', N_CMB=60)
Ps, Pt = pert.power_spectrum()

print(f"Spectral tilts: {pert.spectral_tilts}")
```

---

## Main features

* **Intuitive Interface**: Define inflationary scenarios using text-based mathematical expressions; SymPy handles the analytical derivatives automatically.
* **Flexibility**: Compatible with canonical single-field inflation models.
* **Performance**: Heavily optimized ODE solvers and parallelized numerical integration for SIGWs.
* **Complete Pipeline**: Connects the universe's background dynamics all the way to contemporary observables (PBHs and GWs).

---

## Project Structure

```
primordialpy/
├── src/
│   └── primordialpy/
│       ├── background.py       # Background dynamics solver
│       ├── model.py            # Definition of potentials
│       ├── perturbations.py    # Perturbation and power spectrum calculation
│       ├── pbhabundance.py     # PBH formation and mass fraction
│       ├── pbhconstraints.py   # Observational constraints manager
│       └── inducedGW.py        # SIGW signal calculation
├── notebooks/                  # Jupyter notebooks with detailed tutorials
├── src/primordialpy/constraints_data/ # Observational restriction data
├── tests/                      # Tests for checking the code
├── pyproject.toml              # Modern Python package configuration
├── README.md
└── LICENSE
```

---

## Examples

Check out the `examples/` folder for Jupyter notebooks with detailed use cases, including:

* Standard inflationary models (e.g., Starobinsky, Chaotic).
* Potentials with features (inflection points, bumps) for PBH generation.
* Comparisons with observational data and constraints.

---

## Contributing

Contributions are welcome! If you find a bug or have suggestions:

1. Open an issue.
2. Fork the repository.
3. Create a branch for your feature (`git checkout -b feature/new-feature`).
4. Commit your changes (`git commit -m 'Add new feature'`).
5. Push to the branch (`git push origin feature/new-feature`).
6. Open a Pull Request.

---

## Author

Flavio Pineda
Email: fpineda@xanum.uam.mx
GitHub: [@fpineda94](https://github.com/fpineda94)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use primordialpy in your research, please cite:

```bibtex
@software{primordialpy2026,
  author = {Pineda Arvizu, Flavio Joao},
  title = {primordialpy: A Python library for inflationary dynamics, PBH abundance, and SIGW calculations},
  year = {2026},
  url = {https://github.com/fpineda94/primordialpy}
}
```

## Acknowledgments

This project was developed at UAM-Iztapalapa as part of research on inflationary cosmology and primordial black hole formation.

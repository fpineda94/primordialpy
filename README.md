# primordialpy

A Python library for computing inflationary dynamics, primordial perturbations and PBHs abundances from any single-field inflation model. 

## Description

`primordialpy` allows to the users analyze single-field inflationary models writing the inflationary potential as a simple text string expresion. The code automatically handles the following:  

- Solve the background dynamics
- Compute scalar and tensor primordial perturbations
- Determine and plot the primordial power spectrum 
- Estime the abundance of PBHs for inflationary models

## Instalation

### Requirements

- Python >= 3.9.6
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- sympy >= 1.8.0

### Instalation from GitHub repository

```bash
git clone https://github.com/fpineda94/primordialpy.git
cd primordialpy
pip install -e .
```

## Basic use

```python
from primordialpy.background import Background
from primordialpy.model import PotentialFunction
from primordialpy.perturbations import Perturbations
import matplotlib.pyplot as plt

# Define your potential as a string
# For example, chaoticinflation: V(φ) = (m²/2)φ²
V = '0.5*m**2*phi**2'
param = {'m': 5.9e-6} #Parameters of the model

# Initialize the model
potential = PotentialFunction.from_string(V, param_values= param)

# Background dynamics
bg = Background(potential, phi0 = 17.5)


# Compute perturbations
pert = Perturbations(potential, bg, scale= 'CMB', N_CMB = 60)

#Plot the power spectrum
pert.Plot_spectrum(dpi = 100, spectrum = 'scalar', save = True)



## Main features

- **Intuitive interface**: Define inflationary scenarios using mathematical expressions in text
- **Flexibility**: Compatible with any single-field inflation model
- **Complete calculations**: From the background dynamics to the formation of PBHs
- **Visualization**: Integrated tools for graphing results

## Project structure

```
primordialpy/
├── primordialpy/
│   ├── background.py        # Background
│   ├── model.py            # Definition of potentials
│   ├── perturbations.py    # Perturbation calculation
│   └── ...
├── examples/               # Notebooks with examples
├── constraints_data/       # Observational restriction data
├── paper/                  # Paper and documentation (in preparation)
├── pyproject.toml
├── setup.py
├── README.md
└── LICENSE
```

## Examples

Check out the `examples/` folder for Jupyter notebooks with detailed use cases, including:

- Standard inflationary models
- Potentials with features for primordial black hole (PBH) generation
- Comparison with observational data

## Contributions

Contributions are welcome! If you find a bug or have suggestions:

1. Open an [issue](https://github.com/fpineda94/primordialpy/issues)
2. Fork the repository
3. Create a branch for your feature (`git checkout -b feature/new-feature`)
4. Commit your changes (`git commit -m 'Add new feature'`)
5. Push to the branch (`git push origin feature/new-feature`)
6. Open a Pull Request

## Author

**Flavio Joao Pineda Arvizu**  
Email: fpineda@xanum.uam.mx  
GitHub: [@fpineda94](https://github.com/fpineda94)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use `primordialpy` in your research, please cite:

```bibtex
@software{primordialpy2025,
author = {Pineda Arvizu, Flavio Joao},
title = {primordialpy: A Python library for primordial power spectrum and PBH abundance calculations},
year = {2025},
url = {https://github.com/fpineda94/primordialpy}
}
```

## Acknowledgments

This project was developed at UAM-Iztapalapa as part of research on inflationary cosmology and primordial black hole formation.
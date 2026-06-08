
<img src="docs/assets/logo.png" width="400" height="400">

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-green.svg)
[![image](https://img.shields.io/pypi/v/greapy.svg)](https://pypi.python.org/pypi/greapy)
[![image](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/rcalderonb6/greapy/blob/main/LICENSE) 
[![](https://img.shields.io/badge/arXiv-2509.21491%20-red.svg)](https://arxiv.org/abs/2509.21491)
[![image](https://github.com/rcalderonb6/greapy/workflows/docs/badge.svg)](https://github.com/rcalderonb6/greapy/actions?query=workflow%3Adocs)


**A python-based implementation of the General Relativistic Entropic Acceleration (GREA) theory**

**G**eneral **R**elativistic **E**ntropic **A**cceleration is a theoretical framework that offers an alternative explanation for the observed accelerated expansion of the universe through entropic forces, without invoking a cosmological constant or dark energy. `GREApy` provides the core GREA background cosmology, growth factor computation, a Cobaya theory wrapper for Bayesian parameter estimation, and post-processing utilities for MCMC chains.

-   Free software: MIT License
-   Documentation: https://rcalderonb6.github.io/greapy
    

## Quick start

```python
import numpy as np
from greapy import GREA

# Initialise with default (best-fit) parameters
model = GREA()

# Hubble parameter as a function of redshift
z = np.linspace(0, 2, 100)
H_z = model.H(z)  # km/s/Mpc

# Key derived quantities
print(model.w0)       # effective dark energy equation-of-state today
print(model.rdrag)    # sound horizon at the drag epoch (Mpc)
print(model.alpha)    # GREA alpha parameter
```

## Installation

```bash
uv pip install greapy          # recommended
pip install greapy             # alternative
```

For optional plotting and Cobaya support:

```bash
uv pip install "greapy[extra]"
pip install "greapy[extra]"
```

See the [documentation](https://rcalderonb6.github.io/greapy) for full installation details.

## Citation

If you use `GREApy` in your research, please cite:

```bibtex
@article{Calderon:2025dhj,
    author = "Calderon, R. and others",
    title = "{Constraining GREA, an alternative theory accounting for the present cosmic acceleration}",
    eprint = "2509.21491",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "DESI-2024-0464, IFT-UAM/CSIC-25-99, FERMILAB-PUB-25-0737-PPD",
    month = "9",
    year = "2025"
}
```

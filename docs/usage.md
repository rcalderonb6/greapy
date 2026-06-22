# Usage

This guide shows how to use GREApy for cosmological calculations with the General Relativistic Entropic Acceleration (GREA) model.

## Basic Usage

### Creating a GREA Model

```python
from greapy import GREA

# Default parameters
cosmo = GREA()

# Custom parameters
cosmo = GREA(
    h=0.70,           # Dimensionless Hubble parameter
    omega_cdm=0.12,   # Physical cold dark matter density
    omega_b=0.022,    # Physical baryon density  
    kappa=3.5,        # GREA curvature scale parameter
    omega_g=2.47e-5,  # Physical photon density
    Neff=3.046        # Effective number of neutrino species
)
```

## Key Parameters

### GREA-Specific Parameters

- **`kappa`**: Curvature scale parameter $\sqrt{-k}\eta_0$. Controls the strength of entropic acceleration.
- **`alpha`**: Derived parameter $\alpha = \kappa / (H_0 \eta_0)$. Accessible as `cosmo.alpha`.

### Standard Cosmological Parameters

- **`h`**: Dimensionless Hubble parameter where $H_0 = 100h$ km/s/Mpc
- **`omega_cdm`**: Physical cold dark matter density $\Omega_{cdm} h^2$
- **`omega_b`**: Physical baryon density $\Omega_b h^2$
- **`omega_g`**: Physical photon density $\Omega_\gamma h^2$
- **`Neff`**: Effective number of neutrino species

## Basic Calculations

### Accessing Properties

```python
# Hubble parameter and densities
print(f"H0 = {cosmo.H0:.2f} km/s/Mpc")
print(f"Ωm = {cosmo.Omega_m:.4f}")
print(f"α = {cosmo.alpha:.4f}")
```

### Hubble Parameter Evolution

```python
import numpy as np

# Single redshift
H_z1 = cosmo.H(z=1.0)

# Array of redshifts
z = np.linspace(0, 2, 100)
H_z = cosmo.H(z)
```

### Distance Calculations

```python
z_target = 1.0

# Comoving distance
d_C = cosmo.comoving_distance(z_target)

# Luminosity distance  
d_L = cosmo.luminosity_distance(z_target)

# Angular diameter distance
d_A = cosmo.angular_diameter_distance(z_target)
```

### Sound Horizon

```python
# Sound horizon at recombination and drag epoch
rs_rec = cosmo.rs_rec
rdrag = cosmo.rdrag

# Angular scale
theta_star = cosmo.thetastar
```

### Equation of State

```python
# Scale factors
a = np.linspace(0.2, 1.0, 100)

# Equation of state parameter
w_eff = cosmo.w(a)

# Dark energy density evolution
f_de = cosmo.fde(a)
```

## Using with Cobaya

For parameter estimation:

```python
info = {
    'theory': {'greapy.cobaya.GREA': None},
    'likelihood': {'bao.desi_2024_bao_all': None},
    'params': {
        'h': {'prior': [0.6, 0.8]},
        'omega_cdm': {'prior': [0.1, 0.5]},
        'omega_b': {'prior': [0.01, 0.1]},
        'kappa': {'prior': [2.0, 5.0]},
        'Neff': 3.044
    }
}

from cobaya.run import run
updated_info, sampler = run(info)
```

## Command Line Interface

```bash
# Generate LaTeX tables from MCMC chains
greapy table -c "chain1 chain2" -p "H0 Omega_m kappa alpha" -o results.tex
```

## Plotting

```python
from greapy.plots import plot_distances

z = np.logspace(-2, 1, 100)
ax = plot_distances(z, cosmo)
```

For more detailed examples, see the [tutorial notebooks](examples/First Steps.ipynb).
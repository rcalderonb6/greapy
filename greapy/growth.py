"""Growth factor and structure formation utilities for the GREA model."""

import numpy as np
from scipy.integrate import odeint
from greapy.grea import GREA


def solve_growth(a, Omega_m, sigma8, h, hprime):
    """Solve the linear growth ODE and return $D$, $f$, and $f\sigma_8$.

    Integrates the second-order ODE for the linear density contrast $D(a)$
    in a given background cosmology specified by the Hubble function $h(a)$
    and its derivative $h'(a)$.

    Args:
        a: Array of scale factors at which to evaluate the solution. Need
            not be monotonically increasing — the array is reversed internally
            if necessary.
        Omega_m: Present-day matter density parameter $\Omega_m$.
        sigma8: Amplitude of matter fluctuations $\sigma_8$ used to normalize
            $f\sigma_8$.
        h: Callable $h(a) = H(a) / (100\,\mathrm{km\,s^{-1}\,Mpc^{-1}})$.
        hprime: Callable returning $dh/da$.

    Returns:
        Tuple `(D, f, fs8)` where:

        - `D` — linear growth factor array, shape `(len(a),)`.
        - `f` — growth rate $f = d\ln D / d\ln a$, shape `(len(a),)`.
        - `fs8` — $f\sigma_8$ product, shape `(len(a),)`.
    """
    from greapy.common import is_monotonic_increasing

    def ODE(vec, a):
        # System ODE
        x1, x2 = vec
        x1p = x2
        x2p = (
            3 / 2 * (a ** (-5) * Omega_m) / h(a) ** 2 * x1
            - (3 / a + (hprime(a) / h(a))) * x2
        )
        return np.array([x1p, x2p])

    is_increasing = is_monotonic_increasing(a, strict=True)

    if not is_increasing:
        a = a[::-1]

    # ODE solver parameters
    abserr = 1.0e-8
    relerr = 1.0e-8

    aini = a[0]
    y0 = [aini, 1]
    sols = odeint(ODE, y0, a, atol=abserr, rtol=relerr, h0=10 ** (-10))

    index = np.argmin(np.abs(a - 1))

    d, f, fs8 = (
        sols[:, 0],
        a / sols[:, 0] * sols[:, 1],
        sigma8 * a / (sols[:, 0][index]) * sols[:, 1],
    )
    if not is_increasing:
        return (d[::-1], f[::-1], fs8[::-1])

    return (d, f, fs8)


def gamma(a, f, cosmo: GREA):
    r"""Compute the growth index $\gamma$ from the growth rate $f$ and $\Omega_m(a)$.

    The growth index is defined implicitly by $f \approx \Omega_m(a)^\gamma$,
    so:

    $$\gamma(a) = \frac{\ln f}{\ln \Omega_m(a)}$$

    Args:
        a: Scale factor(s) at which to evaluate $\gamma$.
        f: Growth rate(s) $f = d\ln D / d\ln a$.
        cosmo: `GREA` cosmology object providing `Omega_m`, `Hubble`, and
            `H0` attributes.

    Returns:
        Growth index $\gamma$ evaluated at the given scale factor(s).

    Raises:
        ValueError: If $\Omega_m(a) \leq 0$, $f \leq 0$, or
            $\Omega_m(a) < 10^{-5}$ (to avoid numerical instabilities).
    """
    Omz = cosmo.Omega_m / a**3 / (cosmo.Hubble(a) / cosmo.H0) ** 2

    if np.any(Omz <= 0):
        raise ValueError("Omega_m must be positive for all z")
    if np.any(f <= 0):
        raise ValueError("f must be positive for all z")
    if np.any(Omz < 1e-5):
        raise ValueError(
            "Omega_m must be greater than 1e-5 for all z to avoid numerical issues"
        )

    return np.log(f) / np.log(Omz)


def analytical_D(a, Omega_m: float, normalize=False):
    r"""Compute the linear growth factor $D(a)$ analytically in a flat $\Lambda$CDM background.

    Uses the hypergeometric function solution:

    $$D(a) \propto a \,{}_2F_1\!\left(1, \tfrac{1}{3}; \tfrac{11}{6};\,
    \frac{\Omega_m - 1}{\Omega_m} a^3\right)$$

    Args:
        a: Scale factor(s) at which to evaluate $D$.
        Omega_m: Present-day matter density parameter.
        normalize: If `True`, normalize so that $D(a=1) = 1$.

    Returns:
        Linear growth factor $D(a)$, optionally normalized to unity today.
    """
    from scipy.special import hyp2f1

    x = (Omega_m - 1) / Omega_m
    norm = 1 / hyp2f1(1, 1 / 3, 11 / 6, x) if normalize else 1
    return a * hyp2f1(1, 1 / 3, 11 / 6, x * a**3) * norm


def analytical_Dprime(a, Omega_m: float):
    """Compute $dD/da$ analytically in a flat $\Lambda$CDM background.

    Args:
        a: Scale factor(s) at which to evaluate $D'$.
        Omega_m: Present-day matter density parameter.

    Returns:
        Derivative of the growth factor $D'(a) = dD/da$.
    """
    from scipy.special import hyp2f1

    Dprime = (
        a
        / hyp2f1(1 / 3, 1, 11 / 6, (Omega_m - 1) / Omega_m)
        * (
            hyp2f1(1 / 3, 1, 11 / 6, a**3 * (Omega_m - 1) / Omega_m)
            + 6
            * a**3
            * (Omega_m - 1)
            * hyp2f1(4 / 3, 2, 17 / 6, a**3 * (Omega_m - 1) / Omega_m)
            / (11 * Omega_m)
        )
    )
    return Dprime


def analytical_fsigma8(a, Omega_m, sigma8):
    """Compute $f\sigma_8(a)$ analytically in a flat $\Lambda$CDM background.

    Args:
        a: Scale factor(s) at which to evaluate $f\sigma_8$.
        Omega_m: Present-day matter density parameter.
        sigma8: Amplitude of matter fluctuations $\sigma_8$.

    Returns:
        $f\sigma_8(a) = \sigma_8 D'(a)$ where $D'$ is the normalized growth
        factor derivative.
    """
    Dprime = analytical_Dprime(a, Omega_m)
    return sigma8 * Dprime


def analytical_gamma(a, Omega_m):
    """Compute the growth index $\\gamma(a)$ analytically in a flat $\Lambda$CDM background.

    Args:
        a: Scale factor(s) at which to evaluate $\\gamma$.
        Omega_m: Present-day matter density parameter.

    Returns:
        Growth index $\\gamma(a) = \ln(f) / \ln(\Omega_m(a))$ where
        $\Omega_m(a) = \Omega_m / [\Omega_m + (1-\Omega_m)a^3]$.
    """
    f = analytical_Dprime(a, Omega_m) / analytical_D(a, Omega_m, normalize=True)
    Om = Omega_m / (Omega_m + (1 - Omega_m) * a**3)
    return np.log(f) / np.log(Om)

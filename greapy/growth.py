import numpy as np
from scipy.integrate import odeint
from greapy.grea import GREA


def solve_growth(a, Omega_m, sigma8, h, hprime):
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
    """
    Calculates the growth index gamma as the ratio of the logarithm of the growth rate f to the logarithm of the matter density parameter Omega_m(a).

    Parameters
    ----------
    a : float or array-like
        Scale factor(s) at which to evaluate gamma.
    f : float or array-like
        Growth rate(s), defined as f = d(ln(D))/d(ln(a)), where D is the linear density contrast.
    cosmo : GREA
        Cosmology object providing Omega_m, Hubble(a), and H0 attributes.

    Returns
    -------
    gamma : float or ndarray
        The growth index gamma evaluated at the given scale factor(s).

    Raises
    ------
    ValueError
        If Omega_m(a) is not positive, f is not positive, or Omega_m(a) is less than 1e-5 (to avoid numerical issues).
    """
    """Returns the growth rate gamma"""
    Omz = cosmo.Omega_m / a**3 / (cosmo.Hubble(a) / cosmo.H0) ** 2

    if np.any(Omz <= 0):
        raise ValueError("Omega_m must be positive for all z")
    if np.any(f <= 0):
        raise ValueError("f must be positive for all z")
    if np.any(Omz < 1e-5):
        raise ValueError(
            "Omega_m must be greater than 1e-5 for all z to avoid numerical issues"
        )

    # Calculate gamma as the ratio of the logarithm of f to the logarithm of Omega_m
    # where f is the growth rate f= D' / D = d(ln(D))/d(ln(a)) and Omz = Omega_m(a)
    return np.log(f) / np.log(Omz)


def analytical_D(a, Omega_m: float, normalize=False):
    """Returns the density contrast evolution as a function of a in a LCDM background"""
    from scipy.special import hyp2f1

    x = (Omega_m - 1) / Omega_m
    norm = 1 / hyp2f1(1, 1 / 3, 11 / 6, x) if normalize else 1
    return a * hyp2f1(1, 1 / 3, 11 / 6, x * a**3) * norm


def analytical_Dprime(a, Omega_m: float):
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
    Dprime = analytical_Dprime(a, Omega_m)
    return sigma8 * Dprime


def analytical_gamma(a, Omega_m):
    f = analytical_Dprime(a, Omega_m) / analytical_D(a, Omega_m, normalize=True)
    Om = Omega_m / (Omega_m + (1 - Omega_m) * a**3)
    return np.log(f) / np.log(Om)

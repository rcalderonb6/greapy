def zCMB(omega_m: float, omega_b: float) -> float:
    """Compute a numerical approximation for the CMB decoupling redshift.

    Args:
        omega_m: Physical matter density parameter $\omega_m = \Omega_m h^2$.
        omega_b: Physical baryon density parameter $\omega_b = \Omega_b h^2$.

    Returns:
        Estimated redshift at which CMB photon decoupling occurred.

    Note:
        This is an empirical fitting formula and may not be accurate for all
        cosmological parameter ranges.
    """
    return (omega_m)**(-0.731631) + omega_b**0.93681*(omega_m)**0.0192951 * (937.422/omega_b**0.97966 + 391.672/(omega_m)**0.372296)


def zdrag(omega_m: float, omega_b: float) -> float:
    """Compute the baryon drag redshift $z_d$ using a machine-learning fit.

    Implements Eq. (A2) from [arXiv:2106.00428](https://arxiv.org/abs/2106.00428).

    Args:
        omega_m: Physical matter density parameter $\omega_m = \Omega_m h^2$.
        omega_b: Physical baryon density parameter $\omega_b = \Omega_b h^2$.

    Returns:
        Drag redshift $z_d$, the redshift at which baryons decouple from
        Compton drag, relevant for baryon acoustic oscillation analyses.
    """
    num = 1 + 428.169 * omega_b**(0.256459) * omega_m**(0.616388) + 925.56*omega_m**(0.751615)
    den = omega_m**(0.714129)
    return num/den
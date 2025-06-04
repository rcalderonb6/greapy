def zCMB(omega_m: float, omega_b: float) -> float:
    """
    Calculates a numerical approximation for the CMB (Cosmic Microwave Background) decoupling redshift.

    Parameters:
        omega_m (float): The (physical) matter density  (Ω_m * h^2).
        omega_b (float): The (physical) baryon density (Ω_b * h^2).

    Returns:
        float: The estimated redshift at which CMB decoupling occurred.

    Notes:
        This is an empirical formula and may not be accurate for all cosmological parameter ranges.
    """
    '''
    Numerical approximation for the CMB decoupling redshift.
    '''
    return (omega_m)**(-0.731631) + omega_b**0.93681*(omega_m)**0.0192951 * (937.422/omega_b**0.97966 + 391.672/(omega_m)**0.372296)    

def zdrag(omega_m: float, omega_b: float) -> float:
    """
    Calculates the drag redshift (z_d) using a numerical (machine learning) fit.

    This function implements Eq. (A2) from the paper:
    https://arxiv.org/pdf/2106.00428.pdf

    Parameters
    ----------
    omega_m : float
        The physical matter density parameter (Ω_m*h^2).
    omega_b : float
        The physical baryon density parameter (Ω_b*h^2).

    Returns
    -------
    float
        The drag redshift z_d, a characteristic redshift relevant for baryon acoustic oscillations.

    References
    ----------
    - ArXiv: 2106.00428, Eq. (A2)
    """
    num = 1 + 428.169 * omega_b**(0.256459) * omega_m**(0.616388) + 925.56*omega_m**(0.751615)
    den = omega_m**(0.714129)
    return num/den
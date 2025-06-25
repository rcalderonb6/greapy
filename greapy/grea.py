"""
General Relativistic Entropic Acceleration (GREA) Theory Implementation.

This module provides a comprehensive implementation of the GREA cosmological model,
which describes cosmic acceleration as a consequence of entropic forces. The GREA
theory offers an alternative explanation to dark energy models for the accelerated
expansion of the universe.

The module contains functions and classes to calculate key cosmological parameters,
distances, and observables within the GREA framework, including Hubble parameter,
comoving distances, sound horizon, and equations of state.
"""

import numpy as np
from dataclasses import dataclass
from scipy.integrate import odeint, quad
from scipy.interpolate import UnivariateSpline

from greapy import approximations as approx

# Physical constants
C_KMS: float = 299792.458  # Speed of light in km/s

# Unit conversion factors
H_units_conv_factor: dict[str, float] = {
    "1/Mpc": 1 / C_KMS,  # Convert from km/s/Mpc to 1/Mpc
    "km/s/Mpc": 1,  # No conversion needed
}


@dataclass
class GREA:
    """
    General Relativistic Entropic Acceleration (GREA) model implementation.

    This class encapsulates the GREA cosmological model, providing methods to
    compute various cosmological quantities and observables within this theoretical
    framework. GREA proposes that cosmic acceleration arises from entropic forces
    rather than dark energy.

    Parameters
    ----------
    h : float, default=0.6736
        Dimensionless Hubble parameter (H0/(100 km/s/Mpc)).

    omega_cdm : float, default=0.12
        Physical cold dark matter density (Ωc x h²).

    omega_b : float, default=0.02237
        Physical baryon density parameter (Ωb x h²).

    kappa : float, default=3.55
        Curvature scale parameter in the GREA model, $\\kappa=\\sqrt{-k}\\eta_0$,

    omega_g : float, default=0.0000247739
        Physical photon density parameter (Ωγ x h²).

    Neff : float, default=3.044
        Effective number of neutrino species.

    a_min : float, default=1e-11
        Minimum scale factor for integration.

    Attributes
    ----------
    Numerous derived parameters and cosmological quantities accessible as properties,
    including equation of state parameters, sound horizons, and redshifts at key
    cosmic epochs.
    """

    h: float = 0.6736  # Dimensionless Hubble parameter
    omega_cdm: float = 0.12  # Fractional matter density (Ω_m) today (z=0)
    omega_b: float = 0.02237  # Physical baryon density (Ω_b * h^2)
    kappa: float = 3.55
    omega_g: float = 0.0000247739  # Physical density of photons (Ω_g * h^2)
    Neff: float = 3.044  # Effective number of neutrino species

    a_min: float = 1e-11  # Minimum scale factor for integration

    def __post_init__(self):
        """
        Initialize additional attributes after dataclass initialization.

        Creates a logarithmically spaced array of scale factors from a_min to 0.5
        for numerical integration and interpolation purposes.
        """
        self.a = np.logspace(np.log10(self.a_min), 0.5, 500)
        self.tau_spline = None  # Placeholder for tau spline interpolation
        self._require_update = (
            False  # Flag to indicate if parameters need to be updated
        )

    # def _display_(self):
    #     """Display the current cosmological parameters."""
    #     import marimo as mo

    #     return

    def horizon_distance(self, a):
        """
        Calculate dimensionless horizon distance at scale factor a.

        This function returns the dimensionless horizon distance needed to compute
        the alpha parameter in the GREA model. It's defined as:

        $$D_H(a) = a \\cdot H \\cdot \\eta(a)$$

        Parameters
        ----------
        a : float or array_like
            Scale factor(s) at which to evaluate the horizon distance.

        Returns
        -------
        float or array_like
            Dimensionless horizon distance at the specified scale factor(s).
        """
        return self.Hubble(a) * a * self.tau(a) / (100 * self.h)

    def _system(self, y, a, Omega_m, kappa, aeq):
        """
        Define the ODE system for tau evolution.

        This internal method defines the differential equation system for the
        evolution of tau with respect to scale factor in the GREA model.

        Parameters
        ----------
        y : float
            Current value of tau.
        a : float
            Scale factor.
        Omega_m : float
            Matter density parameter.
        kappa : float
            GREA model parameter.
        aeq : float
            Scale factor at matter-radiation equality.

        Returns
        -------
        float
            Derivative of tau with respect to scale factor.
        """
        den = np.sinh(2 * kappa) - 2 * kappa
        yprime = a**2 * np.sqrt(
            (Omega_m * (1 + aeq / a)) / a**3 + (4 * np.sinh(2 * y)) / (3.0 * a**2) / den
        )
        return 1 / yprime

    def tau(self, a):
        """
        Calculate tau parameter as a function of scale factor.

        Solves the differential equation for tau(a) and creates a spline
        interpolation for efficient evaluation at any scale factor.
        Tau represents a key dynamical variable in the GREA model.

        Parameters
        ----------
        a : float or array_like
            Scale factor(s) at which to evaluate tau.

        Returns
        -------
        float or array_like
            Value of tau at the given scale factor(s).

        Notes
        -----
        This method solves the ODE system defined in _system() and creates
        a spline interpolation for efficient evaluation at arbitrary scale factors.
        The initial condition is set at the minimum scale factor.
        """
        return self._tau(a)

    def _tau(self, a):
        if self.tau_spline is None or self._require_update:
            y0 = [self.a[0] / np.sqrt(self.Omega_g + self.Omega_nu)]
            theta = (self.Omega_bc, self.kappa, self.aeq)
            self.tau_spline = UnivariateSpline(
                self.a, odeint(self._system, y0, self.a, args=theta), s=0
            )
            self._require_update = False
        return self.tau_spline(a)

    def Hubble(self, a, units="km/s/Mpc"):
        r"""
        Calculate Hubble parameter at a given scale factor.

        Computes the Hubble parameter H(a) in the GREA model, including
        contributions from matter, radiation, and the entropic acceleration term.

        The Hubble parameter in GREA is given by:

        $$H(a) = H_0 \sqrt{\frac{\Omega_m(1 + a_{\text{eq}}/a)}{a^3} + \frac{4\sinh(2\tau(a))}{3a^2[\sinh(2k\eta_0) - 2k\eta_0]}}$$

        where $H_0 = 100h$ km/s/Mpc, $\tau(a)$ is the dimensionless horizon distance,
        and $\sqrt{-k}\eta_0$ is a model parameter.

        Parameters
        ----------
        a : float or array_like
            Scale factor(s) at which to evaluate the Hubble parameter.
        units : str, default='km/s/Mpc'
            Units for the output. Options are 'km/s/Mpc' or '1/Mpc'.

        Returns
        -------
        float or array_like
            Hubble parameter at the specified scale factor(s) in the requested units.
        """
        den = np.sinh(2 * self.kappa) - 2 * self.kappa
        E = (
            (self.Omega_bc * (1 + self.aeq / a)) / a**3
            + (4 * np.sinh(2 * self.tau(a))) / (3.0 * a**2) / den
        ) ** (0.5)
        return 100 * self.h * E * H_units_conv_factor[units]

    def H(self, z):
        """
        Calculate Hubble parameter as a function of redshift.

        A convenience method to get the Hubble parameter in terms of redshift z
        rather than scale factor a.

        Parameters
        ----------
        z : float or array_like
            Redshift(s) at which to evaluate the Hubble parameter.

        Returns
        -------
        float or array_like
            Hubble parameter at the specified redshift(s) in km/s/Mpc.

        Notes
        -----
        Converts redshift to scale factor via a = 1/(1+z) and calls Hubble(a).
        """
        return self.Hubble(1 / (1 + z))

    def angular_diameter_distance(self, z):
        r"""
        Calculate angular diameter distance to redshift z.

        The angular diameter distance is the ratio of an object's physical
        transverse size to its angular size in radians.

        Parameters
        ----------
        z : float or array_like
            Redshift(s) to calculate distance to.

        Returns
        -------
        float or array_like
            Angular diameter distance in Mpc.

        Notes
        -----
        Related to comoving distance via $d_A = d_C/(1+z)$.
        """
        return self.comoving_distance(z) / (1 + z)

    def comoving_distance(self, z):
        """
        Calculate comoving distance to redshift z.

        The comoving distance is the distance between two points measured along
        a path defined at the present cosmological time.

        Parameters
        ----------
        z : float or array_like
            Redshift(s) to calculate distance to.

        Returns
        -------
        float or array_like
            Comoving distance in Mpc.

        Notes
        -----
        In GREA, the comoving distance is proportional to the difference in tau
        values between today (a=1) and the target redshift.
        """
        dH = 1 / self.Hubble(1, units="1/Mpc")  # Normalized with correct H(z=0) in Mpc
        return (self.tau(1) - self.tau(1.0 / (1.0 + z))) * dH

    def cs(self, a):
        """
        Calculate the speed of sound in the photon-baryon fluid.

        Computes the speed of sound in the primordial photon-baryon fluid
        as a function of scale factor.

        Parameters
        ----------
        a : float or array_like
            Scale factor(s) at which to evaluate the sound speed.

        Returns
        -------
        float or array_like
            Speed of sound in km/s at the specified scale factor(s).

        Notes
        -----
        The sound speed depends on the baryon-to-photon ratio R = ωb/ωγ
        and decreases as the universe expands due to the increasing influence
        of baryons on the photon-baryon fluid.
        """
        # R is the baryon to photon ratio
        R = self.omega_b / self.omega_g
        return C_KMS * (3 * (1 + (3 / 4 * R * a))) ** (-0.5)

    def _rs_integrand(self, a):
        """
        Integrand for the sound horizon calculation.

        Internal method defining the integrand for computing the sound horizon.

        Parameters
        ----------
        a : float
            Scale factor.

        Returns
        -------
        float
            Value of the sound horizon integrand at scale factor a.
        """
        return self.cs(a) / (a**2 * self.Hubble(a))

    def rs(self, z):
        """
        Calculate the sound horizon at redshift z.

        The sound horizon is the maximum distance that acoustic waves could have
        traveled in the photon-baryon fluid up to a given redshift.

        Parameters
        ----------
        z : float
            Redshift at which to evaluate the sound horizon, typically z_rec
            (recombination) or z_drag (baryon drag epoch).

        Returns
        -------
        float
            Sound horizon distance in Mpc at the specified redshift.

        Notes
        -----
        Computed by integrating the sound speed divided by the expansion rate
        from early times to the specified redshift.
        """
        return quad(self._rs_integrand, self.a.min(), 1 / (1 + z))[0]

    def _alpha(self):
        """
        Calculate the alpha parameter of the GREA model.

        Alpha is a key dimensionless parameter in the GREA model that relates
        the curvature scale (kappa) to the horizon distance at the present time.

        Returns
        -------
        float
            The alpha parameter value.
        """
        return self.kappa / self.horizon_distance(1)

    def _fde(self, a):
        """
        Internal method to calculate the normalized dark energy density function.

        Computes the function that represents the evolution of dark energy density
        in the GREA model, normalized to its present-day value.

        Parameters
        ----------
        a : float or array_like
            Scale factor(s) at which to evaluate the function.

        Returns
        -------
        float or array_like
            Normalized dark energy density at the specified scale factor(s).
        """
        return (np.sinh(2 * self.tau(a)) / a**2) / np.sinh(2 * self.tau(1))

    def fde(self, a):
        """
        Calculate the normalized dark energy density evolution.

        This method returns the ratio of dark energy density at scale factor a
        to its present-day value, providing insight into how the effective dark
        energy evolves in the GREA model.

        Parameters
        ----------
        a : float or array_like
            Scale factor(s) at which to evaluate the dark energy density.

        Returns
        -------
        float or array_like
            Normalized dark energy density at the specified scale factor(s).
        """
        return self._fde(a)

    def w(self, a):
        """
        Calculate the effective equation of state parameter.

        Computes the effective equation of state parameter w(a) of dark energy
        in the GREA model at a given scale factor. The equation of state relates
        pressure to energy density via p = w*ρ.

        The equation of state parameter in GREA is given by:

        $$w(a) = -\\frac{1}{3} \\left( 1 + 2a \\coth(2\\tau(a)) \\tau'(a) \\right)$$

        where $\\tau'(a)$ is the derivative of $\\tau$ with respect to scale factor $a$.

        Parameters
        ----------
        a : float or array_like
            Scale factor(s) at which to evaluate w.

        Returns
        -------
        float or array_like
            Effective equation of state parameter w at the specified scale factor(s).

        Notes
        -----
        This differs from the constant w=-1 of a cosmological constant and allows
        for a dynamical dark energy component.
        """
        w = (
            1
            / 3
            * (-1 - 2 * a * coth(2 * self.tau(a)) * self.tau_spline.derivative()(a))
        )
        return w

    @property
    def aeq(self) -> float:
        """
        Scale factor at matter-radiation equality.

        Returns
        -------
        float
            The scale factor at which the energy densities of matter and radiation
            are equal. This is a key epoch in cosmic history.
        """
        return (self.Omega_g + self.Omega_nu) / self.Omega_bc

    @property
    def omega_bc(self) -> float:
        """
        Physical baryonic and cold dark matter density parameter.

        Returns
        -------
        float
            The physical cold dark matter density parameter ωcdm = Ωcdm * h².
            Calculated by subtracting baryonic density from total matter density.
        """
        return self.omega_b + self.omega_cdm

    @property
    def Omega_bc(self) -> float:
        """
        Fractional cold dark matter and baryonic density parameter.

        Returns
        -------
        float
            The fractional cold dark matter density parameter ωcb = (Ωcdm+Ωb) * h².
            Calculated by adding cold dark matter and baryonic densities.
        """
        return self.omega_bc / self.h**2

    @property
    def Omega_m(self) -> float:
        """
        Fractional matter density parameter today.

        Returns
        -------
        float
            The fractional cold dark matter density parameter ωcdm = Ωcdm * h².
            Calculated by subtracting baryonic density from total matter density.
        """
        return self.omega_bc / (self.H0 / 100) ** 2

    @property
    def Omega_g(self) -> float:
        """
        Photon density parameter.

        Returns
        -------
        float
            The fractional energy density of photons today (Ωγ = ωγ/h²).
        """
        return self.omega_g / self.h**2

    @property
    def Omega_nu(self) -> float:
        """
        Neutrino density parameter.

        Returns
        -------
        float
            The fractional energy density of neutrinos today (Ων).
            Calculated from the effective number of neutrino species Neff and
            the photon density, accounting for temperature and statistical differences.
        """
        return self.Neff * 7 / 8 * (4 / 11) ** (4 / 3) * self.Omega_g

    @property
    def alpha(self) -> float:
        """
        Alpha parameter of the GREA model.

        Returns
        -------
        float
            The dimensionless alpha parameter that characterizes the strength
            of the entropic acceleration mechanism in the GREA model.
        """
        return self._alpha()

    @property
    def w0(self) -> float:
        """
        Present-day equation of state parameter.

        Returns
        -------
        float
            The effective equation of state parameter w at the present time (a=1).
            This is commonly used in dark energy parametrizations.
        """
        return self.w(1)

    @property
    def wa(self) -> float:
        """
        Equation of state evolution parameter.

        Returns
        -------
        float
            The wa parameter in the CPL parametrization w(a) = w0 + wa(1-a).
            This captures the first-order evolution of the equation of state.

        Notes
        -----
        Calculated from the first and second derivatives of tau at a=1.
        """
        tau = self.tau(1)
        taup = self.tau_spline.derivative()(1)
        taupp = self.tau_spline.derivative(n=2)(1)
        factor = -4 * taup**2 + np.sinh(4 * tau) * (taup + taupp)
        return 1 / 3 * csch(2 * tau) ** 2 * factor

    @property
    def z_rec(self) -> float:
        """
        Redshift of recombination.

        Returns
        -------
        float
            The redshift of the last scattering surface (recombination),
            when photons decoupled from the baryon-photon plasma.
            Calculated using fitting formulae dependent on cosmological parameters.
        """
        return approx.zCMB(self.omega_bc, self.omega_b)

    @property
    def z_drag(self) -> float:
        """
        Redshift of the baryon drag epoch.

        Returns
        -------
        float
            The redshift of the baryon drag epoch, when baryons were released
            from the Compton drag of photons. This typically occurs at slightly
            lower redshift than recombination and is relevant for BAO measurements.
        """
        return approx.zdrag(self.omega_bc, self.omega_b)

    @property
    def rdrag(self) -> float:
        """
        Sound horizon at the baryon drag epoch.

        Returns
        -------
        float
            The comoving sound horizon at the baryon drag epoch in Mpc.
            This is the standard ruler used in BAO measurements.
        """
        return self.rs(self.z_drag)

    @property
    def rs_rec(self) -> float:
        """
        Sound horizon at recombination.

        Returns
        -------
        float
            The comoving sound horizon at recombination (last scattering) in Mpc.
            This determines the scale of the acoustic peaks in the CMB.
        """
        return self.rs(self.z_rec)

    @property
    def thetastar(self) -> float:
        """
        Angular scale of the sound horizon at recombination.

        Returns
        -------
        float
            The angular size of the sound horizon at recombination (θ*),
            which is precisely measured by the CMB and serves as a key
            cosmological observable for parameter constraints.
        """
        return self.rs_rec / self.comoving_distance(self.z_rec)

    @property
    def H0(self):
        return self.H(0)


def coth(x):
    """
    Hyperbolic cotangent function.

    Parameters
    ----------
    x : float or array_like
        Input value(s)

    Returns
    -------
    float or array_like
        Hyperbolic cotangent of x
    """
    return np.cosh(x) / np.sinh(x)


def csch(x):
    """
    Hyperbolic cosecant function.

    Parameters
    ----------
    x : float or array_like
        Input value(s)

    Returns
    -------
    float or array_like
        Hyperbolic cosecant of x
    """
    return 1 / np.sinh(x)

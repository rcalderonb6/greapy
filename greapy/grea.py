"""
General Relativistic Entropic Acceleration (GREA) Theory Implementation.

This module provides a comprehensive implementation of the GREA cosmological model,
which describes cosmic acceleration as a consequence of entropic forces [[1](https://arxiv.org/pdf/2106.16012),[2](https://arxiv.org/pdf/2106.16014)]. The GREA
theory offers an alternative explanation to dark energy models for the accelerated expansion of the universe.


The module contains functions and classes to calculate key cosmological parameters,
distances, and observables within the GREA framework, including Hubble parameter,
comoving distances, sound horizon, and equations of state. Throughout the code, we follow the formulation and notation from [[3](https://arxiv.org/pdf/2405.02895)]
"""

import numpy as np
from dataclasses import dataclass
from scipy.integrate import odeint, quad
from scipy.interpolate import UnivariateSpline

from greapy import approximations as approx
from greapy.common import C_KMS

# Unit conversion factors
H_units_conv_factor: dict[str, float] = {
    "1/Mpc": 1 / C_KMS,  # Convert from km/s/Mpc to 1/Mpc
    "km/s/Mpc": 1,  # No conversion needed
}


@dataclass
class GREA:
    """General Relativistic Entropic Acceleration (GREA).

    Encapsulates the GREA cosmological framework from [García-Bellido & Espinosa-Portales (2021)](https://arxiv.org/pdf/2106.16014), providing methods to compute
    cosmologically relevant quantities and observables.

    Args:
        h: Dimensionless Hubble parameter $h = H_0 / (100\,\mathrm{km\,s^{-1}\,Mpc^{-1}})$.
            Default is 0.6736.
        omega_cdm: Physical cold dark matter density $\omega_\mathrm{cdm} = \Omega_\mathrm{cdm} h^2$.
            Default is 0.12.
        omega_b: Physical baryon density $\omega_b = \Omega_b h^2$.
            Default is 0.02237.
        kappa: GREA curvature scale parameter $\kappa = \sqrt{-k}\,\eta_0$.
            Default is 3.55.
        omega_g: Physical photon density $\omega_g = \Omega_g h^2$.
            Default is 0.0000247739.
        Neff: Effective number of neutrino species. Default is 3.044.
        a_min: Minimum scale factor for numerical integration. Default is 1e-11.
    """

    h: float = 0.6736  # Dimensionless Hubble parameter
    omega_cdm: float = (
        0.12  # Physical cold dark matter density (omega_cdm = Omega_cdm * h^2)
    )
    omega_b: float = 0.02237  # Physical baryon density (omega_b = Omega_b * h^2)
    kappa: float = 3.55  # GREA curvature scale parameter (sqrt(-k) * eta_0)
    omega_g: float = 0.0000247739  # Physical photon density (omega_g = Omega_g * h^2)
    Neff: float = 3.044  # Effective number of neutrino species
    a_min: float = 1e-11  # Minimum scale factor for integration

    def __post_init__(self):
        """Initialize additional attributes after dataclass initialization.

        Creates a logarithmically spaced array of scale factors from $\\log_{10}(a_{\\rm min})$
        to $\\log_{10}(0.5)$ for numerical integration and interpolation.
        """
        self.a = np.logspace(np.log10(self.a_min), 0.5, 500)
        self.tau_spline = None  # Placeholder for tau spline interpolation
        self._require_update = (
            False  # Flag to indicate if parameters need to be updated
        )

    def horizon_distance(self, a):
        """Compute the dimensionless horizon distance at scale factor $a$.

        Returns the dimensionless combination $D_H(a) = a H(a) \eta(a)$,
        which enters the definition of the $\\alpha$ parameter in GREA.

        Args:
            a: Scale factor(s) at which to evaluate the horizon distance.

        Returns:
            Dimensionless horizon distance at the specified scale factor(s).
        """
        return self.Hubble(a) * a * self.tau(a) / (100 * self.h)

    def _system(self, y, a, Omega_m, kappa, aeq):
        """Define the ODE system for the $\\tau$ evolution.

        Args:
            y: Current value of $\\tau$.
            a: Scale factor.
            Omega_m: Matter density parameter $\\Omega_m$.
            kappa: GREA model parameter $\\kappa$.
            aeq: Scale factor at matter-radiation equality $a_\mathrm{eq}$.

        Returns:
            Derivative $d\\tau/da$ at the given scale factor.
        """
        den = np.sinh(2 * kappa) - 2 * kappa
        yprime = a**2 * np.sqrt(
            (Omega_m * (1 + aeq / a)) / a**3 + (4 * np.sinh(2 * y)) / (3.0 * a**2) / den
        )
        return 1 / yprime

    def tau(self, a):
        """Evaluate the GREA dynamical variable $\\tau(a)$.

        Solves the ODE for $\\tau(a)$ on first call and returns values from
        a cached spline interpolation on subsequent calls.

        Args:
            a: Scale factor(s) at which to evaluate $\\tau$.

        Returns:
            Value of $\\tau$ at the given scale factor(s).
        """
        return self._tau(a)

    def _build_tau_spline(self):
        y0 = [self.a[0] / np.sqrt(self.Omega_g + self.Omega_nu)]
        theta = (self.Omega_bc, self.kappa, self.aeq)
        self.tau_spline = UnivariateSpline(
            self.a, odeint(self._system, y0, self.a, args=theta), s=0
        )
        self._require_update = False

    def _get_tau_spline(self):
        if self.tau_spline is None or self._require_update:
            self._build_tau_spline()
        return self.tau_spline

    def _tau(self, a):
        return self._get_tau_spline()(a)

    def Hubble(self, a, units="km/s/Mpc"):
        r"""Compute the Hubble parameter $H(a)$ in the GREA model.

        The Hubble parameter includes contributions from matter, radiation,
        and the entropic acceleration term:

        $$H(a) = H_0 \sqrt{\frac{\Omega_m(1 + a_\mathrm{eq}/a)}{a^3}
        + \frac{4\sinh(2\tau(a))}{3a^2[\sinh(2\kappa) - 2\kappa]}}$$

        where $H_0 = 100h\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$ and $\tau(a)$
        is the GREA dynamical variable.

        Args:
            a: Scale factor(s) at which to evaluate $H$.
            units: Output units — `"km/s/Mpc"` (default) or `"1/Mpc"`.

        Returns:
            Hubble parameter at the specified scale factor(s) in the
            requested units.
        """
        den = np.sinh(2 * self.kappa) - 2 * self.kappa
        E = (
            (self.Omega_bc * (1 + self.aeq / a)) / a**3
            + (4 * np.sinh(2 * self.tau(a))) / (3.0 * a**2) / den
        ) ** (0.5)
        return 100 * self.h * E * H_units_conv_factor[units]

    def H(self, z):
        """Compute the Hubble parameter as a function of redshift.

        Convenience wrapper around `Hubble` that accepts redshift $z$
        instead of scale factor $a = 1/(1+z)$.

        Args:
            z: Redshift(s) at which to evaluate $H$.

        Returns:
            Hubble parameter at the specified redshift(s) in km/s/Mpc.
        """
        return self.Hubble(1 / (1 + z))

    def luminosity_distance(self, z):
        """Compute the luminosity distance to redshift $z$.

        Args:
            z: Redshift(s) to compute the distance to.

        Returns:
            Luminosity distance $D_L = D_C (1+z)$ in Mpc.
        """
        return self.comoving_distance(z) * (1 + z)

    def angular_diameter_distance(self, z):
        """Compute the angular diameter distance to redshift $z$.

        Args:
            z: Redshift(s) to compute the distance to.

        Returns:
            Angular diameter distance $D_A = D_C / (1+z)$ in Mpc.
        """
        return self.comoving_distance(z) / (1 + z)

    def comoving_distance(self, z):
        """Compute the comoving distance to redshift $z$.

        In GREA, the comoving distance is proportional to the difference
        in $\\tau$ values between today ($a=1$) and the target redshift:

        $$D_C(z) = \\frac{\\tau(1) - \\tau(1/(1+z))}{H(z=0)/c}$$

        Args:
            z: Redshift(s) to compute the distance to.

        Returns:
            Comoving distance in Mpc.
        """
        dH = 1 / self.Hubble(1, units="1/Mpc")  # Normalized with correct H(z=0) in Mpc
        return (self.tau(1) - self.tau(1.0 / (1.0 + z))) * dH

    def cs(self, a):
        """Compute the speed of sound in the photon-baryon fluid.

        Args:
            a: Scale factor(s) at which to evaluate the sound speed.

        Returns:
            Speed of sound in km/s at the specified scale factor(s).

        Note:
            The sound speed depends on the baryon-to-photon ratio
            $R = \omega_b / \omega_g$ and decreases as the universe expands
            due to the growing baryon contribution.
        """
        # R is the baryon to photon ratio
        R = self.omega_b / self.omega_g
        return C_KMS * (3 * (1 + (3 / 4 * R * a))) ** (-0.5)

    def _rs_integrand(self, a):
        """Integrand for the sound horizon integral.

        Args:
            a: Scale factor.

        Returns:
            Value $c_s(a) / [a^2 H(a)]$ at the given scale factor.
        """
        return self.cs(a) / (a**2 * self.Hubble(a))

    def rs(self, z):
        """Compute the comoving sound horizon at redshift $z$.

        The sound horizon is the maximum distance acoustic waves could
        have traveled in the photon-baryon fluid up to redshift $z$:

        $$r_s(z) = \int_0^{a(z)} \\frac{c_s(a)}{a^2 H(a)} da$$

        Args:
            z: Redshift at which to evaluate $r_s$. Typically the
                recombination redshift $z_\mathrm{rec}$ or the drag
                redshift $z_\mathrm{drag}$.

        Returns:
            Sound horizon distance in Mpc.
        """
        return quad(self._rs_integrand, self.a.min(), 1 / (1 + z))[0]

    def _alpha(self):
        """Compute the GREA $\\alpha$ parameter.

        Returns:
            Dimensionless ratio $\\alpha = \\kappa / D_H(a=1)$.
        """
        return self.kappa / self.horizon_distance(1)

    def _fde(self, a):
        r"""Compute the normalized dark energy density function (internal).

        Args:
            a: Scale factor(s) at which to evaluate the function.

        Returns:
            Ratio $f_\mathrm{de}(a) = [\sinh(2\tau(a))/a^2] / \sinh(2\tau(1))$.
        """
        return (np.sinh(2 * self.tau(a)) / a**2) / np.sinh(2 * self.tau(1))

    def fde(self, a):
        """Compute the normalized dark energy density evolution $f_\mathrm{de}(a)$.

        Returns the ratio of dark energy density at scale factor $a$ to its
        present-day value.

        Args:
            a: Scale factor(s) at which to evaluate $f_\mathrm{de}$.

        Returns:
            Normalized dark energy density at the specified scale factor(s).
        """
        return self._fde(a)

    def w(self, a):
        r"""Compute the effective dark energy equation of state $w(a)$.

        The equation of state in GREA is:

        $$w(a) = -\frac{1}{3}\left(1 + 2a\coth(2\tau(a))\,\tau'(a)\right)$$

        where $\tau'(a) = d\tau/da$.

        Args:
            a: Scale factor(s) at which to evaluate $w$.

        Returns:
            Effective equation of state parameter at the specified scale factor(s).

        Note:
            Unlike the cosmological constant ($w = -1$), this is generically
            dynamical and time-varying.
        """
        spline = self._get_tau_spline()
        w = 1 / 3 * (-1 - 2 * a * _coth(2 * self.tau(a)) * spline.derivative()(a))
        return w

    @property
    def aeq(self) -> float:
        r"""Scale factor at matter-radiation equality.

        Returns:
            Scale factor $a_\mathrm{eq} = (\Omega_g + \Omega_\nu) / \Omega_{bc}$
            at which matter and radiation energy densities are equal.
        """
        return (self.Omega_g + self.Omega_nu) / self.Omega_bc

    @property
    def omega_bc(self) -> float:
        """Physical baryon + cold dark matter density $\omega_{bc}$.

        Returns:
            Sum $\omega_{bc} = \omega_b + \omega_\mathrm{cdm}$.
        """
        return self.omega_b + self.omega_cdm

    @property
    def Omega_bc(self) -> float:
        """Fractional baryon + cold dark matter density $\Omega_{bc}$.

        Returns:
            Dimensionless density $\Omega_{bc} = \omega_{bc} / h^2$.
        """
        return self.omega_bc / self.h**2

    @property
    def Omega_m(self) -> float:
        r"""Fractional total matter density parameter today.

        $$\Omega_m = \frac{\omega_b + \omega_\mathrm{cdm}}{h^2}$$

        Returns:
            Total matter density parameter $\Omega_m$ at $z=0$.
        """
        return self.omega_bc / (self.H0 / 100) ** 2

    @property
    def Omega_g(self) -> float:
        """Fractional photon energy density $\Omega_g = \omega_g / h^2$.

        Returns:
            Dimensionless photon density parameter today.
        """
        return self.omega_g / self.h**2

    @property
    def Omega_nu(self) -> float:
        r"""Fractional neutrino energy density $\Omega_\nu$.

        Computed from the effective number of neutrino species $N_\mathrm{eff}$
        and the photon density, accounting for the neutrino temperature ratio
        $(4/11)^{1/3}$ and Fermi-Dirac statistics.

        Returns:
            Dimensionless neutrino density parameter today.
        """
        return self.Neff * 7 / 8 * (4 / 11) ** (4 / 3) * self.Omega_g

    @property
    def alpha(self) -> float:
        """GREA $\\alpha$ parameter.

        Returns:
            Dimensionless parameter $\\alpha = \\kappa / D_H(a=1)$ characterizing
            the strength of the entropic acceleration mechanism.
        """
        return self._alpha()

    @property
    def w0(self) -> float:
        """Present-day equation of state parameter $w_0 = w(a=1)$.

        Returns:
            Effective equation of state at redshift zero.
        """
        return self.w(1)

    @property
    def wa(self) -> float:
        """CPL equation of state evolution parameter $w_a$.

        Captures the first-order time variation of $w$ in the
        Chevallier-Polarski-Linder parametrization $w(a) = w_0 + w_a(1-a)$.

        Returns:
            $w_a$ evaluated from the first and second derivatives of $\\tau$ at $a=1$.
        """
        spline = self._get_tau_spline()
        tau = self.tau(1)
        taup = spline.derivative()(1)
        taupp = spline.derivative(n=2)(1)
        factor = -4 * taup**2 + np.sinh(4 * tau) * (taup + taupp)
        return 1 / 3 * _csch(2 * tau) ** 2 * factor

    @property
    def z_rec(self) -> float:
        """Redshift of photon decoupling (recombination).

        Returns:
            Redshift $z_\mathrm{rec}$ of the last scattering surface, computed
            from fitting formulae parameterized by $\omega_{bc}$ and $\omega_b$.
        """
        return approx.zCMB(self.omega_bc, self.omega_b)

    @property
    def z_drag(self) -> float:
        """Redshift of the baryon drag epoch.

        Returns:
            Drag redshift $z_\mathrm{drag}$, slightly lower than $z_\mathrm{rec}$,
            at which baryons decouple from Compton drag. This is the relevant
            epoch for BAO standard-ruler measurements.
        """
        return approx.zdrag(self.omega_bc, self.omega_b)

    @property
    def rdrag(self) -> float:
        """Comoving sound horizon at the baryon drag epoch.

        Returns:
            $r_\mathrm{drag} = r_s(z_\mathrm{drag})$ in Mpc — the standard
            ruler used in BAO analyses.
        """
        return self.rs(self.z_drag)

    @property
    def rs_rec(self) -> float:
        """Comoving sound horizon at recombination.

        Returns:
            $r_s(z_\mathrm{rec})$ in Mpc, which sets the angular scale of
            the acoustic peaks in the CMB power spectrum.
        """
        return self.rs(self.z_rec)

    @property
    def thetastar(self) -> float:
        """Angular scale of the sound horizon at recombination $\\theta_*$.

        Returns:
            $\\theta_* = r_s(z_\mathrm{rec}) / D_C(z_\mathrm{rec})$, a
            precisely measured CMB observable used to constrain cosmological
            parameters.
        """
        return self.rs_rec / self.comoving_distance(self.z_rec)

    @property
    def H0(self):
        """Hubble constant $H_0 = H(z=0)$ in km/s/Mpc.

        Returns:
            Present-day value of the Hubble parameter in km/s/Mpc.
        """
        return self.H(0)


def _coth(x):
    return np.cosh(x) / np.sinh(x)


def _csch(x):
    return 1 / np.sinh(x)

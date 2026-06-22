"""
Cobaya integration module for General Relativistic Entropic Acceleration (GREA) theory.

This module provides a Cobaya-compatible wrapper for the GREA cosmological model,
enabling parameter estimation and Bayesian inference using the Cobaya framework.
It also includes utilities for running MCMC and nested sampling analyses.
"""

import numpy as np
from cobaya.theory import Theory

from greapy.grea import GREA as BaseGREA
from greapy.grea import H_units_conv_factor


class GREA(Theory):
    """Cobaya theory wrapper for the GREA cosmological model.

    Provides a Cobaya-compatible interface to the GREA theory, enabling
    parameter estimation and likelihood evaluation within the Cobaya
    framework. Handles parameter dependencies and provides cosmological
    observables required by various likelihoods.

    Attributes:
        h: Dimensionless Hubble parameter $h = H_0 / (100\,\mathrm{km\,s^{-1}\,Mpc^{-1}})$.
            Default is 0.6736.
        omega_cdm: Physical cold dark matter density $\omega_\mathrm{cdm} = \Omega_c h^2$.
            Default is 0.12.
        omega_b: Physical baryon density $\omega_b = \Omega_b h^2$.
            Default is 0.02237.
        kappa: GREA curvature scale parameter $\kappa = \sqrt{-k}\,\eta_0$.
            Default is 3.55.
        Neff: Effective number of neutrino species. Default is 3.044.
    """

    h: float = 0.6736
    omega_cdm: float = 0.12
    omega_b: float = 0.02237
    kappa: float = 3.55
    Neff: float = 3.044

    def initialize(self):
        """Initialize the internal GREA cosmological model.

        Called by Cobaya during setup to create the `BaseGREA` instance
        used for all subsequent cosmological calculations.
        """
        self.cosmo = BaseGREA(
            h=self.h,
            omega_cdm=self.omega_cdm,
            omega_b=self.omega_b,
            kappa=self.kappa,
            Neff=self.Neff,
        )

    def initialize_with_provider(self, provider):
        """Store the Cobaya provider after all theory classes are initialized.

        Args:
            provider: Cobaya `Provider` instance used to retrieve parameter
                values and dependencies from other theory classes.
        """
        self.provider = provider

    def get_requirements(self):
        """Return the parameters this theory class requires from Cobaya.

        Returns:
            Dictionary mapping parameter names to `None` (indicating the raw
            parameter value is needed, not a derived quantity).
        """
        reqs = {
            "h": None,
            "omega_cdm": None,
            "omega_b": None,
            "kappa": None,
            "Neff": None,
        }
        return reqs

    def get_can_provide(self):
        """Return cosmological observables this theory can supply to likelihoods.

        Returns:
            List of observable names computable by this theory class.
        """
        return ["Hubble", "angular_diameter_distance"]

    def get_can_provide_params(self):
        """Return derived parameters this theory class can compute.

        Returns:
            List of derived parameter names. Includes: `alpha`, `H0`,
            `Omega_m`, `rdrag`, `rs_rec`, `ra_rec`, `z_rec`, `DAstar`,
            `rstar`, `zstar`, `thetastar`, `omegam`, `ombh2`, `omch2`,
            `w0`, `wa`, `theta_s_100`.
        """
        derived_params = [
            "alpha",
            "H0",
            "Omega_m",
            "rdrag",
            "rs_rec",
            "ra_rec",
            "z_rec",
            "DAstar",
            "rstar",
            "zstar",
            "thetastar",
            "omegam",
            "ombh2",
            "omch2",
            "w0",
            "wa",
            "theta_s_100",
        ]

        return derived_params

    def calculate(self, state, want_derived=True, **params_values_dict):
        """Compute cosmological observables and derived parameters for Cobaya.

        Called by Cobaya at each likelihood evaluation. Updates the internal
        GREA model with the current parameter values, then populates `state`
        with observables and derived parameters.

        Args:
            state: Cobaya state dictionary where results are stored.
            want_derived: Whether to compute and store derived parameters.
                Default is `True`.
            **params_values_dict: Additional parameter values passed by Cobaya
                (not directly used; parameters are fetched via the provider).
        """
        # Set the values of the Hubble constant, matter densities, etc
        for param in self.get_requirements():
            setattr(self.cosmo, param, self.provider.get_param(param))
        self.cosmo._require_update = True

        # rdrag = self.rs(self.zdrag)
        ra_rec = self.cosmo.angular_diameter_distance(self.cosmo.z_rec)

        state["Hubble"] = self.cosmo.Hubble
        state["angular_diameter_distance"] = self.cosmo.angular_diameter_distance
        state["rdrag"] = self.cosmo.rdrag

        # Store derived parameters
        state["derived"] = {
            "alpha": self.cosmo.alpha,
            # this is not H_0, but H(z=0)
            "H0": self.cosmo.H0,
            # Get the value of w0=w(z=0) and its derivative wa
            "w0": self.cosmo.w0,
            "wa": self.cosmo.wa,
            # These derived quantities are used for the CMB likelihood
            "rdrag": self.cosmo.rdrag,
            "rs_rec": self.cosmo.rs_rec,
            "DAstar": ra_rec * (1 + self.cosmo.z_rec) * 1e-3,
            "ra_rec": ra_rec,
            "z_rec": self.cosmo.z_rec,
            "rstar": self.cosmo.rs_rec,
            "zstar": self.cosmo.z_rec,
            "thetastar": self.cosmo.thetastar,
            # 'thetastar':rs_rec/(ra_rec * (1+z_rec)),
            "theta_s_100": 1e2 * self.cosmo.thetastar,
        }

        state["derived"]["Omega_m"] = self.cosmo.Omega_m
        state["derived"]["ombh2"] = self.cosmo.omega_b
        state["derived"]["omegam"] = self.cosmo.Omega_m
        state["derived"]["omch2"] = self.cosmo.omega_cdm

        # Keep a local reference for direct calls to get_Hubble/get_rdrag in tests
        # and non-Cobaya contexts where the framework cache is not driving state.
        self._current_state = state

    def get_angular_diameter_distance(self, z):
        """Compute angular diameter distance for given redshift(s).

        Args:
            z: Redshift(s) at which to evaluate the angular diameter distance.

        Returns:
            Angular diameter distance(s) in Mpc as a 1D array.
        """
        return np.atleast_1d(
            np.array(self.current_state["angular_diameter_distance"](z))
        )

    def get_Hubble(self, z, units="km/s/Mpc"):
        """Compute the Hubble parameter $H(z)$ for given redshift(s).

        Args:
            z: Redshift(s) at which to evaluate $H$.
            units: Output units — `"km/s/Mpc"` (default) or `"1/Mpc"`.

        Returns:
            Hubble parameter(s) in the specified units as a 1D array.
        """
        a = 1.0 / (1.0 + z)
        return np.atleast_1d(
            np.array(self.current_state["Hubble"](a) * H_units_conv_factor[units])
        )

    def get_rdrag(self):
        """Get the sound horizon at the baryon drag epoch.

        Returns:
            $r_\mathrm{drag}$ in Mpc.
        """
        return self.current_state["rdrag"]


def run_mcmc(
    likelihoods=None,
    model=None,
    priors="baseline",
    method: str = "MCMC",
    output=None,
    resume=True,
    debug=False,
    test=False,
    Rminus1=0.1,
    force=False,
    theory_kwargs=None,
):
    """Run Markov Chain Monte Carlo or Nested Sampling via Cobaya.

    Args:
        likelihoods: Comma-separated string of Cobaya likelihood names.
            If `None`, no likelihoods are included (useful for prior sampling
            or testing).
        model: Cosmological model identifier. Supported values:

            - `None` or `"lcdm"` — standard $\Lambda$CDM via CLASS.
            - `"greapy"` or `"grea"` — GREA model using this package.
            - `"w0wacdm"`, `"cpl"`, `"w0wa"` — $w_0 w_a$ CDM via CLASS.
        priors: Prior specification as a string identifier or a fully
            expanded Cobaya-style dictionary.
        method: Sampling method. Supported values:

            - `"MCMC"`, `"mh"`, `"metropolis-hastings"`, `"mcmc-mh"` — MCMC.
            - `"nested sampling"`, `"nested-sampling"`, `"ns"`, `"pc"`,
              `"polychord"` — nested sampling via PolyChord.
        output: Output directory for chains and results. If `None`, results
            are not written to disk.
        resume: Resume from existing chains if found. Default is `True`.
        debug: Enable Cobaya debug output. Default is `False`.
        test: Run in fast test mode (fewer samples). Default is `False`.
        Rminus1: Gelman-Rubin convergence threshold for MCMC. Sampling
            stops when $R-1 <$ `Rminus1` for all parameters. Default is 0.1.
        force: Overwrite existing output directory. Default is `False`.
        theory_kwargs: Extra keyword arguments forwarded to the theory class.

    Returns:
        Dictionary with keys:

        - `"updated_info"` — updated Cobaya configuration dictionary.
        - `"sampler_info"` — sampler result object.

    Raises:
        ValueError: If an unsupported `method` string is provided.

    Example:
        >>> results = run_mcmc(
        ...     likelihoods="bao.desi_2024_bao_all",
        ...     model="grea",
        ...     method="MCMC",
        ...     output="chains/grea_bao",
        ... )
    """
    from cobaya.run import run

    if model is None or model.lower() in ["lcdm"]:
        theory = {"classy": theory_kwargs}

    elif model.lower() in ["greapy", "grea"]:
        theory = {"greapy.cobaya.GREA": theory_kwargs}

    elif model.lower() in ["w0wacdm", "cpl", "w0wa"]:
        class_settings = {
            "extra_args": {"Omega_Lambda": 0, "Omega_scf": 0},
            **theory_kwargs,
        }
        theory = {"classy": class_settings}

    info = {"theory": theory}

    results = {}
    print(f"Sampling the posterior distribution with {method}:")

    # Handle sampling methods
    if method.lower() in ["mcmc", "mh", "metropolis-hastings", "mcmc-mh"]:
        info["sampler"] = {"mcmc": {"Rminus1_stop": Rminus1}}
    elif method.lower() in [
        "nested sampling",
        "nested-sampling",
        "ns",
        "pc",
        "polychord",
    ]:
        info["sampler"] = {"polychord": None}
    else:
        raise ValueError(
            f"Unknown method {method}. Supported methods are 'MCMC' and 'Nested Sampling'."
        )
    if likelihoods is not None:
        info["likelihood"] = {likelihood: None for likelihood in likelihoods.split(",")}

    if priors is not None:
        info.update(priors)

    info["output"] = output
    updated_info, sampler_info = run(
        info, resume=resume, debug=debug, test=test, force=force
    )

    results["updated_info"] = updated_info
    results["sampler_info"] = sampler_info

    return results

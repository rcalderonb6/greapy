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
    """
    Cobaya theory wrapper for the GREA cosmological model.

    This class provides a Cobaya-compatible interface to the GREA theory
    implementation, enabling parameter estimation and likelihood evaluation
    within the Cobaya framework. It handles parameter dependencies and
    provides cosmological observables required by various likelihoods.

    Attributes
    ----------
    h : float, default=0.6736
        Dimensionless Hubble parameter (H0/(100 km/s/Mpc)).
    omega_cdm : float, default=0.12
        Physical cold dark matter density (Ωc × h²).
    omega_b : float, default=0.02237
        Physical baryon density (Ωb × h²).
    kappa : float, default=3.55
        GREA curvature scale parameter (√(-k)η₀).
    Neff : float, default=3.044
        Effective number of neutrino species.
    """

    h: float = 0.6736
    omega_cdm: float = 0.12
    omega_b: float = 0.02237
    kappa: float = 3.55
    Neff: float = 3.044

    def initialize(self):
        """
        Initialize the GREA cosmological model.

        Called from __init__ to set up the internal GREA instance with
        the current parameter values. This creates the BaseGREA object
        that will be used for all cosmological calculations.
        """
        self.cosmo = BaseGREA(
            h=self.h,
            omega_cdm=self.omega_cdm,
            omega_b=self.omega_b,
            kappa=self.kappa,
            Neff=self.Neff,
        )

    def initialize_with_provider(self, provider):
        """
        Initialize with a provider instance after other components are set up.

        This method is called by Cobaya after other theory classes have been
        initialized. It stores the provider instance which will be used to
        retrieve parameter values during likelihood evaluation.

        Parameters
        ----------
        provider : cobaya.provider.Provider
            Cobaya provider instance used to access parameter values and
            dependencies from other theory classes.
        """
        self.provider = provider

    def get_requirements(self):
        """
        Return dictionary of parameters required by this theory class.

        This method defines which parameters the GREA theory needs to receive
        from the parameter space or other theory classes. These parameters
        will be automatically provided by Cobaya during likelihood evaluation.

        Returns
        -------
        dict
            Dictionary mapping parameter names to their requirements (None
            indicates the parameter value itself is needed).
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
        """
        Return list of observables that this theory class can compute.

        This method defines which cosmological observables the GREA theory
        can provide to likelihood functions. These observables will be
        available for use by various cosmological likelihoods.

        Returns
        -------
        list of str
            List of observable names that can be computed by this theory.
        """
        return ["Hubble", "angular_diameter_distance"]

    def get_can_provide_params(self):
        """
        Return list of derived parameters that this theory class can compute.

        This method defines which derived cosmological parameters the GREA
        theory can calculate and provide to the parameter space. These
        parameters can be used in likelihood evaluation or saved as outputs.

        Returns
        -------
        list of str
            List of derived parameter names that can be computed.

        Notes
        -----
        Derived parameters include:
        - alpha : GREA alpha parameter
        - H0 : Hubble constant today
        - Omega_m : Matter density parameter
        - rdrag : Sound horizon at baryon drag epoch
        - rs_rec, ra_rec, z_rec : Recombination quantities
        - DAstar, rstar, zstar, thetastar : CMB observables
        - omegam, ombh2, omch2 : Density parameters
        - w0, wa : Equation of state parameters
        - theta_s_100 : 100 × angular scale of sound horizon
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
        """
        Perform cosmological calculations for current parameter values.

        This is the main computation method called by Cobaya during likelihood
        evaluation. It updates the GREA model with current parameter values,
        computes observables and derived parameters, and stores results in
        the state dictionary.

        Parameters
        ----------
        state : dict
            Cobaya state dictionary where results will be stored.
        want_derived : bool, default=True
            Whether to compute and store derived parameters.
        **params_values_dict : dict
            Additional parameter values (not typically used).

        Notes
        -----
        The method updates the internal GREA model parameters, forces
        recomputation of derived quantities, and populates the state
        with observables and derived parameters needed by likelihoods.
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
        """
        Compute angular diameter distance for given redshift(s).

        Parameters
        ----------
        z : float or array_like
            Redshift(s) at which to evaluate the angular diameter distance.

        Returns
        -------
        numpy.ndarray
            Angular diameter distance(s) in Mpc. Always returns at least
            a 1D array even for scalar input.
        """
        return np.atleast_1d(
            np.array(self.current_state["angular_diameter_distance"](z))
        )

    def get_Hubble(self, z, units="km/s/Mpc"):
        """
        Compute Hubble parameter for given redshift(s).

        Parameters
        ----------
        z : float or array_like
            Redshift(s) at which to evaluate the Hubble parameter.
        units : str, default="km/s/Mpc"
            Units for the returned Hubble parameter. Supported units:
            - "km/s/Mpc" : kilometers per second per megaparsec
            - "1/Mpc" : inverse megaparsecs

        Returns
        -------
        numpy.ndarray
            Hubble parameter(s) in the specified units. Always returns
            at least a 1D array even for scalar input.
        """
        a = 1.0 / (1.0 + z)
        return np.atleast_1d(
            np.array(self.current_state["Hubble"](a) * H_units_conv_factor[units])
        )

    def get_rdrag(self):
        """
        Get the sound horizon at the baryon drag epoch.

        Returns
        -------
        float
            Sound horizon at baryon drag epoch in Mpc.
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
    """
    Run Markov Chain Monte Carlo or Nested Sampling using Cobaya.

    This function provides a convenient interface for running Bayesian
    parameter estimation with various cosmological models using the
    Cobaya framework. It supports both MCMC and nested sampling methods.

    Parameters
    ----------
    likelihoods : str or None, default=None
        Comma-separated string of likelihood names to use in the analysis.
        If None, no likelihoods are included (useful for testing).
    model : str or None, default=None
        Cosmological model to use. Supported options:
        - None or "lcdm" : Standard ΛCDM model using CLASS
        - "greapy" or "grea" : GREA model using this package
        - "w0wacdm", "cpl", "w0wa" : w0-wa CDM model using CLASS
    priors : str or dict, default="baseline"
        Prior specification. Can be a string identifier or a dictionary
        containing the full prior configuration.
    method : str, default="MCMC"
        Sampling method to use. Supported options:
        - "MCMC", "mh", "metropolis-hastings", "mcmc-mh" : MCMC sampling
        - "nested sampling", "nested-sampling", "ns", "pc", "polychord" : Nested sampling
    output : str or None, default=None
        Output directory path for chains and results. If None, results
        are not saved to disk.
    resume : bool, default=True
        Whether to resume from existing chains if found.
    debug : bool, default=False
        Enable debug mode for detailed output.
    test : bool, default=False
        Run in test mode (faster, less accurate).
    Rminus1 : float, default=0.1
        Convergence criterion for MCMC (R-1 statistic). Sampling stops
        when all parameters have R-1 < Rminus1.
    force : bool, default=False
        Force overwrite of existing output directory.
    theory_kwargs : dict or None, default=None
        Additional keyword arguments to pass to the theory class.

    Returns
    -------
    dict
        Dictionary containing the results with keys:
        - "updated_info" : Updated Cobaya configuration dictionary
        - "sampler_info" : Information about the sampling process

    Raises
    ------
    ValueError
        If an unsupported sampling method is specified.

    Examples
    --------
    >>> # Run MCMC with GREA model and BAO likelihood
    >>> results = run_mcmc(
    ...     likelihoods="bao.desi_2024_bao_all",
    ...     model="grea",
    ...     method="MCMC",
    ...     output="chains/grea_bao"
    ... )

    >>> # Run nested sampling with ΛCDM
    >>> results = run_mcmc(
    ...     likelihoods="bao.desi_2024_bao_all,sn.pantheon_plus",
    ...     model="lcdm",
    ...     method="nested sampling",
    ...     output="chains/lcdm_combined"
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

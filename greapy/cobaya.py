import numpy as np
from cobaya.theory import Theory

from greapy.grea import GREA as BaseGREA
from greapy.grea import H_units_conv_factor


class GREA(Theory):
    """Cobaya's wrapper for the GREA Theory class. Uses the python package greapy internally for cosmological calculations."""

    h: float = 0.6736
    omega_cdm: float = 0.12
    omega_b: float = 0.02237
    kappa: float = 3.55
    Neff: float = 3.044

    def initialize(self):
        """called from __init__ to initialize"""
        self.cosmo = BaseGREA(
            h=self.h,
            omega_cdm=self.omega_cdm,
            omega_b=self.omega_b,
            kappa=self.kappa,
            Neff=self.Neff,
        )

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """
        self.provider = provider

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
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
        Set the observables needed by the different likelihoods
        """
        return ["Hubble", "angular_diameter_distance"]

    def get_can_provide_params(self):
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
        # Set the values of the Hubble constant, matter density, etc
        self.cosmo.h = self.provider.get_param("h")
        self.cosmo.omega_cdm = self.provider.get_param("omega_cdm")
        self.cosmo.omega_b = self.provider.get_param("omega_b")
        self.cosmo.kappa = self.provider.get_param("kappa")
        self.cosmo.Neff = self.provider.get_param("Neff")

        self.cosmo._require_update = (
            True  # Force recomputation of tau and derived parameters
        )

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

    def get_angular_diameter_distance(self, z):
        return np.atleast_1d(
            np.array(self.current_state["angular_diameter_distance"](z))
        )

    def get_Hubble(self, z, units="km/s/Mpc"):
        a = 1.0 / (1.0 + z)
        return np.atleast_1d(
            np.array(self.current_state["Hubble"](a) * H_units_conv_factor[units])
        )

    def get_rdrag(self):
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
    """Run MCMC or Nested Sampling using Cobaya."""
    from cobaya.run import run

    if model is None or model.lower() in ["lcdm"]:
        theory = {"classy": theory_kwargs}

    elif model.lower() in ["greapy", "grea"]:
        theory = {"greapy.cobaya.GREA": theory_kwargs}

    elif model.lower() in ["w0wacdm", "cpl", "w0wa"]:
        w0wa_settings = {
            "extra_args": {"Omega_Lambda": 0, "Omega_scf": 0},
            **theory_kwargs,
        }
        theory = {"classy": w0wa_settings}

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
        info["likelihood"] = {l: None for l in likelihoods.split(",")}

    if priors is not None:
        info.update(priors)

    info["output"] = output
    updated_info, sampler_info = run(
        info, resume=resume, debug=debug, test=test, force=force
    )

    results["updated_info"] = updated_info
    results["sampler_info"] = sampler_info

    return results

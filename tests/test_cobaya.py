from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest import approx

cobaya = pytest.importorskip("cobaya", reason="cobaya not installed")

from greapy.cobaya import GREA as GREACobaya, run_mcmc


@pytest.fixture
def mock_provider():
    p = MagicMock()
    p.get_param.side_effect = lambda name: {
        "h": 0.6736,
        "omega_cdm": 0.12,
        "omega_b": 0.02237,
        "kappa": 3.55,
        "Neff": 3.044,
    }[name]
    return p


@pytest.fixture
def initialized_theory(mock_provider):
    theory = GREACobaya()
    theory.initialize()
    theory.initialize_with_provider(mock_provider)
    return theory


def test_initialize_runs_without_error():
    theory = GREACobaya()
    theory.initialize()


def test_get_requirements_keys():
    theory = GREACobaya()
    reqs = theory.get_requirements()
    assert set(reqs.keys()) == {"h", "omega_cdm", "omega_b", "kappa", "Neff"}


def test_get_can_provide():
    theory = GREACobaya()
    assert set(theory.get_can_provide()) == {"Hubble", "angular_diameter_distance"}


def test_calculate_populates_state(initialized_theory):
    state = {}
    initialized_theory.calculate(state, want_derived=True)
    assert callable(state["Hubble"])
    assert callable(state["angular_diameter_distance"])
    assert "rdrag" in state


def test_calculate_derived_params(initialized_theory):
    state = {}
    initialized_theory.calculate(state, want_derived=True)
    derived = state["derived"]
    for key in ("alpha", "H0", "w0", "wa", "rdrag", "thetastar", "Omega_m"):
        assert key in derived, f"Missing derived param: {key}"
        assert np.isfinite(derived[key]), f"Non-finite derived param: {key}"


def test_get_hubble_at_z0(initialized_theory):
    state = {}
    initialized_theory.calculate(state, want_derived=True)
    H0 = initialized_theory.get_Hubble(0.0)
    assert H0 == approx(state["derived"]["H0"], rel=1e-6)


def test_run_mcmc_raises_on_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        run_mcmc(method="gibbs_sampling", likelihoods=None)
